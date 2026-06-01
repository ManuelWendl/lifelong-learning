"""Humanoid backup policy environment.

Used in Stage 2 of backup policy training. The environment:
  - Initialises from a pre-collected buffer of MuJoCo simulator states (qpos, qvel)
    that were visited by a walk policy.
  - Gives a sparse reward: 0 at every step, 1 the moment the agent enters set A
    (head_height > head_height_threshold AND torso_upright > torso_upright_threshold),
    which also terminates the episode.
  - Runs for a finite horizon H (episode_length) with no discounting.

With discounting=1 in the SAC training config, the optimal Q-function satisfies
    Q*(s) = P(reach A within H steps from s)
which is exactly the probability-of-recovery value function.
"""

import logging
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco_playground._src import mjx_env
from mujoco_playground._src.dm_control_suite import humanoid

from ss2r.common.simulator_states import load_simulator_states


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,
        sim_dt=0.005,
        episode_length=200,
        action_repeat=1,
        vision=False,
        # Set A thresholds: must satisfy BOTH conditions to terminate with reward 1.
        head_height_threshold=1.6,  # full standing head≈1.69m (torso 1.5+0.19); 1.4 only requires ~53% up
        torso_upright_threshold=0.95,  # z-projection of torso orientation (~18° max lean, was 0.9/~26°)
        # Fraction of resets that sample from the fallen-state buffer (1.0 = always
        # use buffer, 0.0 = always use the default standing initialisation).
        ground_start_probability=1.0,
        simulator_states_path="",
        # Only keep buffer states where head height (computed at zero velocity)
        # exceeds this threshold. 65% of the raw buffer has head_h < 0.3 m —
        # fully-flat states from which recovery is impossible in 200 steps, so
        # the Q-function is constant there and gives no actor gradient.
        # Set to 0.0 to keep all states.
        min_head_height_filter=0.8,
    )


class BackupHumanoidEnv(humanoid.Humanoid):
    """Humanoid environment for backup (upright-recovery) policy training.

    Resets from saved simulator states and uses a sparse 0/1 reward that
    terminates on entering the upright set A.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        # move_speed is irrelevant for backup training; we pass 0.0 to satisfy
        # the parent constructor but never use the walk reward.
        super().__init__(
            move_speed=0.0,
            config=config,
            config_overrides=config_overrides,
        )

        self._head_height_threshold = float(config.head_height_threshold)
        self._torso_upright_threshold = float(config.torso_upright_threshold)
        self._ground_start_probability = float(config.ground_start_probability)
        self._min_head_height_filter = float(config.min_head_height_filter)

        # Load saved (qpos, qvel) pairs only when buffer resets are needed.
        states_path = config.simulator_states_path
        if self._ground_start_probability > 0.0:
            if not states_path:
                raise ValueError(
                    "BackupHumanoidEnv requires 'simulator_states_path' in config "
                    "when ground_start_probability > 0. Run Stage 1 first, or set "
                    "ground_start_probability=0.0 to use the standing initialisation."
                )
            qpos_np, _ = load_simulator_states(states_path)
            # Filter to states from which recovery is plausible.
            # Raw buffer: 65% have head_h < 0.3 m (humanoid fully flat, vel ≈ 44 rad/s).
            # Those states provide no gradient because Q ≈ const there.
            # We keep only states above min_head_height_filter.
            if self._min_head_height_filter > 0.0:
                mj = self.mj_model
                md = mujoco.MjData(mj)
                head_id = mj.body("head").id
                head_heights = []
                for qp in qpos_np:
                    md.qpos[:] = qp
                    md.qvel[:] = 0.0
                    mujoco.mj_kinematics(mj, md)
                    head_heights.append(float(md.xpos[head_id, 2]))
                mask = np.array(head_heights) > self._min_head_height_filter
                qpos_np = qpos_np[mask]
                logging.getLogger(__name__).info(
                    "BackupHumanoidEnv: kept %d / %d buffer states "
                    "(head_h > %.2f m).",
                    mask.sum(), len(mask), self._min_head_height_filter,
                )
            self._qpos_buffer = jp.asarray(qpos_np, dtype=jp.float32)
            # Zero velocities: raw buffer states captured mid-fall (vel_norm ≈ 44
            # rad/s) cause simulation instability.
            self._qvel_buffer = jp.zeros(
                (qpos_np.shape[0], self.mjx_model.nv), dtype=jp.float32
            )
            self._n_states = qpos_np.shape[0]

    # ------------------------------------------------------------------
    # Overridden environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, key_choice, idx_key = jax.random.split(rng, 3)

        def buffer_init(idx_key):
            idx = jax.random.randint(idx_key, shape=(), minval=0, maxval=self._n_states)
            return mjx_env.init(
                self.mjx_model,
                qpos=self._qpos_buffer[idx],
                qvel=self._qvel_buffer[idx],
            )

        def standing_init(_):
            return mjx_env.init(self.mjx_model)

        use_buffer = jax.random.uniform(key_choice) < self._ground_start_probability
        data = jax.lax.cond(use_buffer, buffer_init, standing_init, idx_key)
        info = {"rng": rng, "cost": jp.zeros(())}
        metrics = {
            "reward/in_upright_set": jp.zeros(()),
            "reward": jp.zeros(()),
            "cost": jp.zeros(()),
        }
        obs = self._get_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        data = mjx_env.step(self.mjx_model, state.data, action, self.n_substeps)
        obs = self._get_obs(data, state.info)

        # Check membership in set A.
        head_h = self._head_height(data)
        torso_u = self._torso_upright(data)
        in_A = (head_h > self._head_height_threshold) & (
            torso_u > self._torso_upright_threshold
        )

        # Dense reward: product of normalised head height and torso uprightness,
        # so both set-A conditions receive a gradient signal simultaneously.
        # At set A (head_h >= threshold AND torso_u >= threshold) reward = 1.0.
        head_component = jp.clip(head_h / self._head_height_threshold, 0.0, 1.0)
        torso_component = jp.clip(torso_u / self._torso_upright_threshold, 0.0, 1.0)
        reward = head_component * torso_component

        # Terminate on reaching A or on NaN (simulation instability).
        nans = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = (in_A | nans).astype(jp.float32)

        # Cost = indicator of set A membership: Q_c with safety_discounting=1
        # converges to P(reach A within H steps), usable as backup value function.
        cost = in_A.astype(jp.float32)
        metrics = {
            "reward/in_upright_set": in_A.astype(jp.float32),
            "reward": reward,
            "cost": cost,
        }
        info = {**state.info, "cost": cost}
        return mjx_env.State(data, obs, reward, done, metrics, info)


def _make_backup_env(config: config_dict.ConfigDict, **kwargs) -> BackupHumanoidEnv:
    return BackupHumanoidEnv(config=config, **kwargs)


# ---------------------------------------------------------------------------
# Registration – runs at import time so that registry.load("HumanoidBackup")
# resolves correctly after this module is imported in train_backup.py.
# ---------------------------------------------------------------------------
from mujoco_playground import dm_control_suite  # noqa: E402

dm_control_suite.register_environment(
    "HumanoidBackup",
    _make_backup_env,
    default_config,
)
