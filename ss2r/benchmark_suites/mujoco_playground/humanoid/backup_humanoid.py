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

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
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
        # Scale for dense shaping reward (head-height * torso-upright progress).
        # Set to 0.0 to use pure sparse indicator reward.
        dense_reward_scale=1.0,
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
        self._dense_reward_scale = float(config.dense_reward_scale)

        # Load saved (qpos, qvel) pairs only when buffer resets are needed.
        states_path = config.simulator_states_path
        if self._ground_start_probability > 0.0:
            if not states_path:
                raise ValueError(
                    "BackupHumanoidEnv requires 'simulator_states_path' in config "
                    "when ground_start_probability > 0. Run Stage 1 first, or set "
                    "ground_start_probability=0.0 to use the standing initialisation."
                )
            qpos_np, qvel_np = load_simulator_states(states_path)
            self._qpos_buffer = jp.asarray(qpos_np, dtype=jp.float32)  # (N, nq)
            self._qvel_buffer = jp.asarray(qvel_np, dtype=jp.float32)  # (N, nv)
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

        # Dense reward: parent standing reward (standing * upright * dont_move *
        # small_control). move_speed=0.0 in __init__ so dont_move replaces move.
        dense_reward = self._dense_reward_scale * self._get_reward(
            data, action, state.info, {}
        )

        # Reward: dense shaping (or 1.0 on entering A).
        reward = jp.where(in_A, 1.0, dense_reward)

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
