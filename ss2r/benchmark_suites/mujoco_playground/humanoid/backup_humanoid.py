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
        # Path to the .npz file written by save_simulator_states().
        simulator_states_path="",
        # Set A thresholds: must satisfy BOTH conditions to terminate with reward 1.
        head_height_threshold=1.2,   # matches humanoid._STAND_HEIGHT
        torso_upright_threshold=0.9,  # z-projection of torso orientation
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

        # Load saved (qpos, qvel) pairs and pin them as JAX arrays so they can
        # be indexed inside jit/vmap without retracing.
        states_path = config.simulator_states_path
        if not states_path:
            raise ValueError(
                "BackupHumanoidEnv requires 'simulator_states_path' in config. "
                "Run Stage 1 first to collect and save simulator states."
            )
        qpos_np, qvel_np = load_simulator_states(states_path)
        self._qpos_buffer = jp.asarray(qpos_np, dtype=jp.float32)  # (N, nq)
        self._qvel_buffer = jp.asarray(qvel_np, dtype=jp.float32)  # (N, nv)
        self._n_states = qpos_np.shape[0]

    # ------------------------------------------------------------------
    # Overridden environment interface
    # ------------------------------------------------------------------

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, idx_key = jax.random.split(rng)
        idx = jax.random.randint(idx_key, shape=(), minval=0, maxval=self._n_states)
        qpos = self._qpos_buffer[idx]   # (nq,)
        qvel = self._qvel_buffer[idx]   # (nv,)

        data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel)
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

        # Sparse reward: 1 iff in A (episode terminates immediately after).
        reward = in_A.astype(jp.float32)

        # Terminate on reaching A or on NaN (simulation instability).
        nans = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
        done = (in_A | nans).astype(jp.float32)

        cost = jp.zeros(())
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
