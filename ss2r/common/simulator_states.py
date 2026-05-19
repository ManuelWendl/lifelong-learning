"""Utilities for collecting and storing MuJoCo simulator states (qpos, qvel).

Used in backup policy training to accumulate (qpos, qvel) pairs from states
visited throughout Stage 1 (walk training), then use them as initial conditions
for Stage 2 (upright-recovery training).
"""

import functools
import logging

import jax
import jax.numpy as jnp
import numpy as np
from brax.training import acting

_LOG = logging.getLogger(__name__)


def save_simulator_states(qpos: np.ndarray, qvel: np.ndarray, path: str) -> None:
    """Save (qpos, qvel) pairs to a compressed numpy archive."""
    import pathlib
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, qpos=qpos, qvel=qvel)
    _LOG.info(
        f"Saved {qpos.shape[0]} simulator states to {path} "
        f"(qpos {qpos.shape}, qvel {qvel.shape})"
    )


def load_simulator_states(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (qpos, qvel) pairs from a compressed numpy archive."""
    data = np.load(path)
    qpos, qvel = data["qpos"], data["qvel"]
    _LOG.info(
        f"Loaded {qpos.shape[0]} simulator states from {path} "
        f"(qpos {qpos.shape}, qvel {qvel.shape})"
    )
    return qpos, qvel


class EpochStateCollector:
    """Collects diverse simulator states throughout SAC training.

    Designed for backup policy training: by hooking into each training epoch
    (via sac.train's env_state_hook), states are collected from EVERY stage of
    the walk policy's learning — early stumbling, mid-learning, and late walking.
    This gives a much richer distribution than a single post-training rollout.

    Mechanism:
        After each epoch, the hook runs a short rollout (rollout_steps × num_envs
        environment steps) using the CURRENT policy at that point in training.
        All visited (qpos, qvel) pairs are accumulated across all epochs.
        At the end, `get_states(n_states)` returns a random subsample.

    The rollout is compiled once via jax.lax.scan; subsequent epochs with
    different parameter values do NOT retrace (JAX traces on shape/dtype).

    Args:
        env: The wrapped training environment (same object passed to sac.train).
        make_policy_fn: Policy factory returned by sac.train at the end.
            Pass the one returned by training; it is also available via
            sac_networks.make_inference_fn(sac_network).
        n_total_states: Target total states to accumulate before subsampling.
        num_evals: Number of SAC training epochs (= num_evals_after_init in SAC).
            Used to compute rollout_steps so total ≈ n_total_states.
        num_envs: Parallel environments used in training.
        seed: RNG seed (use a different value from the training seed).
    """

    def __init__(
        self,
        env,
        make_policy_fn,
        n_total_states: int,
        num_evals: int,
        num_envs: int,
        seed: int,
    ):
        self._env = env
        self._make_policy_fn = make_policy_fn
        self._n_total_states = n_total_states
        # Number of rollout steps per epoch so that steps × num_envs ≈ n_total / num_evals.
        self._rollout_steps = max(1, -(-n_total_states // (num_evals * num_envs)))
        self._num_envs = num_envs
        self._rng = jax.random.PRNGKey(seed + 12345)

        self._qpos_chunks: list[np.ndarray] = []
        self._qvel_chunks: list[np.ndarray] = []
        self._epoch_count = 0

        # Build a JIT-compiled collection function once (traced on first call).
        self._collect_jit = self._make_collect_fn()

        _LOG.info(
            f"EpochStateCollector: {num_evals} epochs × {self._rollout_steps} steps "
            f"× {num_envs} envs = {num_evals * self._rollout_steps * num_envs} raw states, "
            f"target subsample = {n_total_states}."
        )

    def _make_collect_fn(self):
        """Returns a JIT-compiled fn(env_state, norm_params, policy_params, keys)."""
        env = self._env
        make_policy_fn = self._make_policy_fn

        def collect(env_state, normalizer_params, policy_params, keys):
            policy = make_policy_fn(
                (normalizer_params, policy_params), deterministic=False
            )

            def scan_fn(state, key):
                nstate, _ = acting.actor_step(env, state, policy, key, ())
                # The mujoco_playground wrappers preserve mjx_env.State,
                # so .data.qpos / .data.qvel are accessible directly.
                return nstate, (nstate.data.qpos, nstate.data.qvel)

            _, (qpos_seq, qvel_seq) = jax.lax.scan(scan_fn, env_state, keys)
            return qpos_seq, qvel_seq

        return jax.jit(collect)

    def hook(self, env_state, training_state) -> None:
        """Called by sac.train after each training epoch.

        Runs a short rollout with the current policy and appends all visited
        (qpos, qvel) pairs to the internal buffer.
        """
        self._epoch_count += 1
        self._rng, scan_key = jax.random.split(self._rng)
        keys = jax.random.split(scan_key, self._rollout_steps)

        qpos_seq, qvel_seq = self._collect_jit(
            env_state,
            training_state.normalizer_params,
            training_state.policy_params,
            keys,
        )
        # Transfer to CPU immediately so GPU memory is not accumulated.
        # qpos_seq: (rollout_steps, num_envs, nq)  →  flatten to (N, nq)
        qpos_np = jax.device_get(qpos_seq)
        qvel_np = jax.device_get(qvel_seq)
        n = self._rollout_steps * self._num_envs
        self._qpos_chunks.append(qpos_np.reshape(n, qpos_np.shape[-1]))
        self._qvel_chunks.append(qvel_np.reshape(n, qvel_np.shape[-1]))

        _LOG.info(
            f"[epoch {self._epoch_count}] Collected {n} states "
            f"(total so far: {self._epoch_count * n})"
        )

    def get_states(self, n_states: int | None = None, rng_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Concatenate and (optionally) subsample the collected states.

        Args:
            n_states: If given, randomly subsample down to this many states.
                Defaults to returning all collected states.
            rng_seed: Seed for the subsampling RNG.

        Returns:
            qpos: Array of shape (n_states, nq).
            qvel: Array of shape (n_states, nv).
        """
        qpos_all = np.concatenate(self._qpos_chunks, axis=0)
        qvel_all = np.concatenate(self._qvel_chunks, axis=0)
        total = qpos_all.shape[0]
        _LOG.info(f"EpochStateCollector: {total} total states collected across {self._epoch_count} epochs.")

        if n_states is not None and n_states < total:
            rng_np = np.random.default_rng(rng_seed)
            indices = rng_np.choice(total, size=n_states, replace=False)
            qpos_all = qpos_all[indices]
            qvel_all = qvel_all[indices]

        return qpos_all, qvel_all
