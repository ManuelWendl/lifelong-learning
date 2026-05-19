"""Two-stage backup policy training for humanoid.

Stage 1 (Walk):
  Train a standard humanoid walk policy with SAC for 2M steps.
  After every training epoch, a short rollout with the CURRENT policy collects
  simulator states (qpos, qvel).  Hooking at every epoch (not just the final
  policy) gives a diverse state distribution: early epochs contribute stumbling /
  falling states, late epochs contribute upright walking states.

  Saved artifacts:
    - "policy"  (type: model)  — walk policy checkpoint (normalizer + policy params)

Stage 2 (Backup):
  Train an upright-recovery policy initialised from the Stage-1 states.
  Reward is sparse: 0 at every step, 1 when entering set A (upright) + episode ends.
  With discounting=1.0 the optimal Q-function equals the probability of recovering
  upright from any starting state within the finite horizon H.

  Saved artifacts:
    - "backup_policy"  (type: model)      — full checkpoint
    - "q_reward"       (type: q_function) — Q_r params  (= P(upright) per state)
    - "q_cost"         (type: q_function) — Q_c params  (None when safe=False)

Usage:
  python train_backup.py +experiment=humanoid_backup_stage1
"""

import functools
import logging
from pathlib import Path

import cloudpickle
import hydra
import jax
import jax.nn as jnn
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import sac
from ss2r.algorithms.sac import networks as sac_networks
from ss2r.algorithms.sac import train as sac_train
from ss2r.benchmark_suites.mujoco_playground.humanoid import (  # registers HumanoidBackup
    backup_humanoid,
)
from ss2r.benchmark_suites.mujoco_playground.humanoid.backup_humanoid import (
    BackupHumanoidEnv,
    default_config as backup_default_config,
)
from ss2r.benchmark_suites.mujoco_playground import wrap_for_brax_training
from ss2r.common.logging import TrainingLogger
from ss2r.common.simulator_states import (
    EpochStateCollector,
    save_simulator_states,
)
from ss2r.common.wandb import get_state_path

_LOG = logging.getLogger(__name__)

# Indices into the params tuple returned by sac.train().
_IDX_NORMALIZER = 0
_IDX_POLICY = 1
_IDX_PENALIZER = 2
_IDX_QR = 3
_IDX_QC = 4


class Counter:
    def __init__(self):
        self.count = 0


def _report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


def _locate_last_checkpoint(base_path: str) -> Path | None:
    """Return the most recent checkpoint directory under base_path."""
    ckpt_dir = Path(base_path)
    if not ckpt_dir.exists():
        return None
    checkpoints = [
        p for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 12
    ]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: int(p.name))


def _save_q_functions(params, save_dir: Path) -> None:
    """Pickle reward and cost Q-function params into save_dir."""
    save_dir.mkdir(parents=True, exist_ok=True)
    qr_params = params[_IDX_QR]
    qc_params = params[_IDX_QC]  # None when safe=False

    with open(save_dir / "q_reward.pkl", "wb") as f:
        cloudpickle.dump(
            {
                "normalizer_params": params[_IDX_NORMALIZER],
                "qr_params": qr_params,
            },
            f,
        )
    with open(save_dir / "q_cost.pkl", "wb") as f:
        cloudpickle.dump(
            {
                "normalizer_params": params[_IDX_NORMALIZER],
                "qc_params": qc_params,  # None when no cost critic
            },
            f,
        )
    _LOG.info(
        "Saved Q-functions to %s  (q_cost is None: %s)", save_dir, qc_params is None
    )


# ---------------------------------------------------------------------------
# Stage 1: Walk training + in-training state collection
# ---------------------------------------------------------------------------

def _run_stage1(cfg, checkpoint_path: str, logger: TrainingLogger):
    """Run walk policy training with epoch-level state collection.

    Returns (make_policy, params, collector, steps).
    """
    _LOG.info("=== Stage 1: Walk policy training ===")
    train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(cfg)
    train_env, eval_env = benchmark_suites.make(cfg, train_env_wrap_fn, eval_env_wrap_fn)
    train_fn = sac.get_train_fn(
        cfg,
        checkpoint_path=checkpoint_path,
        restore_checkpoint_path=None,
    )

    num_evals_after_init = max(cfg.training.num_evals - 1, 1)

    def make_collector_hook(train_env, cfg, num_evals_after_init):
        """Returns a hook; EpochStateCollector is built lazily on first call."""
        state = {"collector": None}

        def hook(env_state, training_state):
            if state["collector"] is None:
                import jax.nn as jnn_local
                import ss2r.algorithms.sac.networks as nets

                activation = getattr(jnn_local, cfg.agent.activation)
                make_policy_fn = nets.make_inference_fn(
                    nets.make_sac_networks(
                        observation_size=train_env.observation_size,
                        action_size=train_env.action_size,
                        preprocess_observations_fn=lambda x, _: x,
                        policy_hidden_layer_sizes=tuple(
                            cfg.agent.policy_hidden_layer_sizes
                        ),
                        value_hidden_layer_sizes=tuple(
                            cfg.agent.value_hidden_layer_sizes
                        ),
                        activation=activation,
                        safe=False,
                        use_bro=cfg.agent.use_bro,
                        n_critics=cfg.agent.n_critics,
                    )
                )
                state["collector"] = EpochStateCollector(
                    env=train_env,
                    make_policy_fn=make_policy_fn,
                    n_total_states=cfg.backup.n_states,
                    num_evals=num_evals_after_init,
                    num_envs=cfg.training.num_envs,
                    seed=cfg.training.seed,
                )
            state["collector"].hook(env_state, training_state)

        return hook, state

    hook_fn, hook_state = make_collector_hook(train_env, cfg, num_evals_after_init)

    steps = Counter()
    make_policy, params, metrics = train_fn(
        environment=train_env,
        eval_env=eval_env,
        progress_fn=functools.partial(_report, logger, steps),
        env_state_hook=hook_fn,
    )
    _LOG.info("Stage 1 done. Final metrics: %s", metrics)

    # Upload walk policy checkpoint to WandB as "policy" artifact.
    if cfg.training.store_checkpoint:
        ckpt = _locate_last_checkpoint(checkpoint_path)
        if ckpt:
            logger.log_artifact(str(ckpt), type="model", name="policy")
            _LOG.info("Uploaded Stage 1 checkpoint as 'policy' artifact: %s", ckpt)

    return make_policy, params, hook_state["collector"]


# ---------------------------------------------------------------------------
# Stage 2: Backup policy training
# ---------------------------------------------------------------------------

def _build_backup_env(cfg, states_path: str):
    """Create the BackupHumanoidEnv wrapped for SAC training."""
    s2 = cfg.backup.stage2
    env_config = backup_default_config()
    env_config.episode_length = s2.episode_length
    env_config.simulator_states_path = states_path
    env_config.head_height_threshold = s2.head_height_threshold
    env_config.torso_upright_threshold = s2.torso_upright_threshold

    base_env = BackupHumanoidEnv(config=env_config)
    return wrap_for_brax_training(
        base_env,
        episode_length=s2.episode_length,
        action_repeat=1,
        randomization_fn=None,
        hard_resets=False,
        nonepisodic=False,
    )


def _run_stage2(cfg, states_path: str, checkpoint_path: str, logger: TrainingLogger):
    """Train the backup policy and upload all artifacts."""
    _LOG.info("=== Stage 2: Backup policy training ===")
    s2 = cfg.backup.stage2

    train_env = _build_backup_env(cfg, states_path)
    eval_env = _build_backup_env(cfg, states_path)

    activation = jnn.swish
    network_factory = functools.partial(
        sac_networks.make_sac_networks,
        policy_hidden_layer_sizes=tuple(s2.policy_hidden_layer_sizes),
        value_hidden_layer_sizes=tuple(s2.value_hidden_layer_sizes),
        activation=activation,
        value_obs_key="state",
        policy_obs_key="state",
    )

    steps = Counter()
    make_policy, params, metrics = sac_train.train(
        environment=train_env,
        eval_env=eval_env,
        num_timesteps=s2.num_timesteps,
        episode_length=s2.episode_length,
        action_repeat=1,
        num_envs=cfg.training.num_envs,
        num_eval_envs=cfg.training.num_eval_envs,
        num_eval_episodes=cfg.training.num_eval_episodes,
        learning_rate=s2.learning_rate,
        critic_learning_rate=s2.critic_learning_rate,
        alpha_learning_rate=s2.alpha_learning_rate,
        init_alpha=s2.init_alpha,
        discounting=s2.discounting,
        seed=cfg.training.seed,
        batch_size=s2.batch_size,
        num_evals=s2.num_evals,
        normalize_observations=s2.normalize_observations,
        reward_scaling=s2.reward_scaling,
        tau=s2.tau,
        min_replay_size=s2.min_replay_size,
        max_replay_size=s2.max_replay_size,
        grad_updates_per_step=s2.grad_updates_per_step,
        safe=False,
        use_bro=s2.use_bro,
        n_critics=s2.n_critics,
        network_factory=network_factory,
        checkpoint_logdir=checkpoint_path,
        progress_fn=functools.partial(_report, logger, steps),
    )
    _LOG.info("Stage 2 done. Final metrics: %s", metrics)

    if cfg.training.store_checkpoint:
        # 1. Upload full backup policy checkpoint.
        ckpt = _locate_last_checkpoint(checkpoint_path)
        if ckpt:
            logger.log_artifact(str(ckpt), type="model", name="backup_policy")
            _LOG.info("Uploaded Stage 2 checkpoint as 'backup_policy' artifact: %s", ckpt)

        # 2. Save and upload Q-functions separately.
        #    q_reward: Q_r params  — with discounting=1 this equals P(reach upright)
        #    q_cost:   Q_c params  — None here since safe=False; placeholder for future
        q_dir = Path(checkpoint_path) / "q_functions"
        _save_q_functions(params, q_dir)
        logger.log_artifact(str(q_dir), type="q_function", name="q_reward")
        logger.log_artifact(str(q_dir), type="q_function", name="q_cost")
        _LOG.info("Uploaded Q-function artifacts from %s", q_dir)

    return make_policy, params


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

@hydra.main(
    version_base=None,
    config_path="ss2r/configs",
    config_name="train_brax",
)
def main(cfg):
    _LOG.info(
        "Starting backup policy training with config:\n%s",
        OmegaConf.to_yaml(cfg),
    )

    # Separate checkpoint directories so Stage 1 and Stage 2 don't mix.
    ckpt_base = get_state_path()
    stage1_ckpt = ckpt_base + "/stage1"
    stage2_ckpt = ckpt_base + "/stage2"

    # One shared logger so both stages appear in the same WandB run.
    logger = TrainingLogger(cfg)

    # Stage 1: walk training with in-training state collection.
    with jax.disable_jit(not cfg.jit):
        make_policy, params, collector = _run_stage1(cfg, stage1_ckpt, logger)

    # Subsample to exactly n_states from the full cross-epoch buffer.
    _LOG.info("=== Finalising state buffer ===")
    states_path = cfg.backup.states_save_path
    qpos, qvel = collector.get_states(n_states=cfg.backup.n_states)
    save_simulator_states(qpos, qvel, states_path)

    # Stage 2: backup policy training initialised from the diverse state buffer.
    with jax.disable_jit(not cfg.jit):
        _run_stage2(cfg, states_path, stage2_ckpt, logger)

    _LOG.info("Backup policy training complete.")


if __name__ == "__main__":
    main()
