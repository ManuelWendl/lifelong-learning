import functools
import logging
from pathlib import Path

import hydra
import jax
import jax.nn as jnn
from brax import envs
from omegaconf import OmegaConf

from ss2r import benchmark_suites
from ss2r.algorithms import mbpo, ppo, sac, sbsrl
from ss2r.common.logging import TrainingLogger
from ss2r.common.wandb import get_state_path, get_wandb_checkpoint

_LOG = logging.getLogger(__name__)


def locate_last_checkpoint() -> Path | None:
    ckpt_dir = Path(get_state_path())
    # Get all directories or files that match the 12-digit pattern
    checkpoints = [
        p
        for p in ckpt_dir.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 12
    ]
    if not checkpoints:
        return None  # No checkpoints found
    # Sort by step number (converted from the directory name)
    latest_ckpt = max(checkpoints, key=lambda p: int(p.name))
    return latest_ckpt


def _validate_madrona_args(
    train_env: envs.Env,
    eval_env: envs.Env,
    num_envs: int,
    num_eval_envs: int,
    action_repeat: int,
    num_render_envs: int,
):
    """Validates arguments for Madrona-MJX."""
    if train_env != eval_env:
        raise ValueError("Madrona-MJX requires a fixed environment")
    if num_eval_envs != num_envs != num_render_envs:
        raise ValueError("Madrona-MJX requires a fixed batch size")
    if action_repeat != 1:
        raise ValueError(
            "Implement action_repeat using PipelineEnv's _n_frames to avoid"
            " unnecessary rendering!"
        )


def get_train_fn(cfg):
    if cfg.training.wandb_id:
        restore_checkpoint_path = get_wandb_checkpoint(
            cfg.training.wandb_id, cfg.wandb.entity
        )
    else:
        restore_checkpoint_path = None
    if cfg.agent.name == "sac":
        train_fn = sac.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "ppo":
        train_fn = ppo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "mbpo":
        train_fn = mbpo.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    elif cfg.agent.name == "sbsrl":
        train_fn = sbsrl.get_train_fn(
            cfg,
            restore_checkpoint_path=restore_checkpoint_path,
            checkpoint_path=get_state_path(),
        )
    else:
        raise ValueError(f"Unknown agent name: {cfg.agent.name}")
    return train_fn


def _maybe_make_state_collector_hook(cfg, train_env):
    """Returns (hook_fn, collector_state) if backup state collection is configured, else (None, None)."""
    backup_cfg = getattr(cfg, "backup", None)
    if cfg.agent.name != "sac" or backup_cfg is None or not backup_cfg.get("states_save_path", ""):
        return None, None

    import ss2r.algorithms.sac.networks as sac_networks
    from ss2r.common.simulator_states import EpochStateCollector

    num_evals_after_init = max(cfg.training.num_evals - 1, 1)
    collector_state = {"collector": None}

    def hook(env_state, training_state):
        if collector_state["collector"] is None:
            activation = getattr(jnn, cfg.agent.activation)
            policy_obs_key = "privileged_state" if cfg.training.get("policy_privileged", False) else "state"
            make_policy_fn = sac_networks.make_inference_fn(
                sac_networks.make_sac_networks(
                    observation_size=train_env.observation_size,
                    action_size=train_env.action_size,
                    policy_hidden_layer_sizes=tuple(cfg.agent.policy_hidden_layer_sizes),
                    value_hidden_layer_sizes=tuple(cfg.agent.value_hidden_layer_sizes),
                    activation=activation,
                    safe=False,
                    use_bro=cfg.agent.use_bro,
                    n_critics=cfg.agent.n_critics,
                    policy_obs_key=policy_obs_key,
                )
            )
            collector_state["collector"] = EpochStateCollector(
                env=train_env,
                make_policy_fn=make_policy_fn,
                n_total_states=backup_cfg.n_states,
                num_evals=num_evals_after_init,
                num_envs=cfg.training.num_envs,
                seed=cfg.training.seed,
            )
        collector_state["collector"].hook(env_state, training_state)

    return hook, collector_state


class Counter:
    def __init__(self):
        self.count = 0


def report(logger, step, num_steps, metrics):
    metrics = {k: float(v) for k, v in metrics.items()}
    logger.log(metrics, num_steps)
    step.count = num_steps


@hydra.main(version_base=None, config_path="ss2r/configs", config_name="train_brax")
def main(cfg):
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    logger = TrainingLogger(cfg)
    train_fn = get_train_fn(cfg)
    train_env_wrap_fn, eval_env_wrap_fn = benchmark_suites.get_wrap_env_fn(cfg)
    use_vision = "use_vision" in cfg.agent and cfg.agent.use_vision
    train_env, eval_env = benchmark_suites.make(
        cfg, train_env_wrap_fn, eval_env_wrap_fn
    )
    if use_vision:
        _validate_madrona_args(
            train_env,
            eval_env,
            cfg.training.num_envs,
            cfg.training.num_eval_envs,
            cfg.training.action_repeat,
            cfg.environment.task_params.vision_config.render_batch_size,
        )
    hook, collector_state = _maybe_make_state_collector_hook(cfg, train_env)
    steps = Counter()
    with jax.disable_jit(not cfg.jit):
        make_policy, params, _ = train_fn(
            environment=train_env,
            eval_env=eval_env,
            progress_fn=functools.partial(report, logger, steps),
            **({"env_state_hook": hook} if hook is not None else {}),
        )
        if cfg.training.render:
            rng = jax.random.split(
                jax.random.PRNGKey(cfg.training.seed), cfg.training.num_eval_envs
            )
            if len(params) != 2:
                policy_params = params[:2]
            else:
                policy_params = params
            video = benchmark_suites.render_fns[cfg.environment.task_name](
                eval_env,
                make_policy(policy_params, deterministic=True),
                cfg.training.episode_length,
                rng,
            )
            fps = (
                1 / cfg.environment.task_params.ctrl_dt
                if "task_params" in cfg.environment
                and "ctrl_dt" in cfg.environment.task_params
                else 30.0
            )
            logger.log_video(
                video,
                steps.count,
                "eval/video",
                fps,
            )
        if cfg.training.store_checkpoint:
            artifacts = locate_last_checkpoint()
            if artifacts:
                logger.log_artifact(artifacts, "model", "checkpoint")
    if collector_state is not None and collector_state["collector"] is not None:
        from ss2r.common.simulator_states import save_simulator_states
        collector = collector_state["collector"]
        qpos, qvel = collector.get_states(n_states=cfg.backup.n_states)
        save_simulator_states(qpos, qvel, cfg.backup.states_save_path)
        _LOG.info("Saved simulator states to %s", cfg.backup.states_save_path)
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
