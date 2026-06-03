from typing import Tuple

import jax
import jax.numpy as jnp
from brax import envs
from brax.envs.wrappers.training import VmapWrapper
from brax.training import acting
from brax.training.types import Policy, PRNGKey

from ss2r.algorithms.mbpo.model_env import ModelBasedEnv
from ss2r.algorithms.mbpo.types import TrainingState, TrainingStepFn
from ss2r.algorithms.sac.types import (
    Metrics,
    ReplayBufferState,
    Transition,
    float16,
    float32,
)


def make_non_episodic_training_step(
    env,
    make_planning_policy,
    make_rollout_policy,
    get_rollout_policy_params,
    make_model_env,
    model_replay_buffer,
    sac_replay_buffer,
    alpha_update,
    critic_update,
    cost_critic_update,
    model_update,
    actor_update,
    safe,
    min_alpha,
    reward_q_transform,
    cost_q_transform,
    model_grad_updates_per_step,
    critic_grad_updates_per_step,
    extra_fields,
    get_experience_fn,
    env_steps_per_experience_call,
    tau,
    num_critic_updates_per_actor_update,
    unroll_length,
    num_model_rollouts,
    optimism,
    pessimism,
    model_to_real_data_ratio,
    safety_budget,
    qc_network,
    override_actions,
) -> TrainingStepFn:
    def critic_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_critic = jax.random.split(key)
        transitions = float32(transitions)
        if override_actions:
            behavior_action = transitions.extras["policy_extras"]["behavior_action"]
            transitions = transitions._replace(action=behavior_action)
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        critic_loss, behavior_qr_params, behavior_qr_optimizer_state = critic_update(
            training_state.behavior_qr_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_target_qr_params,
            alpha,
            transitions,
            key_critic,
            reward_q_transform,
            optimizer_state=training_state.behavior_qr_optimizer_state,
            params=training_state.behavior_qr_params,
        )
        if safe and "cost" in transitions.extras["state_extras"]:
            cost = transitions.extras["state_extras"]["cost"]
            new_discount = jnp.where(
                (cost > 0.0) & transitions.discount.astype(bool),
                jnp.ones_like(transitions.discount),
                jnp.zeros_like(transitions.discount),
            )
            backup_transitions = transitions._replace(discount=new_discount)
            (
                backup_cost_critic_loss,
                backup_qc_params,
                backup_qc_optimizer_state,
            ) = cost_critic_update(
                training_state.backup_qc_params,
                training_state.backup_policy_params,
                training_state.normalizer_params,
                training_state.backup_target_qc_params,
                alpha,
                backup_transitions,
                key_critic,
                cost_q_transform,
                True,
                optimizer_state=training_state.backup_qc_optimizer_state,
                params=training_state.backup_qc_params,
            )
            time_to_recovery = qc_network.apply(
                training_state.normalizer_params,
                backup_qc_params,
                transitions.observation,
                transitions.action,
            ).mean()
            cost_metrics = {
                "backup_cost_critic_loss": backup_cost_critic_loss,
                "time_to_recovery": time_to_recovery,
            }
        else:
            cost_metrics = {}
            backup_qc_params = training_state.backup_qc_params
            backup_qc_optimizer_state = training_state.backup_qc_optimizer_state

        polyak = lambda target, new: jax.tree_util.tree_map(
            lambda x, y: x * (1 - tau) + y * tau, target, new
        )
        new_behavior_target_qr_params = polyak(
            training_state.behavior_target_qr_params, behavior_qr_params
        )
        if safe:
            new_backup_target_qc_params = polyak(
                training_state.backup_target_qc_params, backup_qc_params
            )
        else:
            new_backup_target_qc_params = training_state.backup_target_qc_params
        metrics = {
            "critic_loss": critic_loss,
            "fraction_done": 1.0 - transitions.discount.mean(),
            **cost_metrics,
        }
        new_training_state = training_state.replace(  # type: ignore
            behavior_qr_optimizer_state=behavior_qr_optimizer_state,
            behavior_qr_params=behavior_qr_params,
            behavior_target_qr_params=new_behavior_target_qr_params,
            backup_qc_optimizer_state=backup_qc_optimizer_state,
            backup_qc_params=backup_qc_params,
            backup_target_qc_params=new_backup_target_qc_params,
            gradient_steps=training_state.gradient_steps + 1,
        )
        return (new_training_state, key), metrics

    def actor_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, key_alpha, key_actor = jax.random.split(key, 3)
        transitions = float32(transitions)
        if override_actions:
            behavior_action = transitions.extras["policy_extras"]["behavior_action"]
            transitions = transitions._replace(action=behavior_action)
        alpha_loss, alpha_params, alpha_optimizer_state = alpha_update(
            training_state.alpha_params,
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            transitions,
            key_alpha,
            optimizer_state=training_state.alpha_optimizer_state,
        )
        alpha = jnp.exp(training_state.alpha_params) + min_alpha
        (actor_loss, _), new_policy_params, new_policy_optimizer_state = actor_update(
            training_state.behavior_policy_params,
            training_state.normalizer_params,
            training_state.behavior_qr_params,
            training_state.behavior_qc_params,
            alpha,
            transitions,
            key_actor,
            safety_budget,
            None,
            None,
            optimizer_state=training_state.behavior_policy_optimizer_state,
            params=training_state.behavior_policy_params,
        )
        metrics = {
            "actor_loss": actor_loss,
            "alpha_loss": alpha_loss,
            "alpha": jnp.exp(alpha_params),
        }
        new_training_state = training_state.replace(  # type: ignore
            behavior_policy_optimizer_state=new_policy_optimizer_state,
            behavior_policy_params=new_policy_params,
            alpha_optimizer_state=alpha_optimizer_state,
            alpha_params=alpha_params,
        )
        return (new_training_state, key), metrics

    def model_sgd_step(
        carry: Tuple[TrainingState, PRNGKey], transitions: Transition
    ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
        training_state, key = carry
        key, _ = jax.random.split(key)
        transitions = float32(transitions)
        model_loss, model_params, model_optimizer_state = model_update(
            training_state.model_params,
            training_state.normalizer_params,
            transitions,
            optimizer_state=training_state.model_optimizer_state,  # type: ignore
            params=training_state.model_params,
        )
        new_training_state = training_state.replace(  # type: ignore
            model_optimizer_state=model_optimizer_state,
            model_params=model_params,
        )
        return (new_training_state, key), {"model_loss": model_loss}

    def run_experience_step(
        training_state: TrainingState,
        env_state: envs.State,
        buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
        experience_key, training_key = jax.random.split(key)
        normalizer_params, env_state, buffer_state = get_experience_fn(
            env,
            make_rollout_policy,
            get_rollout_policy_params(training_state),
            training_state.normalizer_params,
            model_replay_buffer,
            env_state,
            buffer_state,
            experience_key,
            extra_fields,
        )
        training_state = training_state.replace(  # type: ignore
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + env_steps_per_experience_call,
        )
        return training_state, env_state, buffer_state, training_key

    def generate_model_data(
        planning_env: ModelBasedEnv,
        policy: Policy,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> ReplayBufferState:
        keys = jax.random.split(key, num_model_rollouts + 2)
        key_generate_unroll = keys[1]
        rollout_keys = keys[2:]
        state = planning_env.reset(rollout_keys)
        _, transitions = acting.generate_unroll(
            planning_env,
            state,
            policy,
            key_generate_unroll,
            unroll_length,
            extra_fields=extra_fields,
        )
        transitions = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
        if override_actions:
            # The planning policy is the raw behavior policy (no safety filter), so
            # behavior_action == action. Replace policy_extras entirely to match the
            # replay buffer structure (initialized with dummy_transition that has the
            # full nonepisodic filter fields). Keeping extra keys like log_prob would
            # cause a pytree structure mismatch on insert, storing None for missing
            # fields and later causing len(None) when scanning over transitions.
            n = transitions.action.shape[0]
            transitions.extras["policy_extras"] = {
                "behavior_action": transitions.action,
                "intervention": jnp.zeros(n),
                "policy_distance": jnp.zeros(n),
                "safety_gap": jnp.zeros(n),
                "cumulative_cost": jnp.zeros(n),
                "expected_total_cost": jnp.zeros(n),
                "q_c": jnp.zeros(n),
            }
        sac_buffer_state = sac_replay_buffer.insert(
            sac_buffer_state, float16(transitions)
        )
        return sac_buffer_state

    def relabel_intervention_transitions(transitions: Transition) -> Transition:
        """Terminate and zero the reward wherever the backup policy intervened.

        Analogous to SOOPER relabeling: from the behavior policy's perspective,
        any step where it lost control to the backup is a terminal failure.
        discount=0 stops bootstrapping through the intervention boundary;
        reward=0 provides a lower bound signal (no reward for unsafe states).
        Model rollouts always have intervention=0, so they are unaffected.
        """
        intervention = transitions.extras["policy_extras"].get(
            "intervention", jnp.zeros_like(transitions.reward)
        )
        intervened = intervention > 0.5
        return transitions._replace(
            reward=jnp.where(intervened, jnp.zeros_like(transitions.reward), transitions.reward),
            discount=jnp.where(intervened, jnp.zeros_like(transitions.discount), transitions.discount),
        )

    def relabel_transitions(
        planning_env: ModelBasedEnv,
        transitions: Transition,
    ) -> Tuple[Transition, Metrics]:
        pred_fn = planning_env.model_network.apply
        model_params = planning_env.model_params
        normalizer_params = planning_env.normalizer_params
        vmap_pred_fn = jax.vmap(pred_fn, in_axes=(None, 0, None, None))
        next_obs_pred, reward, cost = vmap_pred_fn(
            normalizer_params, model_params, transitions.observation, transitions.action
        )
        disagreement = (
            next_obs_pred.std(axis=0).mean(-1)
            if isinstance(next_obs_pred, jax.Array)
            else next_obs_pred["state"].std(axis=0).mean(-1)
        )
        new_reward = reward.mean(0) + disagreement * optimism
        if safe:
            cost = cost.mean(0) + disagreement * pessimism
            transitions.extras["state_extras"]["cost"] = cost
        next_obs_pred = jax.tree.map(lambda x: x.mean(0), next_obs_pred)
        return Transition(
            observation=transitions.observation,
            next_observation=next_obs_pred,
            action=transitions.action,
            reward=new_reward,
            discount=transitions.discount,
            extras=transitions.extras,
        ), {"disagreement": disagreement}

    def training_step(
        training_state: TrainingState,
        env_state: envs.State,
        model_buffer_state: ReplayBufferState,
        sac_buffer_state: ReplayBufferState,
        key: PRNGKey,
    ) -> Tuple[
        TrainingState, envs.State, ReplayBufferState, ReplayBufferState, Metrics
    ]:
        # Keep original sac buffer so model-generated data is discarded between steps
        initial_sac_buffer_state = sac_buffer_state
        (
            training_state,
            env_state,
            model_buffer_state,
            training_key,
        ) = run_experience_step(training_state, env_state, model_buffer_state, key)
        # Train world model on real transitions
        model_buffer_state, transitions = model_replay_buffer.sample(model_buffer_state)
        tmp_transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (model_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        (training_state, _), model_metrics = jax.lax.scan(
            model_sgd_step, (training_state, training_key), tmp_transitions
        )
        # Generate model rollouts into sac buffer
        planning_env = make_model_env(
            training_state=training_state,
            transitions=transitions,
        )
        planning_env = VmapWrapper(planning_env)
        policy = make_planning_policy(
            (training_state.normalizer_params, training_state.behavior_policy_params)
        )
        sac_buffer_state = generate_model_data(
            planning_env, policy, sac_buffer_state, training_key
        )
        # Sample from sac buffer and mix with real data
        sac_buffer_state, model_transitions = sac_replay_buffer.sample(sac_buffer_state)
        num_real_transitions = int(
            model_transitions.reward.shape[0] * (1 - model_to_real_data_ratio)
        )
        assert (
            num_real_transitions <= transitions.reward.shape[0]
        ), "More model minibatches than real minibatches"
        if num_real_transitions >= 1:
            transitions = jax.tree_util.tree_map(
                lambda x, y: x.at[:num_real_transitions].set(y[:num_real_transitions]),
                model_transitions,
                transitions,
            )
        else:
            transitions = model_transitions
        transitions = jax.tree_util.tree_map(
            lambda x: jnp.reshape(x, (critic_grad_updates_per_step, -1) + x.shape[1:]),
            transitions,
        )
        transitions = relabel_intervention_transitions(transitions)
        transitions, more_metrics = relabel_transitions(planning_env, transitions)
        (training_state, _), critic_metrics = jax.lax.scan(
            critic_sgd_step, (training_state, training_key), transitions
        )
        num_actor_updates = -(
            -critic_grad_updates_per_step // num_critic_updates_per_actor_update
        )
        assert num_actor_updates > 0, "Actor updates is non-positive"
        transitions = jax.tree_util.tree_map(
            lambda x: x[:num_actor_updates], transitions
        )
        (training_state, _), actor_metrics = jax.lax.scan(
            actor_sgd_step,
            (training_state, training_key),
            transitions,
            length=num_actor_updates,
        )
        # Store backup_qc also as behavior_qc so it's saved correctly to checkpoint
        new_training_state = training_state.replace(  # type: ignore
            behavior_qc_params=training_state.backup_qc_params,
        )
        metrics = {**model_metrics, **critic_metrics, **actor_metrics}
        metrics["buffer_current_size"] = model_replay_buffer.size(model_buffer_state)
        metrics |= env_state.metrics
        metrics |= more_metrics
        return (
            new_training_state,
            env_state,
            model_buffer_state,
            initial_sac_buffer_state,
            metrics,
        )

    return training_step
