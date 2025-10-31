import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from torch.optim import AdamW

from lsy_rl.core.logger import Collector, CollectorList, EmptyLogger, LogCollector, Logger
from lsy_rl.core.replay_buffer import VectorReplayBuffer
from lsy_rl.core.transforms import IdentityTF, Transform
from lsy_rl.sac.policy import SACActor, SACCritic, SACPolicy
from lsy_rl.utils.utils import check_interrupt_sample, checkpoint, set_seeds, tensordict_sample


@torch.no_grad()
def evaluate_agent(
    policy: SACPolicy,
    envs: VectorEnv,
    n_steps: int,
    obs_tf: Transform,
    action_tf: Transform,
    collector: Collector,
    device: torch.device,
    seed: int | None = None,
) -> dict[str, float]:
    """Evaluate the policy on the evaluation environment and log the results."""
    obs, _ = envs.reset(seed=seed)
    policy.eval()
    collector.clear()
    autoreset = torch.zeros(envs.num_envs, dtype=bool, device=device)
    logs = []
    for _ in range(0, n_steps, envs.num_envs):
        action = action_tf(policy.actor.mean_action(obs_tf(obs)))
        next_obs, reward, terminated, truncated, info = envs.step(action)
        collector.collect(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            autoreset=autoreset,
        )
        obs = next_obs
        done = terminated | truncated
        if done.any():
            logs.append(collector.log(done))
        if autoreset.any():
            collector.clear(autoreset)
        autoreset = done
        obs = next_obs
    # Average over all metrics
    avg_log = defaultdict(list)
    for log in logs:
        for k, v in log.items():
            avg_log[k].append(v)
    avg_log = {k: sum(v) / len(v) for k, v in avg_log.items()}
    return avg_log


def sac(
    train_envs: VectorEnv,
    eval_envs: VectorEnv,
    n_steps: int,
    actor_lr: float,
    critic_lr: float,
    replay_buffer: VectorReplayBuffer | None = None,
    buffer_size: int = 1_000_000,
    eps: float = 1e-8,
    train_period: int = 1,
    train_steps: int = 1,
    actor_period: int = 1,
    critic_period: int = 1,
    target_period: int = 1,
    tau: float = 0.005,
    gamma: float = 0.99,
    grad_clip: float | None = None,
    batch_size: int = 256,
    eval_period: int | None = None,
    eval_steps: int = 1000,
    checkpoint_period: int | None = None,
    overwrite_policy: bool = True,
    alpha: float = 0.2,
    autotune_alpha: bool = False,
    alpha_lr: float = 3e-4,
    target_entropy: float | None = None,
    learning_starts: int = 0,
    policy: SACPolicy | None = None,
    obs_tf: Transform = IdentityTF(),
    action_tf: Transform = IdentityTF(),
    train_action_tf: Transform = IdentityTF(),
    eval_action_tf: Transform = IdentityTF(),
    checkpoint_path: Path | None = None,
    checkpoint_buffer: bool = False,
    logger: Logger = EmptyLogger(),
    device: torch.device = torch.device("cpu"),
    eval_collector: Collector | None = None,
    train_collector: Collector | None = None,
    rollout_collector: Collector | None = None,
    seed: int | None = None,
) -> SACPolicy:
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"
    # Move all transforms to the correct device
    obs_tf.to(device=device)
    action_tf.to(device=device)
    train_action_tf.to(device=device)
    eval_action_tf.to(device=device)

    if train_collector is None:
        train_collector = CollectorList()
        train_collector.append(LogCollector(target="actor_loss", log_key="train/actor_loss"))
        train_collector.append(LogCollector(target="critic_loss", log_key="train/critic_loss"))
        train_collector.append(LogCollector(target="alpha_loss", log_key="train/alpha_loss"))
    if rollout_collector is None:
        rollout_collector = CollectorList()
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/reward", reduce="sum")
        )
        rollout_collector.append(
            LogCollector(target="reward", log_key="rollout/step", reduce="cnt")
        )
    if eval_collector is None:
        eval_collector = CollectorList()
        eval_collector.append(LogCollector(target="reward", log_key="eval/reward", reduce="sum"))
        eval_collector.append(LogCollector(target="reward", log_key="eval/step", reduce="cnt"))

    if policy is None:
        obs_shape = train_envs.single_observation_space.shape
        action_shape = train_envs.single_action_space.shape
        actor = SACActor(obs_shape, action_shape)
        critic = SACCritic(obs_shape, action_shape)
        policy = SACPolicy(actor, critic)
    policy.to(device)

    critic_optim = AdamW(policy.critic.parameters(), lr=critic_lr, eps=eps)  # Targets are frozen
    actor_optim = AdamW(policy.actor.parameters(), lr=actor_lr, eps=eps)

    # Automatic entropy tuning
    if autotune_alpha:
        target_entropy = (
            -float(np.prod(train_envs.single_action_space.shape))
            if target_entropy is None
            else target_entropy
        )
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = AdamW([log_alpha], lr=alpha_lr)

    # Create replay buffer
    if replay_buffer is None:
        replay_buffer = VectorReplayBuffer(
            num_envs=train_envs.num_envs, max_size=buffer_size, device=device, seed=seed
        )

    # Create a partial function for checkpoint for more compact calls
    checkpoint_partial = partial(
        checkpoint,
        path=checkpoint_path,
        policy=policy,
        buffer=replay_buffer,
        critic_optimizer=critic_optim,
        actor_optimizer=actor_optim,
        obs_tf=obs_tf,
        checkpoint_buffer=checkpoint_buffer,
    )

    # Stats tracking setup
    n_train_steps = 0
    n_samples = 0
    last_train = 0
    last_eval = 0
    last_checkpoint = 0
    autoreset = torch.zeros(train_envs.num_envs, dtype=bool, device=device)

    # Establish an initial baseline
    log = evaluate_agent(
        policy, eval_envs, eval_steps, obs_tf, eval_action_tf, eval_collector, device, seed
    )
    logger.log(log, step=n_samples)
    if not overwrite_policy and checkpoint_path is not None:
        checkpoint_partial(step=n_samples, overwrite_policy=overwrite_policy)

    obs, _ = train_envs.reset(seed=seed)

    while n_samples < n_steps:
        # Log collection for each iteration. We store all logs and sort them at the end to ensure
        # chronological order, even when training and evaluation happen "before" the rollout. This
        # happens when the rollout step overshoots the required steps for the next training or eval
        # interval.
        logs = ()

        # Sample data
        obs_tf.update(obs)
        with torch.no_grad():
            action, _, _ = policy.actor.action(obs_tf(obs))
        action = action_tf(action)
        next_obs, reward, terminated, truncated, info = train_envs.step(action)
        rollout_collector.collect(
            obs=obs,
            action=action,
            next_obs=next_obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            autoreset=autoreset,
        )
        done = terminated | truncated

        # Vector environments automatically reset. This reset happens on the next step after done.
        # The reset step produces an inconsistent (obs, next_obs) tuple that has to be discarded. To
        # see how this is handled in gymnasium >= 1.0, see
        # https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0.
        mask = ~autoreset
        replay_buffer.add(
            tensordict_sample(
                obs, action, next_obs, reward, terminated, truncated, info, device=device
            )[mask],
            v_idx=torch.nonzero(mask).flatten(),
        )
        n_samples += mask.sum().item()

        done = terminated | truncated
        if done.any():
            logs += ((n_samples, rollout_collector.log(done)),)
        if autoreset.any():
            rollout_collector.clear(autoreset)

        autoreset = done
        obs = next_obs

        # Training.
        train_condition = check_interrupt_sample(
            n_samples, last_train, period=train_period, min_samples=learning_starts
        )
        if train_condition:
            tstart = time.perf_counter()
            policy.train()
            last_train = n_samples - (n_samples % train_period)
            for _ in range(train_steps):
                n_train_steps += 1

                if n_train_steps % critic_period == 0:
                    data = replay_buffer.sample(batch_size)
                    with torch.no_grad():
                        next_obs_t = obs_tf(data["next_obs"])
                        next_state_actions, next_state_log_pi, _ = policy.actor.action(next_obs_t)
                        next_state_actions = train_action_tf(next_state_actions)
                        min_qf_next_target = policy.critic.target(next_obs_t, next_state_actions)
                        min_qf_next_target -= alpha * next_state_log_pi
                        next_q_value = (
                            data["reward"].flatten()
                            + ~data["terminated"].flatten() * gamma * (min_qf_next_target).view(-1)
                        ).float()

                    obs_t = obs_tf(data["obs"])
                    q1_a_values, q2_a_values = policy.critic.values(obs_t, data["action"])
                    q1_a_values, q2_a_values = q1_a_values.view(-1), q2_a_values.view(-1)
                    q1_loss = F.mse_loss(q1_a_values, next_q_value)
                    q2_loss = F.mse_loss(q2_a_values, next_q_value)
                    qf_loss = q1_loss + q2_loss

                    # optimize the model
                    critic_optim.zero_grad()
                    qf_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), grad_clip)
                    critic_optim.step()
                    train_collector.collect(critic_loss=qf_loss.detach())

                if n_train_steps % actor_period == 0:
                    data = replay_buffer.sample(batch_size)
                    obs_t = obs_tf(data["obs"])
                    pi, log_pi, _ = policy.actor.action(obs_t)
                    pi = train_action_tf(pi)
                    min_qf_pi = policy.critic.actor_value(obs_t, pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    if grad_clip is not None:
                        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), grad_clip)
                    actor_optim.step()
                    train_collector.collect(actor_loss=actor_loss.detach())

                    if autotune_alpha:
                        with torch.no_grad():
                            _, log_pi, _ = policy.actor.action(obs_t)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha = log_alpha.exp().item()
                        train_collector.collect(alpha_loss=alpha_loss.detach())

                # Update the target networks
                if n_train_steps % target_period == 0:
                    policy.critic.update_target(tau)

            if log := train_collector.log():
                log["time/train"] = time.perf_counter() - tstart
                logs += ((last_train, log),)
                train_collector.clear()
            policy.eval()

        # Evaluate the agent
        eval_condition = check_interrupt_sample(n_samples, last_eval, period=eval_period)
        if eval_condition:
            tstart = time.perf_counter()
            last_eval = n_samples - (n_samples % eval_period)
            eval_seed = seed if seed is None else n_samples // eval_period
            log = evaluate_agent(
                policy,
                eval_envs,
                eval_steps,
                obs_tf,
                eval_action_tf,
                eval_collector,
                device,
                eval_seed,
            )
            log["time/eval"] = time.perf_counter() - tstart
            logs += ((last_eval, log),)

        # Save training checkpoint
        checkpoint_condition = check_interrupt_sample(
            n_samples, last_checkpoint, period=checkpoint_period
        )
        if checkpoint_condition and checkpoint_path is not None:
            tstart = time.perf_counter()
            last_checkpoint = n_samples - (n_samples % checkpoint_period)
            checkpoint_partial(step=n_samples, overwrite_policy=overwrite_policy)
            logs += ((last_checkpoint, {"time/checkpoint": time.perf_counter() - tstart}),)

        # Log all collected logs in chronological order
        for step, log in sorted(logs, key=lambda x: x[0]):
            logger.log(log, step=step)

    # Save final checkpoint
    if checkpoint_path is not None:
        checkpoint_partial(step=n_samples, overwrite_policy=True)

    logger.flush()
    return policy
