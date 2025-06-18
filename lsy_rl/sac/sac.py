import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from tensordict import TensorDict
from torch.optim import AdamW

from lsy_rl.core.logger import Collector, CollectorList, EmptyLogger, LogCollector, Logger
from lsy_rl.core.replay_buffer import VectorReplayBuffer
from lsy_rl.sac.policy import SACActor, SACCritic, SACPolicy
from lsy_rl.utils import polyak_update_
from lsy_rl.utils.utils import set_seeds, sync_env_normalization


@torch.no_grad()
def evaluate_agent(
    policy: SACPolicy, envs: VectorEnv, n_steps: int, collector: Collector, device: torch.device
) -> dict[str, float]:
    obs, _ = envs.reset()
    policy.eval()
    collector.clear()
    autoreset = torch.zeros(envs.num_envs, dtype=bool, device=device)
    logs = []
    for _ in range(0, n_steps, envs.num_envs):
        action = policy.actor.mean_action(obs)
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
    actor_period: int = 1,
    critic_period: int = 1,
    target_period: int = 1,
    tau: float = 0.005,
    gamma: float = 0.99,
    batch_size: int = 256,
    eval_period: int | None = None,
    eval_steps: int = 1000,
    alpha: float = 0.2,
    autotune_alpha: bool = False,
    alpha_lr: float = 3e-4,
    learning_starts: int = 0,
    policy: SACPolicy | None = None,
    logger: Logger = EmptyLogger(),
    device: torch.device = torch.device("cpu"),
    eval_collector: Collector | None = None,
    train_collector: Collector | None = None,
    rollout_collector: Collector | None = None,
    seed: int | None = None,
) -> SACPolicy:
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"

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

    q1, q2 = policy.critic.q1, policy.critic.q2
    critic_optim = AdamW(policy.critic.parameters(), lr=critic_lr)  # Targets are frozen
    q1_target, q2_target = policy.critic.q1_target, policy.critic.q2_target
    actor_optim = AdamW(policy.actor.parameters(), lr=actor_lr)

    # Automatic entropy tuning
    if autotune_alpha:
        target_entropy = -float(np.prod(train_envs.single_action_space.shape))
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        alpha_optim = AdamW([log_alpha], lr=alpha_lr)

    # Create replay buffer
    if replay_buffer is None:
        replay_buffer = VectorReplayBuffer(
            num_envs=train_envs.num_envs, max_size=buffer_size, device=device, seed=seed
        )

    # Stats tracking setup
    n_samples = 0
    last_train_actor, last_train_critic, last_target = 0, 0, 0
    last_eval = 0
    autoreset = torch.zeros(train_envs.num_envs, dtype=bool, device=device)

    obs, _ = train_envs.reset(seed=seed)

    while n_samples < n_steps:
        # Sample data
        policy.train()
        with torch.no_grad():
            action, _, _ = policy.actor.action(obs)
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

        mask = ~autoreset
        replay_buffer.add(
            TensorDict(
                {
                    "obs": obs[mask],
                    "action": action[mask],
                    "next_obs": next_obs[mask],
                    "reward": reward[mask],
                    "terminated": terminated[mask].float(),
                },
                batch_size=mask.sum().item(),
            ),
            v_idx=torch.nonzero(mask).flatten(),
        )
        n_samples += mask.sum().item()

        done = terminated | truncated
        if done.any():
            logger.log(rollout_collector.log(done), step=n_samples)
        if autoreset.any():
            rollout_collector.clear(autoreset)

        autoreset = done
        obs = next_obs

        # Training.
        if n_samples > learning_starts:
            tstart = time.perf_counter()
            policy.train()
            if n_samples - last_train_critic >= critic_period:
                last_train_critic = n_samples
                data = replay_buffer.sample(batch_size)
                with torch.no_grad():
                    next_state_actions, next_state_log_pi, _ = policy.actor.action(data["next_obs"])
                    q1_next_target = q1_target(data["next_obs"], next_state_actions)
                    q2_next_target = q2_target(data["next_obs"], next_state_actions)
                    min_qf_next_target = (
                        torch.min(q1_next_target, q2_next_target) - alpha * next_state_log_pi
                    )
                    next_q_value = data["reward"].flatten() + (
                        1 - data["terminated"].flatten()
                    ) * gamma * (min_qf_next_target).view(-1)

                q1_a_values = q1(data["obs"], data["action"]).view(-1)
                q2_a_values = q2(data["obs"], data["action"]).view(-1)
                q1_loss = F.mse_loss(q1_a_values, next_q_value)
                q2_loss = F.mse_loss(q2_a_values, next_q_value)
                qf_loss = q1_loss + q2_loss
                train_collector.collect(critic_loss=qf_loss.detach())

                # optimize the model
                critic_optim.zero_grad()
                qf_loss.backward()
                critic_optim.step()

            if n_samples - last_train_actor >= actor_period:  # TD 3 Delayed update support
                last_train_actor = n_samples
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                # TODO: Really? TD3 does not do this
                for _ in range(actor_period):
                    data = replay_buffer.sample(batch_size)
                    pi, log_pi, _ = policy.actor.action(data["obs"])
                    q1_pi = q1(data["obs"], pi)
                    q2_pi = q2(data["obs"], pi)
                    min_qf_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()
                    train_collector.collect(actor_loss=actor_loss.detach())

                    if autotune_alpha:
                        with torch.no_grad():
                            _, log_pi, _ = policy.actor.action(data["obs"])
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha = log_alpha.exp().item()
                        train_collector.collect(alpha_loss=alpha_loss.detach())

            # Update the target networks
            if n_samples - last_target >= target_period:
                last_target = n_samples
                polyak_update_(q1_target, q1, tau)
                polyak_update_(q2_target, q2, tau)

            if log := train_collector.log():
                logger.log(log, step=n_samples)
                train_collector.clear()
            logger.log({"time/train": time.perf_counter() - tstart}, step=n_samples)

        # Evaluate the agent
        if n_samples - last_eval >= eval_period:
            tstart = time.perf_counter()
            last_eval = n_samples
            sync_env_normalization(train_envs, eval_envs)
            log = evaluate_agent(policy, eval_envs, eval_steps, eval_collector, device)
            logger.log(log, step=n_samples)
            logger.log({"time/eval": time.perf_counter() - tstart}, step=n_samples)
    logger.flush()
    return policy
