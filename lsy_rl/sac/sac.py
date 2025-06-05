import time

import numpy as np
import torch
import torch.nn.functional as F
from gymnasium.vector import VectorEnv
from tensordict import TensorDict
from torch.optim import AdamW

from lsy_rl.core.logger import EmptyLogger, Logger
from lsy_rl.core.replay_buffer import VectorReplayBuffer
from lsy_rl.sac.policy import SACActor, SACCritic, SACPolicy
from lsy_rl.utils import polyak_update_
from lsy_rl.utils.utils import set_seeds, sync_env_normalization


def evaluate_agent(
    envs: VectorEnv, policy: SACPolicy, n_steps: int, device: str, seed: int | None = None
) -> tuple[list[float], list[int]]:
    obs, _ = envs.reset(seed=seed)
    ep_rewards = torch.zeros(envs.num_envs, device=device)
    ep_steps = torch.zeros_like(ep_rewards)
    autoreset = torch.zeros(envs.num_envs, dtype=bool, device=device)
    rewards, steps = [], []  # All eval rewards are at the same global step, so we average
    for _ in range(0, n_steps, envs.num_envs):
        with torch.no_grad():
            action = policy.actor.mean_action(obs)
        obs, reward, terminated, truncated, _ = envs.step(action)
        ep_rewards += reward
        ep_steps += 1
        done = terminated | truncated
        rewards.extend([r.item() for r in ep_rewards[done]])
        steps.extend([s.item() for s in ep_steps[done]])
        ep_rewards[autoreset] = 0
        ep_steps[autoreset] = 0
        autoreset = done
    return rewards, steps


def sac(
    train_envs: VectorEnv,
    eval_envs: VectorEnv,
    n_total_steps: int,
    actor_lr: float,
    critic_lr: float,
    eval_period: int,
    buffer_size: int = int(1e6),
    batch_size: int = 256,
    alpha_lr: float = 3e-4,
    gamma: float = 0.99,
    tau: float = 0.005,
    alpha: float = 0.2,
    autotune_alpha: bool = True,
    actor_update_period: int = 1,
    target_network_update_period: int = 1,
    n_eval_steps: int = 1000,
    learning_starts: int = 0,
    device: torch.device = torch.device("cpu"),
    logger: Logger = EmptyLogger(),
    policy: SACPolicy | None = None,
    seed: int | None = None,
) -> SACPolicy:
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"

    # Calculate necessary parameters from the configured parameters
    n_envs = train_envs.num_envs

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
    buffer = VectorReplayBuffer(num_envs=n_envs, max_size=buffer_size, device=device, seed=seed)

    # Stats tracking setup
    global_step, train_step = 0, 0
    ep_rewards = torch.zeros(n_envs, dtype=torch.float32, device=device)
    ep_steps = torch.zeros(n_envs, dtype=torch.int32, device=device)
    last_eval = global_step
    autoreset = torch.zeros(n_envs, dtype=bool, device=device)

    obs, _ = train_envs.reset(seed=seed)

    while global_step < n_total_steps:
        # Sample data
        with torch.no_grad():
            action, _, _ = policy.actor.action(obs)
        next_obs, reward, terminated, truncated, _ = train_envs.step(action)

        mask = ~autoreset
        sample = TensorDict(
            {
                "obs": obs[mask],
                "action": action[mask],
                "next_obs": next_obs[mask],
                "reward": reward[mask],
                "terminated": terminated[mask].float(),
            },
            batch_size=mask.sum().item(),
        )
        buffer.add(sample, torch.arange(n_envs, device=device)[mask])
        global_step += mask.sum().item()

        done = terminated | truncated
        if done.any():
            logger.log(
                {
                    "rollout/ep_reward": ep_rewards[done].mean().item(),
                    "rollout/ep_step": ep_steps[done].float().mean().item(),
                },
                step=global_step,
            )
        ep_rewards[autoreset] = 0
        ep_steps[autoreset] = 0
        autoreset = done
        obs = next_obs

        # Training.
        if global_step > learning_starts:
            data = buffer.sample(batch_size)
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

            # optimize the model
            critic_optim.zero_grad()
            qf_loss.backward()
            critic_optim.step()

            if train_step % actor_update_period == 0:  # TD 3 Delayed update support
                # compensate for the delay by doing 'actor_update_interval' instead of 1
                # TODO: Really? TD3 does not do this
                for _ in range(actor_update_period):
                    pi, log_pi, _ = policy.actor.action(data["obs"])
                    q1_pi = q1(data["obs"], pi)
                    q2_pi = q2(data["obs"], pi)
                    min_qf_pi = torch.min(q1_pi, q2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optim.zero_grad()
                    actor_loss.backward()
                    actor_optim.step()

                    if autotune_alpha:
                        with torch.no_grad():
                            _, log_pi, _ = policy.actor.action(data["obs"])
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        alpha_optim.zero_grad()
                        alpha_loss.backward()
                        alpha_optim.step()
                        alpha = log_alpha.exp().item()

            # Update the target networks
            if train_step % target_network_update_period == 0:
                polyak_update_(q1_target, q1, tau)
                polyak_update_(q2_target, q2, tau)

            logger.log(
                {
                    "train/q1_values": q1_a_values.mean().item(),
                    "train/q2_values": q2_a_values.mean().item(),
                    "train/q1_loss": q1_loss.item(),
                    "train/q2_loss": q2_loss.item(),
                    "train/qf_loss": qf_loss.item() / 2.0,
                    "train/actor_loss": actor_loss.item(),
                    "train/alpha": alpha,
                },
                step=global_step,
            )
            if autotune_alpha:
                logger.log({"train/alpha_loss": alpha_loss.item()}, step=global_step)
            train_step += 1

        # Evaluate the agent
        if global_step - last_eval >= eval_period:
            tstart = time.perf_counter()
            sync_env_normalization(train_envs, eval_envs)
            eval_seed = seed if seed is None else seed + global_step
            eval_rewards, eval_steps = evaluate_agent(
                eval_envs, policy, n_steps=n_eval_steps, device=device, seed=eval_seed
            )
            mean_rewards = np.nan if not eval_rewards else np.mean(eval_rewards)
            mean_steps = np.nan if not eval_steps else np.mean(eval_steps)
            logger.log(
                {"eval/mean_rewards": mean_rewards, "eval/mean_steps": mean_steps}, step=global_step
            )
            last_eval = global_step
            logger.log({"time/eval": time.perf_counter() - tstart}, step=global_step)
    logger.flush()
    return policy
