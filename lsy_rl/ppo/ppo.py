import random
import time

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import NormalizeObservation
from tensordict import TensorDict
from torch.optim import AdamW

from lsy_rl.core.logger import EmptyLogger, Logger
from lsy_rl.core.replay_buffer import TrajectoryBuffer
from lsy_rl.ppo.policy import PPOActor, PPOCritic, PPOPolicy


def set_seeds(seed: int | None = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def unwrap_norm_env(env: VectorEnv) -> NormalizeObservation | None:
    while hasattr(env, "env"):
        if isinstance(env, NormalizeObservation):
            return env
        env = env.env


def sync_envs(train_envs: VectorEnv, eval_envs: VectorEnv):
    """Sync the normalization constants from the train env to the eval env."""
    train_norm_env = unwrap_norm_env(train_envs)
    eval_norm_env = unwrap_norm_env(eval_envs)
    if (train_norm_env is None) != (eval_norm_env is None):
        raise ValueError("Both envs must either have normalization or not have normalization")
    if train_norm_env is None and eval_norm_env is None:  # No normalization, no sync necessary
        return
    eval_norm_env.obs_rms.mean = train_norm_env.obs_rms.mean
    eval_norm_env.obs_rms.var = train_norm_env.obs_rms.var


def evaluate_agent(
    envs: VectorEnv, policy: PPOPolicy, n_steps: int, device: str, seed: int | None = None
) -> tuple[list[float], list[int]]:
    eval_obs, _ = envs.reset(seed=seed)
    ep_rewards = torch.zeros(envs.num_envs, device=device)
    ep_steps = torch.zeros_like(ep_rewards)
    rewards, steps = [], []  # All eval rewards are at the same global step, so we average
    for _ in range(n_steps):
        with torch.no_grad():
            action, _, _, _ = policy.action_and_value(eval_obs, deterministic=True)
        eval_obs, reward, terminated, truncated, _ = envs.step(action)
        ep_rewards += reward
        ep_steps += 1
        done = terminated | truncated
        rewards.extend([r.item() for r in ep_rewards[done]])
        steps.extend([s.item() for s in ep_steps[done]])
        ep_rewards[done] = 0
        ep_steps[done] = 0
    return rewards, steps


def ppo(
    train_envs: VectorEnv,
    eval_envs: VectorEnv,
    n_steps: int,
    n_minibatches: int,
    n_total_steps: int,
    learning_rate: float,
    eval_interval: int,
    clip_coef: float = 0.2,
    ent_coef: float = 0.01,
    vf_coef: float = 0.5,
    max_grad_norm: float = 1.0,
    target_kl: float | None = None,
    n_epochs: int = 10,
    gamma: float = 0.99,
    norm_adv: bool = True,
    clip_vloss: bool = True,
    gae_lambda: float = 0.95,
    n_eval_steps: int = 1000,
    device: torch.device = torch.device("cpu"),
    logger: Logger = EmptyLogger(),
    seed: int | None = None,
):
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"

    # Calculate necessary parameters from the configured parameters
    n_envs = train_envs.num_envs
    batch_size = n_envs * n_steps
    minibatch_size = batch_size // n_minibatches
    n_iterations = n_total_steps // batch_size

    n_total_steps = n_iterations * batch_size
    if n_iterations < 1:
        return

    actor = PPOActor(train_envs.single_observation_space, train_envs.single_action_space)
    critic = PPOCritic(train_envs.single_observation_space)
    agent = PPOPolicy(actor, critic, device)
    optimizer = AdamW(agent.parameters(), lr=learning_rate, eps=1e-5)

    # Create episode buffer
    buffer = TrajectoryBuffer(n_envs, n_steps, device)

    # Stats tracking setup
    global_step = 0
    eval_rewards_hist = []
    eval_rewards_steps = []
    last_eval = global_step
    rewards = torch.zeros(n_envs, dtype=torch.float32, device=device)
    autoreset = torch.zeros(n_envs, dtype=bool, device=device)

    obs, _ = train_envs.reset(seed=seed)

    for iteration in range(1, n_iterations + 1):
        start_time = time.perf_counter()
        steps = torch.zeros(n_envs, dtype=torch.int32, device=device)
        while any(active := (steps < n_steps)):
            action, logprob, _, value = agent.action_and_value(obs)
            next_obs, reward, terminated, truncated, info = train_envs.step(action)
            done = terminated | truncated
            rewards += reward
            rewards[autoreset] = 0
            if done.any():
                # TODO: Add logging
                for r in rewards[done]:
                    ...
            # Add sample to buffer
            mask = active & ~autoreset
            sample = TensorDict(
                {
                    "obs": obs[mask],
                    "action": action[mask],
                    "next_obs": next_obs[mask],
                    "reward": reward[mask],
                    "terminated": terminated[mask],
                    "logprob": logprob[mask],
                    "value": value[mask],
                },
                batch_size=mask.sum().item(),
            )
            buffer.add(sample, steps[mask], mask)
            steps[mask] += 1
            global_step += mask.sum().item()

            obs = next_obs
            autoreset = done

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.value(obs).reshape(1, -1)
            advantages = torch.zeros_like(buffer["reward"]).to(device)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                if t == n_steps - 1:
                    # TODO: Check that terminated is correct instead of dones
                    nextnonterminal = 1.0 - buffer["terminated"][t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer["terminated"][t + 1]
                    nextvalues = buffer["values"][t + 1]
                future_reward = gamma * nextvalues * nextnonterminal
                delta = buffer["reward"][t] + future_reward - buffer["values"][t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + buffer["values"]

        # flatten the batch
        b_obs = buffer["obs"].reshape((-1,) + train_envs.single_observation_space.shape)
        b_logprobs = buffer["logprob"].reshape(-1)
        b_actions = buffer["action"].reshape((-1,) + train_envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer["value"].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.action_and_value(
                    b_obs[mb_inds], b_actions[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -clip_coef, clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optimizer.step()

            if target_kl is not None and approx_kl > target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Evaluate the agent
        if global_step - last_eval >= eval_interval:
            sync_envs(train_envs, eval_envs)
            eval_rewards, eval_steps = evaluate_agent(
                eval_envs, agent, n_steps=n_eval_steps, device=device, seed=seed + iteration
            )
            eval_mean_rewards = np.nan if not eval_rewards else np.mean(eval_rewards)
            eval_mean_steps = np.nan if not eval_steps else np.mean(eval_steps)
            eval_rewards_hist.append(eval_mean_rewards)
            eval_rewards_steps.append(global_step)
            # TODO: Logging
            # wandb.log(
            #     {"eval/mean_rewards": eval_mean_rewards, "eval/mean_steps": eval_mean_steps},
            #     step=global_step,
            # )
            last_eval = global_step

        end_time = time.perf_counter()
        print(f"Iter {iteration}/{n_iterations} took {end_time - start_time:.2f} seconds")
        # TODO: Logging
        # wandb.log(
        #     {
        #         "train/value_loss": v_loss.item(),
        #         "train/policy_loss": pg_loss.item(),
        #         "train/entropy_loss": entropy_loss.item(),
        #         "train/old_approx_kl": old_approx_kl.item(),
        #         "train/approx_kl": approx_kl.item(),
        #         "train/clipfrac": np.mean(clipfracs),
        #         "train/explained_var": explained_var,
        #     },
        #     step=global_step,
        # )
    return agent
