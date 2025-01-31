import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from torch.optim import AdamW

from lsy_rl.core.logger import EmptyLogger, Logger


def set_seeds(seed: int | None = None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def ppo(
    train_envs: VectorEnv,
    eval_envs: VectorEnv,
    config: PPOConfig,
    logger: Logger = EmptyLogger(),
    seed: int | None = None,
):
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"

    config.batch_size = train_envs.num_envs * config.n_steps
    config.minibatch_size = config.batch_size // config.n_minibatches
    config.n_iterations = config.total_timesteps // config.batch_size

    config.total_timesteps = config.n_iterations * config.batch_size
    if config.n_iterations < 1:
        return

    agent = PPOPolicy(train_envs).to(config.device)
    optimizer = AdamW(agent.parameters(), lr=config.learning_rate, eps=1e-5)

    # Create episode buffer
    buffer = EpisodeBuffer(config.n_steps, train_envs.num_envs, config.device)

    # Stats tracking setup
    global_step = 0
    eval_rewards_hist = []
    eval_rewards_steps = []
    last_eval = global_step
    rewards = torch.zeros(config.n_envs, dtype=torch.float32, device=config.device)
    autoreset = torch.zeros(config.n_envs, dtype=bool, device=config.device)

    obs, _ = train_envs.reset(seed=seed)

    for iteration in range(1, config.n_iterations + 1):
        start_time = time.perf_counter()
        steps = torch.zeros(config.n_envs, dtype=torch.int32, device=config.device)
        while any(active := (steps < config.n_steps)):
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
            advantages = torch.zeros_like(buffer["reward"]).to(config.device)
            lastgaelam = 0
            for t in reversed(range(config.n_steps)):
                if t == config.n_steps - 1:
                    # TODO: Check that terminated is correct instead of dones
                    nextnonterminal = 1.0 - buffer["terminated"][t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buffer["terminated"][t + 1]
                    nextvalues = buffer["values"][t + 1]
                future_reward = config.gamma * nextvalues * nextnonterminal
                delta = buffer["reward"][t] + future_reward - buffer["values"][t]
                advantages[t] = lastgaelam = (
                    delta + config.gamma * config.gae_lambda * nextnonterminal * lastgaelam
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
        b_inds = np.arange(config.batch_size)
        clipfracs = []
        for epoch in range(config.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, config.batch_size, config.minibatch_size):
                end = start + config.minibatch_size
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
                    clipfracs += [((ratio - 1.0).abs() > config.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if config.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - config.clip_coef, 1 + config.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds], -config.clip_coef, config.clip_coef
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - config.ent_coef * entropy_loss + v_loss * config.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config.max_grad_norm)
                optimizer.step()

            if config.target_kl is not None and approx_kl > config.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Evaluate the agent
        if global_step - last_eval >= config.eval_interval:
            sync_envs(train_envs, eval_envs)
            eval_rewards, eval_steps = evaluate_agent(
                eval_envs,
                agent,
                n_steps=config.n_eval_steps,
                device=config.device,
                seed=config.seed + iteration,
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
        print(f"Iter {iteration}/{config.n_iterations} took {end_time - start_time:.2f} seconds")
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

    if config.save_model:
        save_model(agent, optimizer, train_envs, Path(__file__).parent / "ppo_checkpoint.pt")
    return agent
