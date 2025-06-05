import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from tensordict import TensorDict
from torch.optim import AdamW

from lsy_rl.core.logger import Collector, CollectorList, EmptyLogger, LogCollector, Logger
from lsy_rl.core.replay_buffer import TrajectoryBuffer
from lsy_rl.ppo.policy import PPOActor, PPOCritic, PPOPolicy
from lsy_rl.utils.utils import set_seeds, sync_env_normalization


def evaluate_agent(
    envs: VectorEnv,
    policy: PPOPolicy,
    n_steps: int,
    device: str,
    collector: Collector,
    seed: int | None = None,
) -> dict[str, float]:
    obs, _ = envs.reset(seed=seed)
    collector.clear(mask=torch.ones(envs.num_envs, dtype=torch.bool))
    autoreset = torch.zeros(envs.num_envs, dtype=bool, device=device)
    logs = []  # All eval logs are at the same global step, so we average
    for _ in range(0, n_steps, envs.num_envs):
        with torch.no_grad():
            action, _, _, _ = policy.action_and_value(obs, deterministic=True)
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
        done = terminated | truncated
        if done.any():
            logs.append(collector.log(done))
        if autoreset.any():
            collector.clear(autoreset)
        autoreset = done
        obs = next_obs
    # Average over all metrics
    avg_log = {}
    for log in logs:
        for k, v in log.items():
            if k not in avg_log:
                avg_log[k] = []
            avg_log[k].append(v)
    avg_log = {k: sum(v) / len(v) for k, v in avg_log.items()}
    return avg_log


def ppo(
    train_envs: VectorEnv,
    eval_envs: VectorEnv,
    n_steps: int,
    n_minibatches: int,
    n_total_steps: int,
    actor_lr: float,
    critic_lr: float,
    eval_period: int,
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
    agent: PPOPolicy | None = None,
    rollout_log_collector: Collector | None = None,
    eval_log_collector: Collector | None = None,
    seed: int | None = None,
) -> PPOPolicy:
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"
    if target_kl is not None and target_kl < 0:
        target_kl = None

    # Calculate necessary parameters from the configured parameters
    n_envs = train_envs.num_envs
    batch_size = n_envs * n_steps
    minibatch_size = batch_size // n_minibatches
    n_iterations = n_total_steps // batch_size

    if n_iterations < 1:
        warnings.warn(
            f"Number of train steps with {n_total_steps:.2e} total steps and {batch_size:.2e} batch"
            " size is < 1, returning without training"
        )
    n_total_steps = n_iterations * batch_size

    if agent is None:
        obs_shape = train_envs.single_observation_space.shape
        action_shape = train_envs.single_action_space.shape
        actor = PPOActor(obs_shape, action_shape)
        critic = PPOCritic(obs_shape)
        agent = PPOPolicy(actor, critic)
    agent.to(device)
    actor_optim = AdamW(agent.actor.parameters(), lr=actor_lr, eps=1e-5)
    critic_optim = AdamW(agent.critic.parameters(), lr=critic_lr, eps=1e-5)

    # Create episode buffer
    buffer = TrajectoryBuffer(n_envs, n_steps, device)

    # Stats tracking setup
    global_step = 0
    last_eval = global_step
    autoreset = torch.zeros(n_envs, dtype=bool, device=device)

    obs, _ = train_envs.reset(seed=seed)

    # Create metric collectors
    if rollout_log_collector is None:
        rollout_log_collector = CollectorList()
        rollout_log_collector.append(LogCollector(target="reward", log_key="rollout/ep_reward"))
        rollout_log_collector.append(LogCollector(target="step", log_key="rollout/ep_steps"))
    if eval_log_collector is None:
        eval_log_collector = CollectorList()
        eval_log_collector.append(LogCollector(target="reward", log_key="eval/ep_reward"))
        eval_log_collector.append(LogCollector(target="step", log_key="eval/ep_steps"))

    for iteration in range(1, n_iterations + 1):
        start_time = time.perf_counter()
        steps = torch.zeros(n_envs, dtype=torch.int32, device=device)
        while any(active := (steps < n_steps)):
            with torch.no_grad():
                action, logprob, _, value = agent.action_and_value(obs)
            next_obs, reward, terminated, truncated, info = train_envs.step(action)
            # Aggregate logs in a customizable way
            rollout_log_collector.collect(
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
            # Add sample to buffer
            mask = active & ~autoreset
            sample = TensorDict(
                {
                    "obs": obs[mask],
                    "action": action[mask],
                    "next_obs": next_obs[mask],
                    "reward": reward[mask],
                    "terminated": terminated[mask].float(),
                    "done": (truncated[mask] | terminated[mask]).float(),
                    "logprob": logprob[mask],
                    "value": value[mask].squeeze(dim=-1),
                },
                batch_size=mask.sum().item(),
            )
            buffer.add(sample, mask)
            steps[mask] += 1
            global_step += mask.sum().item()

            if done.any():
                logger.log(rollout_log_collector.log(done), step=global_step)
            if autoreset.any():
                rollout_log_collector.clear(autoreset)

            autoreset = done
            obs = next_obs
        assert buffer.full(), "Buffer is not full"
        logger.log({"time/rollout": time.perf_counter() - start_time}, step=global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(buffer["reward"], device=device)
            lastgaelam = 0
            for t in reversed(range(n_steps)):
                # Deviation from CleanRL:
                #
                # 1.) We use terminated instead of done because the final state of a truncated
                # episode does not end the episode, therefore not cutting off the bootstrapped value
                # estimate. See https://farama.org/Gymnasium-Terminated-Truncated-Step-API
                #
                # 2.) We store the buffer of terminated returned after stepping without assuming an
                # initial terminated state of all 0s (CleanRL never uses this 0th terminated state).
                # Our version is therefore shifted by 1 to the right, and we do not use
                # buffer["terminated"][t + 1]. In addition, we add the final terminated to the end
                # of the buffer, so we do not need to handle the final value case separately.
                nextnonterminal = 1.0 - buffer["terminated"][t]
                nextvalues = next_value if t == n_steps - 1 else buffer["value"][t + 1]
                delta = buffer["reward"][t] + gamma * nextvalues * nextnonterminal - buffer["value"][t]  # fmt: skip
                # While the value needs to bootstrap across the truncated boundary, we cannot use
                # the gae lambda term for episodes that are done, because that would mix the value
                # estimates from two unrelated episodes. Therefore, we need to mask the gae lambda
                # term for done (truncated or terminated) episodes.
                nextdone = 1 - buffer["done"][t]
                lastgaelam = delta + gamma * gae_lambda * nextdone * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + buffer["value"]

        # flatten the batch
        b_obs = buffer["obs"].flatten(end_dim=-2)
        b_logprobs = buffer["logprob"].reshape(-1)
        b_actions = buffer["action"].flatten(end_dim=-2)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = buffer["value"].reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        tstart = time.perf_counter()
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

                actor_optim.zero_grad()
                critic_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                actor_optim.step()
                critic_optim.step()

            if target_kl is not None and approx_kl > target_kl:
                break
        logger.log({"time/train": time.perf_counter() - tstart}, step=global_step)

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        logger.log(
            {
                "train/value_loss": v_loss.item(),
                "train/policy_loss": pg_loss.item(),
                "train/entropy_loss": entropy_loss.item(),
                "train/old_approx_kl": (-logratio).mean().item(),
                "train/approx_kl": approx_kl.item(),
                "train/clipfrac": np.mean(clipfracs),
                "train/explained_var": explained_var,
            },
            step=global_step,
        )

        # Evaluate the agent
        if global_step - last_eval >= eval_period:
            tstart = time.perf_counter()
            sync_env_normalization(train_envs, eval_envs)
            eval_seed = seed if seed is None else seed + iteration
            logs = evaluate_agent(
                eval_envs,
                agent,
                n_steps=n_eval_steps,
                device=device,
                collector=eval_log_collector,
                seed=eval_seed,
            )
            logger.log(logs, step=global_step)
            last_eval = global_step
            logger.log({"time/eval": time.perf_counter() - tstart}, step=global_step)
        buffer.clear()
    logger.flush()
    return agent
