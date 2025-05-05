import time
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from gymnasium.vector import VectorEnv
from torch.optim import AdamW

from lsy_rl.core.logger import EmptyLogger, Logger
from lsy_rl.core.replay_buffer import VectorReplayBuffer
from lsy_rl.core.transforms import IdentityTF, Transform
from lsy_rl.td3.policy import TD3Actor, TD3Critic, TD3Policy
from lsy_rl.utils.utils import set_seeds, tensordict_sample


def td3(
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
    critic_period: int = 1,
    actor_period: int = 1,
    actor_target_period: int = 2,
    critic_target_period: int = 2,
    tau: float = 0.005,
    gamma: float = 0.99,
    grad_clip: float = 1.0,
    batch_size: int = 128,
    reward_clip: tuple[float, float] | None = None,
    eval_period: int | None = None,
    eval_steps: int = 1000,
    checkpoint_period: int | None = None,
    train_min_samples: int | None = None,
    policy: TD3Policy | None = None,
    obs_tf: Transform = IdentityTF(),
    action_tf: Transform = IdentityTF(),
    train_action_tf: Transform = IdentityTF(),
    eval_action_tf: Transform = IdentityTF(),
    target_action_tf: Transform = IdentityTF(),
    logger: Logger = EmptyLogger(),
    device: torch.device = torch.device("cpu"),
    checkpoint_path: Path | None = None,
    checkpoint_buffer: bool = False,
    seed: int | None = None,
) -> TD3Policy:
    set_seeds(seed)
    assert train_envs is not eval_envs, "Train and eval environments must be different"
    # Move all transforms to the correct device
    obs_tf.to(device=device)
    action_tf.to(device=device)
    train_action_tf.to(device=device)
    eval_action_tf.to(device=device)
    target_action_tf.to(device=device)

    # Calculate log periods to limit the amount of logging
    N_LOGS = 100
    collect_samples_log_period = n_steps // N_LOGS
    # Number of training calls * number of training iterations / number of logs
    train_log_period = ((n_steps // train_period) * train_steps) // N_LOGS

    if policy is None:
        obs_space = train_envs.single_observation_space
        action_space = train_envs.single_action_space
        actor = TD3Actor(obs_space, action_space)
        critic = TD3Critic(obs_space, action_space)
        policy = TD3Policy(actor, critic)
    policy.to(device=device)

    critic_optimizer = AdamW(policy.critic.parameters(), lr=critic_lr, eps=eps)
    actor_optimizer = AdamW(policy.actor.parameters(), lr=actor_lr, eps=eps)

    if replay_buffer is None:
        replay_buffer = VectorReplayBuffer(
            train_envs.num_envs, max_size=buffer_size, device=device, seed=seed
        )

    train_info = {}
    n_train_steps = 0
    n_samples = 0
    evaluate_policy(policy, eval_envs, eval_steps, obs_tf, action_tf, n_samples, logger, device)
    for n_samples, should_train, should_eval, should_checkpoint in collect_samples(
        policy=policy,
        env=train_envs,
        n_steps=n_steps,
        obs_tf=obs_tf,
        action_tf=action_tf,
        replay_buffer=replay_buffer,
        train_period=train_period,
        eval_period=eval_period,
        checkpoint_period=checkpoint_period,
        log_period=collect_samples_log_period,
        train_min_samples=train_min_samples,
        logger=logger,
        device=device,
    ):
        if should_train:
            train_info = train_policy(
                policy=policy,
                replay_buffer=replay_buffer,
                steps=train_steps,
                n_train_steps=n_train_steps,
                n_samples=n_samples,
                critic_period=critic_period,
                actor_period=actor_period,
                actor_target_period=actor_target_period,
                critic_target_period=critic_target_period,
                tau=tau,
                gamma=gamma,
                grad_clip=grad_clip,
                batch_size=batch_size,
                obs_tf=obs_tf,
                action_tf=train_action_tf,
                target_action_tf=target_action_tf,
                critic_optimizer=critic_optimizer,
                actor_optimizer=actor_optimizer,
                info=train_info,
                reward_clip=reward_clip,
                log_period=train_log_period,
                logger=logger,
            )
            n_train_steps += train_steps
        if should_eval:
            evaluate_policy(
                policy,
                eval_envs,
                eval_steps,
                obs_tf,
                eval_action_tf,
                n_samples,
                logger,
                device=device,
            )
        if should_checkpoint and checkpoint_path is not None:
            checkpoint(
                checkpoint_path,
                policy,
                replay_buffer,
                critic_optimizer,
                actor_optimizer,
                obs_tf,
                checkpoint_buffer,
            )
    if checkpoint_path is not None:
        checkpoint(
            checkpoint_path,
            policy,
            replay_buffer,
            critic_optimizer,
            actor_optimizer,
            obs_tf,
            checkpoint_buffer,
        )
    logger.stop()

    return policy


@torch.no_grad()
def collect_samples(
    policy: TD3Policy,
    env: VectorEnv,
    n_steps: int,
    obs_tf: Transform,
    action_tf: Transform,
    replay_buffer: VectorReplayBuffer,
    train_period: int,
    eval_period: int | None,
    checkpoint_period: int | None,
    log_period: int,
    train_min_samples: int | None,
    logger: Logger,
    device: torch.device,
) -> Generator[tuple[int, bool, bool, bool], None, None]:
    """Collect samples from the environment and store them in the replay buffer.

    This function is a generator. It will continue to save samples into the buffer until the maximum
    number of samples is reached. In between samples, it will yield whenever the conditions for
    training, evaluating or checkpointing are met.
    """
    last_train = 0
    last_eval = 0
    last_log = 0
    last_checkpoint = 0
    start_time = time.time()
    log_info = {"ep_steps": 0, "ep_reward": 0, "n_episodes": 0, "last_rewards": []}
    n_samples = 0
    steps = torch.zeros(env.num_envs, device=device)
    rewards = torch.zeros(env.num_envs, device=device)
    autoreset = False

    obs = None

    while n_samples < n_steps:
        policy.eval()

        if obs is None:  # If first rollout, reset the environment
            obs, _ = env.reset()
            obs = obs.to(device)

        obs_tf.update(obs)
        obs_t = obs_tf(obs)
        action = policy.actor(obs_t)
        action = action_tf(action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        sample = tensordict_sample(
            obs, action, next_obs, reward, terminated, truncated, info, device=device
        )
        obs = sample["next_obs"]
        done = terminated | truncated

        # Vector environments automatically reset after T steps. This reset happens on the next
        # step. The reset step produces an inconsistent (obs, next_obs) tuple that has to be
        # discarded. To see how this is handled in gymnasium >= 1.0, see
        # https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0.
        if autoreset:
            autoreset = torch.all(done)
            continue

        # TODO: Add support for variable rollout length
        assert torch.all(done) or not torch.any(done), "Variable rollout length not supported"
        replay_buffer.add(sample)
        autoreset = torch.all(done)

        n_samples += env.num_envs
        steps += 1
        rewards += sample["reward"]
        # If any of the environments are terminated or truncated, log the episode statistics
        if torch.any(done):
            log_info["n_episodes"] += len(done)
            log_info["ep_steps"] += steps[done].sum()
            log_info["ep_reward"] += rewards[done].sum()
            log_info["last_rewards"].extend(sample["reward"][done].tolist())
            steps[done] = 0
            rewards[done] = 0

        # Logging
        if n_samples - last_log >= log_period and log_info["n_episodes"] > 0:
            log = {"rollout/ep_steps": log_info["ep_steps"] / log_info["n_episodes"]}
            log["rollout/ep_reward"] = log_info["ep_reward"] / log_info["n_episodes"]
            elapsed_time = time.time() - start_time
            log["time/time_elapsed"] = elapsed_time
            log["time/total_timesteps"] = n_samples
            log["time/fps"] = n_samples / elapsed_time
            logger.log(log, step=n_samples)
            log_info["ep_steps"], log_info["ep_reward"], log_info["n_episodes"] = (0, 0, 0)
            log_info["last_rewards"] = []
            last_log = n_samples

        train_condition = check_interrupt_sample(
            n_samples, last_train, period=train_period, min_samples=train_min_samples
        )
        eval_condition = check_interrupt_sample(n_samples, last_eval, period=eval_period)
        checkpoint_condition = check_interrupt_sample(
            n_samples, last_checkpoint, period=checkpoint_period
        )
        if train_condition:
            last_train = n_samples
        if eval_condition:
            last_eval = n_samples
        if checkpoint_condition:
            last_checkpoint = n_samples
        if train_condition or eval_condition or checkpoint_condition:
            yield n_samples, train_condition, eval_condition, checkpoint_condition


def train_policy(
    policy: TD3Policy,
    replay_buffer: VectorReplayBuffer,
    steps: int,
    n_train_steps: int,
    n_samples: int,
    critic_period: int,
    actor_period: int,
    actor_target_period: int,
    critic_target_period: int,
    tau: float,
    gamma: float,
    grad_clip: float,
    batch_size: int,
    obs_tf: Transform,
    action_tf: Transform,
    target_action_tf: Transform,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    info: dict,
    log_period: int,
    reward_clip: tuple[float, float] | None,
    logger: Logger,
):
    """Train the policy using the collected samples in the replay buffer."""
    policy.train()  # Critic is always in train mode, not used for inference
    if not info:
        info = {
            "summed_actor_loss": 0,
            "summed_critic_loss": 0,
            "actor_steps_since_log": 0,
            "critic_steps_since_log": 0,
            "n_train_steps": 0,
            "last_log": 0,
        }

    for _ in range(steps):
        # Update 'num_train_steps' at the beginning of the loop so that lower frequency updates
        # do not get executed at the first iteration when 'num_train_steps' is 0
        n_train_steps += 1

        if n_train_steps % critic_period == 0:
            batch = replay_buffer.sample(batch_size)
            # Compute the expected Q values with the reward and the target networks
            with torch.no_grad():
                next_obs_t = obs_tf(batch["next_obs"])
                next_action = policy.actor.target(next_obs_t)
                next_action = target_action_tf(next_action)
                next_q = policy.critic.target(next_obs_t, next_action)
                # Reward, terminated are one-dimensional, so we need to reshape them to avoid
                # broadcasting errors
                reward = batch["reward"].reshape(-1, 1)
                terminated = batch["terminated"].reshape(-1, 1)
                q_target = reward + (gamma * ~terminated * next_q)
                if reward_clip is not None:
                    q_target = torch.clamp(q_target, *reward_clip)
            # Compute the loss as the MSE between the expected Q values and the Q values from
            # the critic
            obs_t = obs_tf(batch["obs"])
            q_1, q_2 = policy.critic.values(obs_t, batch["action"])
            assert q_target.shape == (batch_size, 1), q_target.shape
            assert q_1.shape == q_target.shape, (q_1.shape, q_target.shape)
            assert q_2.shape == q_target.shape, (q_2.shape, q_target.shape)
            q1_loss = (q_target - q_1).pow(2).mean()
            q2_loss = (q_target - q_2).pow(2).mean()
            critic_loss = q1_loss + q2_loss
            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.critic.parameters(), grad_clip)
            critic_optimizer.step()
            info["summed_critic_loss"] += critic_loss.detach()
            info["critic_steps_since_log"] += 1

        if n_train_steps % actor_period == 0:
            batch = replay_buffer.sample(batch_size)
            # Compute the actions for the sample observations, compute the critic value of the
            # observations and actions and compute the actor loss by maximizing the critic value
            obs_t = obs_tf(batch["obs"])
            train_action = policy.actor(obs_t)
            train_action = action_tf(train_action)
            actor_loss = -policy.critic.actor_value(obs_t, train_action).mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), grad_clip)
            actor_optimizer.step()
            info["summed_actor_loss"] += actor_loss.detach()
            info["actor_steps_since_log"] += 1

        # Update the target networks
        if n_train_steps % actor_target_period == 0:
            policy.actor.update_target(tau)
        if n_train_steps % critic_target_period == 0:
            policy.critic.update_target(tau)

        # Log the training statistics
        if n_train_steps - info["last_log"] >= log_period:
            log = {}
            info["last_log"] = n_train_steps
            if info["actor_steps_since_log"] > 0:
                log["train/actor_loss"] = info["summed_actor_loss"] / info["actor_steps_since_log"]
                info["summed_actor_loss"], info["actor_steps_since_log"] = 0, 0
            if info["critic_steps_since_log"] > 0:
                log["train/critic_loss"] = (
                    info["summed_critic_loss"] / info["critic_steps_since_log"]
                )
                info["summed_critic_loss"], info["critic_steps_since_log"] = 0, 0
            if log:
                logger.log(log, step=n_samples)

    info["n_train_steps"] = n_train_steps
    return info


@torch.no_grad()
def evaluate_policy(
    policy: TD3Policy,
    env: VectorEnv,
    n_eval_steps: int,
    obs_tf: Transform,
    action_tf: Transform,
    n_samples: int,
    logger: Logger,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the policy on the evaluation environment and log the results."""
    policy.eval()
    rewards = torch.zeros(env.num_envs, device=device)
    steps = torch.zeros(env.num_envs, device=device)
    all_rewards, ep_rewards, ep_steps, ep_last_rewards = [], [], [], []

    obs, _ = env.reset()
    autoreset = False
    n = 0
    while n < n_eval_steps:
        obs_t = obs_tf(obs)
        action = policy.action(obs_t)
        action = action_tf(action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        sample = tensordict_sample(
            obs, action, next_obs, reward, terminated, truncated, info, device=device
        )
        obs = sample["next_obs"]
        done = terminated | truncated
        if autoreset:  # As in rollout, we discard the reset step of the rollouts for the stats
            autoreset = torch.all(done)
            continue
        autoreset = torch.all(done)

        rewards += sample["reward"]
        all_rewards += sample["reward"].tolist()
        steps += 1

        if torch.any(done):
            ep_steps += steps[done].tolist()
            ep_rewards += rewards[done].tolist()
            ep_last_rewards += sample["reward"][done].tolist()
            steps[done] = 0
            rewards[done] = 0

        n += env.num_envs

    log = {"eval/mean_rewards": np.array(all_rewards).mean()}
    if ep_rewards:
        log["eval/ep_mean_rewards"] = np.array(ep_rewards).mean()
        log["eval/ep_mean_steps"] = np.array(ep_steps).mean()
        log["eval/mean_last_reward"] = np.array(ep_last_rewards).mean()
    logger.log(log, step=n_samples)


def check_interrupt_sample(
    n_samples: int, last_n_samples: int, period: int | None = None, min_samples: int | None = None
) -> bool:
    """Check if we should interrupt sampling based on how many samples we have collected."""
    if min_samples is not None and n_samples < min_samples:
        return False
    if period is not None and n_samples - last_n_samples >= period:
        return True
    return False


def checkpoint(
    path: Path,
    policy: TD3Policy,
    buffer: VectorReplayBuffer,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    obs_tf: Transform,
    checkpoint_buffer: bool = False,
):
    """Save a checkpoint of the policy, replay buffer and optimizers."""
    assert isinstance(path, Path), "The checkpoint path must be a Path object."
    assert path.is_dir(), "The checkpoint path must be a directory."
    policy.save(path / "policy.pt")
    if checkpoint_buffer:
        buffer.save(path / "buffer.pt")
    torch.save(actor_optimizer.state_dict(), path / "actor_opt.pt")
    torch.save(critic_optimizer.state_dict(), path / "critic_opt.pt")
    torch.save(obs_tf.state_dict(), path / "obs_transform.pt")
