import time
from collections import defaultdict
from pathlib import Path
from typing import Generator
from functools import partial

import torch
from gymnasium.vector import VectorEnv
from torch.optim import AdamW

from lsy_rl.core.logger import Collector, CollectorList, EmptyLogger, LogCollector, Logger
from lsy_rl.core.replay_buffer import VectorReplayBuffer
from lsy_rl.core.transforms import IdentityTF, Transform
from lsy_rl.td3.policy import TD3Actor, TD3Critic, TD3Policy
from lsy_rl.utils.utils import set_seeds, tensordict_sample, check_interrupt_sample, checkpoint


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
    overwrite_policy: bool = True,
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
    eval_collector: Collector | None = None,
    train_collector: Collector | None = None,
    rollout_collector: Collector | None = None,
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

    if train_collector is None:
        train_collector = CollectorList()
        train_collector.append(LogCollector(target="actor_loss", log_key="train/actor_loss"))
        train_collector.append(LogCollector(target="critic_loss", log_key="train/critic_loss"))
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

    # Create a partial function for checkpoint for more compact calls
    checkpoint_partial = partial(
        checkpoint, 
        path = checkpoint_path,
        policy = policy,
        buffer = replay_buffer,
        critic_optimizer = critic_optimizer,
        actor_optimizer = actor_optimizer,
        obs_tf = obs_tf,
        checkpoint_buffer = checkpoint_buffer
    )

    n_train_steps = 0
    n_samples = 0
    logs = evaluate_policy(policy, eval_envs, eval_steps, obs_tf, eval_action_tf, eval_collector, device)
    logger.log(logs, step=n_samples)
    if not overwrite_policy and checkpoint_path is not None:
        checkpoint_partial(step=n_samples, overwrite_policy=overwrite_policy)    
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
        collector=rollout_collector,
        train_min_samples=train_min_samples,
        logger=logger,
        device=device,
    ):
        if should_train:
            train_policy(
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
                collector=train_collector,
                reward_clip=reward_clip,
                logger=logger,
            )
            n_train_steps += train_steps
        if should_eval:
            log = evaluate_policy(
                policy, eval_envs, eval_steps, obs_tf, eval_action_tf, eval_collector, device=device
            )
            logger.log(log, step=n_samples)
        if should_checkpoint and checkpoint_path is not None:
            checkpoint_partial(step=n_samples, overwrite_policy=overwrite_policy)
    if checkpoint_path is not None:
        checkpoint_partial(step=n_samples, overwrite_policy=True)
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
    collector: Collector,
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
    last_checkpoint = 0
    start_time = time.time()
    n_samples = 0
    autoreset = torch.zeros(env.num_envs, dtype=bool, device=device)

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

        if done.any():
            logger.log(collector.log(done), step=n_samples)
        if autoreset.any():
            collector.clear(autoreset)

        autoreset = done
        obs = next_obs

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
            elapsed_time = time.time() - start_time
            logger.log(
                {
                    "time/elapsed": elapsed_time,
                    "time/steps": n_samples,
                    "time/fps": n_samples / elapsed_time,
                },
                step=n_samples,
            )
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
    collector: Collector,
    reward_clip: tuple[float, float] | None,
    logger: Logger,
):
    """Train the policy using the collected samples in the replay buffer."""
    policy.train()  # Critic is always in train mode, not used for inference

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
            collector.collect(critic_loss=critic_loss.detach())

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
            collector.collect(actor_loss=actor_loss.detach())

        # Update the target networks
        if n_train_steps % actor_target_period == 0:
            policy.actor.update_target(tau)
        if n_train_steps % critic_target_period == 0:
            policy.critic.update_target(tau)

        # Log the training statistics
        if log := collector.log():
            logger.log(log, step=n_samples)
            collector.clear()


@torch.no_grad()
def evaluate_policy(
    policy: TD3Policy,
    envs: VectorEnv,
    n_steps: int,
    obs_tf: Transform,
    action_tf: Transform,
    collector: Collector,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate the policy on the evaluation environment and log the results."""
    obs, _ = envs.reset()
    policy.eval()
    collector.clear(mask=torch.ones(envs.num_envs, dtype=torch.bool))
    autoreset = torch.zeros(envs.num_envs, dtype=bool, device=device)
    logs = []
    for _ in range(0, n_steps, envs.num_envs):
        action = action_tf(policy.action(obs_tf(obs)))
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