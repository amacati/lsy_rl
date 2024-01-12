import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from gymnasium.vector import VectorEnv

from lsy_rl.core import Algorithm
from lsy_rl.core.logger import Logger
from lsy_rl.utils import space_info
from lsy_rl.wrappers.tensor_wrapper import TensorWrapper
from lsy_rl.ddpg.config import DDPGConfig, EnvConfig, TrainConfig, EvalConfig, CheckpointConfig
from lsy_rl.ddpg.config import RolloutConfig
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic, DDPGPolicy

logger = logging.getLogger(__name__)


class DDPG(Algorithm):

    def __init__(self,
                 env: VectorEnv,
                 eval_env: VectorEnv,
                 config: SimpleNamespace,
                 logger: Logger | None = None):
        super().__init__()
        assert hasattr(env, "num_envs"), "The environment must have a 'num_envs' attribute."
        self.config = self._parse_config(config)

        # Create wrapped environments so that the observations and actions are always Tensors
        self.env = TensorWrapper(env, device=self.config.train.device)
        self.eval_env = TensorWrapper(eval_env, device=self.config.train.device)
        self.logger = logger

        # Initialize the policy with actor and critic networks
        obs_shape, _ = space_info(env, mode="obs")
        action_shape, _ = space_info(env, mode="action")
        self.config.train.actor_kwargs |= {"obs_dim": obs_shape[0], "action_dim": action_shape[0]}
        actor = self.config.train.actor_class(**self.config.train.actor_kwargs)
        self.config.train.policy_kwargs["actor"] = actor
        self.config.train.critic_kwargs |= {"obs_dim": obs_shape[0], "action_dim": action_shape[0]}
        critic = self.config.train.critic_class(**self.config.train.critic_kwargs)
        self.config.train.policy_kwargs["critic"] = critic
        self.policy = DDPGPolicy(**self.config.train.policy_kwargs, device=self.config.train.device)
        # Initialize the optimizers
        self.actor_optimizer = torch.optim.AdamW(self.policy.actor.parameters(),
                                                 lr=self.config.train.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic.parameters(),
                                                  lr=self.config.train.critic_lr)
        # Initialize the replay buffer
        self.config.rollout.replay_buffer_kwargs |= {"env": env, "device": self.config.train.device}
        buffer_cls = self.config.rollout.replay_buffer_class
        self.buffer = buffer_cls(**self.config.rollout.replay_buffer_kwargs)

        # Save rollout, train, eval and checkpoint info into separate dictionaries
        self.rollout_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.config.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.config.train.device),
        }
        self.train_info = {"num_samples": 0, "num_gradient_steps": 0}
        self.eval_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.config.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.config.train.device),
        }
        self.checkpoint_info = {"num_samples": 0}

    @property
    def stop_condition(self):
        max_samples = self.rollout_info["num_samples"] >= self.config.rollout.max_samples
        return max_samples

    @property
    def train_condition(self):
        if self.rollout_info["num_samples"] < self.config.train.batch_size:
            return False
        num_samples = self.rollout_info["num_samples"] - self.train_info["num_samples"]
        if num_samples >= self.config.train.actor_freq:
            return True
        if num_samples >= self.config.train.critic_freq:
            return True
        return False

    @property
    def eval_condition(self):
        num_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        return num_samples >= self.config.eval.freq

    @property
    def checkpoint_condition(self):
        if self.config.checkpoint.freq is None:
            return False
        num_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
        return num_samples >= self.config.checkpoint.freq

    def train(self):
        while not self.stop_condition:
            self.collect_samples()
            if self.train_condition:
                self.train_policy()
            if self.eval_condition:
                self.evaluate_policy()
            if self.checkpoint_condition:
                self.save_checkpoint()
        self.logger.stop()

    @torch.no_grad()
    def collect_samples(self):
        self.policy.actor.eval()

        # If first rollout, reset the environment
        if not "obs" in self.rollout_info:
            self.rollout_info["obs"], _ = self.env.reset()
        obs = self.rollout_info["obs"]

        # Calculate how many samples to collect before we need to interrupt for any callbacks
        required_samples = self.rollout_info["num_samples"] + self._next_required_samples()
        while self.rollout_info["num_samples"] < required_samples:
            action = self.policy.actor(obs)
            action += torch.randn_like(action) * self.config.rollout.action_noise
            action = torch.clamp(action, -1, 1)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            self.buffer.add(obs, action, reward, next_obs, terminated, truncated)
            obs = next_obs
            self.rollout_info["num_samples"] += self.env.num_envs
            self.rollout_info["steps"] += 1
            self.rollout_info["rewards"] += reward

            # If any of the environments are terminated or truncated, log the episode statistics
            if any(terminated) or any(truncated):
                ep_steps = self.rollout_info["steps"][terminated | truncated].mean()
                ep_reward = self.rollout_info["rewards"][terminated | truncated].mean()
                data = {"rollout/ep_steps": ep_steps, "rollout/ep_reward": ep_reward}
                self.logger.log(data, step=self.rollout_info["num_samples"])
                self.rollout_info["steps"][terminated | truncated] = 0
                self.rollout_info["rewards"][terminated | truncated] = 0

        self.rollout_info["obs"] = obs
        self.policy.actor.train()

    def train_policy(self):
        self.policy.actor.train()  # Critic is always in train mode, not used for inference
        for _ in range(self.config.train.gradient_steps):
            # Sample experience from the replay buffer
            batch = self.buffer.sample(self.config.train.gradient_steps)
            obs, action, reward, next_obs, terminated, truncated = batch

            if self.train_info["num_gradient_steps"] % self.config.train.critic_freq == 0:
                # Compute the expected Q values with the reward and the target networks
                with torch.no_grad():
                    next_action = self.policy.actor.target(next_obs)
                    next_q_target = self.policy.critic.target(next_obs, next_action)
                    q_target = reward + (self.config.train.gamma * ~terminated * next_q_target)
                    # TODO: include reward clipping?
                # Compute the loss as the MSE between the expected Q values and the Q values from
                # the critic
                q_expected = self.policy.critic(obs, action)
                critic_loss = torch.mean((q_expected - q_target)**2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

            if self.train_info["num_gradient_steps"] % self.config.train.actor_freq == 0:
                # Compute the actions for the sample observations, compute the critic value of the
                # observations and actions and compute the actor loss by maximizing the critic value
                train_action = self.policy.actor(obs)
                action_noise = torch.randn_like(train_action) * self.config.train.action_noise
                train_action = torch.clamp(train_action + action_noise, -1, 1)
                actor_loss = -self.policy.critic(obs, train_action).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

            # Update 'num_gradient_steps' before target networks to prevent updating them at the
            # first training step
            self.train_info["num_gradient_steps"] += 1

            # Update the target networks
            if self.train_info["num_gradient_steps"] % self.config.train.actor_target_freq == 0:
                self.policy.actor.update_target(self.config.train.tau)
            if self.train_info["num_gradient_steps"] % self.config.train.critic_target_freq == 0:
                self.policy.critic.update_target(self.config.train.tau)
        self._update_train_info()

    @torch.no_grad()
    def evaluate_policy(self):
        self.policy.actor.eval()
        obs, _ = self.eval_env.reset()
        num_samples = 0
        rewards, ep_rewards, ep_steps = [], [], []
        while num_samples < self.config.eval.steps:
            action = self.policy.action(obs)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            self.eval_info["rewards"] += reward
            rewards += reward.tolist()
            self.eval_info["steps"] += 1

            if any(terminated) or any(truncated):
                ep_steps += self.eval_info["steps"][terminated | truncated].tolist()
                ep_rewards += self.eval_info["rewards"][terminated | truncated].tolist()
                self.eval_info["steps"][terminated | truncated] = 0
                self.eval_info["rewards"][terminated | truncated] = 0

            num_samples += self.eval_env.num_envs

        data = {
            "eval/ep_steps": np.array(ep_steps).mean(),
            "eval/ep_reward": np.array(ep_rewards).mean(),
            "eval/mean_reward": np.array(rewards).mean(),
        }
        self.logger.log(data, step=self.rollout_info["num_samples"])
        self._update_eval_info()
        self.policy.actor.train()

    def save_checkpoint(self):
        assert self.config.checkpoint.path.is_dir(), "The checkpoint path must be a directory."
        self.policy.save(self.config.checkpoint.path / "policy.pt")
        self.buffer.save(self.config.checkpoint.path / "buffer.pt")
        self.checkpoint_info["num_samples"] = self.rollout_info["num_samples"]

    def _next_required_samples(self):
        # Calculate required samples for next training step
        current_samples = self.rollout_info["num_samples"] - self.train_info["num_samples"]
        actor_train_samples = self.config.train.actor_freq - current_samples
        critic_train_samples = self.config.train.critic_freq - current_samples
        train_samples = min([actor_train_samples, critic_train_samples])
        # Check if we have enough samples for a batch. If not, collect as many samples as required
        # to fill a batch
        if self.rollout_info["num_samples"] - self.config.train.batch_size < 0:
            if train_samples < self.config.train.batch_size - self.rollout_info["num_samples"]:
                train_samples = self.config.train.batch_size - self.rollout_info["num_samples"]
        if train_samples == 0:
            train_samples = min((self.config.train.actor_freq, self.config.train.critic_freq))

        # Calculate required samples for next eval step
        current_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        eval_samples = self.config.eval.freq - current_samples
        eval_samples = eval_samples if eval_samples > 0 else self.config.eval.freq
        # Calculate required samples for next checkpoint
        if self.config.checkpoint.freq is None:
            checkpoint_samples = np.inf
        else:
            current_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
            checkpoint_samples = self.config.checkpoint.freq - current_samples
        return min([train_samples, eval_samples, checkpoint_samples])

    def _update_train_info(self):
        self.train_info["num_samples"] = self.rollout_info["num_samples"]

    def _update_eval_info(self):
        self.eval_info["num_samples"] = self.rollout_info["num_samples"]
        # Reset the steps and rewards for future eval runs. Otherwise, the next eval run adds to
        # the values from the previous run
        self.eval_info["steps"][...] = 0
        self.eval_info["rewards"][...] = 0

    def _parse_config(self, config: SimpleNamespace) -> DDPGConfig:
        # Create env config
        env_config = EnvConfig(**vars(config.env))
        rollout_config = RolloutConfig(**vars(config.rollout))
        train_config = TrainConfig(**vars(config.train))
        eval_config = EvalConfig(**vars(config.eval))
        checkpoint_config = CheckpointConfig(config.checkpoint.freq, Path(config.checkpoint.path))

        # Check if the config is valid
        for freq in ("actor_freq", "actor_target_freq", "critic_freq", "critic_target_freq"):
            freq = getattr(train_config, freq)
            assert freq > 0, "All training frequencies must be greater than 0."
            if not freq % env_config.kwargs["num_envs"] == 0:
                raise ValueError((f"The frequency ({freq}) must be divisible by 'num_envs' "
                                  f"({env_config.num_envs})."))
        if not eval_config.freq % env_config.kwargs["num_envs"] == 0:
            raise ValueError((f"The 'eval_freq' ({eval_config.freq}) must be divisible by "
                              f"'num_envs' ({env_config.num_envs})."))
        if checkpoint_config.freq is not None:
            if checkpoint_config.freq is None:
                raise ValueError("If 'checkpoint_freq' is not None, 'checkpoint_path' must be "
                                 "specified.")
            if not checkpoint_config.freq % env_config.kwargs["num_envs"] == 0:
                raise ValueError((f"The 'checkpoint_freq' ({checkpoint_config.freq}) must be "
                                  f"divisible by 'num_envs' ({env_config.num_envs})."))
        return DDPGConfig(env_config, rollout_config, train_config, eval_config, checkpoint_config)
