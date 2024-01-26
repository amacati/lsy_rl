import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
from gymnasium.vector import VectorEnv

import lsy_rl
from lsy_rl.core import Algorithm
from lsy_rl.core.logger import Logger, EmptyLogger
from lsy_rl.wrappers.tensordict_wrapper import TensorDictWrapper
from lsy_rl.ddpg.config import DDPGConfig, EnvConfig, TrainConfig, EvalConfig, CheckpointConfig
from lsy_rl.ddpg.config import RolloutConfig
from lsy_rl.ddpg.policy import DDPGPolicy
from lsy_rl.core.replay_buffer import HerVectorReplayBuffer

logger = logging.getLogger(__name__)


class DDPG(Algorithm):

    def __init__(self,
                 env: VectorEnv,
                 eval_env: VectorEnv,
                 config: SimpleNamespace,
                 logger: Logger = EmptyLogger()):
        super().__init__()
        assert hasattr(env, "num_envs"), "The environment must have a 'num_envs' attribute."
        self.config = self._parse_config(config)

        # Create wrapped environments so that the observations and actions are always Tensors
        self.env = TensorDictWrapper(env, device=self.config.train.device)
        self.eval_env = TensorDictWrapper(eval_env, device=self.config.train.device)
        self.separate_eval_env = eval_env is not env
        self.logger = logger

        # Initialize the policy with actor and critic networks
        spaces = {"obs_space": env.observation_space, "action_space": env.action_space}
        self.config.train.actor_kwargs |= spaces
        actor = self.config.train.actor_class(**self.config.train.actor_kwargs)
        self.config.train.policy_kwargs["actor"] = actor
        self.config.train.critic_kwargs |= spaces
        critic = self.config.train.critic_class(**self.config.train.critic_kwargs)
        self.config.train.policy_kwargs["critic"] = critic
        self.policy = DDPGPolicy(**self.config.train.policy_kwargs, device=self.config.train.device)
        # Initialize the optimizers
        self.actor_optimizer = torch.optim.AdamW(self.policy.actor.parameters(),
                                                 lr=self.config.train.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic.parameters(),
                                                  lr=self.config.train.critic_lr)
        # Initialize the replay buffer
        self.config.rollout.replay_buffer_kwargs |= {
            "num_envs": self.env.num_envs,
            "device": self.config.train.device
        }
        buffer_cls = self.config.rollout.replay_buffer_class
        self.buffer = buffer_cls(**self.config.rollout.replay_buffer_kwargs)

        # Save rollout, train, eval and checkpoint info into separate dictionaries
        self.rollout_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.config.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.config.train.device),
        }
        self.train_info = {
            "num_samples": 0,
            "num_gradient_steps": 0,
            "actor_loss": 0,
            "actor_steps_since_log": 0,
            "critic_loss": 0,
            "critic_steps_since_log": 0
        }
        freq = min((self.config.train.actor_freq, self.config.train.critic_freq))
        grad_steps = (self.config.rollout.max_samples // freq) * self.config.train.gradient_steps
        self.train_info["log_freq"] = max(1, grad_steps // 1000)  # Log 1000 times during training
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
            self.rollout_info["obs"] = self.env.reset()["obs"]
        obs = self.rollout_info["obs"]

        # Calculate how many samples to collect before we need to interrupt for any callbacks
        required_samples = self.rollout_info["num_samples"] + self._next_required_samples()
        while self.rollout_info["num_samples"] < required_samples:
            action = self.policy.actor(obs)
            action = action + torch.randn_like(action) * self.config.rollout.action_noise
            action = torch.clamp(action, -1, 1)
            sample = self.env.step(action)
            sample["obs"], sample["action"] = obs, action
            self.buffer.add(sample)
            obs = sample["next_obs"]
            self.rollout_info["num_samples"] += self.env.num_envs
            self.rollout_info["steps"] += 1
            self.rollout_info["rewards"] += sample["reward"]

            # If any of the environments are terminated or truncated, log the episode statistics
            if any(sample["terminated"]) or any(sample["truncated"]):
                idx = sample["terminated"] | sample["truncated"]
                ep_steps = self.rollout_info["steps"][idx].mean()
                ep_reward = self.rollout_info["rewards"][idx].mean()
                data = {"rollout/ep_steps": ep_steps, "rollout/ep_reward": ep_reward}
                self.logger.log(data, step=self.rollout_info["num_samples"])
                self.rollout_info["steps"][idx] = 0
                self.rollout_info["rewards"][idx] = 0

        self.rollout_info["obs"] = obs
        self.policy.actor.train()

    def train_policy(self):
        self.policy.actor.train()  # Critic is always in train mode, not used for inference
        for _ in range(self.config.train.gradient_steps):
            # Sample experience from the replay buffer

            if self.train_info["num_gradient_steps"] % self.config.train.critic_freq == 0:
                batch = self.buffer.sample(self.config.train.batch_size)
                # Compute the expected Q values with the reward and the target networks
                with torch.no_grad():
                    next_action = self.policy.actor.target(batch["next_obs"])
                    next_q_target = self.policy.critic.target(batch["next_obs"], next_action)
                    # Reward, terminated are one-dimensional, so we need to reshape them to avoid
                    # broadcasting errors
                    reward = batch["reward"].reshape(-1, 1)
                    terminated = batch["terminated"].reshape(-1, 1)
                    q_target = reward + (self.config.train.gamma * ~terminated * next_q_target)
                    # TODO: include reward clipping?
                # Compute the loss as the MSE between the expected Q values and the Q values from
                # the critic
                q_expected = self.policy.critic(batch["obs"], batch["action"])
                assert q_target.shape == (self.config.train.batch_size, 1), q_target.shape
                critic_loss = torch.mean((q_expected - q_target)**2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                self.train_info["critic_loss"] += critic_loss.detach()
                self.train_info["critic_steps_since_log"] += 1

            if self.train_info["num_gradient_steps"] % self.config.train.actor_freq == 0:
                batch = self.buffer.sample(self.config.train.batch_size)
                # Compute the actions for the sample observations, compute the critic value of the
                # observations and actions and compute the actor loss by maximizing the critic value
                train_action = self.policy.actor(batch["obs"])
                action_noise = torch.randn_like(train_action) * self.config.train.action_noise
                train_action = torch.clamp(train_action + action_noise, -1, 1)
                actor_loss = -self.policy.critic(batch["obs"], train_action).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.train_info["actor_loss"] += actor_loss.detach()
                self.train_info["actor_steps_since_log"] += 1
            # Update 'num_gradient_steps' before target networks to prevent updating them at the
            # first training step
            self.train_info["num_gradient_steps"] += 1

            self._rate_limit_train_log()
            # Update the target networks
            if self.train_info["num_gradient_steps"] % self.config.train.actor_target_freq == 0:
                self.policy.actor.update_target(self.config.train.tau)
            if self.train_info["num_gradient_steps"] % self.config.train.critic_target_freq == 0:
                self.policy.critic.update_target(self.config.train.tau)

        self._update_train_info()

    @torch.no_grad()
    def evaluate_policy(self):
        self.policy.actor.eval()
        obs = self.eval_env.reset()["obs"]
        num_samples = 0
        rewards, ep_rewards, ep_steps = [], [], []
        while num_samples < self.config.eval.steps:
            action = self.policy.action(obs)
            sample = self.eval_env.step(action)
            obs = sample["next_obs"]
            self.eval_info["rewards"] += sample["reward"]
            rewards += sample["reward"].tolist()
            self.eval_info["steps"] += 1

            if any(sample["terminated"]) or any(sample["truncated"]):
                idx = sample["terminated"] | sample["truncated"]
                ep_steps += self.eval_info["steps"][idx].tolist()
                ep_rewards += self.eval_info["rewards"][idx].tolist()
                self.eval_info["steps"][idx] = 0
                self.eval_info["rewards"][idx] = 0

            num_samples += self.eval_env.num_envs

        data = {"eval/mean_reward": np.array(rewards).mean()}
        if ep_rewards:
            data["eval/ep_mean_reward"] = np.array(ep_rewards).mean()
            data["eval/ep_mean_steps"] = np.array(ep_steps).mean()
        self.logger.log(data, step=self.rollout_info["num_samples"])
        self._update_eval_info()
        # If the train and eval envs are the same, we need to reset the env and save the obs to the
        # rollout info. Otherwise, the next rollout will start from the last state of the eval env,
        # but will still use the last observation from the latest rollout
        if not self.separate_eval_env:
            self.rollout_info["obs"] = self.env.reset()["obs"]
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

    def _rate_limit_train_log(self):
        if self.train_info["num_gradient_steps"] % self.train_info["log_freq"] == 0:
            data = {}
            if self.train_info["actor_steps_since_log"] > 0:
                data["train/actor_loss"] = (self.train_info["actor_loss"] /
                                            self.train_info["actor_steps_since_log"])
                self.train_info["actor_loss"] = 0
                self.train_info["actor_steps_since_log"] = 0
            if self.train_info["critic_steps_since_log"] > 0:
                data["train/critic_loss"] = (self.train_info["critic_loss"] /
                                             self.train_info["critic_steps_since_log"])
                self.train_info["critic_loss"] = 0
                self.train_info["critic_steps_since_log"] = 0
            if data:
                self.logger.log(data, step=self.rollout_info["num_samples"])

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
        if hasattr(config.rollout, "replay_buffer_class"):
            if isinstance(config.rollout.replay_buffer_class, str):
                config.rollout.replay_buffer_class = getattr(lsy_rl.core.replay_buffer,
                                                             config.rollout.replay_buffer_class)
            if config.rollout.replay_buffer_class is HerVectorReplayBuffer:
                if not "reward_fn" in config.rollout.replay_buffer_kwargs:
                    raise ValueError("If 'replay_buffer_class' is HerVectorReplayBuffer, "
                                     "'reward_fn' must be specified.")
        rollout_config = RolloutConfig(**vars(config.rollout))
        train_config = TrainConfig(**vars(config.train))
        eval_config = EvalConfig(**vars(config.eval))
        if hasattr(config.checkpoint, "path"):
            config.checkpoint.path = Path(config.checkpoint.path)
        checkpoint_config = CheckpointConfig(**vars(config.checkpoint))

        # Check if the config is valid
        for freq in ("actor_freq", "actor_target_freq", "critic_freq", "critic_target_freq"):
            _freq = getattr(train_config, freq)
            assert _freq > 0, "All training frequencies must be greater than 0."
            if not _freq % env_config.kwargs["num_envs"] == 0:
                raise ValueError((f"The frequency ({freq}) must be divisible by 'num_envs' "
                                  f"({env_config.kwargs['num_envs']})."))
        if not eval_config.freq % env_config.kwargs["num_envs"] == 0:
            raise ValueError((f"The 'eval_freq' ({eval_config.freq}) must be divisible by "
                              f"'num_envs' ({env_config.kwargs['num_envs']})."))
        if checkpoint_config.freq is not None:
            if checkpoint_config.freq is None:
                raise ValueError("If 'checkpoint_freq' is not None, 'checkpoint_path' must be "
                                 "specified.")
            if not checkpoint_config.freq % env_config.kwargs["num_envs"] == 0:
                raise ValueError((f"The 'checkpoint_freq' ({checkpoint_config.freq}) must be "
                                  f"divisible by 'num_envs' ({env_config.kwargs['num_envs']})."))
        return DDPGConfig(env_config, rollout_config, train_config, eval_config, checkpoint_config)
