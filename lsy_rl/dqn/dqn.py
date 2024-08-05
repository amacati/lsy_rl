from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from lsy_rl.core import Algorithm
from lsy_rl.core.logger import Logger
from lsy_rl.dqn.config import (
    CheckpointConfig,
    DQNConfig,
    EnvConfig,
    EvalConfig,
    RolloutConfig,
    TrainConfig,
)
from lsy_rl.dqn.policy import DQNPolicy
from lsy_rl.utils import space_info
from lsy_rl.wrappers.tensordict_wrapper import TensorDictWrapper


class DQN(Algorithm):
    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        config: SimpleNamespace,
        logger: Logger | None = None,
    ):
        super().__init__()
        assert hasattr(env, "num_envs"), "The environment must have a 'num_envs' attribute."
        self.config = self._parse_config(config)
        # Create wrapped environments so that the observations and actions are always Tensors
        self.env = TensorDictWrapper(env, device=self.config.train.device)
        self.eval_env = TensorDictWrapper(eval_env, device=self.config.train.device)
        # Check if the action space is multi-discrete. Discrete action spaces are converted to
        # multi-discrete for vectorized environments, and we only support vectorized environments
        if not isinstance(self.env.action_space, spaces.MultiDiscrete):
            raise TypeError(
                ("The action space must be multi-discrete, is type " f"{self.env.action_space}.")
            )
        assert all(nvec == self.env.action_space.nvec[0] for nvec in self.env.action_space.nvec)
        self.logger = logger

        # Initialize the policy
        obs_shape, _ = space_info(env, mode="obs")
        self.num_actions = self.env.action_space.nvec[0]
        self.config.train.net_kwargs |= {"obs_dim": obs_shape[0], "action_dim": self.num_actions}
        network = self.config.train.net_cls(**self.config.train.net_kwargs)
        self.config.train.policy_kwargs["network"] = network
        self.policy = DQNPolicy(**self.config.train.policy_kwargs, device=self.config.train.device)
        # Initialize the optimizers
        self.optimizer = torch.optim.AdamW(self.policy.dqn.parameters(), lr=self.config.train.lr)
        # Initialize the replay buffer
        self.config.rollout.replay_buffer_kwargs |= {"env": env, "device": self.config.train.device}
        buffer_cls = self.config.rollout.replay_buffer_cls
        self.buffer = buffer_cls(**self.config.rollout.replay_buffer_kwargs)

        # Save rollout, train, eval and checkpoint info into separate dictionaries
        self.rollout_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.config.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.config.train.device),
        }
        self.train_info = {"num_samples": 0, "num_gradient_steps": 0, "loss": 0}
        # Reduce the number of log entries during training for performance reasons
        train_steps = self.config.rollout.max_samples // self.config.train.period
        grad_steps = train_steps * self.config.train.gradient_steps
        self.train_info["log_period"] = max(1, grad_steps // 1000)  # Log 1000 times during training
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
        if num_samples >= self.config.train.period:
            return True
        return False

    @property
    def eval_condition(self):
        num_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        return num_samples >= self.config.eval.period

    @property
    def checkpoint_condition(self):
        if self.config.checkpoint.period is None:
            return False
        num_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
        return num_samples >= self.config.checkpoint.period

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
        self.policy.dqn.eval()

        # If first rollout, reset the environment
        if not "obs" in self.rollout_info:
            self.rollout_info["obs"], _ = self.env.reset()
        obs = self.rollout_info["obs"]

        # Calculate how many samples to collect before we need to interrupt for any callbacks
        required_samples = self.rollout_info["num_samples"] + self._next_required_samples()
        while self.rollout_info["num_samples"] < required_samples:
            action = self.policy.action(obs)
            # Add exploration noise to the action
            rand_actions = torch.randint_like(action, 0, self.num_actions)
            random_mask = torch.rand_like(action, dtype=torch.float32) < self.config.rollout.epsilon
            action[random_mask] = rand_actions[random_mask]
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
        self.policy.dqn.train()

    def train_policy(self):
        self.policy.dqn.train()
        for _ in range(self.config.train.gradient_steps):
            # Sample experience from the replay buffer
            batch = self.buffer.sample(self.config.train.batch_size)
            obs, action, reward, next_obs, terminated, truncated = batch
            dqn, target_dqn = self.policy.dqn.get_network_and_target()
            self.optimizer.zero_grad()
            q = dqn(obs)[range(self.config.train.batch_size), action]
            with torch.no_grad():
                a_next = torch.max(dqn(next_obs), 1).indices
                q_next = target_dqn(next_obs)[range(self.config.train.batch_size), a_next]
                q_next = torch.clamp(q_next, -self.config.train.q_clip, self.config.train.q_clip)
                q_td = reward + self.config.train.gamma * q_next * ~terminated
            loss = (q - q_td).pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), self.config.train.grad_clip)
            self.optimizer.step()
            self.train_info["num_gradient_steps"] += 1
            self.train_info["loss"] += loss.detach()  # Accumulate loss for logging
            if self.train_info["num_gradient_steps"] % self.train_info["log_period"] == 0:
                data = {"train/loss": self.train_info["loss"] / self.train_info["log_period"]}
                self.logger.log(data, step=self.rollout_info["num_samples"])
                self.train_info["loss"] = 0
        self._update_train_info()

    @torch.no_grad()
    def evaluate_policy(self):
        self.policy.dqn.eval()
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
        self.policy.dqn.train()

    def save_checkpoint(self):
        assert self.config.checkpoint.path.is_dir(), "The checkpoint path must be a directory."
        self.policy.save(self.config.checkpoint.path / "policy.pt")
        self.buffer.save(self.config.checkpoint.path / "buffer.pt")
        self.checkpoint_info["num_samples"] = self.rollout_info["num_samples"]

    def _next_required_samples(self):
        # Calculate required samples for next training step
        current_samples = self.rollout_info["num_samples"] - self.train_info["num_samples"]
        train_samples = self.config.train.period - current_samples
        # Check if we have enough samples for a batch. If not, collect as many samples as required
        # to fill a batch
        if self.rollout_info["num_samples"] - self.config.train.batch_size < 0:
            if train_samples < self.config.train.batch_size - self.rollout_info["num_samples"]:
                train_samples = self.config.train.batch_size - self.rollout_info["num_samples"]
        if train_samples == 0:
            train_samples = self.config.train.period
        # Calculate required samples for next eval step
        current_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        eval_samples = self.config.eval.period - current_samples
        eval_samples = eval_samples if eval_samples > 0 else self.config.eval.period
        # Calculate required samples for next checkpoint
        if self.config.checkpoint.period is None:
            checkpoint_samples = np.inf
        else:
            current_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
            checkpoint_samples = self.config.checkpoint.period - current_samples
        return min([train_samples, eval_samples, checkpoint_samples])

    def _update_train_info(self):
        self.train_info["num_samples"] = self.rollout_info["num_samples"]

    def _update_eval_info(self):
        self.eval_info["num_samples"] = self.rollout_info["num_samples"]
        # Reset the steps and rewards for future eval runs. Otherwise, the next eval run adds to
        # the values from the previous run
        self.eval_info["steps"][...] = 0
        self.eval_info["rewards"][...] = 0

    def _parse_config(self, config: SimpleNamespace) -> DQNConfig:
        # Create env config
        env_config = EnvConfig(**vars(config.env))
        rollout_config = RolloutConfig(**vars(config.rollout))
        train_config = TrainConfig(**vars(config.train))
        eval_config = EvalConfig(**vars(config.eval))
        checkpoint_config = CheckpointConfig(config.checkpoint.period, Path(config.checkpoint.path))

        # Check if the config is valid
        assert train_config.period > 0, "The training period must be greater than 0."
        if not train_config.period % env_config.kwargs["num_envs"] == 0:
            raise ValueError(
                (
                    f"The train period ({train_config.period}) must be divisible by "
                    f" 'num_envs' ({env_config.num_envs})."
                )
            )
        if not eval_config.period % env_config.kwargs["num_envs"] == 0:
            raise ValueError(
                (
                    f"The eval period ({eval_config.period}) must be divisible by "
                    f"'num_envs' ({env_config.num_envs})."
                )
            )
        if checkpoint_config.period is not None:
            if checkpoint_config.period is None:
                raise ValueError(
                    "If 'checkpoint_period' is not None, 'checkpoint_path' must be " "specified."
                )
            if not checkpoint_config.period % env_config.kwargs["num_envs"] == 0:
                raise ValueError(
                    (
                        f"The 'checkpoint_period' ({checkpoint_config.period}) must be "
                        f"divisible by 'num_envs' ({env_config.num_envs})."
                    )
                )
        return DQNConfig(env_config, rollout_config, train_config, eval_config, checkpoint_config)
