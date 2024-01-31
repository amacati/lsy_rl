import logging
from types import SimpleNamespace

import torch
import numpy as np
from gymnasium.vector import VectorEnv

from lsy_rl.core import Algorithm
from lsy_rl.core.logger import Logger, EmptyLogger
from lsy_rl.wrappers.tensordict_wrapper import TensorDictWrapper, DefaultTensorDictWrapper
from lsy_rl.ddpg.config import DDPGConfig, EnvConfig, TrainConfig, EvalConfig, CheckpointConfig
from lsy_rl.ddpg.config import RolloutConfig
from lsy_rl.ddpg.policy import DDPGPolicy

logger = logging.getLogger(__name__)


class DDPG(Algorithm):

    num_logs: int = 200

    def __init__(self,
                 env: VectorEnv,
                 eval_env: VectorEnv,
                 config: SimpleNamespace,
                 logger: Logger = EmptyLogger()):
        super().__init__()
        assert hasattr(env, "num_envs"), "The environment must have a 'num_envs' attribute."
        self.cfg = self._parse_config(config)

        # Create wrapped environments so that the observations and actions are always Tensors
        if not isinstance(env, TensorDictWrapper):
            env = DefaultTensorDictWrapper(env, device=self.cfg.train.device)
        self.env = env
        if not isinstance(eval_env, TensorDictWrapper):
            eval_env = DefaultTensorDictWrapper(eval_env, device=self.cfg.train.device)
        self.eval_env = eval_env
        self.separate_eval_env = eval_env is not env
        self.logger = logger

        # Initialize the policy with actor and critic networks
        spaces = {"obs_space": env.observation_space, "action_space": env.action_space}
        self.cfg.train.actor_kwargs |= spaces
        actor = self.cfg.train.actor_cls(**self.cfg.train.actor_kwargs)
        self.cfg.train.policy_kwargs["actor"] = actor
        self.cfg.train.critic_kwargs |= spaces
        critic = self.cfg.train.critic_cls(**self.cfg.train.critic_kwargs)
        self.cfg.train.policy_kwargs["critic"] = critic
        self.policy = DDPGPolicy(**self.cfg.train.policy_kwargs, device=self.cfg.train.device)
        # Initialize the optimizers
        self.actor_optimizer = torch.optim.AdamW(self.policy.actor.parameters(),
                                                 lr=self.cfg.train.actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.policy.critic.parameters(),
                                                  lr=self.cfg.train.critic_lr)
        # Put transforms on the same device as the policy
        device = self.cfg.train.device
        self.cfg.rollout.action_transform = self.cfg.rollout.action_transform.to(device)
        self.cfg.train.action_transform = self.cfg.train.action_transform.to(device)
        self.cfg.train.target_action_transform = self.cfg.train.target_action_transform.to(device)
        self.cfg.eval.action_transform = self.cfg.eval.action_transform.to(device)

        # Initialize the replay buffer
        self.cfg.rollout.replay_buffer_kwargs |= {
            "num_envs": self.env.num_envs,
            "device": self.cfg.train.device
        }
        buffer_cls = self.cfg.rollout.replay_buffer_cls
        self.buffer = buffer_cls(**self.cfg.rollout.replay_buffer_kwargs)

        # Save rollout, train, eval and checkpoint info into separate dictionaries
        self.rollout_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.cfg.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.cfg.train.device),
            "log_freq": max(1, self.cfg.rollout.max_samples // self.num_logs),
            "log": {
                "ep_steps": 0,
                "ep_reward": 0,
                "ep_count": 0
            }
        }
        self.train_info = {
            "num_samples": 0,
            "num_train_steps": 0,
            "log": {
                "actor_loss": 0,
                "critic_loss": 0,
                "actor_steps_since_log": 0,
                "critic_steps_since_log": 0
            }
        }
        num_trainings = (self.cfg.rollout.max_samples // self.cfg.train.freq)
        total_train_steps = num_trainings * self.cfg.train.steps
        self.train_info["log_freq"] = max(1, total_train_steps // self.num_logs)
        self.eval_info = {
            "num_samples": 0,
            "rewards": torch.zeros(self.env.num_envs, device=self.cfg.train.device),
            "steps": torch.zeros(self.env.num_envs, device=self.cfg.train.device),
        }
        self.checkpoint_info = {"num_samples": 0}

    @property
    def stop_condition(self):
        max_samples = self.rollout_info["num_samples"] >= self.cfg.rollout.max_samples
        return max_samples

    @property
    def train_condition(self):
        if self.rollout_info["num_samples"] < self.cfg.train.batch_size:
            return False  # Make sure we have enough samples for a batch
        num_samples = self.rollout_info["num_samples"] - self.train_info["num_samples"]
        if num_samples >= self.cfg.train.freq:
            return True
        return False

    @property
    def eval_condition(self):
        num_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        return num_samples >= self.cfg.eval.freq

    @property
    def checkpoint_condition(self):
        if self.cfg.checkpoint.freq is None:
            return False
        num_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
        return num_samples >= self.cfg.checkpoint.freq

    def train(self):
        self.evaluate_policy()  # Establish an initial baseline
        while not self.stop_condition:
            self.collect_samples()
            if self.train_condition:
                self.train_policy()
            if self.eval_condition:
                self.evaluate_policy()
            if self.checkpoint_condition:
                self.save_checkpoint()
        if self.cfg.checkpoint.path is not None:
            self.save_checkpoint()  # Save the final checkpoint even if we don't reach the freq
        self.logger.stop()

    @torch.no_grad()
    def collect_samples(self):
        self.policy.actor.eval()

        # If first rollout, reset the environment
        if not "obs" in self.rollout_info:
            self.rollout_info["obs"] = self.env.reset()
        obs = self.rollout_info["obs"]

        # Calculate how many samples to collect before we need to interrupt for any callbacks
        required_samples = self.rollout_info["num_samples"] + self._next_required_samples()
        while self.rollout_info["num_samples"] < required_samples:
            action = self.policy.actor(obs["obs"])
            action, _ = self.cfg.rollout.action_transform(action, obs)
            sample = self.env.step(action)
            sample["obs"], sample["action"] = obs["obs"], action
            self.buffer.add(sample)
            obs["obs"] = sample["next_obs"]
            self.rollout_info["num_samples"] += self.env.num_envs
            self.rollout_info["steps"] += 1
            self.rollout_info["rewards"] += sample["reward"]

            # If any of the environments are terminated or truncated, log the episode statistics
            if any(sample["terminated"]) or any(sample["truncated"]):
                idx = sample["terminated"] | sample["truncated"]
                self.rollout_info["log"]["ep_steps"] += (self.rollout_info["steps"][idx].sum())
                self.rollout_info["log"]["ep_reward"] += (self.rollout_info["rewards"][idx].sum())
                self.rollout_info["log"]["ep_count"] += len(idx)
                self.rollout_info["steps"][idx] = 0
                self.rollout_info["rewards"][idx] = 0

            self._rate_limit_rollout_log()

        self.rollout_info["obs"] = obs
        self.policy.actor.train()

    def train_policy(self):
        self.policy.actor.train()  # Critic is always in train mode, not used for inference

        for _ in range(self.cfg.train.steps):
            # Update 'num_train_steps' at the beginning of the loop so that lower frequency updates
            # do not get executed at the first iteration when 'num_train_steps' is 0
            self.train_info["num_train_steps"] += 1

            if self.train_info["num_train_steps"] % self.cfg.train.critic_freq == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the expected Q values with the reward and the target networks
                with torch.no_grad():
                    next_action = self.policy.actor.target(batch["next_obs"])
                    next_action, _ = self.cfg.train.target_action_transform(next_action, batch)
                    next_q_target = self.policy.critic.target(batch["next_obs"], next_action)
                    # Reward, terminated are one-dimensional, so we need to reshape them to avoid
                    # broadcasting errors
                    reward = batch["reward"].reshape(-1, 1)
                    terminated = batch["terminated"].reshape(-1, 1)
                    q_target = reward + (self.cfg.train.gamma * ~terminated * next_q_target)
                    q_target = torch.clamp(q_target, self.cfg.train.reward_clip[0],
                                           self.cfg.train.reward_clip[1])
                # Compute the loss as the MSE between the expected Q values and the Q values from
                # the critic
                q_expected = self.policy.critic(batch["obs"], batch["action"])
                assert q_target.shape == (self.cfg.train.batch_size, 1), q_target.shape
                critic_loss = torch.mean((q_expected - q_target)**2)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(),
                                               self.cfg.train.grad_clip)
                self.critic_optimizer.step()
                self.train_info["log"]["critic_loss"] += critic_loss.detach()
                self.train_info["log"]["critic_steps_since_log"] += 1

            if self.train_info["num_train_steps"] % self.cfg.train.actor_freq == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the actions for the sample observations, compute the critic value of the
                # observations and actions and compute the actor loss by maximizing the critic value
                train_action = self.policy.actor(batch["obs"])
                train_action, _ = self.cfg.train.action_transform(train_action, batch)
                actor_loss = -self.policy.critic(batch["obs"], train_action).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(),
                                               self.cfg.train.grad_clip)
                self.actor_optimizer.step()
                self.train_info["log"]["actor_loss"] += actor_loss.detach()
                self.train_info["log"]["actor_steps_since_log"] += 1

            self._rate_limit_train_log()
            # Update the target networks
            if self.train_info["num_train_steps"] % self.cfg.train.actor_target_freq == 0:
                self.policy.actor.update_target(self.cfg.train.tau)
            if self.train_info["num_train_steps"] % self.cfg.train.critic_target_freq == 0:
                self.policy.critic.update_target(self.cfg.train.tau)

        self._update_train_info()

    @torch.no_grad()
    def evaluate_policy(self):
        self.policy.actor.eval()
        obs = self.eval_env.reset()
        num_samples = 0
        rewards, ep_rewards, ep_steps = [], [], []
        while num_samples < self.cfg.eval.steps:
            action = self.policy.action(obs["obs"])
            action, _ = self.cfg.eval.action_transform(action, obs)
            sample = self.eval_env.step(action)
            obs["obs"] = sample["next_obs"]
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
            self.rollout_info["obs"] = self.env.reset()
        self.policy.actor.train()

    def save_checkpoint(self):
        assert self.cfg.checkpoint.path.is_dir(), "The checkpoint path must be a directory."
        self.policy.save(self.cfg.checkpoint.path / "policy.pt")
        self.buffer.save(self.cfg.checkpoint.path / "buffer.pt")
        self.checkpoint_info["num_samples"] = self.rollout_info["num_samples"]

    def _next_required_samples(self):
        # Calculate required samples for next training step
        current_samples = self.rollout_info["num_samples"] - self.train_info["num_samples"]
        train_samples = self.cfg.train.freq - current_samples
        # Check if we have enough samples for a batch. If not, collect as many samples as required
        # to fill a batch
        if self.rollout_info["num_samples"] - self.cfg.train.batch_size < 0:
            if train_samples < self.cfg.train.batch_size - self.rollout_info["num_samples"]:
                train_samples = self.cfg.train.batch_size - self.rollout_info["num_samples"]
        if train_samples == 0:
            train_samples = self.cfg.train.freq

        # Calculate required samples for next eval step
        current_samples = self.rollout_info["num_samples"] - self.eval_info["num_samples"]
        eval_samples = self.cfg.eval.freq - current_samples
        eval_samples = eval_samples if eval_samples > 0 else self.cfg.eval.freq
        # Calculate required samples for next checkpoint
        if self.cfg.checkpoint.freq is None:
            checkpoint_samples = np.inf
        else:
            current_samples = self.rollout_info["num_samples"] - self.checkpoint_info["num_samples"]
            checkpoint_samples = self.cfg.checkpoint.freq - current_samples
        return min([train_samples, eval_samples, checkpoint_samples])

    def _rate_limit_rollout_log(self):
        if self.rollout_info["num_samples"] % self.rollout_info["log_freq"] == 0:
            ep_steps = self.rollout_info["log"]["ep_steps"]
            ep_reward = self.rollout_info["log"]["ep_reward"]
            ep_count = self.rollout_info["log"]["ep_count"]
            if ep_count > 0:
                data = {
                    "rollout/ep_steps": ep_steps / ep_count,
                    "rollout/ep_reward": ep_reward / ep_count
                }
                self.logger.log(data, step=self.rollout_info["num_samples"])
                self.rollout_info["log"]["ep_steps"] = 0
                self.rollout_info["log"]["ep_reward"] = 0
                self.rollout_info["log"]["ep_count"] = 0

    def _rate_limit_train_log(self):
        if self.train_info["num_train_steps"] % self.train_info["log_freq"] == 0:
            data = {}
            if self.train_info["log"]["actor_steps_since_log"] > 0:
                data["train/actor_loss"] = (self.train_info["log"]["actor_loss"] /
                                            self.train_info["log"]["actor_steps_since_log"])
                self.train_info["log"]["actor_loss"] = 0
                self.train_info["log"]["actor_steps_since_log"] = 0
            if self.train_info["log"]["critic_steps_since_log"] > 0:
                data["train/critic_loss"] = (self.train_info["log"]["critic_loss"] /
                                             self.train_info["log"]["critic_steps_since_log"])
                self.train_info["log"]["critic_loss"] = 0
                self.train_info["log"]["critic_steps_since_log"] = 0
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
        rollout_config = RolloutConfig(**vars(config.rollout))
        train_config = TrainConfig(**vars(config.train))
        eval_config = EvalConfig(**vars(config.eval))
        checkpoint_config = CheckpointConfig(**vars(config.checkpoint))

        # Check if the config is valid
        for cfg in (train_config, eval_config, checkpoint_config):
            if cfg.freq is not None and cfg.freq % env_config.kwargs["num_envs"] != 0:
                raise ValueError(f"Config {cfg} frequency ({cfg.freq}) must be multiple of "
                                 f"'num_envs' ({env_config.kwargs['num_envs']}).")
        return DDPGConfig(env_config, rollout_config, train_config, eval_config, checkpoint_config)
