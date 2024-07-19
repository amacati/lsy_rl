import logging
import random
import time
from types import SimpleNamespace

import gymnasium
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from munch import Munch

from lsy_rl.core import Algorithm
from lsy_rl.core.logger import EmptyLogger, Logger
from lsy_rl.ddpg.config import (
    CheckpointConfig,
    DDPGConfig,
    EnvConfig,
    EvalConfig,
    RolloutConfig,
    TrainConfig,
)
from lsy_rl.ddpg.policy import DDPGPolicy
from lsy_rl.utils.utils import unique_folder
from lsy_rl.wrappers.wrapper import wrap_env

logger = logging.getLogger(__name__)


class DDPG(Algorithm):
    """Deep Deterministic Policy Gradient (DDPG) algorithm."""

    num_logs: int = 200

    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        config: SimpleNamespace,
        logger: Logger = EmptyLogger(),
        seed: int | None = None,
    ):
        """Initialize the DDPG algorithm.

        Args:
            env: Training environment.
            eval_env: Evaluation environment.
            config: Configuration of the algorithm. See `DDPGConfig` for details.
            logger: Logger for keeping track of results. Defaults to an empty logger.
            seed: Random seed used for reproducibility. Defaults to None, i.e. no seed.
        """
        super().__init__()
        # torch.set_float32_matmul_precision('high')  # TODO: Check if this impacts performance
        assert hasattr(env, "num_envs"), "The environment must have a 'num_envs' attribute."
        self.cfg = self._parse_config(config, env)
        # Create wrapped environments so that the observations and actions are always Tensors
        self.env = wrap_env(env, device=self.cfg.train.device)
        self.eval_env = wrap_env(eval_env, device=self.cfg.train.device)
        self.separate_eval_env = self.env.unwrapped is not self.eval_env.unwrapped

        # Set random seeds
        self.seed = seed
        self._set_seed(seed)

        self.logger = logger
        self.policy = self._init_policy()  # Initialize the policy with actor and critic networks
        # Initialize the optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(), lr=self.cfg.train.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(), lr=self.cfg.train.critic_lr
        )

        # Initialize the replay buffer
        self.cfg.rollout.replay_buffer_kwargs |= {
            "num_envs": self.env.num_envs,
            "device": self.cfg.train.device,
        }
        buffer_cls = self.cfg.rollout.replay_buffer_cls
        self.buffer = buffer_cls(**self.cfg.rollout.replay_buffer_kwargs)

        # Allocate rollout, train, eval and checkpoint info
        self.rollout_info = self._init_rollout_info()
        self.train_info = self._init_train_info()
        self.eval_info = self._init_eval_info()
        self.checkpoint_info = self._init_checkpoint_info()

        # Don't overwrite the checkpoint path in the config in case it gets reused for multiple runs
        self.checkpoint_path = unique_folder(self.cfg.checkpoint.path)

    @property
    def stop_condition(self) -> bool:
        """Check if we should stop training based on how many samples we have collected."""
        max_samples = self.rollout_info.n_samples >= self.cfg.rollout.max_samples
        return max_samples

    @property
    def train_condition(self) -> bool:
        """Check if we should train the policy based on how many samples we have collected."""
        if self.rollout_info.n_samples < self.cfg.train.min_samples:
            return False
        return self.rollout_info.n_samples - self.train_info.n_samples >= self.cfg.train.freq

    @property
    def eval_condition(self) -> bool:
        """Check if we should evaluate the policy based on how many samples we have collected."""
        return self.rollout_info.n_samples - self.eval_info.n_samples >= self.cfg.eval.freq

    @property
    def checkpoint_condition(self) -> bool:
        """Check if we should save a checkpoint based on how many samples we have collected."""
        if self.cfg.checkpoint.freq is None:
            return False
        n_samples = self.rollout_info.n_samples - self.checkpoint_info.n_samples
        return n_samples >= self.cfg.checkpoint.freq

    def train(self):
        """Train the policy using the DDPG algorithm."""
        self.evaluate_policy()  # Establish an initial baseline
        while not self.stop_condition:
            self.collect_samples()
            if self.train_condition:
                self.train_policy()
            if self.eval_condition:
                self.evaluate_policy()
            if self.checkpoint_condition:
                self.save_checkpoint()
        if self.checkpoint_path is not None:
            self.save_checkpoint()  # Save the final checkpoint even if we don't reach the freq
        self.logger.stop()

    @torch.no_grad()
    def collect_samples(self):
        """Collect samples from the environment and store them in the replay buffer."""
        self.policy.actor.eval()

        if "obs" not in self.rollout_info:  # If first rollout, reset the environment
            self.rollout_info.obs = self.env.reset()
        obs = self.rollout_info.obs

        # Calculate how many samples to collect before we need to interrupt for any callbacks
        required_samples = self.rollout_info.n_samples + self._next_required_samples()
        while self.rollout_info.n_samples < required_samples:
            self.cfg.rollout.obs_transform.update(obs["obs"])
            obs_t, _ = self.cfg.rollout.obs_transform(obs["obs"])
            action = self.policy.actor(obs_t)
            action, _ = self.cfg.rollout.action_transform(action, obs)
            sample = self.env.step(action)
            sample["obs"], sample["action"] = obs["obs"], action
            # Vector environments automatically reset after T steps. The last observation is
            # already the first observation of the next episode. We have to handle two cases:
            # 1) When we continue sampling, we want obs["obs"] to be the first observation of the
            # next episode.
            # 2) The sample added to the buffer should have the terminal observation as next_obs.
            obs["obs"] = sample["next_obs"]  # Avoid cloning if we don't need to

            done = sample["terminated"] | sample["truncated"]
            if torch.any(done):  # Case 2: Replace sample next_obs with the final observation
                # Case 1: Make obs["obs"] the first observation of the next episode. We need to
                # clone the sample because we will modify it in the next step
                obs["obs"] = sample["next_obs"].clone()
                # Case 2: Replace the next_obs with the final observation
                sample["next_obs"][done] = sample["info", "final_observation"][done]
            self.buffer.add(sample)

            self.rollout_info.n_samples += self.env.num_envs
            self.rollout_info.steps += 1
            self.rollout_info.rewards += sample["reward"]
            # If any of the environments are terminated or truncated, log the episode statistics
            if torch.any(done):
                self.rollout_info.log.ep_steps += self.rollout_info.steps[done].sum()
                self.rollout_info.log.ep_reward += self.rollout_info.rewards[done].sum()
                self.rollout_info.log.n_episodes += len(done)
                self.rollout_info.log.last_rewards.extend(sample["reward"][done].tolist())
                self.rollout_info.steps[done] = 0
                self.rollout_info.rewards[done] = 0

            self._log_rollout()

        self.rollout_info.obs = obs
        self.policy.actor.train()

    def train_policy(self):
        """Train the policy using the collected samples in the replay buffer."""
        self.policy.actor.train()  # Critic is always in train mode, not used for inference

        for _ in range(self.cfg.train.steps):
            # Update 'num_train_steps' at the beginning of the loop so that lower frequency updates
            # do not get executed at the first iteration when 'num_train_steps' is 0
            self.train_info.n_train_steps += 1

            if self.train_info.n_train_steps % self.cfg.train.critic_freq == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the expected Q values with the reward and the target networks
                with torch.no_grad():
                    next_obs_t, _ = self.cfg.train.obs_transform(batch["next_obs"])
                    next_action = self.policy.actor.target(next_obs_t)
                    next_action, _ = self.cfg.train.target_action_transform(next_action, batch)
                    next_q_target = self.policy.critic.target(next_obs_t, next_action)
                    # Reward, terminated are one-dimensional, so we need to reshape them to avoid
                    # broadcasting errors
                    reward = batch["reward"].reshape(-1, 1)
                    terminated = batch["terminated"].reshape(-1, 1)
                    q_target = reward + (self.cfg.train.gamma * ~terminated * next_q_target)
                    q_target = torch.clamp(q_target, *self.cfg.train.reward_clip)
                # Compute the loss as the MSE between the expected Q values and the Q values from
                # the critic
                obs_t, _ = self.cfg.train.obs_transform(batch["obs"])
                q_expected = self.policy.critic(obs_t, batch["action"])
                assert q_target.shape == (self.cfg.train.batch_size, 1), q_target.shape
                assert q_expected.shape == q_target.shape, (q_expected.shape, q_target.shape)
                critic_loss = (q_target - q_expected).pow(2).mean()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.cfg.train.grad_clip
                )
                self.critic_optimizer.step()
                self.train_info.log.critic_loss += critic_loss.detach()
                self.train_info.log.critic_steps_since_log += 1

            if self.train_info.n_train_steps % self.cfg.train.actor_freq == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the actions for the sample observations, compute the critic value of the
                # observations and actions and compute the actor loss by maximizing the critic value
                obs_t, _ = self.cfg.train.obs_transform(batch["obs"])
                train_action = self.policy.actor(obs_t)
                train_action, _ = self.cfg.train.action_transform(train_action, batch)
                actor_loss = -self.policy.critic(obs_t, train_action).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.cfg.train.grad_clip
                )
                self.actor_optimizer.step()
                self.train_info.log.actor_loss += actor_loss.detach()
                self.train_info.log.actor_steps_since_log += 1

            self._log_train()
            # Update the target networks
            if self.train_info.n_train_steps % self.cfg.train.actor_target_freq == 0:
                self.policy.actor.update_target(self.cfg.train.tau)
            if self.train_info.n_train_steps % self.cfg.train.critic_target_freq == 0:
                self.policy.critic.update_target(self.cfg.train.tau)

        self.train_info.n_samples = self.rollout_info.n_samples

    @torch.no_grad()
    def evaluate_policy(self):
        """Evaluate the policy on the evaluation environment and log the results."""
        self.policy.actor.eval()
        obs = self.eval_env.reset()
        n_samples = 0
        rewards, ep_rewards, ep_steps, ep_last_rewards = [], [], [], []
        while n_samples < self.cfg.eval.steps:
            obs_t, _ = self.cfg.eval.obs_transform(obs["obs"])
            action = self.policy.action(obs_t)
            action, _ = self.cfg.eval.action_transform(action, obs)
            sample = self.eval_env.step(action)
            obs["obs"] = sample["next_obs"]
            self.eval_info.rewards += sample["reward"]
            rewards += sample["reward"].tolist()
            self.eval_info.steps += 1

            if torch.any(sample["terminated"]) or torch.any(sample["truncated"]):
                idx = sample["terminated"] | sample["truncated"]
                ep_steps += self.eval_info.steps[idx].tolist()
                ep_rewards += self.eval_info.rewards[idx].tolist()
                ep_last_rewards += sample["reward"][idx].tolist()
                self.eval_info.steps[idx] = 0
                self.eval_info.rewards[idx] = 0

            n_samples += self.eval_env.num_envs

        data = {"eval/mean_reward": np.array(rewards).mean()}
        if ep_rewards:
            data["eval/ep_mean_reward"] = np.array(ep_rewards).mean()
            data["eval/ep_mean_steps"] = np.array(ep_steps).mean()
            if self.cfg.eval.success_criteria is not None:
                success = self.cfg.eval.success_criteria(ep_last_rewards)
                data["eval/success_rate"] = success.mean()
        self.logger.log(data, step=self.rollout_info.n_samples)
        # Reset the steps and rewards for future eval runs. Otherwise, the next eval run adds to
        # the values from the previous run
        self.eval_info.steps[...], self.eval_info.rewards[...] = 0, 0
        self.eval_info.n_samples = self.rollout_info.n_samples
        # If the train and eval envs are the same, we need to reset the env and save the obs to the
        # rollout info. Otherwise, the next rollout will start from the last state of the eval env,
        # but will still use the last observation from the latest rollout
        if not self.separate_eval_env:
            self.rollout_info.obs = self.env.reset()
        self.policy.actor.train()

    def save_checkpoint(self):
        """Save a checkpoint of the policy, replay buffer and optimizers."""
        assert self.checkpoint_path.is_dir(), "The checkpoint path must be a directory."
        self.policy.save(self.checkpoint_path / "policy.pt")
        self.buffer.save(self.checkpoint_path / "buffer.pt")
        torch.save(self.actor_optimizer.state_dict(), self.checkpoint_path / "actor_opt.pt")
        torch.save(self.critic_optimizer.state_dict(), self.checkpoint_path / "critic_opt.pt")
        torch.save(
            self.cfg.rollout.obs_transform.state_dict(), self.checkpoint_path / "obs_transform.pt"
        )
        self.checkpoint_info.n_samples = self.rollout_info.n_samples

    def _next_required_samples(self) -> int:
        """Calculate the number of samples until training, evaluation or checkpointing."""
        # Calculate required samples for next training step
        samples_since = self.rollout_info.n_samples - self.train_info.n_samples
        train_samples = self.cfg.train.freq - samples_since
        train_samples = train_samples if train_samples > 0 else self.cfg.train.freq
        # If training should start after a minimum number of samples, calculate the difference
        if self.cfg.train.min_samples is not None:
            if self.rollout_info.n_samples < self.cfg.train.min_samples:
                train_samples = self.cfg.train.min_samples - self.rollout_info.n_samples
        # Calculate required samples for next eval step
        eval_samples = self.cfg.eval.freq - (self.rollout_info.n_samples - self.eval_info.n_samples)
        eval_samples = eval_samples if eval_samples > 0 else self.cfg.eval.freq
        # Calculate required samples for next checkpoint
        checkpoint_samples = np.inf
        if self.cfg.checkpoint.freq is not None:
            current_samples = self.rollout_info.n_samples - self.checkpoint_info.n_samples
            checkpoint_samples = self.cfg.checkpoint.freq - current_samples
        return min([train_samples, eval_samples, checkpoint_samples])

    def _log_rollout(self):
        """Log the rollout statistics and the time elapsed.

        Rate limited to avoid logging too frequently.
        """
        info, log = self.rollout_info, self.rollout_info.log
        if info.n_samples - info.last_log < self.rollout_info.log_freq:  # Rate limit logging
            return
        if log.n_episodes == 0:  # No episodes finished since last log
            return
        data = {"rollout/ep_steps": log.ep_steps / log.n_episodes}
        data["rollout/ep_reward"] = log.ep_reward / log.n_episodes
        if self.cfg.rollout.success_criteria is not None:
            success = self.cfg.rollout.success_criteria(log.last_rewards)
            data["rollout/success_rate"] = success.mean()
        self.logger.log(data, step=info.n_samples)
        # Reset the log values
        info.ep_steps, info.ep_reward, info.n_episodes, info.last_rewards = 0, 0, 0, []
        elapsed_time = time.time() - info.start_time
        data = {"time/time_elapsed": elapsed_time, "time/total_timesteps": info.n_samples}
        data["time/fps"] = info.n_samples / elapsed_time
        self.logger.log(data, step=info.n_samples)
        info.last_log = info.n_samples

    def _log_train(self):
        """Log the training statistics.

        Rate limited to avoid logging too frequently.
        """
        info, log = self.train_info, self.train_info.log
        if info.n_train_steps - info.last_log < info.log_freq:  # Rate limit logging
            return
        data = {}
        if log.actor_steps_since_log > 0:
            data["train/actor_loss"] = log.actor_loss / log.actor_steps_since_log
            log.actor_loss, log.actor_steps_since_log = 0, 0
        if log.critic_steps_since_log > 0:
            data["train/critic_loss"] = log.critic_loss / log.critic_steps_since_log
            log.critic_loss, log.critic_steps_since_log = 0, 0
        if data:
            self.logger.log(data, step=self.rollout_info.n_samples)
        info.last_log = info.n_train_steps

    def _set_seed(self, seed: int | None):
        if seed is not None:
            assert isinstance(seed, int), "The seed must be an integer."
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            env_seed = [i + seed for i in range(self.env.num_envs)]
            self.env.reset(seed=env_seed)  # Reset once to set the correct RNG state
            # Make sure the seeds for the eval envs are different from the train envs
            eval_env_seed = [i + seed + self.env.num_envs for i in range(self.eval_env.num_envs)]
            if self.separate_eval_env:  # Only reset if the eval env is not also the train env
                self.eval_env.reset(seed=eval_env_seed)

    def _parse_config(self, config: SimpleNamespace, env: gymnasium.vector.VectorEnv) -> DDPGConfig:
        env_config = EnvConfig(**config.env)
        rollout_config = RolloutConfig(**config.rollout, env=env.env_fns[0]())
        train_config = TrainConfig(**config.train)
        eval_config = EvalConfig(**config.eval)
        checkpoint_config = CheckpointConfig(**config.checkpoint)

        # Check if the config is valid
        for cfg in (train_config, eval_config, checkpoint_config):
            if cfg.freq is not None and cfg.freq % env_config.n_envs != 0:
                raise ValueError(
                    f"Config {cfg} frequency ({cfg.freq}) must be multiple of "
                    f"'n_envs' ({env_config.n_envs})."
                )
        return DDPGConfig(env_config, rollout_config, train_config, eval_config, checkpoint_config)

    def _init_policy(self) -> DDPGPolicy:
        spaces = {"obs_space": self.env.observation_space, "action_space": self.env.action_space}
        self.cfg.train.actor_kwargs |= spaces
        actor = self.cfg.train.actor_cls(**self.cfg.train.actor_kwargs)
        self.cfg.train.policy_kwargs["actor"] = actor
        self.cfg.train.critic_kwargs |= spaces
        critic = self.cfg.train.critic_cls(**self.cfg.train.critic_kwargs)
        self.cfg.train.policy_kwargs["critic"] = critic
        return DDPGPolicy(**self.cfg.train.policy_kwargs, device=self.cfg.train.device)

    def _init_rollout_info(self) -> Munch:
        """Initialize a container to store rollout information for flow control and logging."""
        info = Munch()
        info.n_samples = 0
        info.steps = torch.zeros(self.env.num_envs, device=self.cfg.train.device)
        info.rewards = torch.zeros(self.env.num_envs, device=self.cfg.train.device)
        info.log_freq = max(1, self.cfg.rollout.max_samples // self.num_logs)
        info.last_log = 0
        info.log = Munch({"ep_steps": 0, "ep_reward": 0, "n_episodes": 0, "last_rewards": []})
        info.start_time = time.time()
        return info

    def _init_train_info(self) -> Munch:
        """Initialize a container to store training information for flow control and logging."""
        info = Munch()
        info.n_samples = 0
        info.n_train_steps = 0
        num_trainings = self.cfg.rollout.max_samples // self.cfg.train.freq
        total_train_steps = num_trainings * self.cfg.train.steps
        info.log_freq = max(1, total_train_steps // self.num_logs)
        info.last_log = 0
        log = Munch()
        log.actor_loss, log.actor_steps_since_log = 0, 0
        log.critic_loss, log.critic_steps_since_log = 0, 0
        info.log = log
        return info

    def _init_eval_info(self) -> Munch:
        """Initialize a container to store evaluation information for flow control and logging."""
        info = Munch()
        info.n_samples = 0
        info.steps = torch.zeros(self.eval_env.num_envs, device=self.cfg.train.device)
        info.rewards = torch.zeros(self.eval_env.num_envs, device=self.cfg.train.device)
        return info

    def _init_checkpoint_info(self) -> Munch:
        """Initialize a container to store checkpoint information for flow control and logging."""
        return Munch({"n_samples": 0})
