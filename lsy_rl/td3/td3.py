import logging
from types import SimpleNamespace

import gymnasium
import torch
from gymnasium.vector import VectorEnv

from lsy_rl.core.logger import EmptyLogger, Logger
from lsy_rl.ddpg.ddpg import DDPG
from lsy_rl.td3.config import (
    CheckpointConfig,
    EnvConfig,
    EvalConfig,
    RolloutConfig,
    TD3Config,
    TrainConfig,
)
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils.utils import unique_folder
from lsy_rl.wrappers.wrapper import wrap_env

logger = logging.getLogger(__name__)


class TD3(DDPG):
    def __init__(
        self,
        env: VectorEnv,
        eval_env: VectorEnv,
        config: SimpleNamespace,
        logger: Logger = EmptyLogger(),
        seed: int | None = None,
    ):
        """Initialize the TD3 algorithm.

        Args:
            env: Training environment.
            eval_env: Evaluation environment.
            config: Configuration of the algorithm. See `TD3Config` for details.
            logger: Logger for keeping track of results. Defaults to an empty logger.
            seed: Random seed used for reproducibility. Defaults to None, i.e. no seed.
        """
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
        self.time_info = self._init_time_info()

        # Don't overwrite the checkpoint path in the config in case it gets reused for multiple runs
        self.checkpoint_path = unique_folder(self.cfg.checkpoint.path)

    def train_policy(self):
        """Train the policy using the collected samples in the replay buffer."""
        self.policy.actor.train()  # Critic is always in train mode, not used for inference
        self.policy.actor.mode = "train"

        for _ in range(self.cfg.train.steps):
            # Update 'num_train_steps' at the beginning of the loop so that lower frequency updates
            # do not get executed at the first iteration when 'num_train_steps' is 0
            self.train_info.n_train_steps += 1

            if self.train_info.n_train_steps % self.cfg.train.critic_period == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the expected Q values with the reward and the target networks
                with torch.no_grad():
                    next_obs_t = self.cfg.train.obs_transform(batch["next_obs"])
                    next_action = self.policy.actor.target(next_obs_t)
                    next_action = self.cfg.train.target_action_transform(next_action)
                    next_q = self.policy.critic.target(next_obs_t, next_action)
                    # Reward, terminated are one-dimensional, so we need to reshape them to avoid
                    # broadcasting errors
                    reward = batch["reward"].reshape(-1, 1)
                    terminated = batch["terminated"].reshape(-1, 1)
                    q_target = reward + (self.cfg.train.gamma * ~terminated * next_q)
                    q_target = torch.clamp(q_target, *self.cfg.train.reward_clip)
                # Compute the loss as the MSE between the expected Q values and the Q values from
                # the critic
                obs_t = self.cfg.train.obs_transform(batch["obs"])
                q_1, q_2 = self.policy.critic.values(obs_t, batch["action"])
                assert q_target.shape == (self.cfg.train.batch_size, 1), q_target.shape
                assert q_1.shape == q_target.shape, (q_1.shape, q_target.shape)
                assert q_2.shape == q_target.shape, (q_2.shape, q_target.shape)
                q1_loss = (q_target - q_1).pow(2).mean()
                q2_loss = (q_target - q_2).pow(2).mean()
                critic_loss = q1_loss + q2_loss
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.critic.parameters(), self.cfg.train.grad_clip
                )
                self.critic_optimizer.step()
                self.train_info.log.critic_loss += critic_loss.detach()
                self.train_info.log.critic_steps_since_log += 1

            if self.train_info.n_train_steps % self.cfg.train.actor_period == 0:
                batch = self.buffer.sample(self.cfg.train.batch_size)
                # Compute the actions for the sample observations, compute the critic value of the
                # observations and actions and compute the actor loss by maximizing the critic value
                obs_t = self.cfg.train.obs_transform(batch["obs"])
                train_action = self.policy.actor(obs_t)
                train_action = self.cfg.train.action_transform(train_action)
                actor_loss = -self.policy.critic.actor_value(obs_t, train_action).mean()

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
            if self.train_info.n_train_steps % self.cfg.train.actor_target_period == 0:
                self.policy.actor.update_target(self.cfg.train.tau)
            if self.train_info.n_train_steps % self.cfg.train.critic_target_period == 0:
                self.policy.critic.update_target(self.cfg.train.tau)

        self.train_info.n_samples = self.rollout_info.n_samples

    def _parse_config(self, config: SimpleNamespace, env: gymnasium.vector.VectorEnv) -> TD3Config:
        env_config = EnvConfig(**config.env)
        env_config.env = env
        rollout_config = RolloutConfig(**config.rollout)
        train_config = TrainConfig(**config.train)
        eval_config = EvalConfig(**config.eval)
        checkpoint_config = CheckpointConfig(**config.checkpoint)

        # Check if the config is valid
        for cfg in (train_config, eval_config, checkpoint_config):
            if cfg.period is not None and cfg.period % env_config.n_envs != 0:
                raise ValueError(
                    f"Config {cfg} period ({cfg.period}) must be multiple of "
                    f"'n_envs' ({env_config.n_envs})."
                )
        return TD3Config(env_config, rollout_config, train_config, eval_config, checkpoint_config)

    def _init_policy(self) -> TD3Policy:
        spaces = {"obs_space": self.env.observation_space, "action_space": self.env.action_space}
        self.cfg.train.actor_kwargs |= spaces
        actor = self.cfg.train.actor_cls(**self.cfg.train.actor_kwargs)
        self.cfg.train.policy_kwargs["actor"] = actor
        self.cfg.train.critic_kwargs |= spaces
        critic = self.cfg.train.critic_cls(**self.cfg.train.critic_kwargs)
        self.cfg.train.policy_kwargs["critic"] = critic
        return TD3Policy(**self.cfg.train.policy_kwargs, device=self.cfg.train.device)
