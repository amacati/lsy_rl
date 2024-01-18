from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import torch
from torch import IntTensor
from tensordict import TensorDict
import numpy as np

from lsy_rl.utils import space_info, torchify_dtype

if TYPE_CHECKING:
    from gymnasium import Space, Env

logger = logging.getLogger(__name__)


class ReplayBuffer(ABC):

    def __init__(self, env: Env):
        self.env = env
        self.num_samples = 0

    @abstractmethod
    def add(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    @abstractmethod
    def sample(self, batch_size) -> tuple[torch.Tensor, ...]:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def __len__(self) -> int:
        pass


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(self,
                 env: Env,
                 maxlen: int,
                 device: torch.device = torch.device("cpu"),
                 seed: int | None = None):
        super().__init__(env)
        self.maxlen = maxlen
        # Allocate buffers
        self.device = device
        obs_shape, obs_type = space_info(env, mode="obs")
        act_shape, act_type = space_info(env, mode="action")
        obs_type, act_type = torchify_dtype(obs_type), torchify_dtype(act_type)
        self.buffer: TensorDict[torch.Tensor] = TensorDict(
            {
                "obs": torch.zeros((maxlen, *obs_shape), dtype=obs_type),
                "action": torch.zeros((maxlen, *act_shape), dtype=act_type),
                "reward": torch.zeros((maxlen, 1), dtype=torch.float32),
                "next_obs": torch.zeros((maxlen, *obs_shape), dtype=obs_type),
                "terminated": torch.zeros((maxlen, 1), dtype=torch.bool),
                "truncated": torch.zeros((maxlen, 1), dtype=torch.bool),
            },
            batch_size=maxlen,
            device=device)
        # Helper indices to implement a ring buffer
        self._idx = 0  # Points to the current index of the ring buffer
        self._maxidx = -1  # Points to the last valid sample
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)

    def add(self, obs, action, reward, next_obs, terminated, truncated):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        # Convert all incoming data to tensors
        obs = torch.as_tensor(obs)
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs)
        terminated = torch.as_tensor(terminated)
        truncated = torch.as_tensor(truncated)
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        if terminated.ndim == 1:
            terminated = terminated.unsqueeze(-1)
        if truncated.ndim == 1:
            truncated = truncated.unsqueeze(-1)
        num_samples = obs.shape[0]
        assert num_samples < self.maxlen, "Vectorized sample size must be smaller than the buffer"
        # If num_samples + self._idx > self.maxlen, the index wraps around to the beginning of the
        # buffer. We take the index vector modulo ``self.maxlen`` to implement this behavior.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.maxlen
        self._copy_to_buffer(idx, obs, action, reward, next_obs, terminated, truncated)
        # Update the helper indices
        self._idx = (self._idx + num_samples) % self.maxlen
        self._maxidx = min(self._maxidx + num_samples, self.maxlen - 1)

    def _copy_to_buffer(self, idx: IntTensor, obs, action, reward, next_obs, terminated, truncated):
        assert idx.ndim == 1, "Index vector must be one dimensional"
        assert obs.shape[0] == idx.shape[0], "Sample size must match the index"
        self.buffer["obs"][idx, ...] = obs
        self.buffer["action"][idx, ...] = action
        self.buffer["reward"][idx, ...] = reward
        self.buffer["next_obs"][idx, ...] = next_obs
        self.buffer["terminated"][idx, ...] = terminated
        self.buffer["truncated"][idx, ...] = truncated

    def sample(self, batch_size):
        assert batch_size <= self._maxidx + 1, "Not enough samples in the buffer"
        idx = self.rng.choice(self._maxidx + 1, size=(batch_size,), replace=False)
        return tuple(self.buffer[idx].values())

    def clear(self):
        for b in self.buffer.values():
            b.zero_()
        self._idx = 0
        self._maxidx = -1

    @staticmethod
    def space_info(obs_space: Space):
        return obs_space.shape, obs_space.dtype

    def save(self, path):
        save_dict = {"idx": self._idx, "maxidx": self._maxidx, "buffer": self.buffer}
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self._idx, self._maxidx = save_dict["idx"], save_dict["maxidx"]
        self.buffer = save_dict["buffer"]
        assert self.buffer.batch_size == self.maxlen, "Loaded buffer has wrong size"

    def __len__(self) -> int:
        return self._maxidx + 1


class VectorReplayBuffer(ReplayBuffer):

    def __init__(self,
                 env: Env,
                 maxlen: int,
                 device: torch.device = torch.device("cpu"),
                 seed: int | None = None):
        super().__init__(env)
        self.num_envs = env.num_envs
        assert maxlen > self.num_envs, "Buffer size must be larger than the number of environments"
        self.bufflen = maxlen // self.num_envs
        if maxlen % self.num_envs != 0:
            logger.warning(f"Buffer size ({maxlen}) is not a multiple of the number of environments"
                           f" ({self.num_envs}). Buffer size reduced to "
                           f"{self.bufflen * self.num_envs}.")
        # Allocate buffers
        self.device = device
        obs_shape, obs_type = space_info(env, mode="obs")
        act_shape, act_type = space_info(env, mode="action")
        obs_type, act_type = torchify_dtype(obs_type), torchify_dtype(act_type)
        self.buffer: TensorDict[torch.Tensor] = TensorDict(
            {
                "obs": torch.zeros((self.num_envs, self.bufflen, *obs_shape), dtype=obs_type),
                "action": torch.zeros((self.num_envs, self.bufflen, *act_shape), dtype=act_type),
                "reward": torch.zeros((self.num_envs, self.bufflen, 1), dtype=torch.float32),
                "next_obs": torch.zeros((self.num_envs, self.bufflen, *obs_shape), dtype=obs_type),
                "terminated": torch.zeros((self.num_envs, self.bufflen, 1), dtype=torch.bool),
                "truncated": torch.zeros((self.num_envs, self.bufflen, 1), dtype=torch.bool),
            },
            batch_size=self.num_envs,
            device=device)
        # Allocate helper for default environment indexing
        self._default_env_idx = torch.arange(self.num_envs, dtype=int, device=device)
        # Helper indices to implement a vectorized ring buffer
        # Points to the current ring buffer index
        self._idx = torch.zeros(self.num_envs, dtype=int, device=device)
        # Points to the last valid sample
        self._maxidx = -torch.ones(self.num_envs, dtype=int, device=device)
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)

    def add(self,
            obs: np.ndarray | torch.Tensor,
            action: np.ndarray | torch.Tensor,
            reward: np.ndarray | torch.Tensor,
            next_obs: np.ndarray | torch.Tensor,
            terminated: np.ndarray | torch.Tensor,
            truncated: np.ndarray | torch.Tensor,
            env_idx: IntTensor | None = None):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        # If no explicit environment index given, assume one sample per env
        env_idx = env_idx or self._default_env_idx
        for tensor in (obs, action, reward, next_obs, terminated, truncated):
            assert tensor.shape[0] == env_idx.shape[0], "Sample size must match the env index"
        # Convert all incoming data to tensors
        obs = torch.as_tensor(obs)
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward, dtype=torch.float32)
        next_obs = torch.as_tensor(next_obs)
        terminated = torch.as_tensor(terminated)
        truncated = torch.as_tensor(truncated)
        # Count the number of samples per environment
        num_samples = torch.bincount(env_idx, minlength=self.num_envs).to(self.device)
        assert torch.max(num_samples) < self.bufflen, "Sample sizes must be smaller than the buffer"
        # Compute the indices for each sample
        idx = torch.zeros(len(env_idx), dtype=torch.long)
        for i, n_samples in enumerate(num_samples):
            idx_update = self._idx[i] + torch.arange(n_samples, device=self.device)
            idx[env_idx == i] = idx_update % self.bufflen
            self._idx[i] = (self._idx[i] + n_samples) % self.bufflen
        # Indices need to be 2D for indexing
        self._copy_to_buffer(env_idx.unsqueeze(1), idx.unsqueeze(1), obs, action, reward, next_obs,
                             terminated, truncated)
        # Update the helper indices
        self._idx[env_idx] = (self._idx[env_idx] + num_samples[env_idx]) % self.bufflen
        self._maxidx[env_idx] = torch.min(self._maxidx[env_idx] + num_samples[env_idx],
                                          torch.ones_like(self._maxidx) * (self.bufflen - 1))

    def _copy_to_buffer(self, env_idx: IntTensor, idx: IntTensor, obs, action, reward, next_obs,
                        terminated, truncated):
        assert env_idx.ndim == 2, "Environment index vector must be 2D"
        assert env_idx.shape[1] == 1, "Environment index vector must be single column"
        assert idx.ndim == 2, "Index vector must be 2D"
        assert idx.shape[1] == 1, "Index vector must be single column"
        assert obs.shape[0] == idx.shape[0], "Sample size must match the index"
        self.buffer["obs"][env_idx, idx, ...] = obs.unsqueeze(1)
        self.buffer["action"][env_idx, idx, ...] = action.unsqueeze(1)
        self.buffer["reward"][env_idx, idx, ...] = reward.reshape((-1, 1, 1))
        self.buffer["next_obs"][env_idx, idx, ...] = next_obs.unsqueeze(1)
        self.buffer["terminated"][env_idx, idx, ...] = terminated.reshape((-1, 1, 1))
        self.buffer["truncated"][env_idx, idx, ...] = truncated.reshape((-1, 1, 1))

    def sample(self, batch_size):
        assert batch_size <= self._maxidx + 1, "Not enough samples in the buffer"
        idx = self.rng.choice(self._maxidx + 1, size=(batch_size,), replace=False)
        return tuple(self.buffer[idx].values())

    def clear(self):
        for b in self.buffer.values():
            b.zero_()
        self._idx = 0
        self._maxidx = -1

    @staticmethod
    def space_info(obs_space: Space):
        return obs_space.shape, obs_space.dtype

    def save(self, path):
        save_dict = {"idx": self._idx, "maxidx": self._maxidx, "buffer": self.buffer}
        torch.save(save_dict, path)

    def load(self, path):
        save_dict = torch.load(path, map_location=self.device)
        self._idx, self._maxidx = save_dict["idx"], save_dict["maxidx"]
        self.buffer = save_dict["buffer"]
        assert self.buffer.batch_size == self.bufflen, "Loaded buffer has wrong size"

    def __len__(self) -> int:
        return torch.sum(self._maxidx + 1).item()
