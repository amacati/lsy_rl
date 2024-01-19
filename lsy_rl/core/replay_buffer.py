from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

import torch
from torch import IntTensor
from tensordict import TensorDict
import numpy as np

if TYPE_CHECKING:
    from gymnasium import Space

logger = logging.getLogger(__name__)


class ReplayBuffer(ABC):

    def __init__(self):
        ...

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
                 num_envs: int,
                 max_size: int,
                 device: torch.device = torch.device("cpu"),
                 seed: int | None = None):
        super().__init__()
        self.num_envs = num_envs
        self.max_size = max_size
        # Allocate buffers
        self.device = device
        self.buffer: TensorDict[torch.Tensor] = TensorDict({}, batch_size=max_size, device=device)
        # Helper indices to implement a ring buffer
        self._idx = 0  # Points to the current index of the ring buffer
        self._maxidx = -1  # Points to the last valid sample
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)

    def add(self, sample: TensorDict[torch.Tensor]):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        # Convert all incoming data to tensors
        sample = sample.flatten_keys()
        num_samples = sample.batch_size[0]
        assert num_samples < self.max_size, "Vectorized sample size must be smaller than the buffer"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        for key in sample.keys():
            if key not in self.buffer.keys():
                self.buffer[key] = torch.empty((self.max_size, *sample[key].shape[1:]),
                                               dtype=sample[key].dtype)
        # If num_samples + self._idx > self.max_size, the index wraps around to the beginning of the
        # buffer. We take the index vector modulo ``self.max_size`` to implement this behavior.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.max_size
        self._copy_to_buffer(idx, sample)
        # Update the helper indices
        self._idx = (self._idx + num_samples) % self.max_size
        self._maxidx = min(self._maxidx + num_samples, self.max_size - 1)

    def _copy_to_buffer(self, idx: IntTensor, sample: TensorDict[torch.Tensor]):
        assert idx.ndim == 1, "Index vector must be one dimensional"
        assert sample.batch_size[0] == idx.shape[0], "Sample size must match the index"
        for key, value in sample.items():
            # Cast to the correct type to avoid type errors when sample keys change dtype in some
            # episodes (e.g. when the reward changes from float to int at the end of an episode)
            self.buffer[key][idx, ...] = value.type(self.buffer[key].dtype)

    def sample(self, batch_size):
        assert batch_size <= self._maxidx + 1, "Not enough samples in the buffer"
        idx = self.rng.choice(self._maxidx + 1, size=(batch_size,), replace=False)
        return self.buffer[idx]

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
        assert self.buffer.batch_size == self.max_size, "Loaded buffer has wrong size"

    def __len__(self) -> int:
        return self._maxidx + 1


class VectorReplayBuffer(ReplayBuffer):

    def __init__(self,
                 num_envs: int,
                 max_size: int,
                 device: torch.device = torch.device("cpu"),
                 seed: int | None = None):
        super().__init__()
        self.num_envs = num_envs
        assert max_size > self.num_envs, "Buffer size must be larger than the number of envs"
        self.bufflen = max_size // self.num_envs
        if max_size % self.num_envs != 0:
            logger.warning(f"Buffer size ({max_size}) is not a multiple of the number of envs"
                           f" ({self.num_envs}). Buffer size reduced to "
                           f"{self.bufflen * self.num_envs}.")
        # Allocate buffers
        self.device = device
        self.buffer = TensorDict({}, batch_size=(self.num_envs, self.bufflen), device=device)
        # Allocate helper for default environment indexing
        self._default_env_idx = torch.arange(self.num_envs, dtype=int)
        # Helper indices to implement a vectorized ring buffer
        # Points to the current ring buffer index
        self._idx = torch.zeros(self.num_envs, dtype=int)
        # Points to the last valid sample
        self._maxidx = -torch.ones(self.num_envs, dtype=int)
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)

    def add(self, sample: TensorDict[torch.Tensor], env_idx: IntTensor | None = None):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        # If no explicit environment index given, assume one sample per env
        env_idx = env_idx or self._default_env_idx
        sample.batch_size[0] == env_idx.shape[0], "Sample size must match the env index"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        for key in sample.keys():
            if key not in self.buffer.keys():
                self.buffer[key] = torch.empty(
                    (self.num_envs, self.bufflen, *sample[key].shape[1:]), dtype=sample[key].dtype)
        # Count the number of samples per environment
        num_samples = torch.bincount(env_idx, minlength=self.num_envs)
        assert torch.max(num_samples) < self.bufflen, "Sample sizes must be smaller than the buffer"
        # Compute the indices for each sample
        idx = self._categorical_cumsum(env_idx) + self._idx[env_idx] % self.bufflen
        # Indices need to be 2D for indexing
        self._copy_to_buffer(env_idx, idx, sample)
        # Update the helper indices
        self._idx[env_idx] = (self._idx[env_idx] + num_samples[env_idx]) % self.bufflen
        self._maxidx[env_idx] = torch.min(self._maxidx[env_idx] + num_samples[env_idx],
                                          torch.ones_like(self._maxidx) * (self.bufflen - 1))

    @staticmethod
    def _categorical_cumsum(x: torch.Tensor) -> torch.Tensor:
        """Compute the cumulative sum per category of a tensor of integers.

        Example:
            >>> x = torch.tensor([2, 3, 1, 1, 4, 2, 3, 3])
            >>> VectorReplayBuffer._categorical_cumsum(x)
            tensor([1, 1, 1, 2, 1, 2, 2, 3])
        """
        unique, inverse = torch.unique(x, return_inverse=True)
        idx = torch.zeros((len(unique), len(x)), dtype=torch.int64, device=x.device)
        idx[inverse, torch.arange(len(x))] = 1
        return idx.cumsum(dim=1)[inverse, torch.arange(len(x))] - 1

    def _copy_to_buffer(self, env_idx: IntTensor, idx: IntTensor, sample: TensorDict[torch.Tensor]):
        assert env_idx.ndim == 1, "Environment index vector must be 1D"
        assert idx.ndim == 1, "Index vector must be 1D"
        assert sample.batch_size[0] == idx.shape[0], "Sample size must match the index"
        for key, value in sample.items():
            self.buffer[key][env_idx, idx, ...] = value

    def sample(self, batch_size: int):
        assert batch_size <= torch.sum(self._maxidx + 1), "Not enough samples in the buffer"
        env_idx = self.rng.choice(self.num_envs, size=batch_size, replace=True)
        idx = self.rng.integers(self._maxidx[env_idx] + 1, size=batch_size)
        env_idx = torch.tensor(env_idx).to(self.device)
        idx = torch.tensor(idx).to(self.device)
        return self.buffer[env_idx, idx]

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
