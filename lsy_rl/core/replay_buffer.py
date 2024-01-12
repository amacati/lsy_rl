from __future__ import annotations

from typing import TYPE_CHECKING

from abc import ABC, abstractmethod
import torch
from tensordict import TensorDict
import numpy as np

from lsy_rl.utils import space_info, torchify_dtype

if TYPE_CHECKING:
    from gymnasium import Space, Env


class ReplayBuffer(ABC):

    def __init__(self, env: Env):
        self.env = env
        self.num_samples = 0

    @abstractmethod
    def add(self, obs, action, reward, next_obs, terminated, truncated):
        pass

    @abstractmethod
    def sample(self, batch_size):
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
        reward = torch.as_tensor(reward)
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
        # If num_samples + self._idx <= self.maxlen, we can simply write the next rows. Otherwise,
        # we need to split the samples into two parts. The first part is written to the end of the
        # buffer. Then we wrap around and write the second part to the beginning of the buffer.
        if num_samples + self._idx <= self.maxlen:
            self._copy_to_buffer(self._idx, self._idx + num_samples, obs, action, reward, next_obs,
                                 terminated, truncated)
        else:
            split_idx = self.maxlen - self._idx
            self._copy_to_buffer(self._idx, self.maxlen, obs[:split_idx], action[:split_idx],
                                 reward[:split_idx], next_obs[:split_idx], terminated[:split_idx],
                                 truncated[:split_idx])
            self._copy_to_buffer(0, num_samples - split_idx, obs[split_idx:], action[split_idx:],
                                 reward[split_idx:], next_obs[split_idx:], terminated[split_idx:],
                                 truncated[split_idx:])
        self._idx = (self._idx + num_samples) % self.maxlen
        self._maxidx = min(self._maxidx + num_samples, self.maxlen - 1)

    def _copy_to_buffer(self, idx_start: int, idx_end: int, obs, action, reward, next_obs,
                        terminated, truncated):
        assert idx_end >= idx_start, "Slice must be non-negative"
        assert idx_end <= self.maxlen, "Index out of bounds"
        assert obs.shape[0] == idx_end - idx_start, "Slice must match the length of the contents"
        self.buffer["obs"][idx_start:idx_end] = obs
        self.buffer["action"][idx_start:idx_end] = action
        self.buffer["reward"][idx_start:idx_end] = reward
        self.buffer["next_obs"][idx_start:idx_end] = next_obs
        self.buffer["terminated"][idx_start:idx_end] = terminated
        self.buffer["truncated"][idx_start:idx_end] = truncated

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
