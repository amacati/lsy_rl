from __future__ import annotations

import logging
import random
import sys
from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from tensordict import TensorDict
from torch import IntTensor

from lsy_rl.utils.utils import module_type_from_string

logger = logging.getLogger(__name__)

replay_buffer_cls: Callable[[str], type[ReplayBuffer]] = module_type_from_string(__name__)


class ReplayBuffer(ABC):
    def __init__(self):
        pass

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
    def __init__(
        self,
        num_envs: int,
        max_size: int,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
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
        num_samples = sample.batch_size[0]
        assert num_samples < self.max_size, "Vectorized sample size must be smaller than the buffer"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        self._allocate_buffers(sample)
        # If num_samples + self._idx > self.max_size, the index wraps around to the beginning of the
        # buffer. We take the index vector modulo ``self.max_size`` to implement this behavior.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.max_size
        self.buffer[idx] = sample
        # Update the helper indices
        self._idx = (self._idx + num_samples) % self.max_size
        self._maxidx = min(self._maxidx + num_samples, self.max_size - 1)

    def _allocate_buffers(self, sample: TensorDict):
        for key, val in sample.items():
            if key not in self.buffer.keys():
                if isinstance(val, torch.Tensor):
                    self.buffer[key] = torch.zeros((self.max_size, *val.shape[1:]), dtype=val.dtype)
                elif isinstance(val, TensorDict):
                    self.buffer[key] = TensorDict({}, batch_size=self.max_size, device=self.device)
                else:
                    raise TypeError(f"Unsupported type {type(val)}")

    def sample(self, batch_size):
        if batch_size > self._maxidx + 1:
            idx = np.random.randint(0, self._maxidx + 1, batch_size)
        else:
            idx = np.array(random.sample(range(self._maxidx + 1), batch_size))
        return self.buffer[idx]

    def clear(self):
        for b in self.buffer.values():
            b.zero_()
        self._idx = 0
        self._maxidx = -1

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
    def __init__(
        self,
        num_envs: int,
        max_size: int,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        super().__init__()
        self.num_envs = num_envs
        assert max_size > self.num_envs, "Buffer size must be larger than the number of envs"
        self.bufflen = max_size // self.num_envs
        if max_size % self.num_envs != 0:
            logger.warning(
                f"Buffer size ({max_size}) is not a multiple of the number of envs"
                f" ({self.num_envs}). Buffer size reduced to "
                f"{self.bufflen * self.num_envs}."
            )
        # Allocate buffers
        self.device = device
        self.buffer = TensorDict({}, batch_size=(self.num_envs, self.bufflen), device=device)
        # Allocate helper for default environment indexing
        self._default_v_idx = torch.arange(self.num_envs, dtype=int, device=device)
        # Helper indices to implement a vectorized ring buffer
        # Points to the current ring buffer index
        self._idx = torch.zeros(self.num_envs, dtype=int, device=device)
        # Points to the last valid sample
        self._maxidx: IntTensor = -torch.ones(self.num_envs, dtype=int, device=device)
        # Reproducible random number generator
        self.rng = np.random.default_rng(seed=seed)

    def add(self, sample: TensorDict[torch.Tensor], v_idx: IntTensor | None = None):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        # If no explicit environment index given, assume one sample per env
        v_idx = v_idx or self._default_v_idx
        sample.batch_size[0] == v_idx.shape[0], "Sample size must match the env index"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        self._allocate_buffers(sample)
        # Count the number of samples per environment
        num_samples = torch.bincount(v_idx, minlength=self.num_envs).to(self.device)
        assert torch.max(num_samples) < self.bufflen, "Sample sizes must be smaller than the buffer"
        # Compute the indices for each sample
        idx = (self._categorical_cumsum(v_idx) + self._idx[v_idx]) % self.bufflen
        # Indices need to be 2D for indexing
        self.buffer[v_idx, idx] = sample
        # Update the helper indices
        self._idx[v_idx] = (self._idx[v_idx] + num_samples[v_idx]) % self.bufflen
        self._maxidx[v_idx] = torch.min(
            self._maxidx[v_idx] + num_samples[v_idx],
            torch.ones_like(self._maxidx) * (self.bufflen - 1),
        )

    def _allocate_buffers(self, sample: TensorDict):
        for key, val in sample.items():
            if key not in self.buffer.keys():
                if isinstance(val, torch.Tensor):
                    self.buffer[key] = torch.zeros(
                        (self.num_envs, self.bufflen, *val.shape[1:]), dtype=val.dtype
                    )
                elif isinstance(val, TensorDict):
                    self.buffer[key] = TensorDict(
                        {}, batch_size=(self.num_envs, self.bufflen), device=self.device
                    )
                else:
                    raise TypeError(f"Unsupported type {type(val)}")

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

    def sample(self, batch_size: int) -> TensorDict[torch.Tensor]:
        assert batch_size <= torch.sum(self._maxidx + 1), "Not enough samples in the buffer"
        v_idx = torch.randint(self.num_envs, size=(batch_size,), device=self.device)
        idx = self.rng.integers(self._maxidx[v_idx].cpu() + 1, size=batch_size)
        idx = torch.tensor(idx).to(self.device)
        return self.buffer[v_idx, idx]

    def clear(self):
        for b in self.buffer.values():
            b.zero_()
        self._idx = 0
        self._maxidx = -1

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


class HerVectorReplayBuffer(VectorReplayBuffer):
    def __init__(
        self,
        num_envs: int,
        max_size: int,
        reward_fn: Callable,
        p_her: float = 0.8,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        super().__init__(num_envs, max_size, device, seed)
        self.reward_fn = reward_fn
        self.p_her = p_her
        # We track the remaining steps to the end of the episode for each environment. This has two
        # purposes: First, we need the remaining steps to sample a virtual goal from the same
        # trajectory. Second, we need to know when we are about to overwrite an old trajectory. In
        # that case, we invalidate the indices of the whole next episode by setting the remaining
        # steps to -1
        self._idx = 0
        self._maxidx = -1
        self._remaining_steps = torch.empty(
            (num_envs, max_size // num_envs), dtype=int, device=self.device
        )
        self._remaining_steps[:] = -1
        self._running_steps = torch.zeros(num_envs, dtype=int, device=self.device)
        self._invalid_idx = torch.zeros((num_envs, 2), dtype=int, device=self.device)
        self._invalid_idx[:, 1] = self.bufflen - 1

    def add(self, sample: TensorDict[torch.Tensor]):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        self._allocate_buffers(sample)
        # A sample must contain exactly one sample per vector entry
        v_idx = self._default_v_idx
        idx = self._idx
        assert sample.batch_size[0] == v_idx.shape[0], "Sample size must match the env index"
        # +1 because we overwrite to 0 inclusive
        overwrite_len = self._remaining_steps[v_idx, (idx + 1) % self.bufflen] + 1
        for i in torch.nonzero(overwrite_len).flatten():
            ep_idx = (torch.arange(overwrite_len[i], device=self.device) + idx + 1) % self.bufflen
            self._remaining_steps[i, ep_idx] = -1
            # Invalid indices at 0 point to the end of the last completed episode, so we don't need
            # to update them
            self._invalid_idx[i, 1] = (ep_idx[-1] + 1) % self.bufflen
        self._running_steps += 1
        done = (sample["terminated"] | sample["truncated"]).squeeze()  # Remove batch dimension
        # If the episode is done, we need to update the remaining steps to the end of the episode
        # for the current and all previous samples
        for i in torch.nonzero(done).flatten():
            # +1 because we overwrite to 0 inclusive
            ep_idx = (
                torch.arange(-self._running_steps[i] + 1, 1, device=self.device) + idx
            ) % self.bufflen
            assert len(ep_idx) <= self.bufflen, "Episode length exceeds buffer length"
            # Create a descending range of remaining steps to the end of the episode
            steps = torch.arange(self._running_steps[i] - 1, -1, -1, device=self.device)
            self._remaining_steps[i, ep_idx] = steps
            self._invalid_idx[i, 0] = (ep_idx[-1] + 1) % self.bufflen
        self._running_steps[done] = 0
        self.buffer[v_idx, idx] = sample
        # Update the helper indices
        self._idx = (self._idx + 1) % self.bufflen
        self._maxidx = min(self._maxidx + 1, self.bufflen - 1)

    def sample(self, batch_size: int) -> TensorDict[torch.Tensor]:
        """Sample a batch of hindsight experience transitions from the buffer.

        Args:
            batch_size: The batch size.
        """
        assert len(self) >= batch_size, "Not enough samples in the buffer"
        # Hindsight sample selection:
        # We need to sample from the valid indices. We track the invalid indices in the buffer with
        # the _invalid_idx helper. To sample only valid indices, we take the following steps:
        #
        # 1.) Randomly sample a vector index
        #
        # 2.) Compute the number of invalid samples per vector entry
        # a.) If the invalid index wraps around the buffer, we need to sum the samples from the end
        #     of the buffer with those from the beginning.
        # b.) If the invalid index does not wrap around the buffer, we can simply subtract the end
        #     index from the start index.
        #
        # 3.) Sample random integers from the range [0, maxidx + 1 - n_invalid). We then offset the
        #     sampled indices by the end index of the invalid index modulo  the maximum index.
        #
        # 4.) Clone the batch to prevent the overwriting of the original data
        #
        # 5.) Sample HER indices where we replace the goal with a virtual goal from the same
        #     trajectory
        #
        # 6.) Compute a random offset to a future sample of the same trajectory for the HER samples.
        #     We track the remaining steps to the end of the episode for each sample and use this
        #     information to randomly offset the HER samples within the same trajectory.
        #
        # 7.) Set the desired goal of the HER samples to the virtual goal and recalculate the reward
        v_idx = torch.randint(self.num_envs, size=(batch_size,), device=self.device)
        invalid_idx = self._invalid_idx[v_idx]
        # Compute the number of invalid samples per vector entry
        n_invalid = (invalid_idx[:, 1] - invalid_idx[:, 0] + 1) % self.bufflen
        # Sample random indices within the valid range
        offset_range = self.bufflen - n_invalid
        offsets = (torch.rand(batch_size, device=self.device) * offset_range).long()
        idx = (invalid_idx[:, 1] + 1 + offsets) % self.bufflen
        # Check if all samples are valid
        assert torch.all(self._remaining_steps[v_idx, idx] >= 0), "Invalid samples in the batch"
        # Clone the batch and sample HER indices
        batch = self.buffer[v_idx, idx].clone()
        her_idx = torch.randperm(batch_size, device=self.device)[: int(batch_size * self.p_her)]
        # Compute the random offset for the HER samples
        # Add +1 because we later multiply with torch.rand, which samples from [0, 1), so the final
        # offset is in the range [0, offset_interval)
        offset_interval = self._remaining_steps[v_idx, idx] + 1
        offset = (torch.rand(batch_size, device=self.device) * offset_interval).long()
        offset_idx = (idx + offset) % self.bufflen
        # Update the batch with virtual goals for HER samples
        virtual_goals = self.buffer["obs", "achieved_goal"][v_idx[her_idx], offset_idx[her_idx]]
        # desired goal needs to be replaced in both obs and next_obs
        batch["obs", "desired_goal"][her_idx] = virtual_goals
        batch["next_obs", "desired_goal"][her_idx] = virtual_goals
        achieved_goals = batch["next_obs", "achieved_goal"]
        # Overwrite all rewards. Updates the rewards in-place. If we overwrite only those rewards
        # that are affected by HER, we would need to copy the buffer first
        batch["reward"] = self.reward_fn(achieved_goals, batch["obs", "desired_goal"])
        return batch

    def __len__(self) -> int:
        n_invalid = (self._invalid_idx[:, 1] - self._invalid_idx[:, 0] + 1) % self.bufflen
        return (self.bufflen - n_invalid).sum().item()
