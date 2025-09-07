from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from tensordict import TensorDict
from torch import IntTensor

from lsy_rl.utils.utils import module_type_from_string

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

replay_buffer_cls: Callable[[str], type[ReplayBuffer]] = module_type_from_string(__name__)


class ReplayBuffer(ABC):
    """Abstract base class for replay buffers."""

    def __init__(self):
        """Initialize the replay buffer."""
        pass

    @abstractmethod
    def add(self, sample: TensorDict[torch.Tensor]):
        """Add a sample to the buffer."""
        pass

    @abstractmethod
    def clear(self):
        """Clear the replay buffer."""
        pass

    @abstractmethod
    def save(self, path: Path):
        """Save the replay buffer to a file."""
        pass

    @abstractmethod
    def load(self, path: Path):
        """Load the replay buffer from a file."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of valid samples in the buffer."""
        pass


def _allocate_buffers(buffer: TensorDict, sample: TensorDict) -> TensorDict:
    """Allocate buffers for unknown keys in the sample."""
    n_batch_dims = len(buffer.batch_size) - 1
    for key, val in sample.items():
        if key not in buffer.keys():
            if isinstance(val, torch.Tensor):
                buffer[key] = torch.zeros(
                    (*buffer.batch_size, *val.shape[n_batch_dims:]), dtype=val.dtype
                )
            elif isinstance(val, TensorDict):
                buffer[key] = TensorDict({}, batch_size=(buffer.batch_size), device=buffer.device)
            else:
                raise TypeError(f"Unsupported type {type(val)}")
    return buffer


class SimpleReplayBuffer(ReplayBuffer):
    """Simple replay buffer for a single environment."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        """Initialize the replay buffer.

        Args:
            num_envs: Number of environments.
            max_size: Maximum size of the buffer.
            device: Buffer device.
            seed: Random seed.
        """
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
        _allocate_buffers(self.buffer, sample)
        # If num_samples + self._idx > self.max_size, the index wraps around to the beginning of the
        # buffer. We take the index vector modulo ``self.max_size`` to implement this behavior.
        idx = torch.arange(self._idx, self._idx + num_samples) % self.max_size
        self.buffer[idx] = sample
        # Update the helper indices
        self._idx = (self._idx + num_samples) % self.max_size
        self._maxidx = min(self._maxidx + num_samples, self.max_size - 1)

    def sample(self, batch_size: int) -> TensorDict[torch.Tensor]:
        """Sample a batch of transitions from the buffer."""
        if batch_size > self._maxidx + 1:
            idx = np.random.randint(0, self._maxidx + 1, batch_size)
        else:
            idx = np.array(random.sample(range(self._maxidx + 1), batch_size))
        return self.buffer[idx]

    def clear(self):
        """Clear the replay buffer."""
        for b in self.buffer.values():
            b.zero_()
        self._idx = 0
        self._maxidx = -1

    def save(self, path: Path):
        """Save the replay buffer to a file.

        Args:
            path: The path to the file.
        """
        save_dict = {"idx": self._idx, "maxidx": self._maxidx, "buffer": self.buffer}
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Load the replay buffer from a file.

        Args:
            path: The path to the file.
        """
        # Note: weights_only=False is required to recover the buffer when loading
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self._idx, self._maxidx = save_dict["idx"], save_dict["maxidx"]
        self.buffer = save_dict["buffer"]
        assert self.buffer.batch_size == self.max_size, "Loaded buffer has wrong size"

    def __len__(self) -> int:
        """Return the number of valid samples in the buffer."""
        return self._maxidx + 1


class TrajectoryBuffer(ReplayBuffer):
    """Vectorized trajectory buffer."""

    def __init__(
        self, num_envs: int, trajectory_len: int, device: torch.device = torch.device("cpu")
    ):
        """Initialize the vectorized trajectory buffer.

        Args:
            num_envs: Number of environments.
            max_size: Maximum size of the buffer.
            device: Buffer device.
        """
        super().__init__()
        self.num_envs = num_envs
        self.trajectory_len = trajectory_len
        # Allocate buffers
        self.device = device
        self.buffer = TensorDict({}, batch_size=(self.trajectory_len, self.num_envs), device=device)
        # Allocate helper for default environment indexing
        self._env_idx = torch.arange(self.num_envs, dtype=int, device=device)
        self._mask = torch.ones(self.num_envs, dtype=torch.bool, device=device)
        # Helper indices pointing to the current buffer write position
        self._idx = torch.zeros(self.num_envs, dtype=int, device=device)

    def add(self, sample: TensorDict[torch.Tensor], mask: torch.Tensor | None = None):
        """Add a vector sample to the buffer."""
        # If no mask is given, assume one sample per env
        mask = self._mask if mask is None else mask
        v_idx = self._env_idx[mask]
        assert torch.all(self._idx[v_idx] < self.trajectory_len), "Buffer overflow"
        assert sample.batch_size[0] == v_idx.shape[0], "Sample size must match the env index"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        _allocate_buffers(self.buffer, sample)
        # Compute the indices for each sample
        self.buffer[self._idx[v_idx], v_idx] = sample
        # Update the helper indices
        self._idx[v_idx] += 1

    def clear(self):
        """Clear the replay buffer."""
        for b in self.buffer.values():
            b.zero_()
        self._idx[...] = 0

    def save(self, path: Path):
        """Save the replay buffer to a file.

        Args:
            path: The path to the file.
        """
        save_dict = {"idx": self._idx, "buffer": self.buffer}
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Load the replay buffer from a file.

        Args:
            path: The path to the file.
        """
        # Note: weights_only=False is required to recover the buffer when loading
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self._idx, self.buffer = save_dict["idx"], save_dict["buffer"]
        assert self.buffer.batch_size == (self.trajectory_len, self.num_envs), (
            "Loaded buffer has wrong size"
        )

    def full(self) -> bool:
        """Check if the buffer is full."""
        return torch.all(self._idx == self.trajectory_len)

    def __len__(self) -> int:
        """Return the number of valid samples in the buffer."""
        return torch.sum(self._idx).item()

    def __getitem__(self, key: str) -> torch.Tensor:
        """Get a sample from the buffer."""
        return self.buffer[key]


class VectorReplayBuffer(ReplayBuffer):
    """Vectorized replay buffer for multiple environments."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        """Initialize the vectorized replay buffer.

        Args:
            num_envs: Number of environments.
            max_size: Maximum size of the buffer.
            device: Buffer device.
            seed: Random seed.
        """
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
        v_idx = self._default_v_idx if v_idx is None else v_idx
        assert sample.batch_size[0] == v_idx.shape[0], "Sample size must match the env index"
        # Check if there are unknown keys in the sample and allocate buffers for them if necessary
        _allocate_buffers(self.buffer, sample)
        # Count the number of samples per environment
        num_samples = torch.bincount(v_idx, minlength=self.num_envs).to(self.device)
        assert torch.max(num_samples) < self.bufflen, "Sample sizes must be smaller than the buffer"
        # Compute the indices for each sample
        idx = (self._categorical_cumsum(v_idx) + self._idx[v_idx]) % self.bufflen
        # Indices need to be 2D for indexing
        self.buffer[v_idx, idx] = sample
        # Update the helper indices
        self._idx[v_idx] = (self._idx[v_idx] + num_samples[v_idx]) % self.bufflen
        self._maxidx[v_idx] = torch.clip(
            self._maxidx[v_idx] + num_samples[v_idx], max=self.bufflen - 1
        )

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
        """Sample a batch of transitions from the buffer.

        Args:
            batch_size: The batch size.
        """
        v_idx = torch.randint(self.num_envs, size=(batch_size,), device=self.device)
        idx = self.rng.integers(self._maxidx[v_idx].cpu() + 1, size=batch_size)
        idx = torch.tensor(idx).to(self.device)
        return self.buffer[v_idx, idx]

    def clear(self):
        """Clear the replay buffer."""
        for b in self.buffer.values():
            b.zero_()
        self._idx[:] = 0
        self._maxidx[:] = -1

    def save(self, path: Path):
        """Save the replay buffer to a file.

        Args:
            path: The path to the file.
        """
        save_dict = {"idx": self._idx, "maxidx": self._maxidx, "buffer": self.buffer}
        torch.save(save_dict, path)

    def load(self, path: Path):
        """Load the replay buffer from a file.

        Args:
            path: The path to the file.
        """
        # Note: weights_only=False is required to recover the buffer when loading
        save_dict = torch.load(path, map_location=self.device, weights_only=False)
        self._idx, self._maxidx = save_dict["idx"], save_dict["maxidx"]
        self.buffer = save_dict["buffer"]
        assert self.buffer.batch_size == (self.num_envs, self.bufflen), (
            "Loaded buffer has wrong size"
        )

    def __len__(self) -> int:
        """Return the number of valid samples in the buffer."""
        return torch.sum(self._maxidx + 1).item()


class HerVectorReplayBuffer(VectorReplayBuffer):
    """Hindsight Experience Replay buffer for vectorized environments."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        reward_fn: Callable,
        p_her: float = 0.8,
        device: torch.device = torch.device("cpu"),
        seed: int | None = None,
    ):
        """Initialize the HER replay buffer.

        Args:
            num_envs: Number of environments.
            max_size: Maximum size of the buffer.
            reward_fn: Reward function for HER.
            p_her: Probability of replacing the goal of a sample with an achieved goal.
            device: Buffer device.
            seed: Random seed.
        """
        super().__init__(num_envs, max_size, device, seed)
        assert isinstance(reward_fn, Callable), "Reward function must be a callable"
        self.reward_fn = reward_fn
        self.p_her = p_her
        # We track the remaining steps to the end of the trajectory for each environment. The is
        # necessary to sample a virtual goal from future states of the same trajectory
        self._idx = torch.zeros((num_envs,), dtype=int, device=self.device)
        self._maxidx = -torch.ones((num_envs,), dtype=int, device=self.device)
        self._remaining_steps = torch.empty(
            (num_envs, max_size // num_envs), dtype=int, device=self.device
        )
        self._remaining_steps[:] = -1
        self._running_steps = torch.zeros(num_envs, dtype=int, device=self.device)

    def __len__(self) -> int:
        """Return the number of valid samples in the buffer."""
        return torch.sum(self._maxidx + 1).item()

    def add(self, sample: TensorDict[torch.Tensor], v_idx: IntTensor | None = None):
        """Add a vectorized sample to the buffer.

        If the buffer is full, overwrite the oldest samples.
        """
        _allocate_buffers(self.buffer, sample)
        if v_idx is not None:
            assert len(v_idx) == len(torch.unique(v_idx)), "v_idx must contain unique indices"
            v_idx = v_idx.to(self.device)
        else:
            v_idx = self._default_v_idx
        # A sample must contain exactly one sample per vector entry
        assert sample.batch_size[0] == v_idx.shape[0], "Sample size must match the env index"
        # +1 because we overwrite to 0 inclusive
        self._remaining_steps[v_idx, self._idx[v_idx]] = 0
        col_idx, row_idx = self._col_row_idx(self._idx, self._running_steps, self.bufflen)
        self._remaining_steps[col_idx, row_idx] += 1
        self._running_steps += 1
        done = (sample["terminated"] | sample["truncated"]).squeeze()  # Remove batch dimension
        self._running_steps[v_idx[done.to(self.device)]] = 0
        self.buffer[v_idx, self._idx[v_idx]] = sample
        # Update the helper indices
        self._idx[v_idx] = (self._idx[v_idx] + 1) % self.bufflen
        self._maxidx[v_idx] = torch.clip(self._maxidx[v_idx] + 1, max=self.bufflen - 1)

    @staticmethod
    @torch.jit.script
    def _col_row_idx(
        row_end: torch.Tensor, row_lengths: torch.Tensor, max_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create column and row index vectors for a 2D matrix.

        Equivalent of indexing x[col_idx, (row_end - row_lengths) : row_end] with non-homogeneous
        row lengths. PyTorch does not support this kind of indexing, so we need to create the row
        and column indices manually. The row indices are cyclic and wrap around the matrix.

        Note:
            This function is implemented in TorchScript to be used in the JIT compiler so that the
            loops are less expensive.

        Args:
            row_end: The end of the row index.
            row_lengths: The lengths of the rows.
            max_len: The maximum length of the rows.
        """
        assert row_end.shape == row_lengths.shape, (
            "row_end and row_lengths must have the same shape"
        )
        device = row_end.device
        row_idx = torch.cat(
            [
                (torch.arange(-row_lengths[i], 0, device=device) + row_end[i]) % max_len
                for i in range(len(row_lengths))
            ]
        )
        col_idx = torch.cat([torch.full([int(row_lengths[i])], i) for i in range(len(row_lengths))])
        return col_idx, row_idx

    def sample(self, batch_size: int) -> TensorDict[torch.Tensor]:
        """Sample a batch of hindsight experience transitions from the buffer.

        Args:
            batch_size: The batch size.
        """
        # Hindsight sample selection:
        #
        # 1.) Randomly sample a vector index
        #
        # 2.) Sample random, valid indices for each vector.
        #
        # 3.) Clone the batch to prevent the overwriting of the original data
        #
        # 4.) Sample HER indices where we replace the goal with a virtual goal from the same
        #     trajectory
        #
        # 5.) Compute a random offset to a future sample of the same trajectory for the HER samples.
        #     We track the remaining steps to the end of the episode for each sample and use this
        #     information to randomly offset the HER samples within the same trajectory.
        #
        # 6.) Set the desired goal of the HER samples to the virtual goal and recalculate the reward
        v_idx = torch.randint(self.num_envs, size=(batch_size,), device=self.device)
        idx = (torch.rand(batch_size, device=self.device) * self._maxidx[v_idx]).long()
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
