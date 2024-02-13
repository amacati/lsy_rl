from typing import Callable, Any
import logging

from gymnasium import Env

import torch
from torch import Tensor
from tensordict import TensorDict
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper

logger = logging.getLogger(__name__)


class OrbitWrapper(DefaultTensorDictWrapper):
    """Make Orbit envs compatible with the expected gymnasium interface.

    Orbit does not return the final observation by default when the episode is terminated or
    truncated. This wrapper ensures that the final observation is returned in the info dict under
    the 'final_observation' key.
    """

    def __init__(self, env: Env, max_episode_steps: int,
                 device: torch.device = torch.device("cpu")):
        super().__init__(env, device)
        self._num_steps = 0
        self.max_episode_steps = max_episode_steps

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = super().step(action)
        self._num_steps += 1
        # Ensure that no episode is terminated or truncated. If it does, Orbit does a partial reset
        # internally which we want to avoid.
        assert torch.all(~sample["terminated"] & ~sample["truncated"])
        sample["truncated"][:] = self._num_steps >= self.max_episode_steps
        if torch.any(sample["terminated"] | sample["truncated"]):
            sample["info", "final_observation"] = sample["next_obs"].clone()
            # Orbit reset does not update the buffers correctly, therefore we reset manually
            new_sample = self.reset()
            sample["next_obs"] = new_sample["obs"]
        return sample

    def reset(self,
              *,
              seed: list[int] | None = None,
              options: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
        """Patch Orbit's reset by taking an additional step in the environment.

        IsaacSim does not update its buffers correctly on resets. As a consequence, the reset
        observation might be incorrect. Link buffers contain either zeros on initialization, or
        carry over stale information from the previous episode.

        To bring the buffers back in sync, we have to take a simulation step. For more information
        see https://github.com/NVIDIA-Omniverse/orbit/issues/240.

        Warning:
            This effectively shortens the environment horizon by one!
        """
        seed = None if seed is None else seed[0]
        self.env.reset(seed=seed, options=options)
        action = torch.zeros_like(self.action_space.sample())  # Try to take a zero action
        if action.cpu().numpy() not in self.env.action_space:  # If zero action is invalid, sample
            self.action_space.seed(seed=seed)  # Make random initial action reproducible
            action = self.action_space.sample()
        obs, _, _, _, info = self.env.step(self.transform_action(action))
        sample = TensorDict({}, batch_size=self.num_envs, device=self.device)
        sample["obs"] = self.transform_obs(obs).clone()
        sample["info"] = self.transform_info(info).clone()
        self._num_steps = 0
        return sample
