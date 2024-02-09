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

    def __init__(self, env: Env, device: torch.device = torch.device("cpu")):
        super().__init__(env, device)

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = super().step(action)
        if torch.any(sample["terminated"] | sample["truncated"]):
            sample["info", "final_observation"] = sample["next_obs"].clone()
        return sample
