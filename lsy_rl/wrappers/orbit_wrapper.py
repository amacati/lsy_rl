from typing import Callable, Any
import logging

from gymnasium import Env

import torch
from torch import Tensor
from tensordict import TensorDict
from gymnasium import Wrapper

logger = logging.getLogger(__name__)


class OrbitWrapper(Wrapper):

    def __init__(self, env: Env, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        assert env.is_vector_env, "OrbitWrapper only supports vectorized environments"
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        self.device = device
        self.observation_space.sample = self._patch_space(self.env.observation_space.sample)
        self.action_space.sample = self._patch_space(self.env.action_space.sample)
        self._use_info = True

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = TensorDict({"action": action}, batch_size=self.num_envs, device=self.device)
        next_obs, reward, terminated, truncated, info = self.env.step(action.cuda())
        if self._use_info:  # If info has failed once, we disable it for the rest of the run
            try:
                sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
            except RuntimeError:
                logger.warning("Failed to convert info to TensorDict. Disabling info.")
                self._use_info = False
        sample["next_obs"] = self.transform_obs(next_obs)
        sample["reward"] = torch.as_tensor(reward)
        sample["terminated"] = torch.as_tensor(terminated)
        sample["truncated"] = torch.as_tensor(truncated)
        return sample

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        sample = TensorDict({}, batch_size=self.num_envs, device=self.device)
        if self._use_info:
            try:
                sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
            except RuntimeError:
                logger.warning("Failed to convert info to TensorDict. Disabling info.")
                self._use_info = False
        sample["obs"] = self.transform_obs(obs)
        return sample

    def transform_obs(self, obs: Tensor | dict[str:Tensor]) -> Tensor | TensorDict:
        match obs:
            case dict():
                return TensorDict(obs, batch_size=self.num_envs, device=self.device)
            case Tensor():
                return obs.to(self.device)
            case _:
                raise TypeError(f"Unsupported type {type(obs)}")

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return wrapper
