from typing import Any, Callable

from gymnasium import Env
from tensordict import TensorDict

import torch
from torch import Tensor
import numpy as np


class TensorWrapper(Env):
    """A wrapper that converts the actions and observations to Tensors.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self,
                 env: Env,
                 device: torch.device = torch.device("cpu"),
                 info_keys: list[str] = []):
        super().__init__()
        self.env = env
        # Only keep wanted keys in infodict to prevent undesired memory usage or Tensor conversion
        # errors, e.g. when trying to convert numpy object array
        self.info_keys = info_keys

        # Infer the device of the environment. If the environment action space is a numpy array,
        # we need to convert the step() action to a numpy array before passing it to the
        # environment. If the environment action space is a Tensor, we ensure that it is on the
        # correct device
        self.env_device = torch.device("cpu")
        self.env_mode = "np"
        sample_action = self.env.action_space.sample()
        if isinstance(sample_action, Tensor):
            self.env_mode = "torch"
            self.env_device = sample_action.device

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        self.device = device
        if self.env_mode == "np":  # Patch the sample() methods to return Tensors on the device
            self.observation_space.sample = self._patch_space(self.env.observation_space.sample)
            self.action_space.sample = self._patch_space(self.env.action_space.sample)

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = TensorDict({}, batch_size=self.num_envs, device=self.device)
        action = self._convert_action(action)  # Convert to np if necessary or send to env_device
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        if self.info_keys:
            info = {key: value for key, value in info.items() if key in self.info_keys}
            sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
        sample["next_obs"] = torch.as_tensor(next_obs)
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
        if self.info_keys:
            info = {key: value for key, value in info.items() if key in self.info_keys}
            sample["info"] = TensorDict(info, batch_size=self.num_envs, device=self.device)
        sample["obs"] = torch.as_tensor(obs)
        return sample

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _convert_action(self, action: Tensor) -> Tensor | np.ndarray:
        if self.env_mode == "np":
            return action.cpu().numpy()
        return action.to(self.env_device)

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return wrapper
