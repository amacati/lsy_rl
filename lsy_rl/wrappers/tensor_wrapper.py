from typing import Any, Callable

from gymnasium import Env

import torch
from torch import FloatTensor, BoolTensor, Tensor
import numpy as np


class TensorWrapper(Env):
    """A wrapper that converts the actions and observations to Tensors.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self, env: Env, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.env = env

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

    def step(self,
             action: Tensor) -> tuple[Tensor, FloatTensor, BoolTensor, BoolTensor, dict[str, Any]]:
        action = self._convert_action(action)  # Convert to np if necessary or send to env_device
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = torch.as_tensor(obs, device=self.device)
        reward = torch.as_tensor(reward, device=self.device)
        terminated = torch.as_tensor(terminated, dtype=torch.bool, device=self.device)
        truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
        info = self._dict_to_device(info)
        return obs, reward, terminated, truncated, info

    def reset(self,
              *,
              seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[Tensor, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return torch.as_tensor(obs, device=self.device), info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def _convert_action(self, action: Tensor) -> Tensor | np.ndarray:
        if self.env_mode == "np":
            return action.cpu().numpy()
        return action.to(self.env_device)

    def _dict_to_device(self, data: dict) -> dict:
        for key, value in data.items():
            if isinstance(value, dict):
                data[key] = self._dict_to_device(value)
            elif isinstance(value, list):
                data[key] = [self._dict_to_device(v) for v in value]
            elif isinstance(value, np.ndarray):
                if value.dtype != np.object_:  # Don't convert object arrays
                    data[key] = torch.as_tensor(value, device=self.device)
            elif isinstance(value, Tensor):
                data[key] = value.to(self.output_device)
        return data

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return wrapper
