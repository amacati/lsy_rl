from typing import Any, Callable
import logging

from gymnasium import Env
from gymnasium import Wrapper
from tensordict import TensorDict

import torch
from torch import Tensor
import numpy as np

logger = logging.getLogger(__name__)


class TensorDictWrapper(Wrapper):

    def __init__(self, env: Env):
        super().__init__(env)


class DefaultTensorDictWrapper(TensorDictWrapper):
    """A wrapper that converts the actions and observations to Tensors.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self, env: Env, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        self.num_envs = env.num_envs
        self.device = device

        self.observation_space.sample = self._patch_space(self.env.observation_space.sample)
        self.action_space.sample = self._patch_space(self.env.action_space.sample)

        self._use_info = True

        # Infer the device of the environment. If the environment action space is a numpy array,
        # we need to convert the step() action to a numpy array before passing it to the
        # environment. If the environment action space is a Tensor, we ensure that it is on the
        # correct device
        self.env_mode, self.env_device = self._determine_env_mode(env)

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.num_envs = env.num_envs
        if self.env_mode == "np":  # Patch the sample() methods to return Tensors on the device
            self.observation_space.sample = self._patch_space(self.env.observation_space.sample)
            self.action_space.sample = self._patch_space(self.env.action_space.sample)

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = TensorDict({"action": action}, batch_size=self.num_envs, device=self.device)
        action = self.transform_action(action)  # Convert to np if necessary or send to env_device
        next_obs, reward, terminated, truncated, info = self.env.step(action)
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

    def transform_obs(self, obs: np.ndarray | Tensor | dict[str:np.ndarray]) -> Tensor | TensorDict:
        match obs:
            case np.ndarray():
                return torch.as_tensor(obs, device=self.device)
            case dict():
                return TensorDict(obs, batch_size=self.num_envs, device=self.device)
            case Tensor():
                return obs.to(self.device)
            case _:
                raise TypeError(f"Unsupported type {type(obs)}")

    def transform_action(self, action: Tensor) -> Tensor | np.ndarray:
        if self.env_mode == "np":
            return action.cpu().numpy()
        return action.to(self.env_device)

    def _patch_space(self, fn: Callable) -> Callable:

        def wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return wrapper

    def _determine_env_mode(self, env: Env) -> tuple[str, torch.device]:
        """Determine the input type and device of the environment.

        Some environments expect numpy arrays as input, others expect Tensors. If Tensors are
        expected, we ensure that they are on the correct device to avoid unnecessary data transfers.
        """
        try:
            from omni.isaac.orbit.envs import RLTaskEnv
            if isinstance(env.unwrapped, RLTaskEnv):
                return "torch", torch.device("cuda")
        except ImportError:  # IsaacSim is not installed or not open
            pass
        if isinstance(env.action_space.sample(), np.ndarray):
            return "np", torch.device("cpu")
        elif isinstance(env.action_space.sample(), Tensor):
            return "torch", env.action_space.sample().device
        raise TypeError(f"Unsupported action space {type(env.action_space.sample())}")
