import copy
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
from gymnasium import Env, Wrapper
from tensordict import TensorDict
from torch import Tensor

logger = logging.getLogger(__name__)


class TensorDictWrapper(Wrapper, ABC):
    def __init__(self, env: Env):
        super().__init__(env)

    @abstractmethod
    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        ...

    @abstractmethod
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> TensorDict:
        ...


class DefaultTensorDictWrapper(TensorDictWrapper):
    """A wrapper that converts the actions and observations to Tensors.

    If the environment expects numpy arrays, actions are converted to numpy arrays before being
    passed to the environment. If the environment expects Tensors, the actions are sent to the
    device of the environment. If both the environment and the training are on the same device, this
    wrapper is a no-op. Observations are always converted to Tensors on the training device.
    """

    def __init__(self, env: Env, device: torch.device = torch.device("cpu")):
        super().__init__(env)
        self.num_envs = env.num_envs
        self.device = device

        # Infer the device of the environment. If the environment action space is a numpy array,
        # we need to convert the step() action to a numpy array before passing it to the
        # environment. If the environment action space is a Tensor, we ensure that it is on the
        # correct device
        self.env_mode, self.env_device = self._determine_env_mode(env)

        # Patch the sample() methods to return Tensors on the device. Use copy.deepcopy to avoid
        # modifying the original spaces.
        self.observation_space = copy.deepcopy(env.observation_space)
        self.action_space = copy.deepcopy(env.action_space)
        self.observation_space.sample = self._patch_space(self.observation_space.sample)
        self.action_space.sample = self._patch_space(self.action_space.sample)
        self._failed_info_keys = set()  # Keep track of info keys that failed to convert

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        sample = TensorDict({"action": action}, batch_size=self.num_envs, device=self.device)
        action = self.transform_action(action)  # Convert to np if necessary or send to env_device
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        sample["next_obs"] = self.transform_obs(next_obs).clone()
        sample["reward"] = torch.as_tensor(reward, dtype=torch.float64).clone()
        sample["terminated"] = torch.as_tensor(terminated).clone()
        sample["truncated"] = torch.as_tensor(truncated).clone()
        sample["info"] = self.transform_info(info).clone()
        return sample

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Tensor, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        sample = TensorDict({}, batch_size=self.num_envs, device=self.device)
        sample["obs"] = self.transform_obs(obs).clone()
        sample["info"] = self.transform_info(info).clone()
        return sample

    def transform_obs(
        self, obs: np.ndarray | Tensor | dict[str : np.ndarray]
    ) -> Tensor | TensorDict:
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
        assert isinstance(action, Tensor), "Action input must be a tensor"
        if self.env_mode == "np":
            return action.detach().cpu().numpy()
        return action.to(self.env_device)

    def transform_info(self, info: dict) -> TensorDict:
        assert isinstance(info, dict), f"Expected dict, got {type(info)}"
        info_tf = {}
        for key, value in info.items():
            match value:
                case np.ndarray():
                    if value.dtype == np.object_:
                        info_tf[key] = self._transform_np_object(value)
                    else:
                        info_tf[key] = torch.as_tensor(value, device=self.device)
                case Tensor():
                    info_tf[key] = value.to(self.device)
                case _:
                    if key not in self._failed_info_keys:
                        self._failed_info_keys.add(key)  # Only log once per key
                        logger.warning(
                            (
                                f"Dropping info key '{key}' with unsupported conversion "
                                f"type {type(value)}"
                            )
                        )
        return TensorDict(info_tf, batch_size=self.num_envs, device=self.device)

    def _transform_np_object(self, value: np.ndarray) -> dict[str, np.ndarray] | np.ndarray:
        """Converts a numpy array with dtype np.object to a list of Tensors."""
        assert isinstance(value, np.ndarray), f"Expected np.ndarray, got {type(value)}"
        assert value.dtype == object, f"Expected dtype np.object, got {value.dtype}"
        match value[0]:
            case dict():
                return {k: np.stack([d[k] for d in value]) for k in value[0].keys()}
            case np.ndarray():
                return np.stack([x for x in value])
            case _:
                raise TypeError(f"Unsupported type {type(value[0])}")

    def _patch_space(self, fn: Callable) -> Callable:
        def space_wrapper():
            return torch.as_tensor(fn(), device=self.device)

        return space_wrapper

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
