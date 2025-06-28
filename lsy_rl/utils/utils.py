from __future__ import annotations

import datetime
import inspect
import logging
import random
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable, TypeVar, TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector import NormalizeObservation
from ml_collections import ConfigDict
from tensordict import TensorDict
from torch import Tensor

if TYPE_CHECKING:
    from lsy_rl.core.policy import Policy
    from lsy_rl.core.replay_buffer import ReplayBuffer
    from lsy_rl.core.transforms import Transform

logger = logging.getLogger(__name__)


def polyak_update_(target_net: nn.Module, net: nn.Module, tau: float):
    """Update the target network with the current weights.

    Note:
        This function is in-place.

    Args:
        target_net: The target network to update.
        net: The network to copy the weights from.
        tau: The polyak factor, where tau is the weight of the network weights and (1 - tau) is the
            weight of the target network weights. Must be in [0, 1].
    """
    assert 0.0 <= tau <= 1.0, "tau must be in [0, 1]"
    for target_param, param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def module_type_from_string(module_name: str) -> Callable[[str], type]:
    """Get a module type factory converting strings to types within the module.

    Args:
        module_name: The name of the module to get the types from.

    Example:
        >>> np_type = module_type_from_string("numpy")
        >>> x = np_type("array")([1, 2, 3])

    Returns:
        A factory function that converts a string to a type within the module.
    """

    def _module_type_from_string(name: str) -> type:
        return getattr(sys.modules[module_name], name)

    return _module_type_from_string


def torchify(x: np.ndarray, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    match x:
        case np.ndarray():
            return torch.as_tensor(x, device=device)
        case torch.Tensor():
            return x
        case _:
            raise TypeError(f"Unsupported type {type(x)}")


def torchify_dtype(dtype: np.dtype) -> torch.dtype:
    # np.bool is deprecated, but bool cannot be used in a match statement. This is a workaround.
    if dtype == bool:
        return torch.bool
    match dtype:
        case np.uint8:
            return torch.uint8
        case np.int8:
            return torch.int8
        case np.int16:
            return torch.int16
        case np.int32:
            return torch.int32
        case np.int64:
            return torch.int64
        case np.float16:
            return torch.float16
        case np.float32:
            return torch.float32
        case np.float64:
            return torch.float64
        case np.complex64:
            return torch.complex64
        case np.complex128:
            return torch.complex128
        case _:
            raise ValueError(f"Unsupported dtype {dtype}")


def load_config(path: Path) -> ConfigDict:
    """Load a toml configuration file and convert it to a ConfigDict object.

    Args:
        path: The path to the configuration file.

    Returns:
        The configuration as a ConfigDict object providing key access via dot/member syntax.
    """
    with open(path, "rb") as f:
        config = tomllib.load(f)
    return ConfigDict(config, type_safe=False)


def unique_folder(dir: Path | None) -> Path | None:
    """Create a unique folder in the directory.

    The name is a timestamp in the format 'YYYY_MM_DD_HH_MM'. If the folder already exists, we
    append a number to the timestamp, e.g. '2021_01_01_12_00_(1)'.

    Args:
        dir: The directory where the folder will be created.

    Returns:
        A unique folder.
    """
    if dir is None:
        return
    uid = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
    if (dir / uid).is_dir():
        t = 1
        while (dir / f"{uid}_({t})").is_dir():
            t += 1
        uid = f"{uid}_({t})"
    (dir / uid).mkdir(parents=True, exist_ok=False)
    return dir / uid


T = TypeVar("T")


def to_cls(
    cls: T | str, factory: Callable[[str], T] | None = None, expected_type: T | None = None
) -> T:
    """Convert the input that might be a class or a string of the class name to a class.

    Args:
        cls: The class type or name to convert.
        factory: A factory function to convert a string to a class.
        expected_type: The expected type of the value.

    Returns:
        The class type.
    """
    if not isinstance(cls, type):
        if isinstance(cls, str) and factory is not None:
            return factory(cls)
    elif issubclass(cls, expected_type):
        return cls
    raise TypeError(f"Invalid type {cls} (expected type {expected_type} or string)")


def check_kwargs(kwargs: dict[str, Any], cls: type, ignore: list[str] = []):
    """Check if all required arguments are present in the kwargs.

    Args:
        kwargs: The keyword arguments to check.
        cls: The class to check the arguments against.
        ignore: Any arguments to ignore.
    """
    assert isinstance(kwargs, dict), "Kwargs must be a dict"
    for x in required_args(cls):
        if x in ignore or x in ("args", "kwargs"):
            continue
        if x not in kwargs:
            raise ValueError(f"Missing required argument '{x}' for {cls}")


def required_args(cls: type) -> list[str]:
    """Get the required arguments for a class.

    Args:
        cls: The class to get the arguments for.

    Returns:
        The required arguments for the class.
    """
    return [p.name for p in inspect.signature(cls).parameters.values() if p.default == p.empty]


def set_seeds(seed: int | None = None):
    """Set the seeds for random, numpy, and torch.

    Args:
        seed: The seed value.
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def tensordict_sample(
    obs: Tensor,
    action: Tensor,
    next_obs: Tensor,
    reward: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    info: Tensor,
    device: torch.device | None = None,
) -> TensorDict:
    """Create a TensorDict sample from the environment.

    Expects tensors to be batched over environments.

    Args:
        obs: The observation.
        action: The action.
        next_obs: The next observation.
        reward: The reward.
        terminated: The terminated flag.
        truncated: The truncated flag.
        info: The info.
        device: The device to store the TensorDict on.

    Returns:
        The TensorDict sample.
    """
    return TensorDict(
        {
            "obs": obs,
            "action": action,
            "next_obs": next_obs,
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "info": info,
        },
        batch_size=obs.shape[0],
        device=device,
    )


def unwrap_norm_env(env: VectorEnv) -> NormalizeObservation | None:
    """Return the normalization wrapper if it exists, otherwise return None."""
    while hasattr(env, "env"):
        if isinstance(env, NormalizeObservation):
            return env
        env = env.env


def sync_env_normalization(train_envs: VectorEnv, eval_envs: VectorEnv):
    """Sync the normalization constants from the train env to the eval env."""
    train_norm_env = unwrap_norm_env(train_envs)
    eval_norm_env = unwrap_norm_env(eval_envs)
    if (train_norm_env is None) != (eval_norm_env is None):
        raise ValueError("Both envs must either have normalization or not have normalization")
    if train_norm_env is None and eval_norm_env is None:  # No normalization, no sync necessary
        return
    eval_norm_env.obs_rms.mean = train_norm_env.obs_rms.mean
    eval_norm_env.obs_rms.var = train_norm_env.obs_rms.var


def check_interrupt_sample(
    n_samples: int, last_n_samples: int, period: int | None = None, min_samples: int | None = None
) -> bool:
    """Check if we should interrupt sampling based on how many samples we have collected."""
    if min_samples is not None and n_samples < min_samples:
        return False
    if period is not None and n_samples - last_n_samples >= period:
        return True
    return False


def checkpoint(
    path: Path,
    policy: Policy,
    buffer: ReplayBuffer,
    critic_optimizer: torch.optim.Optimizer,
    actor_optimizer: torch.optim.Optimizer,
    obs_tf: Transform,
    checkpoint_buffer: bool = False,
):
    """Save a checkpoint of the policy, replay buffer and optimizers."""
    assert isinstance(path, Path), "The checkpoint path must be a Path object."
    assert path.exists(), f"Checkpoint path {path} doesn't exist."
    assert path.is_dir(), f"The checkpoint path {path} must be a directory."
    policy.save(path / "policy.pt")
    if checkpoint_buffer:
        buffer.save(path / "buffer.pt")
    torch.save(actor_optimizer.state_dict(), path / "actor_opt.pt")
    torch.save(critic_optimizer.state_dict(), path / "critic_opt.pt")
    torch.save(obs_tf.state_dict(), path / "obs_transform.pt")
