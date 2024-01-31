import logging
from typing import Callable

import numpy as np
import sys
import torch
import torch.nn as nn

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
