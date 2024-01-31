from __future__ import annotations

from numbers import Number
import sys

import torch
import torch.nn as nn
from torch import Tensor

from lsy_rl.core.noise import Noise, noise_cls
from lsy_rl.utils.utils import module_type_from_string

transform_cls: type[Transform] = module_type_from_string(__name__)


def transform_cls(name: str) -> type[Transform]:
    return getattr(sys.modules[__name__], name)


class Transform(nn.Module):

    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict()

    def reset(self):
        ...

    def __call__(self, x: Tensor) -> Tensor:
        ...


class ChainedTF(Transform):

    def __init__(self, transforms: list[Transform]):
        super().__init__()
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.Sequential(*transforms)

    def reset(self):
        for transform in self.params["transforms"]:
            transform.reset()

    def __call__(self, x: Tensor) -> Tensor:
        return self.params["transforms"](x)


class IdentityTF(Transform):

    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor) -> Tensor:
        return x


class ClipTF(Transform):

    def __init__(self, min: Number, max: Number):
        super().__init__()
        assert isinstance(min, Number) and isinstance(max, Number), "min and max must be Numbers"
        self.params["min"] = nn.Parameter(torch.tensor(min, dtype=torch.float32),
                                          requires_grad=False)
        self.params["max"] = nn.Parameter(torch.tensor(max, dtype=torch.float32),
                                          requires_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return torch.clamp(x, self.params["min"], self.params["max"])


class AdditiveNoiseTF(Transform):

    def __init__(self, noise: Noise | dict):
        super().__init__()
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def __call__(self, x: Tensor) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x + self.params["noise"](x)
