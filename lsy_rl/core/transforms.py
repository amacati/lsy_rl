from __future__ import annotations

from numbers import Number
import sys
from typing import Any, Iterable

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

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        ...


class ChainedTF(Transform):

    def __init__(self, transforms: list[Transform]):
        super().__init__()
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

    def reset(self):
        for transform in self.params["transforms"]:
            transform.reset()

    def forward(self, x: Tensor, *args) -> Tensor:
        for transform in self.params["transforms"]:
            x, args = transform(x, *args)
        return x, args


class IdentityTF(Transform):

    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, *args) -> Tensor:
        return x, args


class ClipTF(Transform):

    def __init__(self, min: Number, max: Number):
        super().__init__()
        assert isinstance(min, Number) and isinstance(max, Number), "min and max must be Numbers"
        self.params["min"] = nn.Parameter(torch.tensor(min, dtype=torch.float32),
                                          requires_grad=False)
        self.params["max"] = nn.Parameter(torch.tensor(max, dtype=torch.float32),
                                          requires_grad=False)

    def forward(self, x: Tensor, *args) -> Tensor:
        assert isinstance(x, Tensor), f"Input must be a Tensor, is {type(x)} {x}"
        return torch.clamp(x, self.params["min"], self.params["max"]), args


class AdditiveNoiseTF(Transform):

    def __init__(self, noise: Noise | dict):
        super().__init__()
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def forward(self, x: Tensor, *args) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x + self.params["noise"](x), args


class ScaleTF(Transform):

    def __init__(self, scale: Number | Iterable[Number]):
        super().__init__()
        assert isinstance(scale, (Number, Iterable)), "scale must be a Number or Iterable"
        self.params["scale"] = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, x: Tensor, *args) -> Tensor:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x * self.params["scale"], args
