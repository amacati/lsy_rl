from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch import Tensor

from lsy_rl.utils.utils import module_type_from_string

noise_cls: Callable[[str], type[Noise]] = module_type_from_string(__name__)


class Noise(torch.nn.Module, ABC):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict()

    def reset(self):
        ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


class UniformNoise(Noise):
    def __init__(self, min: Number | list[Number], max: Number | list[Number]):
        super().__init__()
        assert isinstance(min, (Number, list)), f"min must be a float or list of floats, got {min}"
        assert isinstance(max, (Number, list)), f"max must be a float or list of floats, got {max}"
        min, max = torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)
        self.params["min"] = nn.Parameter(min, requires_grad=False)
        self.params["diff"] = nn.Parameter(max - min, requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return torch.rand(x.shape, device=x.device) * self.params["diff"] + self.params["min"]


class NormalNoise(Noise):
    def __init__(self, mean: Number | list[Number], std: Number | list[Number]):
        super().__init__()
        assert isinstance(mean, (Number, list)), f"mean must be float or list of floats, got {mean}"
        assert isinstance(std, (Number, list)), f"std must be float or list of floats, got {std}"
        mean, std = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
        self.params["mean"] = nn.Parameter(mean, requires_grad=False)
        self.params["std"] = nn.Parameter(std, requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        x = torch.randn(x.shape, device=x.device) * self.params["std"] + self.params["mean"]
        return x


class EpsilonNoise(Noise):
    def __init__(self, noise: Noise, epsilon: Number):
        super().__init__()
        assert isinstance(noise, Noise), "noise must be a Noise object"
        assert isinstance(epsilon, Number), "epsilon must be a Number"
        self.params["noise"] = noise
        eps = torch.tensor(epsilon, dtype=torch.float32)
        self.params["epsilon"] = nn.Parameter(eps, requires_grad=False)

    def __call__(self, x: Tensor):
        choice = torch.rand(x.shape[0], device=x.device) < self.params["epsilon"]
        return torch.where(choice[:, None], self.params["noise"](x), 0)


class ZeroNoise(Noise):
    def __init__(self):
        super().__init__()

    def __call__(self, x: Tensor):
        return torch.zeros_like(x)


class HybridNoise(Noise):
    def __init__(self, noise: list[Noise | dict], prob: Iterable[Number]):
        """Sample noise from a list of noise with given probability."""
        super().__init__()
        prob = torch.tensor(prob, dtype=torch.float32)
        assert len(noise) == len(prob), "noise and prob must have the same length"
        # Convert potential dicts to Noise objects
        noise = [noise_cls(n["type"])(**n["kwargs"]) if isinstance(n, dict) else n for n in noise]
        assert all(isinstance(n, Noise) for n in noise), "noise must be a list of Noise objects"
        assert prob.sum() == 1.0, "prob must sum to 1"
        self.params["noise"] = nn.ModuleList(noise)
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)

    def __call__(self, x: Tensor):
        noise_idx = torch.multinomial(self.params["prob"], x.shape[0], replacement=True)
        noise = torch.zeros_like(x)
        for i, noise_idx in enumerate(noise_idx):
            noise[i] = self.params["noise"][noise_idx](x[i])
        return noise
