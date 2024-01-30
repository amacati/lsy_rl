from __future__ import annotations

from abc import ABC, abstractmethod
import sys
from typing import Iterable

import torch
from torch import Tensor
import torch.nn as nn


def noise_cls(name: str) -> type[Noise]:
    return getattr(sys.modules[__name__], name)


class Noise(torch.nn.Module, ABC):

    def __init__(self,):
        super().__init__()
        self.params = nn.ParameterDict()

    def reset(self):
        ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


class UniformNoise(Noise):

    def __init__(
        self,
        low: float,
        high: float,
    ):
        super().__init__()
        assert isinstance(low, float) and isinstance(high, float), "low and high must be floats"
        self.params["low"] = nn.Parameter(torch.tensor(low), requires_grad=False)
        self.params["diff"] = nn.Parameter(torch.tensor(high - low), requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return torch.rand(x.shape, device=x.device) * self.params["diff"] + self.params["low"]


class NormalNoise(Noise):

    def __init__(self, mean: float, std: float):
        super().__init__()
        assert isinstance(mean, float) and isinstance(std, float), "mean and std must be floats"
        self.params["mean"] = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.params["std"] = nn.Parameter(torch.tensor(std), requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        x = torch.randn(x.shape, device=x.device) * self.params["std"] + self.params["mean"]
        return x


class EpsilonNoise(Noise):

    def __init__(self, noise: Noise, epsilon: float):
        super().__init__()
        assert isinstance(noise, Noise), "noise must be a Noise object"
        assert isinstance(epsilon, float), "epsilon must be a float"
        self.params["noise"] = noise
        self.params["epsilon"] = nn.Parameter(torch.tensor(epsilon), requires_grad=False)

    def __call__(self, x: Tensor):
        choice = torch.rand(x.shape[0], device=x.device) < self.params["epsilon"]
        return torch.where(choice[:, None], self.params["noise"](x), 0)


class HybridNoise(Noise):

    def __init__(self, noise: list[Noise], prob: Tensor):
        """Sample noise from a list of noise with given probability."""
        super().__init__()
        prob = torch.tensor(prob, dtype=torch.float32)
        assert len(noise) == len(prob), "noise and prob must have the same length"
        assert all(isinstance(n, Noise) for n in noise), "noise must be a list of Noise objects"
        assert all(n.shape == noise[0].shape for n in noise), "all noise must have the same shape"
        assert prob.sum() == 1.0, "prob must sum to 1"
        self.params["noise"] = nn.ModuleList(noise)
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)
        self.shape = noise[0].shape

    def __call__(self):
        return self.params["noise"][torch.multinomial(self.params["prob"], 1)]()
