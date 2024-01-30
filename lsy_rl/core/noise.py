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

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self._device = device
        self.params = nn.ParameterDict()

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device
        self.params.to(device)

    def reset(self):
        ...

    @abstractmethod
    def __call__(self, x: Tensor) -> Tensor:
        ...


class UniformNoise(Noise):

    def __init__(self, low: float, high: float, device: torch.device = torch.device("cpu")):
        super().__init__(device=device)
        assert isinstance(low, float) and isinstance(high, float), "low and high must be floats"
        self.params["low"] = nn.Parameter(torch.tensor(low), requires_grad=False)
        self.params["diff"] = nn.Parameter(torch.tensor(high - low), requires_grad=False)
        self.params.to(device)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return torch.rand(x.shape, device=x.device) * self.params["diff"] + self.params["low"]


class NormalNoise(Noise):

    def __init__(self, mean: float, std: float, device: torch.device = torch.device("cpu")):
        super().__init__(device=device)
        assert isinstance(mean, float) and isinstance(std, float), "mean and std must be floats"
        self.params["mean"] = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.params["std"] = nn.Parameter(torch.tensor(std), requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        x = torch.randn(x.shape, device=x.device) * self.params["std"] + self.params["mean"]
        return x


class HybridNoise(Noise):

    def __init__(self, noise: list[Noise], prob: Tensor,
                 device: torch.device = torch.device("cpu")):
        """Sample noise from a list of noise with given probability."""
        super().__init__(device=device)
        prob = torch.tensor(prob, device=device, dtype=torch.float32)
        assert len(noise) == len(prob), "noise and prob must have the same length"
        assert all(isinstance(n, Noise) for n in noise), "noise must be a list of Noise objects"
        assert all(n.shape == noise[0].shape for n in noise), "all noise must have the same shape"
        assert prob.sum() == 1.0, "prob must sum to 1"
        self.params["noise"] = nn.ModuleList(noise)
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)
        self.shape = noise[0].shape
        self.device = device  # Update device of all parameters

    def __call__(self):
        return self.params["noise"][torch.multinomial(self.params["prob"], 1)]()
