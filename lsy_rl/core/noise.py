from abc import ABC, abstractmethod

import torch
from torch import Tensor
import torch.nn as nn


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
    def __call__(self):
        ...


class UniformNoise(Noise):

    def __init__(self, low: Tensor, high: Tensor, device: torch.device = torch.device("cpu")):
        super().__init__()
        assert low.shape == high.shape, "low and high must have the same shape"
        self.params["low"] = nn.Parameter(low, requires_grad=False)
        self.params["diff"] = nn.Parameter(high - low, requires_grad=False)
        self.params.to(device)
        self.shape = low.shape

    def __call__(self):
        return torch.rand(self.shape) * self.params["diff"] + self.params["low"]


class NormalNoise(Noise):

    def __init__(self, mean: Tensor, std: Tensor, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.params["mean"] = nn.Parameter(mean, requires_grad=False)
        self.params["std"] = nn.Parameter(std, requires_grad=False)
        assert mean.shape == std.shape, "mean and std must have the same shape"
        self.shape = self.params["mean"].shape

    def __call__(self):
        x = torch.randn(self.shape, device=self.device) * self.params["std"] + self.params["mean"]
        return x


class HybridNoise(Noise):

    def __init__(self, noise: list[Noise], prob: Tensor,
                 device: torch.device = torch.device("cpu")):
        """Sample noise from a list of noise with given probability."""
        super().__init__(device=device)
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
