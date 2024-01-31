from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable
from numbers import Number

import torch
from torch import Tensor
import torch.nn as nn

from lsy_rl.utils.utils import module_type_from_string

noise_cls: type[Noise] = module_type_from_string(__name__)


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

    def __init__(self, min: Number, max: Number):
        super().__init__()
        assert isinstance(min, Number) and isinstance(max, Number), "min and max must be floats"
        self.params["min"] = nn.Parameter(torch.tensor(min, dtype=torch.float32),
                                          requires_grad=False)
        self.params["diff"] = nn.Parameter(torch.tensor(max - min, dtype=torch.float32),
                                           requires_grad=False)

    def __call__(self, x: Tensor):
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return torch.rand(x.shape, device=x.device) * self.params["diff"] + self.params["min"]


class NormalNoise(Noise):

    def __init__(self, mean: Number, std: Number):
        super().__init__()
        assert isinstance(mean, Number) and isinstance(std, Number), "mean and std must be floats"
        self.params["mean"] = nn.Parameter(torch.tensor(mean), requires_grad=False)
        self.params["std"] = nn.Parameter(torch.tensor(std), requires_grad=False)

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
        self.params["epsilon"] = nn.Parameter(torch.tensor(epsilon, torch.float32),
                                              requires_grad=False)

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
