from __future__ import annotations

from numbers import Number
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch import Tensor

from lsy_rl.core.noise import Noise, noise_cls
from lsy_rl.utils.utils import module_type_from_string

# TODO: Replace with plain transform
transform_cls: type[Transform] = module_type_from_string(__name__)


# TODO: Remove transform *args passing in forward
class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.params = nn.ParameterDict()

    def reset(self): ...

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]: ...

    def update(self, x: Tensor): ...


class ChainedTF(Transform):
    def __init__(self, transforms: list[Transform]):
        super().__init__()
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

    def reset(self):
        for transform in self.params["transforms"]:
            transform.reset()

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        for transform in self.params["transforms"]:
            x, args = transform(x, *args)
        return x, args


class IdentityTF(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        return x, args


class ClipTF(Transform):
    def __init__(self, min: Number, max: Number):
        super().__init__()
        assert isinstance(min, Number) and isinstance(max, Number), "min and max must be Numbers"
        self.params["min"] = nn.Parameter(
            torch.tensor(min, dtype=torch.float32), requires_grad=False
        )
        self.params["max"] = nn.Parameter(
            torch.tensor(max, dtype=torch.float32), requires_grad=False
        )

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        assert isinstance(x, Tensor), f"Input must be a Tensor, is {type(x)} {x}"
        return torch.clamp(x, self.params["min"], self.params["max"]), args


class AdditiveNoiseTF(Transform):
    def __init__(self, noise: Noise | dict):
        super().__init__()
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x + self.params["noise"](x), args


class ChoiceTF(Transform):
    def __init__(self, transforms: list[Transform | dict], prob: list[float]):
        super().__init__()
        # Convert potential dicts to Transform objects
        for i, transform in enumerate(transforms):
            if isinstance(transform, dict):
                transform = transform_cls(transform["type"])(**(transform.get("kwargs") or {}))
                transforms[i] = transform

        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)
        prob = torch.tensor(prob, dtype=torch.float32)
        assert torch.all(prob >= 0), "p must be non-negative"
        assert torch.isclose(prob.sum(), torch.tensor(1.0)), "p must sum to 1"
        assert len(prob) == len(transforms), "prob must have the same length as transforms"
        prob = prob / prob.sum()
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        tf_idx = torch.multinomial(self.params["prob"], x.shape[0], replacement=True)
        x = torch.stack(
            [self.params["transforms"][j](x[i], *args)[0] for i, j in enumerate(tf_idx)]
        )
        return x, args


class ReplaceWithNoiseTF(Transform):
    def __init__(self, noise: Noise | dict):
        super().__init__()
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        return self.params["noise"](x), args


class ScaleTF(Transform):
    def __init__(self, scale: Number | Iterable[Number]):
        super().__init__()
        assert isinstance(scale, (Number, Iterable)), "scale must be a Number or Iterable"
        self.params["scale"] = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x * self.params["scale"], args


class TensorNormTF(Transform):
    def __init__(self):
        super().__init__()
        self.eps2 = 1e-4
        self._is_init = False

    def forward(self, x: Tensor, *args) -> tuple[Tensor, Any]:
        assert isinstance(x, Tensor), f"Expected input to be a Tensor, is {type(x)}"
        self._lazy_init(x)
        return (x - self.params["mean"]) / self.params["std"], args

    def update(self, x: Tensor):
        assert isinstance(x, Tensor), f"Expected input to be a Tensor, is {type(x)}"
        self._lazy_init(x)
        # A batched variant of Welford's algorithm
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        self.params["count"] += x.shape[0]
        delta = x - self.params["mean"]
        self.params["mean"] += torch.sum(delta / self.params["count"], axis=0)
        self.params["m2"] += torch.sum(delta * (x - self.params["mean"]), axis=0)
        std2 = torch.maximum(self.eps2, self.params["m2"] / self.params["count"])  # Num. stability
        self.params["std"] = torch.sqrt(std2)

    def _lazy_init(self, x: Tensor):
        assert isinstance(x, Tensor), f"Expected input to be a Tensor, is {type(x)}"
        if not self._is_init:
            shape = x.shape[1:]
            self.params["mean"] = nn.Parameter(
                torch.zeros(shape, dtype=x.dtype, device=x.device), requires_grad=False
            )
            self.params["std"] = nn.Parameter(
                torch.ones(shape, dtype=x.dtype, device=x.device), requires_grad=False
            )
            self.params["m2"] = nn.Parameter(
                torch.zeros(shape, dtype=x.dtype, device=x.device), requires_grad=False
            )
            self.params["count"] = nn.Parameter(
                torch.zeros(1, dtype=torch.int64, device=x.device), requires_grad=False
            )
            self._is_init = True


class TensorDictNormTF(Transform):
    def __init__(self):
        super().__init__()
        self._is_init = False

    @torch.no_grad()
    def forward(self, x: TensorDict, *args) -> tuple[Tensor, Any]:
        assert isinstance(x, TensorDict), "Input must be a Tensor"
        self._lazy_init(x)
        norm_td = TensorDict(
            {
                k: (v - self.params[f"{k}_mean"]) / self.params[f"{k}_std"]
                for k, v in x.flatten_keys().items()
            },
            batch_size=x.batch_size,
            device=x.device,
        ).unflatten_keys()
        return norm_td, args

    @torch.no_grad()
    def update(self, x: TensorDict):
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        assert len(x.batch_size) == 1, f"Batch size must be a scalar, is {x.batch_size}"
        self._lazy_init(x)
        self.params["count"] += x.batch_size[0]
        # A batched variant of Welford's algorithm
        # See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        for key, value in x.flatten_keys().items():
            delta = value - self.params[f"{key}_mean"]
            self.params[f"{key}_mean"] += torch.sum(delta / self.params["count"], axis=0)
            self.params[f"{key}_m2"] += torch.sum(
                delta * (value - self.params[f"{key}_mean"]), axis=0
            )
            std2 = torch.maximum(
                self.params["eps2"], self.params[f"{key}_m2"] / self.params["count"]
            )
            self.params[f"{key}_std"].copy_(torch.sqrt(std2))

    def _lazy_init(self, x: TensorDict):
        assert isinstance(x, TensorDict), f"Expected input to be a TensorDict, is {type(x)}"
        if not self._is_init:
            for key, value in x.flatten_keys().items():
                shape = value.shape[1:]
                self.params[f"{key}_mean"] = nn.Parameter(
                    torch.zeros(shape, dtype=value.dtype, device=value.device), requires_grad=False
                )
                self.params[f"{key}_std"] = nn.Parameter(
                    torch.ones(shape, dtype=value.dtype, device=value.device), requires_grad=False
                )
                assert self.params[f"{key}_std"].requires_grad is False, "std must not require grad"
                self.params[f"{key}_m2"] = nn.Parameter(
                    torch.zeros(shape, dtype=value.dtype, device=value.device), requires_grad=False
                )
            self.params["count"] = nn.Parameter(
                torch.zeros(1, dtype=torch.int64, device=value.device), requires_grad=False
            )
            self.params["eps2"] = nn.Parameter(
                torch.tensor(1e-4, dtype=value.dtype, device=value.device), requires_grad=False
            )
            self._is_init = True

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        if not self._is_init:  # Parameters have to be created before loading the state_dict
            param_dict = {}
            for key, value in state_dict.items():
                key = key.split(".")[1]  # Remove "params."
                if key.endswith("_mean"):  # Infer the previous keys and shapes from the state_dict
                    param_dict |= {key.removesuffix("_mean"): value.unsqueeze(0)}
            self._lazy_init(TensorDict(param_dict, batch_size=1))  # Initialize all parameters
        return super().load_state_dict(state_dict, strict, assign)
