from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
from ml_collections import ConfigDict
from tensordict import TensorDict
from torch import Tensor

from lsy_rl.core.noise import Noise, noise_cls
from lsy_rl.utils.utils import module_type_from_string, to_cls

if TYPE_CHECKING:
    from torch.nn.modules.module import _IncompatibleKeys

transform: Callable[[str], type[Transform]] = module_type_from_string(__name__)
transform_cls = transform  # Alias for modules that use transform as a variable name


def to_transforms(transforms: list[Transform | dict] | Transform) -> ChainedTF:
    """Convert a transform or a list of transforms to a Transform object.

    Args:
        transforms: The transform or list of transforms to convert. Transforms can also be given as
            dictionaries with the keys "type" and "kwargs".

    Returns:
        The fused Transform.
    """
    if isinstance(transforms, ChainedTF):
        return transforms
    if isinstance(transforms, Transform):
        return ChainedTF([transforms])
    assert isinstance(transforms, Iterable), "transforms must be an iterable if not a Transform"
    transform = ChainedTF()
    for tf in transforms:
        if isinstance(tf, Transform):
            transform.append(tf)
            continue
        assert isinstance(tf, Mapping | ConfigDict), f"Unsupported transform type {type(tf)}"
        tf_cls = to_cls(tf["type"], factory=transform_cls, expected_type=Transform)
        transform.append(tf_cls(**(tf.get("kwargs") or {})))
    return transform


def share_transforms(transforms: list[ChainedTF], exclude: list[Transform] | None = None):
    """Share the parameters of a list of chained transforms.

    Args:
        transforms: The list of chained transforms to share parameters between.
    """
    exclude = [] if exclude is None else exclude
    assert all(isinstance(x, ChainedTF) for x in transforms), "All elements must be ChainedTFs"
    for i, ctf in enumerate(transforms):
        assert isinstance(ctf, ChainedTF), "All elements must be ChainedTFs"
        for tf in ctf:
            if not tf.shared:
                continue
            if tf in exclude:
                continue
            for j in range(len(transforms)):
                if i != j:
                    transforms[j].append(tf)
            exclude.append(tf)


class Transform(nn.Module):
    """Base class for all transforms."""

    def __init__(self, shared: bool = False):
        """Initialize a parameter dictionary which stores all parameters."""
        super().__init__()
        self.params = nn.ParameterDict()
        self.shared = shared

    def reset(self):
        """Reset the state of the transform."""
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Apply the transform to the input Tensor."""
        ...

    def update(self, x: Tensor):
        """Update the transform with a batch of data."""
        ...


class ChainedTF(Transform):
    """Chain multiple transforms together into a sequence."""

    def __init__(self, transforms: list[Transform] = [], shared: bool = False):
        """Initialize the transforms.

        Args:
            transforms: The list of transforms to chain together.
        """
        super().__init__(shared=shared)
        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)

    def __len__(self) -> int:
        """Return the number of transforms in the chain."""
        return len(self.params["transforms"])

    def __getitem__(self, idx: int) -> Transform:
        """Return the transform at the given index."""
        return self.params["transforms"][idx]

    def reset(self):
        """Reset the state of all transforms in the chain."""
        for transform in self.params["transforms"]:
            transform.reset()

    def forward(self, x: Tensor) -> Tensor:
        """Sequentially apply all transforms in the chain."""
        for transform in self.params["transforms"]:
            x = transform(x)
        return x

    def update(self, x: Tensor | TensorDict):
        """Update all transforms in the chain with a batch of data."""
        for transform in self.params["transforms"]:
            transform.update(x)

    def append(self, transform: Transform):
        """Append a transform to the chain."""
        assert isinstance(transform, Transform), "transform must be a Transform"
        self.params["transforms"].append(transform)

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        """Copy parameters and buffers from state_dict into this module and its descendants.

        Args:
            state_dict: A dict containing parameters and persistent buffers.
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned
                by this module's state_dict() function.
            assign: Whether to assign the parameters directly or to copy them.
        """
        for i, transform in enumerate(self.params["transforms"]):
            sd = {
                k.removeprefix(f"params.transforms.{i}."): v
                for k, v in state_dict.items()
                if k.startswith(f"params.transforms.{i}.")
            }
            transform.load_state_dict(sd, strict, assign)


class IdentityTF(Transform):
    """Identity transform that does nothing.

    Useful for, e.g., choice transforms where one option is to do nothing.
    """

    def __init__(self, shared: bool = False):
        """Initialize the identity transform."""
        super().__init__(shared=shared)

    def forward(self, x: Tensor) -> Tensor:
        """Return the input Tensor unchanged."""
        return x


class ClipTF(Transform):
    """Clip the input Tensor to a given range."""

    def __init__(
        self, min: Number | list[Number], max: Number | list[Number], shared: bool = False
    ):
        """Initialize the clipping parameters.

        Args:
            min: The minimum value or list of minimum values for each dimension.
            max: The maximum value or list of maximum values for each dimension.
        """
        super().__init__(shared=shared)
        assert isinstance(min, (Number, list)) and isinstance(max, (Number, list)), (
            "min and max must be floats or lists of floats"
        )
        min, max = torch.tensor(min, dtype=torch.float32), torch.tensor(max, dtype=torch.float32)
        self.params["min"] = nn.Parameter(min, requires_grad=False)
        self.params["max"] = nn.Parameter(max, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Clip the input Tensor to the given range."""
        assert isinstance(x, Tensor), f"Input must be a Tensor, is {type(x)} {x}"
        return torch.clamp(x, self.params["min"], self.params["max"])


class AdditiveNoiseTF(Transform):
    """Add noise to the input Tensor."""

    def __init__(self, noise: Noise | dict | ConfigDict, shared: bool = False):
        """Initialize the noise.

        Args:
            noise: The noise object or dict with the keys "type" and "kwargs". If a dict, the noise
                is created with noise_cls(noise["type"])(**noise["kwargs"]).
        """
        super().__init__(shared=shared)
        if isinstance(noise, dict | ConfigDict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def forward(self, x: Tensor) -> Tensor:
        """Add noise to the input Tensor."""
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x + self.params["noise"](x)


class ChoiceTF(Transform):
    """Apply one transform from a list of transforms with a given probability."""

    def __init__(self, transforms: list[Transform | dict], prob: list[float], shared: bool = False):
        """Initialize the transforms and probabilities.

        Args:
            transforms: The list of transforms or dicts with the keys "type" and "kwargs". If dicts,
                transforms are created with transform_cls(transform["type"])(**transform["kwargs"]).
            prob: The probabilities for each transform. Must sum to 1.
        """
        super().__init__(shared=shared)
        # Convert potential dicts to Transform objects
        for i, tf in enumerate(transforms):
            if isinstance(tf, dict):
                transforms[i] = transform(tf["type"])(**(tf.get("kwargs") or {}))

        assert all(isinstance(x, Transform) for x in transforms), "All elements must be Transforms"
        self.params["transforms"] = nn.ModuleList(transforms)
        prob = torch.tensor(prob, dtype=torch.float32)
        assert torch.all(prob >= 0), "p must be non-negative"
        assert torch.isclose(prob.sum(), torch.tensor(1.0)), "p must sum to 1"
        assert len(prob) == len(transforms), "prob must have the same length as transforms"
        prob = prob / prob.sum()
        self.params["prob"] = nn.Parameter(prob, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Apply one of the transforms with the given probability."""
        tf_idx = torch.multinomial(self.params["prob"], x.shape[0], replacement=True)
        x = torch.stack([self.params["transforms"][j](x[i]) for i, j in enumerate(tf_idx)])
        return x


class ReplaceWithNoiseTF(Transform):
    """Replace the input Tensor with noise of the same."""

    def __init__(self, noise: Noise | dict, shared: bool = False):
        """Initialize the noise.

        Args:
            noise: The noise object or dict with the keys "type" and "kwargs". If a dict, the noise
                is created with noise_cls(noise["type"])(**noise["kwargs"]).
        """
        super().__init__(shared=shared)
        if isinstance(noise, dict):
            noise = noise_cls(noise["type"])(**(noise.get("kwargs") or {}))
        assert isinstance(noise, Noise), "noise must be a Noise object"
        self.params["noise"] = noise

    def forward(self, x: Tensor) -> Tensor:
        """Replace the input Tensor with noise sampled from the noise module."""
        return self.params["noise"](x)


class ScaleTF(Transform):
    """Scale the input Tensor by a constant factor."""

    def __init__(self, scale: Number | Iterable[Number], shared: bool = False):
        """Initialize the scaling parameters.

        Args:
            scale: The scaling factor. If a single number, all dimensions are scaled by the same
                factor. If an Iterable, the dimensions are scaled element-wise.
        """
        super().__init__(shared=shared)
        assert isinstance(scale, (Number, Iterable)), "scale must be a Number or Iterable"
        self.params["scale"] = nn.Parameter(torch.tensor(scale), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """Scale the input Tensor."""
        assert isinstance(x, Tensor), "Input must be a Tensor"
        return x * self.params["scale"]


class UnitNormTF(Transform):
    """Scale the input Tensor to unit norm."""

    def __init__(self, shared: bool = False):
        """Initialize the scaling parameters."""
        super().__init__(shared=shared)

    def forward(self, x: Tensor) -> Tensor:
        """Scale the input Tensor."""
        assert isinstance(x, Tensor), "Input must be a Tensor"
        x = x / torch.norm(x, dim=-1, keepdim=True)
        return x


class FunctionalTF(Transform):
    """Apply a functional transformation to the input Tensor."""

    def __init__(self, fn: Callable[[Tensor], Tensor], shared: bool = False):
        """Initialize the functional transformation."""
        super().__init__(shared=shared)
        self.params["fn"] = fn

    def forward(self, x: Tensor) -> Tensor:
        """Apply the functional transformation to the input Tensor."""
        return self.params["fn"](x)


class TensorNormTF(Transform):
    """Normalize Tensors with running statistics of the mean and standard deviation."""

    def __init__(self, shared: bool = False):
        """Parameters are created lazily during the first forward pass or update."""
        super().__init__(shared=shared)
        self.eps2 = 1e-4
        self._is_init = False

    def forward(self, x: Tensor) -> Tensor:
        """Normalize the input Tensor with the computed statistics."""
        assert isinstance(x, Tensor), f"Expected input to be a Tensor, is {type(x)}"
        self._lazy_init(x)
        return (x - self.params["mean"]) / self.params["std"]

    def update(self, x: Tensor):
        """Update the normalization statistics with a batch of data."""
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
    """Normalize TensorDicts with running statistics of the mean and standard deviation."""

    def __init__(self, shared: bool = False):
        """The parameters are created lazily during the first forward pass or update."""
        super().__init__(shared=shared)
        self._is_init = False

    @torch.no_grad()
    def forward(self, x: TensorDict) -> Tensor:
        """Normalize the input TensorDict with the computed statistics."""
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
        return norm_td

    @torch.no_grad()
    def update(self, x: TensorDict):
        """Update the normalization statistics with a batch of data."""
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
    ) -> _IncompatibleKeys:
        """Copy parameters and buffers from state_dict into this module and its descendants.

        The parameters might not be initialized yet since they are created lazily. Therefore, we
        infer the shapes from the state dict and initialize the parameters if necessary before
        loading the state dict.

        Args:
            state_dict: A dict containing parameters and persistent buffers.
            strict: Whether to strictly enforce that the keys in state_dict match the keys returned
                by this module's state_dict() function.
            assign: Whether to assign the parameters directly or to copy them.
        """
        if not self._is_init:  # Parameters have to be created before loading the state_dict
            param_dict = {}
            for key, value in state_dict.items():
                key = key.split(".")[1]  # Remove "params."
                if key.endswith("_mean"):  # Infer the previous keys and shapes from the state_dict
                    param_dict |= {key.removesuffix("_mean"): value.unsqueeze(0)}
            assert param_dict, "No parameters found in state_dict"
            self._lazy_init(TensorDict(param_dict, batch_size=1))  # Initialize all parameters
        return super().load_state_dict(state_dict, strict, assign)
