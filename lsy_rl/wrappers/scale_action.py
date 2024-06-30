"""Wrapper for rescaling actions to within a max and min action."""

from typing import Any

import gymnasium
import torch
from gymnasium.spaces import Box
from tensordict import TensorDict
from torch import Tensor

from lsy_rl.wrappers.tensordict_wrapper import TensorDictWrapper


class ScaleAction(TensorDictWrapper):
    def __init__(self, env: TensorDictWrapper, scale: float | Tensor):
        super().__init__(env)
        assert isinstance(env, TensorDictWrapper), "Environment must be a TensorDictWrapper"
        assert isinstance(self.action_space, Box), "Environment action space must be a Box"
        if isinstance(scale, float):
            scale = torch.tensor(scale, dtype=torch.float32)
        self.scale = scale.to(self.device)
        np_scale = self.scale.cpu().numpy()
        self.action_space.low *= np_scale
        self.action_space.high *= np_scale

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        return self.env.step(self.transform_action(action))

    def transform_action(self, action: Tensor) -> Tensor:
        return action * self.scale

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> TensorDict[str, Tensor]:
        return self.env.reset(seed=seed, options=options)
