"""Wrapper for rescaling actions to within a max and min action."""
from typing import Union

import numpy as np
import torch
from torch import Tensor

from tensordict import TensorDict
import gymnasium
from gymnasium.spaces import Box
from lsy_rl.wrappers.tensordict_wrapper import TensorDictWrapper


class ScaleAction(TensorDictWrapper):

    def __init__(self, env: TensorDictWrapper, scale: float | Tensor):
        super().__init__(env)
        assert isinstance(env, TensorDictWrapper), "Environment must be a TensorDictWrapper"
        assert isinstance(env.action_space, Box), "Environment action space must be a Box"
        if isinstance(scale, float):
            scale = torch.tensor(scale, dtype=torch.float32)
        self.scale = scale.to(self.device)
        np_scale = self.scale.cpu().numpy()
        self.env.action_space = gymnasium.spaces.Box(low=env.action_space.low * np_scale,
                                                     high=env.action_space.high * np_scale,
                                                     shape=env.action_space.shape,
                                                     dtype=env.action_space.dtype)

    def step(self, action: Tensor) -> TensorDict[str, Tensor]:
        return self.env.step(self.transform_action(action))

    def transform_action(self, action: Tensor) -> Tensor:
        return action * self.scale

    def reset(self) -> TensorDict[str, Tensor]:
        return self.env.reset()
