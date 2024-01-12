"""Wrapper for rescaling actions to within a max and min action."""
from typing import Union

import numpy as np
import torch
from torch import Tensor

import gymnasium as gym
from gymnasium.spaces import Box


class TransformAction(gym.ActionWrapper):

    def __init__(self, env: gym.Env, action_transform: callable):
        super().__init__(env)
        self.action_transform = action_transform

    def action(self, action: Tensor) -> Tensor:
        return self.action_transform(action)
