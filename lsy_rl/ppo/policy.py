from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from lsy_rl.core.policy import Policy

if TYPE_CHECKING:
    from gymnasium.spaces import Box


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(nn.Module):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__()
        self.mean = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, torch.tensor(action_space.shape).prod()), std=0.01),
        )
        self.logstd = nn.Parameter(torch.zeros(1, torch.tensor(action_space.shape).prod()))


class PPOCritic(nn.Module):
    def __init__(self, obs_space: Box):
        super().__init__()
        self.network = nn.Sequential(
            layer_init(nn.Linear(torch.tensor(obs_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def forward(self, obs: Tensor) -> Tensor:
        return self.network(obs)


class PPOPolicy(Policy, nn.Module):
    def __init__(self, actor: PPOActor, critic: PPOCritic, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.actor = torch.compile(actor.to(self.device))
        self.critic = torch.compile(critic.to(self.device))

    def action(self, obs: Tensor) -> Tensor:
        return self.actor.mean(obs)

    def action_and_value(
        self, obs: Tensor, action: Tensor | None = None, deterministic: bool = False
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mean = self.actor.mean(obs)
        action_logstd = self.actor.logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(obs)

    def to(self, device: str) -> PPOPolicy:
        self.device = torch.device(device)
        self.actor.to(self.device)
        self.critic.to(self.device)
        return self
