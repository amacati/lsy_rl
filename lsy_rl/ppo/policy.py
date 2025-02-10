from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from lsy_rl.core.policy import Policy


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        assert isinstance(obs_shape, tuple), "obs_shape must be a tuple"
        assert isinstance(action_shape, tuple), "action_shape must be a tuple"
        self.network = PPOActorNet(obs_shape, action_shape)
        self.logstd = nn.Parameter(torch.zeros(1, torch.tensor(action_shape).prod()))

    def mean(self, obs: Tensor) -> Tensor:
        return self.network(obs)


class PPOActorNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        self.network = nn.ModuleDict(
            {
                "in": layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
                "f_in": nn.Tanh(),
                "hidden1": layer_init(nn.Linear(64, 64)),
                "f_hidden1": nn.Tanh(),
                "out": layer_init(nn.Linear(64, torch.tensor(action_shape).prod()), std=0.01),
                "f_out": nn.Identity(),
            }
        )

    def forward(self, obs: Tensor) -> Tensor:
        x = obs.float()
        for layer in self.network.values():
            x = layer(x)
        return x


class PPOCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...]):
        super().__init__()
        self.network = PPOCriticNet(obs_shape)

    def forward(self, obs: Tensor) -> Tensor:
        return self.network(obs)


class PPOCriticNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...]):
        super().__init__()
        self.network = nn.ModuleDict(
            {
                "input": layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
                "f_input": nn.Tanh(),
                "hidden1": layer_init(nn.Linear(64, 64)),
                "f_hidden1": nn.Tanh(),
                "output": layer_init(nn.Linear(64, 1), std=1.0),
            }
        )

    def forward(self, obs: Tensor) -> Tensor:
        x = obs.float()
        for layer in self.network.values():
            x = layer(x)
        return x


class PPOPolicy(Policy, nn.Module):
    def __init__(self, actor: PPOActor, critic: PPOCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic

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
