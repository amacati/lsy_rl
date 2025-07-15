from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.distributions import Normal

from lsy_rl.core.policy import Policy


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class PPOActor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...], use_logstd_net: bool = False):
        super().__init__()
        assert isinstance(obs_shape, tuple), "obs_shape must be a tuple"
        assert isinstance(action_shape, tuple), "action_shape must be a tuple"
        actor_net_cls = PPOActorNetWithStd if use_logstd_net else PPOActorNet
        self.network = actor_net_cls(obs_shape, action_shape)

    def mean_logstd(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        return self.network(obs)
    
    def mean(self, obs: Tensor) -> Tensor:
        return self.mean_logstd(obs)[0]


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
        self.logstd = nn.Parameter(-2*torch.ones(1, torch.tensor(action_shape).prod()))

    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = obs.float()
        for layer in self.network.values():
            x = layer(x)
        logstd = self.logstd.expand_as(x)
        return x, logstd


class PPOActorNetWithStd(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        self.shared_layers = nn.ModuleDict(
            {
                "in": layer_init(nn.Linear(torch.tensor(obs_shape).prod(), 64)),
                "f_in": nn.Tanh(),
                "hidden1": layer_init(nn.Linear(64, 64)),
                "f_hidden1": nn.Tanh(),
            }
        )
        self.mean_head = nn.ModuleDict(
            {
                "out": layer_init(nn.Linear(64, torch.tensor(action_shape).prod()), std=0.01),
                "f_out": nn.Identity(),
            }
        )
        self.logstd_head = nn.ModuleDict(
            {
                "out": layer_init(nn.Linear(64, torch.tensor(action_shape).prod())),
                "f_out": nn.Tanh(),
            }
        )
    
    @property # To maintain backwards compatability with code using PPOActorNet
    def network(self) -> nn.ModuleDict:
        return self.mean_head
    
    def forward(self, obs: Tensor) -> tuple[Tensor, Tensor]:
        x = obs.float()
        for layer in self.shared_layers.values():
            x = layer(x)
        mean = x
        for layer in self.mean_head.values():
            mean = layer(mean)
        logstd = x
        for layer in self.logstd_head.values():
            logstd = layer(logstd)
        # Same method used in SAC's implementation for more stable training
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)
        return mean, logstd
    

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
        action_mean, action_logstd = self.actor.mean_logstd(obs)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample() if not deterministic else action_mean
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)

    def value(self, obs: Tensor) -> Tensor:
        return self.critic(obs)
    
    def save(self, path: Path):
        save_dict = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(save_dict, path)

    def load(self, path: Path):
        save_dict = torch.load(path, weights_only=True)
        self.actor.load_state_dict(save_dict["actor"])
        self.critic.load_state_dict(save_dict["critic"])