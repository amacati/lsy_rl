from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from lsy_rl.core.policy import Policy
from lsy_rl.utils import polyak_update_

class SACActor(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        assert isinstance(obs_shape, tuple), "obs_shape must be a tuple"
        assert isinstance(action_shape, tuple), "action_shape must be a tuple"
        self.network = SACActorNet(obs_shape, action_shape)

    def action(self, obs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, logstd = self.network(obs)
        normal = torch.distributions.Normal(mean, logstd.exp())
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(1.0 * (1 - action.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)
        return action, log_prob, mean

    def mean_action(self, obs: Tensor) -> Tensor:
        x = obs
        for layer in self.network.shared_layers.values():
            x = layer(x)
        for layer in self.network.mean_head.values():
            x = layer(x)
        return x


class SACActorNet(nn.Module):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -5

    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        self.shared_layers = nn.ModuleDict(
            {
                "in": nn.Linear(torch.tensor(obs_shape).prod(), 256),
                "f_in": nn.ReLU(),
                "hidden1": nn.Linear(256, 256),
                "f_hidden1": nn.ReLU(),
                "hidden2": nn.Linear(256, 256),
                "f_hidden2": nn.ReLU(),
            }
        )
        self.mean_head = nn.ModuleDict({"mean": nn.Linear(256, torch.tensor(action_shape).prod())})
        self.logstd_layer = nn.ModuleDict(
            {"logstd": nn.Linear(256, torch.tensor(action_shape).prod()), "f_logstd": nn.Tanh()}
        )
        # Actions MUST be in the range [-1, 1]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.shared_layers.values():
            x = layer(x)
        mean = x
        for layer in self.mean_head.values():
            mean = layer(mean)
        logstd = x
        for layer in self.logstd_layer.values():
            logstd = layer(logstd)
        # From SpinUp / Denis Yarats
        logstd = self.LOG_STD_MIN + 0.5 * (self.LOG_STD_MAX - self.LOG_STD_MIN) * (logstd + 1)
        return mean, logstd


class SACCritic(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        self.q1 = SACCriticNet(obs_shape, action_shape)
        self.q2 = SACCriticNet(obs_shape, action_shape)
        self.q1_target = SACCriticNet(obs_shape, action_shape)
        for param in self.q1_target.parameters():  # Freeze target network parameters
            param.requires_grad = False
        self.q2_target = SACCriticNet(obs_shape, action_shape)
        for param in self.q2_target.parameters():  # Freeze target network parameters
            param.requires_grad = False
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
    
    def values(self, obs: Tensor, action: Tensor) -> tuple[Tensor, Tensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def actor_value(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.minimum(self.q1(x), self.q2(x))
    
    def target(self, obs: Tensor, action: Tensor) -> Tensor:
        x = torch.cat([obs, action], dim=-1)
        return torch.minimum(self.q1_target(x), self.q2_target(x))
    
    def update_target(self, tau: float):
        polyak_update_(self.q1_target, self.q1, tau)
        polyak_update_(self.q2_target, self.q2, tau)

class SACCriticNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_shape: tuple[int, ...]):
        super().__init__()
        obs_dim, action_dim = torch.tensor(obs_shape).prod(), torch.tensor(action_shape).prod()
        self.network = nn.ModuleDict(
            {
                "input": nn.Linear(obs_dim + action_dim, 256),
                "f_input": nn.ReLU(),
                "hidden1": nn.Linear(256, 256),
                "f_hidden1": nn.ReLU(),
                "hidden2": nn.Linear(256, 256),
                "f_hidden2": nn.ReLU(),
                "output": nn.Linear(256, 1),
            }
        )

    def forward(self, obs_action: Tensor) -> Tensor:
        x = obs_action
        for layer in self.network.values():
            x = layer(x)
        return x


class SACPolicy(Policy, nn.Module):
    def __init__(self, actor: SACActor, critic: SACCritic):
        super().__init__()
        self.actor = actor
        self.critic = critic

    def action(self, obs: Tensor) -> Tensor:
        return self.actor.mean_action(obs)
    
    def save(self, path: Path):
        save_dict = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(save_dict, path)

    def load(self, path: Path):
        save_dict = torch.load(path, weights_only=True)
        self.actor.load_state_dict(save_dict["actor"])
        self.critic.load_state_dict(save_dict["critic"])