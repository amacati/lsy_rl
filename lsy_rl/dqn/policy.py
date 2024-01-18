from abc import abstractmethod, ABC

import torch
from torch import FloatTensor, IntTensor
import torch.nn as nn

from lsy_rl.core.policy import Policy
from pathlib import Path


class DQNet(ABC, nn.Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_network_and_target(self) -> tuple[nn.Module, nn.Module]:
        ...


class DoubleDQNet(DQNet):

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.networks = nn.ModuleDict({
            "dqn1":
                nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                              nn.Linear(128, action_dim)),
            "dqn2":
                nn.Sequential(nn.Linear(obs_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                              nn.Linear(128, action_dim))
        })

    def forward(self, obs: FloatTensor) -> FloatTensor:
        return (self.networks["dqn1"](obs) + self.networks["dqn2"](obs)) / 2

    def get_network_and_target(self) -> tuple[nn.Module, nn.Module]:
        if torch.rand(1) > 0.5:  # Randomly choose the network to update
            return self.networks["dqn1"], self.networks["dqn2"]
        return self.networks["dqn2"], self.networks["dqn1"]


class DQNPolicy(Policy):

    def __init__(self, network: DQNet, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.dqn = network.to(self.device)

    def action(self, obs: FloatTensor) -> IntTensor:
        return torch.argmax(self.dqn(obs), dim=-1)

    def save(self, path: Path):
        torch.save(self.dqn.state_dict(), path)

    def load(self, path: Path):
        self.dqn.load_state_dict(torch.load(path))
