import torch
from torch import FloatTensor
import torch.nn as nn

from lsy_rl.core.policy import Policy
from lsy_rl.utils import polyak_update_
from pathlib import Path


class DDPGActor(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = DDPGActorNetwork(obs_dim, action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGActorNetwork(obs_dim, action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs):
        return self.network(obs)

    def target(self, obs):
        return self.target_network(obs)

    def update_target(self, tau: float):
        """Update the target network with the current weights."""
        polyak_update_(self.target_network, self.network, tau)


class DDPGActorNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = nn.ModuleDict({
            "input": nn.Linear(input_dim, 256),
            "f_input": nn.ReLU(),
            "hidden1": nn.Linear(256, 256),
            "f_hidden1": nn.ReLU(),
            "hidden2": nn.Linear(256, 256),
            "f_hidden2": nn.ReLU(),
            "output": nn.Linear(256, output_dim),
            "f_output": nn.Tanh()
        })

    def forward(self, obs: FloatTensor) -> FloatTensor:
        x = obs
        for layer in self.network.values():
            x: FloatTensor = layer(x)
        return x


class DDPGCritic(nn.Module):

    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.network = DDPGCriticNetwork(obs_dim + action_dim)
        # Initialize the target network and synchronize the weights
        self.target_network = DDPGCriticNetwork(obs_dim + action_dim)
        self.target_network.load_state_dict(self.network.state_dict())

    def forward(self, obs: FloatTensor, action: FloatTensor):
        return self.network(torch.cat([obs, action], dim=-1))

    def target(self, obs: FloatTensor, action: FloatTensor):
        return self.target_network(torch.cat([obs, action], dim=-1))

    def update_target(self, tau: float):
        polyak_update_(self.target_network, self.network, tau)


class DDPGCriticNetwork(nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.network = nn.ModuleDict({
            "input": nn.Linear(input_dim, 256),
            "f_input": nn.ReLU(),
            "hidden1": nn.Linear(256, 256),
            "f_hidden1": nn.ReLU(),
            "hidden2": nn.Linear(256, 256),
            "f_hidden2": nn.ReLU(),
            "output": nn.Linear(256, 1)
        })

    def forward(self, obs_action: FloatTensor) -> FloatTensor:
        x = obs_action
        for layer in self.network.values():
            x = layer(x)
        return x


class DDPGPolicy(Policy):

    def __init__(self, actor: DDPGActor, critic: DDPGCritic, device: str = "cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

    def action(self, obs: FloatTensor) -> FloatTensor:
        return self.actor(obs)

    def save(self, path: Path):
        save_dict = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(save_dict, path)

    def load(self, path: Path):
        save_dict = torch.load(path)
        self.actor.load_state_dict(save_dict["actor"])
        self.critic.load_state_dict(save_dict["critic"])
