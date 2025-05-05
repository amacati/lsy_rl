from pathlib import Path

import torch
import torch.nn as nn
from gymnasium.spaces import Box
from torch import FloatTensor

from lsy_rl.core.policy import Policy
from lsy_rl.ddpg.policy import DDPGActor, DDPGCriticNetwork
from lsy_rl.utils import polyak_update_


class TD3Actor(DDPGActor):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__(obs_space, action_space)


class TD3Critic(nn.Module):
    def __init__(self, obs_space: Box, action_space: Box):
        super().__init__()
        assert isinstance(obs_space, Box), f"Invalid obs space type {type(obs_space)}"
        assert isinstance(action_space, Box), f"Invalid action space type {type(action_space)}"
        obs_ndim, action_ndim = len(obs_space.shape), len(action_space.shape)
        assert obs_ndim == 1, f"Invalid obs space dimension {obs_ndim}"
        assert action_ndim == 1, f"Invalid action space dimension {action_ndim}"
        obs_dim, action_dim = obs_space.shape[0], action_space.shape[0]
        self.q1 = DDPGCriticNetwork(obs_dim + action_dim)
        self.q2 = DDPGCriticNetwork(obs_dim + action_dim)
        # Initialize the target network and synchronize the weights
        self.q1_target = DDPGCriticNetwork(obs_dim + action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target = DDPGCriticNetwork(obs_dim + action_dim)
        self.q2_target.load_state_dict(self.q2.state_dict())

    def values(self, obs: FloatTensor, action: FloatTensor) -> tuple[FloatTensor, FloatTensor]:
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def actor_value(self, obs: FloatTensor, action: FloatTensor) -> FloatTensor:
        assert obs.dtype == torch.float32, f"Invalid dtype {obs.dtype}"
        assert action.dtype == torch.float32, f"Invalid dtype {action.dtype}"
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)

    def target(self, obs: FloatTensor, action: FloatTensor) -> FloatTensor:
        assert obs.dtype == torch.float32, f"Invalid dtype {obs.dtype}"
        assert action.dtype == torch.float32, f"Invalid dtype {action.dtype}"
        x = torch.cat([obs, action], dim=-1)
        return torch.minimum(self.q1_target(x), self.q2_target(x))

    def update_target(self, tau: float):
        polyak_update_(self.q1_target, self.q1, tau)
        polyak_update_(self.q2_target, self.q2, tau)


class TD3Policy(Policy):
    def __init__(self, actor: TD3Actor, critic: TD3Critic):
        super().__init__()
        # Compile disabled for now. Does not yield any performance improvements
        self.actor = actor
        # self.actor = torch.compile(self.actor)
        self.critic = critic
        # self.critic = torch.compile(self.critic)

    def action(self, obs: FloatTensor) -> FloatTensor:
        return self.actor(obs)

    def save(self, path: Path):
        save_dict = {"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}
        torch.save(save_dict, path)

    def load(self, path: Path):
        save_dict = torch.load(path, weights_only=True)
        self.actor.load_state_dict(save_dict["actor"])
        self.critic.load_state_dict(save_dict["critic"])
