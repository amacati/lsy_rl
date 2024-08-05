from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from lsy_rl.core.noise import ClippedNormalNoise
from lsy_rl.core.transforms import AdditiveNoiseTF, IdentityTF, Transform, to_transforms
from lsy_rl.ddpg.config import CheckpointConfig as DDPGCheckpointConfig
from lsy_rl.ddpg.config import EnvConfig as DDPGEnvConfig
from lsy_rl.ddpg.config import EvalConfig as DDPGEvalConfig
from lsy_rl.ddpg.config import RolloutConfig as DDPGRolloutConfig
from lsy_rl.td3.policy import TD3Actor, TD3Critic
from lsy_rl.utils.utils import check_kwargs, to_cls


@dataclass
class TD3Config:
    env: EnvConfig
    rollout: RolloutConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig

    def __post_init__(self):
        dev = self.train.device
        self.rollout.action_transform = self.rollout.action_transform.to(dev)
        self.rollout.obs_transform = self.rollout.obs_transform.to(dev)
        self.eval.action_transform = self.eval.action_transform.to(dev)
        self.eval.obs_transform = self.eval.obs_transform.to(dev)


@dataclass
class EnvConfig(DDPGEnvConfig):
    ...


@dataclass
class RolloutConfig(DDPGRolloutConfig):
    ...


@dataclass
class TrainConfig:
    period: int = 1
    steps: int = 1
    actor_period: int = 2
    critic_period: int = 1
    actor_target_period: int = 4
    critic_target_period: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    min_samples: int = 1
    actor_cls: type[TD3Actor] = TD3Actor
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    critic_cls: type[TD3Critic] = TD3Critic
    critic_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    target_action_transform: Transform = field(
        default_factory=lambda: AdditiveNoiseTF(ClippedNormalNoise(0, 0.2, -0.5, 0.5))
    )
    gamma: float = 0.99
    tau: float = 1e-3
    reward_clip: tuple[float, float] = (-torch.inf, torch.inf)
    grad_clip: float = torch.inf
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        assert self.critic_period < self.actor_period, "TD3 specifies critic_period < actor_period"
        self.actor_cls = to_cls(self.actor_cls, expected_type=torch.nn.Module)
        check_kwargs(self.actor_kwargs, self.actor_cls, ignore=["obs_space", "action_space"])
        self.critic_cls = to_cls(self.critic_cls, expected_type=torch.nn.Module)
        check_kwargs(self.critic_kwargs, self.critic_cls, ignore=["obs_space", "action_space"])
        self.obs_transform = to_transforms(self.obs_transform).to(self.device)
        self.action_transform = to_transforms(self.action_transform).to(self.device)
        self.target_action_transform = to_transforms(self.target_action_transform).to(self.device)


@dataclass
class EvalConfig(DDPGEvalConfig):
    ...


@dataclass
class CheckpointConfig(DDPGCheckpointConfig):
    ...
