from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field

import torch
from typing import Any
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic
from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer


@dataclass
class DDPGConfig:

    env: EnvConfig
    rollout: RolloutConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig


@dataclass
class EnvConfig:

    name: str
    seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=lambda: {"num_envs": 1})


@dataclass
class RolloutConfig:

    max_samples: int
    action_noise: float = 0.1
    replay_buffer_class: type[ReplayBuffer] = SimpleReplayBuffer
    replay_buffer_kwargs: dict[str, Any] = field(default_factory=lambda: {"max_size": 1_000_000})


@dataclass
class TrainConfig:

    actor_class: type[DDPGActor] = DDPGActor
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    actor_freq: int = 2
    actor_target_freq: int = 2
    actor_lr: float = 1e-4
    critic_class: type[DDPGCritic] = DDPGCritic
    critic_kwargs: dict[str, Any] = field(default_factory=dict)
    critic_freq: int = 1
    critic_target_freq: int = 2
    critic_lr: float = 1e-3
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    gradient_steps: int = 1
    batch_size: int = 64
    action_noise: float = 0.01
    gamma: float = 0.99
    tau: float = 1e-3
    device: torch.device = torch.device("cpu")


@dataclass
class EvalConfig:

    freq: int
    steps: int


@dataclass
class CheckpointConfig:

    freq: int | None = None
    path: Path | None = None
