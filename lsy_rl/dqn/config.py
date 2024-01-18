from __future__ import annotations

from dataclasses import dataclass, field

import torch
from typing import Any
from lsy_rl.dqn.policy import DQNet, DoubleDQNet
from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer


@dataclass
class DQNConfig:

    env: EnvConfig
    rollout: RolloutConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig


@dataclass
class EnvConfig:

    name: str
    seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=dict())


@dataclass
class RolloutConfig:

    max_samples: int
    replay_buffer_class: type[ReplayBuffer] = SimpleReplayBuffer
    replay_buffer_kwargs: dict[str, Any] = field(default_factory={"maxsize": 1_000_000})
    epsilon: float = 0.05


@dataclass
class TrainConfig:

    freq: int
    lr: float = 1e-4
    net_class: type[DQNet] = DoubleDQNet
    net_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    gradient_steps: int = 1
    batch_size: int = 64
    grad_clip: float = float("inf")
    q_clip: float = float("inf")
    gamma: float = 0.99
    device: torch.device = torch.device("cpu")


@dataclass
class EvalConfig:

    freq: int
    steps: int


@dataclass
class CheckpointConfig:

    freq: int | None = None
    path: str | None = None
