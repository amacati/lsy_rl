from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import TypeVar, Callable
import inspect

import torch
from typing import Any
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic
from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer, replay_buffer_cls

T = TypeVar("T")


def maybe_str_to_cls(value: T | str,
                     factory: Callable[[str], T] | None = None,
                     expected_type: T | None = None) -> T:
    if not isinstance(value, type):
        if isinstance(value, str) and factory is not None:
            return factory(value)
    elif issubclass(value, expected_type):
        return value
    raise TypeError(f"Invalid type {value} (expected type {expected_type} or string)")


def required_args(cls: type) -> list[str]:
    return [p.name for p in inspect.signature(cls).parameters.values() if p.default == p.empty]


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

    def __post_init__(self):
        self.replay_buffer_class = maybe_str_to_cls(self.replay_buffer_class, replay_buffer_cls,
                                                    ReplayBuffer)
        assert isinstance(self.replay_buffer_kwargs, dict), "Invalid replay buffer kwargs type"
        for x in required_args(self.replay_buffer_class):
            if x not in self.replay_buffer_kwargs:
                raise ValueError(f"Missing required argument '{x}' for {self.replay_buffer_class}")


@dataclass
class TrainConfig:

    train_freq: int = 1
    train_steps: int = 1
    actor_freq: int = 1
    critic_freq: int = 1
    actor_target_freq: int = 2
    critic_target_freq: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_class: type[DDPGActor] = DDPGActor
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    critic_class: type[DDPGCritic] = DDPGCritic
    critic_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    action_noise: float = 0.01
    gamma: float = 0.99
    tau: float = 1e-3
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.actor_class = maybe_str_to_cls(self.actor_class, expected_type=torch.nn.Module)
        self.critic_class = maybe_str_to_cls(self.critic_class, expected_type=torch.nn.Module)


@dataclass
class EvalConfig:

    freq: int
    steps: int


@dataclass
class CheckpointConfig:

    freq: int | None = None
    path: Path | None = None

    def __post_init__(self):
        if self.freq is not None and self.path is None:
            raise ValueError("If 'checkpoint_freq' is not None, 'checkpoint_path' must be set")
        if isinstance(self.path, str):
            self.path = Path(self.path)
