from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import TypeVar, Callable
import inspect

import torch
from typing import Any
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic
from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer, replay_buffer_cls
from lsy_rl.core.noise import Noise, NormalNoise, noise_cls

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


def check_kwargs(kwargs: dict[str, Any], cls: type, ignore: list[str] = []):
    assert isinstance(kwargs, dict), "Kwargs must be a dict"
    for x in required_args(cls):
        if x in ignore:
            continue
        if x not in kwargs:
            raise ValueError(f"Missing required argument '{x}' for {cls}")


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
    noise_cls: type[Noise] | str = NormalNoise
    noise_kwargs: dict[str, Any] = field(
        default_factory=lambda: {torch.tensor([0.0]), torch.tensor([0.1])})
    action_clip_low: float = -1.0
    action_clip_high: float = 1.0
    replay_buffer_cls: type[ReplayBuffer] = SimpleReplayBuffer
    replay_buffer_kwargs: dict[str, Any] = field(default_factory=lambda: {
        "max_size": 1_000_000,
        "num_envs": 1
    })

    def __post_init__(self):
        self.replay_buffer_cls = maybe_str_to_cls(self.replay_buffer_cls, replay_buffer_cls,
                                                  ReplayBuffer)
        check_kwargs(self.replay_buffer_kwargs, self.replay_buffer_cls)
        self.noise_cls = maybe_str_to_cls(self.noise_cls, factory=noise_cls, expected_type=Noise)
        check_kwargs(self.noise_kwargs, self.noise_cls)


@dataclass
class TrainConfig:

    freq: int = 1
    steps: int = 1
    actor_freq: int = 1
    critic_freq: int = 1
    actor_target_freq: int = 2
    critic_target_freq: int = 2
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    actor_cls: type[DDPGActor] = DDPGActor
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    critic_cls: type[DDPGCritic] = DDPGCritic
    critic_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    action_noise_cls: type[Noise] | str = NormalNoise
    action_noise_kwargs: dict[str, Any] = field(
        default_factory=lambda: {torch.tensor([0.0]), torch.tensor([0.01])})
    action_clip_low: float = -1.0
    action_clip_high: float = 1.0
    gamma: float = 0.99
    tau: float = 1e-3
    reward_clip_low: float = -torch.inf
    reward_clip_high: float = torch.inf
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.actor_cls = maybe_str_to_cls(self.actor_cls, expected_type=torch.nn.Module)
        check_kwargs(self.actor_kwargs, self.actor_cls, ignore=["obs_space", "action_space"])
        self.critic_cls = maybe_str_to_cls(self.critic_cls, expected_type=torch.nn.Module)
        check_kwargs(self.critic_kwargs, self.critic_cls, ignore=["obs_space", "action_space"])
        self.action_noise_cls = maybe_str_to_cls(self.action_noise_cls,
                                                 factory=noise_cls,
                                                 expected_type=Noise)
        check_kwargs(self.action_noise_kwargs, self.action_noise_cls)


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
