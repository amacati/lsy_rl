from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, TypeVar

import gymnasium
import numpy as np
import torch

from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer, replay_buffer_cls
from lsy_rl.core.transforms import ChainedTF, IdentityTF, Transform, transform_cls
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic

T = TypeVar("T")


def maybe_str_to_cls(
    value: T | str, factory: Callable[[str], T] | None = None, expected_type: T | None = None
) -> T:
    if not isinstance(value, type):
        if isinstance(value, str) and factory is not None:
            return factory(value)
    elif issubclass(value, expected_type):
        return value
    raise TypeError(f"Invalid type {value} (expected type {expected_type} or string)")


def check_kwargs(kwargs: dict[str, Any], cls: type, ignore: list[str] = []):
    assert isinstance(kwargs, dict), "Kwargs must be a dict"
    for x in required_args(cls):
        if x in ignore or x in ("args", "kwargs"):
            continue
        if x not in kwargs:
            raise ValueError(f"Missing required argument '{x}' for {cls}")


def required_args(cls: type) -> list[str]:
    return [p.name for p in inspect.signature(cls).parameters.values() if p.default == p.empty]


def convert_transforms(transforms: list[Transform | dict] | Transform) -> Transform:
    if isinstance(transforms, Transform):
        return transforms
    tfs = []
    for transform in transforms:
        if isinstance(transform, Transform):
            tfs.append(transform)
            continue
        assert isinstance(transform, dict)
        tf_cls = maybe_str_to_cls(transform["type"], factory=transform_cls, expected_type=Transform)
        tfs.append(tf_cls(**(transform.get("kwargs") or {})))
    return ChainedTF(tfs)


@dataclass
class DDPGConfig:
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
class EnvConfig:
    name: str
    seed: int | None = None
    kwargs: dict[str, Any] = field(default_factory=lambda: {"num_envs": 1})


@dataclass
class RolloutConfig:
    max_samples: int
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    replay_buffer_cls: type[ReplayBuffer] = SimpleReplayBuffer
    replay_buffer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"max_size": 1_000_000, "num_envs": 1}
    )
    env: gymnasium.Env | None = None
    success_criteria: Callable[[list[float]], np.ndarray] | None = None

    def __post_init__(self):
        self.replay_buffer_cls = maybe_str_to_cls(
            self.replay_buffer_cls, replay_buffer_cls, ReplayBuffer
        )
        if "reward_fn" in self.replay_buffer_kwargs:
            self.replay_buffer_kwargs["reward_fn"] = self.env.unwrapped.compute_reward
        check_kwargs(
            self.replay_buffer_kwargs, self.replay_buffer_cls, ignore=["num_envs", "device"]
        )
        self.obs_transform = convert_transforms(self.obs_transform)
        self.action_transform = convert_transforms(self.action_transform)


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
    min_samples: int = 1
    actor_cls: type[DDPGActor] = DDPGActor
    actor_kwargs: dict[str, Any] = field(default_factory=dict)
    critic_cls: type[DDPGCritic] = DDPGCritic
    critic_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)
    batch_size: int = 64
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    target_action_transform: Transform = field(default_factory=IdentityTF)
    gamma: float = 0.99
    tau: float = 1e-3
    reward_clip: tuple[float, float] = (-torch.inf, torch.inf)
    grad_clip: float = torch.inf
    device: torch.device = torch.device("cpu")

    def __post_init__(self):
        self.actor_cls = maybe_str_to_cls(self.actor_cls, expected_type=torch.nn.Module)
        check_kwargs(self.actor_kwargs, self.actor_cls, ignore=["obs_space", "action_space"])
        self.critic_cls = maybe_str_to_cls(self.critic_cls, expected_type=torch.nn.Module)
        check_kwargs(self.critic_kwargs, self.critic_cls, ignore=["obs_space", "action_space"])
        self.obs_transform = convert_transforms(self.obs_transform).to(self.device)
        self.action_transform = convert_transforms(self.action_transform).to(self.device)
        self.target_action_transform = convert_transforms(self.target_action_transform).to(
            self.device
        )


@dataclass
class EvalConfig:
    freq: int
    steps: int
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    success_criteria: Callable[[list[float]], np.ndarray] | None = None

    def __post_init__(self):
        self.obs_transform = convert_transforms(self.obs_transform)
        self.action_transform = convert_transforms(self.action_transform)


@dataclass
class CheckpointConfig:
    freq: int | None = None
    path: Path | None = None

    def __post_init__(self):
        if self.freq is False:  # TOML can't represent None, use False instead and convert to None
            self.freq = None
        if self.freq is not None and self.path is None:
            raise ValueError("If 'checkpoint_freq' is not None, 'checkpoint_path' must be set")
        if isinstance(self.path, str):
            self.path = Path(self.path).absolute()
            assert self.path.is_absolute(), "Checkpoint path must be an absolute path"
