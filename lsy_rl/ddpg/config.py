from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import gymnasium
import gymnasium.vector.async_vector_env
import torch

from lsy_rl.core.replay_buffer import ReplayBuffer, SimpleReplayBuffer, replay_buffer_cls
from lsy_rl.core.transforms import IdentityTF, Transform, share_transforms, to_transforms
from lsy_rl.ddpg.policy import DDPGActor, DDPGCritic
from lsy_rl.utils.utils import check_kwargs, to_cls

if TYPE_CHECKING:
    import numpy as np


@dataclass
class DDPGConfig:
    env: EnvConfig
    rollout: RolloutConfig
    train: TrainConfig
    eval: EvalConfig
    checkpoint: CheckpointConfig

    def __post_init__(self):
        dev = self.train.device
        share_transforms(
            (self.rollout.obs_transform, self.eval.obs_transform, self.train.obs_transform)
        )
        share_transforms(
            (
                self.rollout.action_transform,
                self.eval.action_transform,
                self.train.action_transform,
                self.train.target_action_transform,
            )
        )
        self.rollout.action_transform = self.rollout.action_transform.to(dev)
        self.rollout.obs_transform = self.rollout.obs_transform.to(dev)
        self.eval.action_transform = self.eval.action_transform.to(dev)
        self.eval.obs_transform = self.eval.obs_transform.to(dev)


@dataclass
class EnvConfig:
    name: str
    seed: int | None = None
    n_envs: int = 1
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: gymnasium.Env | None = None


@dataclass
class RolloutConfig:
    max_samples: int
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    replay_buffer_cls: type[ReplayBuffer] = SimpleReplayBuffer
    replay_buffer_kwargs: dict[str, Any] = field(
        default_factory=lambda: {"max_size": 1_000_000, "num_envs": 1}
    )
    success_criteria: Callable[[list[float]], np.ndarray] | None = None

    def __post_init__(self):
        self.replay_buffer_cls = to_cls(self.replay_buffer_cls, replay_buffer_cls, ReplayBuffer)
        self.obs_transform = to_transforms(self.obs_transform)
        self.action_transform = to_transforms(self.action_transform)
        check_kwargs(
            self.replay_buffer_kwargs,
            self.replay_buffer_cls,
            ignore=["num_envs", "device", "reward_fn"],
        )


@dataclass
class TrainConfig:
    period: int = 1
    steps: int = 1
    actor_period: int = 1
    critic_period: int = 1
    actor_target_period: int = 2
    critic_target_period: int = 2
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
        self.actor_cls = to_cls(self.actor_cls, expected_type=torch.nn.Module)
        check_kwargs(self.actor_kwargs, self.actor_cls, ignore=["obs_space", "action_space"])
        self.critic_cls = to_cls(self.critic_cls, expected_type=torch.nn.Module)
        check_kwargs(self.critic_kwargs, self.critic_cls, ignore=["obs_space", "action_space"])
        self.obs_transform = to_transforms(self.obs_transform).to(self.device)
        self.action_transform = to_transforms(self.action_transform).to(self.device)
        self.target_action_transform = to_transforms(self.target_action_transform).to(self.device)


@dataclass
class EvalConfig:
    period: int
    steps: int
    obs_transform: Transform = field(default_factory=IdentityTF)
    action_transform: Transform = field(default_factory=IdentityTF)
    success_criteria: Callable[[list[float]], np.ndarray] | None = None
    post_callback: Callable[[list[float], int], None] | None = None

    def __post_init__(self):
        self.obs_transform = to_transforms(self.obs_transform)
        self.action_transform = to_transforms(self.action_transform)


@dataclass
class CheckpointConfig:
    period: int | None = None
    path: Path | None = None
    save_buffer: bool = False

    def __post_init__(self):
        if self.period is False:  # TOML can't represent None, use False instead and convert to None
            self.period = None
        if self.period is not None and self.path is None:
            raise ValueError("If 'checkpoint_period' is not None, 'checkpoint_path' must be set")
        if isinstance(self.path, str):
            self.path = Path(self.path).absolute()
            assert self.path.is_absolute(), "Checkpoint path must be an absolute path"
