from pathlib import Path

import gymnasium
import pytest
import torch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

from lsy_rl.core import Algorithm
from lsy_rl.ddpg import DDPG
from lsy_rl.utils import load_config


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


maybe_cuda = pytest.param(
    "cuda", marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available.")
)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.parametrize("vectorization_mode", ("sync", "async"))
@pytest.mark.integration
def test_init(device: torch.device, vectorization_mode: str):
    env = NumpyToTorch(
        gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode),
        device=device,
    )
    config = load_config(Path(__file__).parent / "data/ddpg_config.toml")
    config.train.device = device
    ddpg = DDPG(env, env, config)
    assert isinstance(ddpg, Algorithm)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.parametrize("vectorization_mode", ("sync", "async"))
@pytest.mark.integration
def test_training(device: torch.device, vectorization_mode: str):
    env = NumpyToTorch(
        gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode),
        device=device,
    )
    eval_env = NumpyToTorch(
        gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode),
        device=device,
    )
    config = load_config(Path(__file__).parent / "data/ddpg_config.toml")
    config.train.device = device
    ddpg = DDPG(env, eval_env, config)
    ddpg.train()
    assert isinstance(ddpg, Algorithm)
