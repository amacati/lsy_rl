from pathlib import Path

import gymnasium
import pytest
import torch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

from lsy_rl.ppo import ppo
from lsy_rl.utils import load_config


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


maybe_cuda = pytest.param(
    "cuda", marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available.")
)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.integration
def test_training(device: torch.device):
    env = gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="sync")
    eval_env = gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode="sync")
    env, eval_env = NumpyToTorch(env, device), NumpyToTorch(eval_env, device)
    config = load_config(Path(__file__).parent / "data/ppo_config.toml")
    config.device = device
    ppo(env, eval_env, **config)
