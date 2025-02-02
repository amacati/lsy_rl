from pathlib import Path

import gymnasium
import pytest
import torch

from lsy_rl.core import Algorithm
from lsy_rl.td3 import TD3
from lsy_rl.utils import load_config


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


@pytest.mark.parametrize(
    "device",
    (
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."),
        ),
    ),
)
@pytest.mark.parametrize("vectorization_mode", ("sync", "async"))
@pytest.mark.integration
def test_init(device: torch.device, vectorization_mode: str):
    env = gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode)
    config = load_config(Path(__file__).parent / "data/td3_config.toml")
    config.train.device = device
    td3 = TD3(env, env, config)
    assert isinstance(td3, Algorithm)


@pytest.mark.parametrize(
    "device",
    (
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."),
        ),
    ),
)
@pytest.mark.parametrize("vectorization_mode", ("sync", "async"))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.integration
def test_training(device: torch.device, vectorization_mode: str, batch_size: int):
    env = gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode)
    eval_env = gymnasium.make_vec("Pendulum-v1", num_envs=10, vectorization_mode=vectorization_mode)
    config = load_config(Path(__file__).parent / "data/td3_config.toml")
    config.train.batch_size = batch_size
    config.train.device = device
    td3 = TD3(env, eval_env, config)
    td3.train()
    assert isinstance(td3, Algorithm)
