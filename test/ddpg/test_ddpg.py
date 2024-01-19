from pathlib import Path

import pytest
import gymnasium
import torch

from lsy_rl.core import Algorithm
from lsy_rl.ddpg import DDPG
from lsy_rl.utils.wandb import load_config


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_init(device):
    env = gymnasium.vector.make("Pendulum-v1", num_envs=10)
    config = load_config(Path(__file__).parent / "data/ddpg_config.yaml")
    config.train.device = device
    ddpg = DDPG(env, env, config)
    assert isinstance(ddpg, Algorithm)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_training(device):
    env = gymnasium.vector.make("Pendulum-v1", num_envs=10)
    eval_env = gymnasium.vector.make("Pendulum-v1", num_envs=10)
    config = load_config(Path(__file__).parent / "data/ddpg_config.yaml")
    config.train.device = device
    ddpg = DDPG(env, eval_env, config)
    ddpg.train()
    assert isinstance(ddpg, Algorithm)
