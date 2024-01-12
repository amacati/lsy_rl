import pytest
import gymnasium
import torch

from lsy_rl.core import Algorithm
from lsy_rl.ddpg import DDPG


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


def test_init():
    env = gymnasium.vector.make("Pendulum-v1", num_envs=10)
    ddpg = DDPG(env)
    assert isinstance(ddpg, Algorithm)


@pytest.mark.skipif(cuda_not_available(), reason="Cuda not available.")
def test_init_cuda():
    env = gymnasium.vector.make("Pendulum-v1", num_envs=10)
    policy_kwargs = {"device": "cuda"}
    buffer_kwargs = {"maxlen": 100_000, "device": "cuda"}
    ddpg = DDPG(env, policy_kwargs=policy_kwargs, replay_buffer_kwargs=buffer_kwargs)
    assert isinstance(ddpg, Algorithm)
