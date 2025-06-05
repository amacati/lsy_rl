from pathlib import Path

import gymnasium
import pytest
import torch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch

from lsy_rl.core.transforms import to_transforms
from lsy_rl.td3 import td3
from lsy_rl.td3.policy import TD3Policy
from lsy_rl.utils import load_config


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


maybe_cuda = pytest.param(
    "cuda", marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available.")
)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.parametrize("vectorization_mode", ("sync", "async"))
@pytest.mark.parametrize("batch_size", (1, 3))
@pytest.mark.integration
def test_training(device: torch.device, vectorization_mode: str, batch_size: int):
    vector_kwargs = {}
    if vectorization_mode == "async":
        vector_kwargs = {"context": "spawn"}

    env = gymnasium.make_vec(
        "Pendulum-v1",
        num_envs=3,
        vectorization_mode=vectorization_mode,
        vector_kwargs=vector_kwargs,
    )
    eval_env = gymnasium.make_vec(
        "Pendulum-v1",
        num_envs=3,
        vectorization_mode=vectorization_mode,
        vector_kwargs=vector_kwargs,
    )
    env, eval_env = NumpyToTorch(env, device=device), NumpyToTorch(eval_env, device=device)
    config = load_config(Path(__file__).parent / "data/td3_config.toml")
    config.target_action_tf = to_transforms([config.target_action_tf])
    config.batch_size = batch_size
    config.device = device
    policy = td3(env, eval_env, **config)
    assert isinstance(policy, TD3Policy)
