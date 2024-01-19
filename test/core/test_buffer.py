import pytest
import torch
import gymnasium
from tensordict import merge_tensordicts

from lsy_rl.core.replay_buffer import VectorReplayBuffer, SimpleReplayBuffer
from lsy_rl.wrappers.tensor_wrapper import TensorWrapper


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_simple_init(device):
    SimpleReplayBuffer(num_envs=2, max_size=10, device=device)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_simple_add(device):
    num_envs = 2
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    obs = env.reset()["obs"]
    action = env.action_space.sample()
    sample = env.step(action)
    sample["obs"] = obs
    buffer.add(sample)
    assert len(buffer) == num_envs
    assert buffer.buffer.keys() == sample.keys()


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_simple_add_wrap(device):
    num_envs, max_size = 2, 11  # Wraps around after 5 steps
    assert max_size % num_envs != 0, "max_size must not be divisible by num_envs for wrap test"
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
    obs = env.reset()["obs"]
    action = env.action_space.sample()
    sample = env.step(action)
    sample["obs"] = obs
    sample["obs"][...] = 1
    for _ in range(max_size // num_envs):
        buffer.add(sample)
    sample["obs"][...] = 2
    buffer.add(sample)  # Wraps around
    assert len(buffer) == max_size
    assert torch.all(buffer.buffer["obs"][0, ...] == 2)
    assert torch.all(buffer.buffer["obs"][-1, ...] == 2)
    assert torch.all(buffer.buffer["obs"][1:-1, ...] == 1)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_vector_init(device):
    VectorReplayBuffer(num_envs=2, max_size=10, device=device)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_vector_add(device):
    num_envs = 2
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = VectorReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    assert isinstance(env.observation_space.sample(), torch.Tensor)
    sample = env.reset()
    action = env.action_space.sample()
    next_sample = env.step(action)
    sample = merge_tensordicts(sample, next_sample)
    buffer.add(sample)
    assert len(buffer) == num_envs


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_vector_sample(device):
    num_envs = 2
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = VectorReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    assert isinstance(env.observation_space.sample(), torch.Tensor)
    sample = env.reset()
    action = env.action_space.sample()
    next_sample = env.step(action)
    sample = merge_tensordicts(sample, next_sample)
    for _ in range(10):
        buffer.add(sample)
    buffer.sample(10)
