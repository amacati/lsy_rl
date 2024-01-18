import pytest
import torch
import gymnasium

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
    env = gymnasium.vector.make("Pendulum-v1", num_envs=2)
    SimpleReplayBuffer(env=env, maxlen=10, device=device)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_simple_add(device):
    num_envs = 2
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(env=env, maxlen=10, device=device)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(obs, action, reward, next_obs, terminated, truncated)
    assert len(buffer) == num_envs


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_simple_add_wrap(device):
    num_envs, maxlen = 2, 11  # Wraps around after 5 steps
    assert maxlen % num_envs != 0, "maxlen must not be divisible by num_envs for wrap test"
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(env=env, maxlen=maxlen, device=device)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    obs = torch.ones_like(obs)
    for _ in range(maxlen // num_envs):
        buffer.add(obs, action, reward, next_obs, terminated, truncated)
    obs = torch.ones_like(obs) * 2
    buffer.add(obs, action, reward, next_obs, terminated, truncated)  # Wraps around
    assert len(buffer) == maxlen
    assert torch.all(buffer.buffer["obs"][0, ...] == 2)
    assert torch.all(buffer.buffer["obs"][-1, ...] == 2)
    assert torch.all(buffer.buffer["obs"][1:-1, ...] == 1)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_vector_init(device):
    env = gymnasium.vector.make("Pendulum-v1", num_envs=2)
    VectorReplayBuffer(env=env, maxlen=10, device=device)


@pytest.mark.parametrize(
    "device",
    (torch.device("cpu"),
     pytest.param(torch.device("cuda"),
                  marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."))))
def test_vector_add(device):
    num_envs = 2
    env = TensorWrapper(gymnasium.vector.make("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = VectorReplayBuffer(env=env, maxlen=10, device=device)
    assert isinstance(env.observation_space.sample(), torch.Tensor)
    obs, info = env.reset()
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    buffer.add(obs, action, reward, next_obs, terminated, truncated)
    assert len(buffer) == num_envs
