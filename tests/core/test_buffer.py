import gymnasium
import pytest
import torch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from tensordict import TensorDict

from lsy_rl.core.replay_buffer import HerVectorReplayBuffer, SimpleReplayBuffer, VectorReplayBuffer
from lsy_rl.utils.utils import tensordict_sample


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
@pytest.mark.unit
def test_simple_init(device):
    SimpleReplayBuffer(num_envs=2, max_size=10, device=device)


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
@pytest.mark.unit
def test_simple_add(device):
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    buffer.add(sample)
    assert len(buffer) == num_envs
    assert all([kb == ko for kb, ko in zip(buffer.buffer.keys(), sample.keys())])


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
@pytest.mark.unit
def test_simple_add_wrap(device):
    num_envs, max_size = 2, 11  # Wraps around after 5 steps
    assert max_size % num_envs != 0, "max_size must not be divisible by num_envs for wrap test"
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
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
    (
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available."),
        ),
    ),
)
@pytest.mark.unit
def test_vector_init(device):
    VectorReplayBuffer(num_envs=2, max_size=10, device=device)


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
@pytest.mark.unit
def test_vector_add(device):
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = VectorReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    buffer.add(sample)
    assert len(buffer) == num_envs


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
@pytest.mark.unit
def test_vector_sample(device):
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = VectorReplayBuffer(num_envs=num_envs, max_size=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    for _ in range(10):
        buffer.add(sample)
    sample = buffer.sample(10)
    assert sample["obs"].shape == (10, 3)


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
@pytest.mark.unit
def test_vector_add_wrap(device):
    num_envs, max_size = 2, 10  # Wraps around after 5 steps
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    sample["next_obs"][...] = 1
    for _ in range(max_size // num_envs):
        buffer.add(sample)
    sample["next_obs"][...] = 2
    buffer.add(sample)  # Wraps around
    assert len(buffer) == max_size
    assert torch.all(buffer.buffer["next_obs"][0, ...] == 2)
    assert torch.all(buffer.buffer["next_obs"][1, ...] == 2)
    assert torch.all(buffer.buffer["next_obs"][2:, ...] == 1)
    buffer.sample(10)


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
@pytest.mark.unit
def test_her_vector_replay_buffer_sample(device):
    num_envs, max_size = 2, 10  # Wraps around after 5 steps
    batch_size = 8
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = SimpleReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
    buffer = HerVectorReplayBuffer(
        num_envs=num_envs, max_size=max_size, reward_fn=lambda x, y: torch.zeros(batch_size)
    )
    for _ in range(2):
        obs, _ = env.reset()
        action = torch.tensor(env.action_space.sample())
        next_obs, reward, terminated, truncated, info = env.step(action)
        sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
        sample["obs"] = TensorDict(
            {
                "obs": sample["next_obs"].clone(),
                "desired_goal": sample["next_obs"].clone(),
                "achieved_goal": sample["next_obs"].clone(),
            },
            batch_size=num_envs,
        )
        sample["next_obs"] = sample["obs"].clone()
        for _ in range(max_size // num_envs - 1):
            buffer.add(sample)
        sample["truncated"][...] = True
        buffer.add(sample)
    assert len(buffer) == max_size
    sample = buffer.sample(batch_size)
