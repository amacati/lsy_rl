import gymnasium
import pytest
import torch
from gymnasium.wrappers.vector.numpy_to_torch import NumpyToTorch
from tensordict import TensorDict

from lsy_rl.core.replay_buffer import (
    HerVectorReplayBuffer,
    SimpleReplayBuffer,
    TrajectoryBuffer,
    VectorReplayBuffer,
)
from lsy_rl.utils.utils import tensordict_sample


def cuda_not_available() -> bool:
    return not torch.cuda.is_available()


maybe_cuda = pytest.param(
    "cuda", marks=pytest.mark.skipif(cuda_not_available(), reason="Cuda not available.")
)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_simple_init(device):
    SimpleReplayBuffer(num_envs=2, max_size=10, device=device)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_vector_init(device):
    VectorReplayBuffer(num_envs=2, max_size=10, device=device)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
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


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_trajectory_init(device):
    """Test TrajectoryBuffer initialization."""
    TrajectoryBuffer(num_envs=2, trajectory_len=10, device=device)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_trajectory_add(device):
    """Test adding samples to TrajectoryBuffer."""
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = TrajectoryBuffer(num_envs=num_envs, trajectory_len=1, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    buffer.add(sample)
    assert len(buffer) == num_envs
    assert all([kb == ko for kb, ko in zip(buffer.buffer.keys(), sample.keys())])
    with pytest.raises(AssertionError):
        buffer.add(sample)


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_trajectory_add_with_mask(device):
    """Test adding samples with mask to TrajectoryBuffer."""
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = TrajectoryBuffer(num_envs=num_envs, trajectory_len=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)

    # Test adding with mask
    mask = torch.tensor([True, False], device=device)
    buffer.add(sample[0:1], mask)  # Add only first sample
    assert len(buffer) == 1


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_trajectory_indexing(device):
    """Test indexing the TrajectoryBuffer."""
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = TrajectoryBuffer(num_envs=num_envs, trajectory_len=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample(), device=device)
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)
    # Add multiple samples
    for _ in range(3):
        buffer.add(sample)
    # Test indexing
    assert torch.all(buffer["obs"][:, 0, ...] == sample["obs"])
    assert torch.all(buffer["action"][:, 0, ...] == sample["action"])
    assert torch.all(buffer["next_obs"][:, 0, ...] == sample["next_obs"])
    assert torch.all(buffer["reward"][:, 0, ...] == sample["reward"])
    assert torch.all(buffer["terminated"][:, 0, ...] == sample["terminated"])
    assert torch.all(buffer["truncated"][:, 0, ...] == sample["truncated"])


@pytest.mark.parametrize("device", ("cpu", maybe_cuda))
@pytest.mark.unit
def test_trajectory_clear(device):
    """Test clearing the TrajectoryBuffer."""
    num_envs = 2
    env = NumpyToTorch(gymnasium.make_vec("Pendulum-v1", num_envs=num_envs), device=device)
    buffer = TrajectoryBuffer(num_envs=num_envs, trajectory_len=10, device=device)
    obs, _ = env.reset()
    action = torch.tensor(env.action_space.sample())
    next_obs, reward, terminated, truncated, info = env.step(action)
    sample = tensordict_sample(obs, action, next_obs, reward, terminated, truncated, info)

    # Add samples then clear
    buffer.add(sample)
    assert len(buffer) > 0
    buffer.clear()
    assert len(buffer) == 0
    assert torch.all(buffer._idx == 0)
