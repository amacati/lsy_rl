import timeit

import torch
from tensordict import TensorDict

from lsy_rl.core.replay_buffer import HerVectorReplayBuffer, VectorReplayBuffer

# Replay buffer settings
num_envs = 32
max_size = 100_000
device = torch.device("cpu")


def create_sample():
    obs = TensorDict(
        {
            "achieved_goal": torch.randn(num_envs, 10),
            "desired_goal": torch.randn(num_envs, 10),
            "obs": torch.randn(num_envs, 100),
        },
        batch_size=num_envs,
        device="cpu",
    )
    sample = TensorDict(
        {
            "obs": obs,
            "action": torch.randn(num_envs, 10),
            "reward": torch.randn(num_envs, 1),
            "next_obs": obs,
            "terminated": torch.rand(num_envs, 1) > 0.9,
            "truncated": torch.rand(num_envs, 1) > 0.9,
        },
        batch_size=num_envs,
        device="cpu",
    )
    return sample


def reward_fn(x, y):
    return torch.ones(x.shape[0], 1)


vector_rb = VectorReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
her_rb = HerVectorReplayBuffer(
    num_envs=num_envs, max_size=max_size, reward_fn=reward_fn, device=device
)
vector_rb.add(create_sample())  # Allocate buffers on first call
her_rb.add(create_sample())


def benchmark_add():
    # Set the limit for the sum of squares calculation
    num_calls = 10_000
    # Benchmark the first implementation
    setup = "from __main__ import vector_rb, create_sample\n"
    stmt = "vector_rb.add(create_sample())"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<VectorReplayBuffer.add> took an average of {time_vec_append:.2e}s per call.")

    setup = "from __main__ import her_rb, create_sample\n"
    stmt = "her_rb.add(create_sample())"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<HERVectorReplayBuffer.add> took an average of {time_vec_append:.2e}s per call.")
    # 3.69e-04s per call.


def benchmark_sample():
    # Set the limit for the sum of squares calculation
    num_calls = 10_000
    # Benchmark the first implementation
    setup = """from __main__ import vector_rb, create_sample
[vector_rb.add(create_sample()) for _ in range(10000)]"""
    stmt = "vector_rb.sample(256)"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<VectorReplayBuffer.sample> took an average of {time_vec_append:.2e}s per call.")

    setup = """from __main__ import her_rb, create_sample
[her_rb.add(create_sample()) for _ in range(10000)]"""
    stmt = "her_rb.sample(256)"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<HERVectorReplayBuffer.sample> took an average of {time_vec_append:.2e}s per call.")
    # 3.69e-04s per call.


if __name__ == "__main__":
    print("### Benchmarking replay buffer methods. ###")
    for func in [benchmark_add, benchmark_sample]:
        print(f"\n{func.__name__} (y/n)?")
        if input().lower() == "y":
            func()
