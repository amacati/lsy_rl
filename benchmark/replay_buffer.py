import timeit

import torch
from tensordict import TensorDict

from lsy_rl.core.replay_buffer import VectorReplayBuffer, HerVectorReplayBuffer

# Replay buffer settings
num_envs = 32
max_size = 100_000
device = torch.device("cpu")


def create_sample():
    sample = TensorDict(
        {
            "obs": torch.randn(num_envs, 100),
            "action": torch.randn(num_envs, 10),
            "reward": torch.randn(num_envs, 1),
            "next_obs": torch.randn(num_envs, 100),
            "terminated": torch.rand(num_envs, 1) > 0.9,
            "truncated": torch.rand(num_envs, 1) > 0.9
        },
        batch_size=num_envs,
        device="cpu")
    return sample


def reward_fn(*args, **kwargs):
    return 1


vector_rb = VectorReplayBuffer(num_envs=num_envs, max_size=max_size, device=device)
her_rb = HerVectorReplayBuffer(num_envs=num_envs,
                               max_size=max_size,
                               reward_fn=reward_fn,
                               device=device)
vector_rb.add(create_sample())  # Allocate buffers on first call
her_rb.add(create_sample())


def main():
    # Set the limit for the sum of squares calculation
    num_calls = 10_000
    # Benchmark the first implementation
    setup = ("from __main__ import vector_rb, create_sample\n")
    stmt = "vector_rb.add(create_sample())"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<VectorReplayBuffer.add> took an average of {time_vec_append:.2e}s per call.")

    setup = ("from __main__ import her_rb, create_sample\n")
    stmt = "her_rb.add(create_sample())"
    time_vec_append = timeit.timeit(stmt=stmt, setup=setup, number=num_calls) / num_calls
    print(f"<HERVectorReplayBuffer.add> took an average of {time_vec_append:.2e}s per call.")


if __name__ == "__main__":
    main()
