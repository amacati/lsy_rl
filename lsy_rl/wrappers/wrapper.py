import torch
from gymnasium.vector import VectorEnv

from lsy_rl.wrappers.orbit_wrapper import OrbitWrapper
from lsy_rl.wrappers.tensordict_wrapper import DefaultTensorDictWrapper, TensorDictWrapper


def contains_wrapper(env: VectorEnv, wrapper_type: type) -> bool:
    """Check if the environment contains a wrapper of the given type."""
    while env is not env.unwrapped:
        if isinstance(env, wrapper_type):
            return True
        env = env.env
    return False


def wrap_env(env: VectorEnv, device: torch.device = torch.device("cpu")) -> TensorDictWrapper:
    """Wrap the environment with the appropriate wrappers."""
    # Determine if the environment is an Orbit environment
    try:
        # Import Orbit lazily to prevent errors when Orbit is not installed or IsaacSim is not
        # running
        from omni.isaac.orbit.envs.rl_task_env import RLTaskEnv

        if isinstance(env.unwrapped, RLTaskEnv):
            if contains_wrapper(env, OrbitWrapper):
                return env  # Already wrapped
            return OrbitWrapper(env, device)
    except ImportError:
        pass
    if contains_wrapper(env, TensorDictWrapper):
        return env  # Already wrapped
    return DefaultTensorDictWrapper(env, device)
