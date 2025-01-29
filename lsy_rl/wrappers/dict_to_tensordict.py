from functools import partial

from gymnasium.spaces import Space
from gymnasium.vector import VectorEnv
from gymnasium.wrappers.vector.vectorize_observation import TransformObservation
from tensordict import TensorDict


class DictToTensorDict(TransformObservation):
    """Transform the observation of an env from a dict to a TensorDict."""

    def __init__(
        self, env: VectorEnv, observation_space: Space | None = None, device: str | None = None
    ):
        transform = partial(dict_to_tensordict, batch_size=env.num_envs, device=device)
        super().__init__(env, transform, observation_space)


def dict_to_tensordict(
    obs: dict, batch_size: int | tuple[int], device: str | None = None
) -> TensorDict:
    return TensorDict(obs, batch_size=batch_size, device=device)
