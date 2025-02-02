from abc import ABC, abstractmethod

from torch import Tensor


class Policy(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def action(self, obs: Tensor) -> Tensor: ...
