from abc import ABC, abstractmethod
from pathlib import Path

from torch import Tensor, FloatTensor


class Policy(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def action(self, obs: FloatTensor) -> Tensor:
        ...

    @abstractmethod
    def save(self, path: Path):
        ...

    @abstractmethod
    def load(self, path: Path):
        ...
