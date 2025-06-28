from abc import abstractmethod
from pathlib import Path

from torch import Tensor
from torch.nn import Module


class Policy(Module):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def action(self, obs: Tensor) -> Tensor: ...

    @abstractmethod
    def save(self, path: Path) -> None: ...