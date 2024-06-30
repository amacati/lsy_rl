from abc import ABC, abstractmethod


class Algorithm(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train(self): ...
