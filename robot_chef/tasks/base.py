from abc import ABC, abstractmethod

from ..config import MainConfig
from ..sim import Simulator


class Task(ABC):

    def __init__(self, sim: Simulator, cfg: MainConfig):
        self.sim = sim
        self.cfg = cfg

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def plan(self) -> None:
        pass

    @abstractmethod
    def execute(self) -> None:
        pass

    @abstractmethod
    def metrics(self) -> None:
        pass
