"""Task interface for robot chef demonstrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class Task(ABC):
    name: str

    @abstractmethod
    def setup(self, world, cfg) -> None:
        raise NotImplementedError

    @abstractmethod
    def plan(self, world, cfg) -> None:
        raise NotImplementedError

    @abstractmethod
    def execute(self, world, cfg) -> None:
        raise NotImplementedError

    @abstractmethod
    def metrics(self, world) -> Dict[str, float]:
        raise NotImplementedError


__all__ = ["Task"]
