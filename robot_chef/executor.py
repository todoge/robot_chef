"""Finite state machine executor for the preset recipe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from .actions import ActionPrimitive, ActionResult
from .simulation import RobotChefSimulation


@dataclass
class ExecutionLog:
    results: List[ActionResult] = field(default_factory=list)

    def append(self, result: ActionResult) -> None:
        self.results.append(result)

    def summary(self) -> str:
        lines = [f"{result.name}: {'OK' if result.success else 'FAIL'} - {result.details}" for result in self.results]
        return "\n".join(lines)


class RecipeExecutor:
    """Executes a sequence of action primitives sequentially."""

    def __init__(self, sim: RobotChefSimulation, actions: Sequence[ActionPrimitive]):
        self.sim = sim
        self.actions = list(actions)
        self.log = ExecutionLog()

    def run(self) -> ExecutionLog:
        for action in self.actions:
            result = action.run(self.sim)
            self.log.append(result)
        return self.log
