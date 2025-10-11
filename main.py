"""Entry point for running the robot chef fried rice demo."""

from __future__ import annotations

import argparse
from contextlib import suppress

from robot_chef.actions import RECIPE_ACTIONS
from robot_chef.executor import RecipeExecutor
from robot_chef.simulation import RobotChefSimulation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot chef fried rice demonstration")
    parser.add_argument("--headless", action="store_true", help="Run the simulation without a GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sim = RobotChefSimulation(gui=not args.headless)
    try:
        executor = RecipeExecutor(sim, RECIPE_ACTIONS)
        log = executor.run()
        print(log.summary())
    finally:
        with suppress(Exception):
            sim.disconnect()


if __name__ == "__main__":
    main()
