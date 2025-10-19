"""Command-line entry point for robot chef tasks."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

from robot_chef import actions, config, executor, simulation
from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn

LOGGER = logging.getLogger("robot_chef")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot chef simulation runner")
    parser.add_argument(
        "--task",
        choices=["fried_rice", "pour_bowl_into_pan"],
        default="fried_rice",
        help="Task to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to task-specific configuration file",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run PyBullet in DIRECT mode without GUI",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity level",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values using dotted keys (e.g. scene.world_yaw_deg=45)",
    )
    return parser.parse_args()


def _parse_override_args(entries) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise ValueError(f"Invalid override '{entry}'. Expected format key=value.")
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key cannot be empty in '{entry}'.")
        overrides[key] = value.strip()
    return overrides


def run_fried_rice(gui: bool) -> None:
    sim = simulation.RobotChefSimulation(gui=gui)
    try:
        exec = executor.RecipeExecutor(sim, actions.RECIPE_ACTIONS)
        log = exec.run()
        print(log.summary())
    finally:
        sim.disconnect()


def _format_transfer_line(metrics: Dict[str, float]) -> str:
    """
    Produce a stable 'Transfer Ratio: X.XX (inside/total)' line even if some keys are absent.
    Known optional keys:
      - 'transfer_ratio' (float in [0,1])
      - 'in_pan' or 'particles_in_pan' (int)
      - 'total' or 'particles_total' (int)
    """
    # Pull inside/total with fallbacks.
    inside = int(metrics.get("in_pan", metrics.get("particles_in_pan", 0)))
    total = int(metrics.get("total", metrics.get("particles_total", 0)))

    # Prefer provided transfer_ratio; otherwise compute if possible.
    ratio = metrics.get("transfer_ratio", None)
    if ratio is None:
        ratio = float(inside) / float(total) if total > 0 else 0.0
    else:
        try:
            ratio = float(ratio)
        except Exception:
            ratio = 0.0

    return "Transfer Ratio: {ratio:.2f} ({inside}/{total})".format(
        ratio=ratio, inside=inside, total=total
    )


def run_pour_task(gui: bool, config_path: Path, overrides: Dict[str, str], task_cls) -> None:
    task_cfg = config.load_pour_task_config(config_path, overrides=overrides or None)
    sim = simulation.RobotChefSimulation(gui=gui, recipe=task_cfg)
    task = task_cls()
    try:
        task.setup(sim, task_cfg)
        planned = task.plan(sim, task_cfg)
        if not planned:
            LOGGER.warning("Planning failed; continuing to execution for diagnostics.")
        executed = task.execute(sim, task_cfg)
        if not executed:
            LOGGER.warning("Execution returned False.")

        # Metrics are optional; protect access.
        metrics: Dict[str, float] = {}
        try:
            m = task.metrics(sim)
            if isinstance(m, dict):
                metrics = m
        except Exception:
            LOGGER.exception("metrics() raised; using empty metrics.")

        print(_format_transfer_line(metrics))
    finally:
        sim.disconnect()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    default_config = Path("config/recipes/pour_demo.yaml")
    try:
        overrides = _parse_override_args(args.overrides)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)

    if args.task == "fried_rice":
        run_fried_rice(gui=not args.headless)
    elif args.task == "pour_bowl_into_pan":
        config_path = Path(args.config) if args.config else default_config
        if not config_path.is_file():
            print(f"Configuration file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
        run_pour_task(
            gui=not args.headless,
            config_path=config_path,
            overrides=overrides,
            task_cls=PourBowlIntoPanAndReturn,
        )
    else:
        print(f"Unsupported task: {args.task}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
