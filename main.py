import argparse
import logging
import sys
from pathlib import Path
from time import sleep

import pybullet as p

from robot_chef import config
from robot_chef.simulator import StirSimulator
from robot_chef.tasks.stir import TaskStir


def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    ap.add_argument("--config", help="Path to YAML config", default="config/recipes/stir.yml")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = config.load_main_config(Path(args.config))
    sim = StirSimulator(gui=not args.headless, cfg=cfg)
    task = TaskStir()

    try:
        sim.setup()
        task.setup(sim, cfg)
        planned = task.plan()
        ok = task.execute() if planned else False
        task.metrics(print=True)
        sleep(60)
        sys.exit(0 if ok else 1)
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()
