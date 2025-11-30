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
    ap.add_argument("--config", required=True, help="Path to YAML config")
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
        
        x_slider = p.addUserDebugParameter("x", -1, 1, 0)
        y_slider = p.addUserDebugParameter("y", -1, 1, 0)
        z_slider = p.addUserDebugParameter("z", 0, 1, 0.5)

        from time import sleep
        while True:
            x = p.readUserDebugParameter(x_slider)
            y = p.readUserDebugParameter(y_slider)
            z = p.readUserDebugParameter(z_slider)
            p.addUserDebugLine([x, y, 0], [x, y, z], [1, 0, 0])
            print(f"Current position: ({x:.3f}, {y:.3f}, {z:.3f})")
            sleep(0.5)

        sleep(60)
        sys.exit(0 if ok else 1)
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()
