# main.py
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from time import sleep

import pybullet as p


def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    ap.add_argument("--task", choices=["a2"], default="a2")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from robot_chef import config  # agent will create
    from robot_chef.sim import Simulator  # agent will create
    from robot_chef.tasks.a2 import A2

    cfg = config.load_stir_task_config(Path(args.config))
    sim = Simulator(gui=not args.headless, cfg=cfg)
    task = A2()
    try:
        task.setup(sim, cfg)
        planned = task.plan()
        ok = task.execute() if planned else False

        # x_slider = p.addUserDebugParameter("x", -1, 1, 0)
        # y_slider = p.addUserDebugParameter("y", -1, 1, 0)
        # z_slider = p.addUserDebugParameter("z", 0, 1, 0.5)

        # from time import sleep
        # while True:
        #     x = p.readUserDebugParameter(x_slider)
        #     y = p.readUserDebugParameter(y_slider)
        #     z = p.readUserDebugParameter(z_slider)
        #     p.addUserDebugLine([x, y, 0], [x, y, z], [1, 0, 0])
        #     print(f"Current position: ({x:.3f}, {y:.3f}, {z:.3f})")
        #     sleep(0.5)

        # metrics = task.metrics(sim)
        # print("Transfer Ratio: {ratio:.2f} ({inside}/{total})".format(
        #     ratio=float(metrics.get("transfer_ratio", 0.0)),
        #     inside=int(metrics.get("in_pan", 0)),
        #     total=int(metrics.get("total", 0)),
        # ))
        
        sleep(60)
        sys.exit(0 if ok else 1)
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()
