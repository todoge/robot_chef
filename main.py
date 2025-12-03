# main.py
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    ap.add_argument("--task", choices=["pour_bowl_into_pan"], default="pour_bowl_into_pan")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn
    from robot_chef import config
    from robot_chef.simulation import RobotChefSimulation

    cfg = config.load_pour_task_config(Path(args.config))
    sim = RobotChefSimulation(gui=not args.headless, recipe=cfg)
    task = PourBowlIntoPanAndReturn()
    try:
        task.setup(sim, cfg)
        planned = task.plan(sim, cfg)
        ok = task.execute(sim, cfg) if planned else False
        metrics = task.metrics(sim)
        print("Transfer Ratio: {ratio:.2f} ({inside}/{total})".format(
            ratio=float(metrics.get("transfer_ratio", 0.0)),
            inside=int(metrics.get("in_pan", 0)),
            total=int(metrics.get("total", 0)),
        ))
        sys.exit(0 if ok else 1)
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()
