from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    ap.add_argument("--task", choices=["pour_bowl_into_pan", "path_planning"], default="pour_bowl_into_pan")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    
    from robot_chef import config
    from robot_chef.simulation import RobotChefSimulation

    cfg = config.load_pour_task_config(Path(args.config))
    sim = RobotChefSimulation(gui=not args.headless, recipe=cfg)

    if args.task == "pour_bowl_into_pan":
        from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn
        task = PourBowlIntoPanAndReturn()
    elif args.task == "path_planning":
        from robot_chef.tasks.under_arm_reach import UnderArmReachingTask
        task = UnderArmReachingTask()
    else:
        raise ValueError(f"Unknown task: {args.task}")

    try:
        task.setup(sim, cfg)
        planned = task.plan(sim, cfg)
        ok = task.execute(sim, cfg) if planned else False
        
        metrics = task.metrics(sim)
        if metrics:
            print("Metrics:", metrics)
            
        sys.exit(0 if ok else 1)
    finally:
        sim.disconnect()

if __name__ == "__main__":
    main()