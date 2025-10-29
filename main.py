# main.py
from __future__ import annotations
import argparse, logging, sys
from pathlib import Path
import math
import pybullet as p

def parse_args():
    ap = argparse.ArgumentParser(description="Robot Chef")
    #ap.add_argument("--task", choices=["pour_bowl_into_pan"], default="pour_bowl_into_pan")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG","INFO","WARNING","ERROR"])
    return ap.parse_args()

def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Defer imports so the agent can regenerate these modules:
    from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn
    from robot_chef import config  # agent will create
    from robot_chef.isolated_simulation import RobotChefSimulation  # agent will create

    cfg = config.load_pour_task_config(Path(args.config))
    sim = RobotChefSimulation(gui=not args.headless, recipe=cfg)
    sim._setup_environment()
    down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
    sim.move_arm_to_pose("left", [0.5,0.3,2], down_orn, max_secs=5.0)
    ee_pos, ee_orn = sim.get_eef_state("left")
    sim.setup_camera(ee_pos, ee_orn)
    rgb, depth_buffer, depth_norm, _= sim.camera.get_rgbd()
    _, center_coord = sim.obj_detector.detect_object(rgb)
    #bbox_mask= sim.obj_detector.get_bbox_mask()
    #pixel_row, pixel_col, grasp_angle, grasp_width, quality, quality_map, angle_map, width_map = sim.grasping_predictor.predict_grasp(depth_norm, bbox_mask)
    #sim.grasping_predictor.visualize_grasp_predictions(depth_norm, quality_map, angle_map, width_map, pixel_row, pixel_col)
    coord = sim.pixel_to_world(center_coord[0], center_coord[1], depth_buffer)
    sim.set_recalibration_keypose(coord)
    sim.move_arm_to_pose("left", sim.keyposes["recalibration"], down_orn, max_secs=3.0)
    sim.gripper_open(arm="left")
    ee_pos, ee_orn = sim.get_eef_state("left")
    sim.setup_camera(ee_pos, ee_orn)
    rgb, depth_buffer, depth_norm, _= sim.camera.get_rgbd()
    pixel_row, pixel_col, grasp_angle, grasp_width, quality, quality_map, angle_map, width_map = sim.grasping_predictor.predict_grasp(depth_norm)
    sim.grasping_predictor.visualize_grasp_predictions(depth_norm, quality_map, angle_map, width_map, pixel_row, pixel_col, "grasp_visualisation_recali.png")
    coord = sim.pixel_to_world(pixel_row, pixel_col, depth_buffer)
    sim.set_grasping_keyposes(coord)
    sim.move_arm_to_pose("left", sim.keyposes["pregrasp"], down_orn, max_secs=3.0)
    grasp_euler = [math.pi, 0, grasp_angle]
    grasp_orn = p.getQuaternionFromEuler(grasp_euler)
    sim.move_gripper_straight_down(sim.keyposes["pregrasp"], sim.keyposes["grasp"], grasp_orn, "left")
    sim.gripper_close(arm="left", force=200.0)
    sim.move_arm_to_pose("left", sim.keyposes["lift"], down_orn, max_secs=3.0)
    pour_orn = p.getQuaternionFromEuler([-math.pi/2, 0, 0])
    sim.move_arm_to_pose("left", sim.keyposes["lift"], pour_orn, max_secs=4.0)


    #task = PourBowlIntoPanAndReturn()
    '''
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
    '''
    while True:
        x = 1 

if __name__ == "__main__":
    main()
