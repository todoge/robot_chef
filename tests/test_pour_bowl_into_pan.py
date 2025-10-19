"""Integration test for the pour bowl into pan task."""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pytest

from robot_chef import config, simulation
from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn


@pytest.mark.slow
def test_pour_bowl_into_pan_transfer_and_return():
    cfg_path = Path("config/recipes/pour_demo.yaml")
    cfg = config.load_pour_task_config(cfg_path)
    sim = simulation.RobotChefSimulation(gui=False, recipe=cfg)
    task = PourBowlIntoPanAndReturn()
    try:
        task.setup(sim, cfg)
        task.plan(sim, cfg)
        task.execute(sim, cfg)
        metrics = task.metrics(sim)
        assert metrics["transfer_ratio"] >= 0.6

        bowl_id = sim.objects["rice_bowl"].body_id
        final_pos, final_orn = p.getBasePositionAndOrientation(bowl_id, physicsClientId=sim.client_id)
        dx = final_pos[0] - cfg.bowl_pose.x
        dy = final_pos[1] - cfg.bowl_pose.y
        xy_error = math.hypot(dx, dy)
        assert xy_error <= cfg.tolerances.place_back_xy

        final_yaw = p.getEulerFromQuaternion(final_orn)[2]
        yaw_error = ((final_yaw - cfg.bowl_pose.yaw + math.pi) % (2.0 * math.pi)) - math.pi
        assert abs(yaw_error) <= math.radians(cfg.tolerances.place_back_yaw_deg)
    finally:
        sim.disconnect()
