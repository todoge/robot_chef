"""Integration test for perception-enabled pan pouring task."""

from __future__ import annotations

import math
from pathlib import Path

import pybullet as p
import pytest

from robot_chef import config, simulation
from robot_chef.tasks.pour_bowl_into_pan import PourBowlIntoPanAndReturn


def _rotate_xy(x: float, y: float, yaw_deg: float) -> tuple[float, float]:
    yaw_rad = math.radians(yaw_deg)
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    rx = cos_yaw * x - sin_yaw * y
    ry = sin_yaw * x + cos_yaw * y
    return rx, ry


@pytest.mark.slow
@pytest.mark.parametrize("world_yaw_deg", [0, 45])
@pytest.mark.parametrize("view_name", ["top", "oblique"])
def test_pour_bowl_into_pan_perception(world_yaw_deg: int, view_name: str) -> None:
    overrides = {
        "scene.world_yaw_deg": str(world_yaw_deg),
        "camera.active_view": view_name,
        "perception.enable": "true",
    }
    cfg = config.load_pour_task_config(Path("config/recipes/pour_demo.yaml"), overrides=overrides)
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
        target_x, target_y = _rotate_xy(cfg.bowl_pose.x, cfg.bowl_pose.y, world_yaw_deg)
        xy_error = math.hypot(final_pos[0] - target_x, final_pos[1] - target_y)
        assert xy_error <= cfg.tolerances.place_back_xy

        final_yaw = p.getEulerFromQuaternion(final_orn)[2]
        expected_yaw = cfg.bowl_pose.yaw + math.radians(world_yaw_deg)
        yaw_error = ((final_yaw - expected_yaw + math.pi) % (2.0 * math.pi)) - math.pi
        assert abs(math.degrees(yaw_error)) <= cfg.tolerances.place_back_yaw_deg
    finally:
        sim.disconnect()
