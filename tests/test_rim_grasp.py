"""Unit tests for the bowl rim detection pipeline."""

from __future__ import annotations

import pytest
import pybullet as p

from robot_chef import config, simulation
from robot_chef.perception import Camera, configure_scene_context, detect_bowl_rim


@pytest.mark.parametrize("world_yaw_deg", [0, 45, 90])
@pytest.mark.parametrize("view_name", ["top", "oblique"])
def test_detect_bowl_rim_candidates(world_yaw_deg: int, view_name: str) -> None:
    overrides = {
        "scene.world_yaw_deg": str(world_yaw_deg),
        "camera.active_view": view_name,
        "perception.enable": "true",
    }
    cfg = config.load_pour_task_config("config/recipes/pour_demo.yaml", overrides=overrides)
    sim = simulation.RobotChefSimulation(gui=False, recipe=cfg)
    try:
        view_cfg = cfg.camera.get_view(view_name)
        camera = Camera(
            client_id=sim.client_id,
            view_name=view_name,
            position=view_cfg.xyz,
            rpy_rad=view_cfg.rpy_rad,
            fov_deg=view_cfg.fov_deg,
            resolution=view_cfg.resolution,
            near=cfg.camera.near,
            far=cfg.camera.far,
            depth_noise_std=0.0,
            depth_drop_prob=0.0,
            seed=cfg.seed,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb, depth, _ = camera.get_rgbd()
        bowl = sim.objects["rice_bowl"]
        bowl_aabb = p.getAABB(bowl.body_id, physicsClientId=sim.client_id)
        configure_scene_context(
            bowl_aabb_min=bowl_aabb[0],
            bowl_aabb_max=bowl_aabb[1],
            world_from_cam=camera.world_from_cam,
            roi_margin=cfg.perception.bowl_roi_margin_m,
            sample_count=cfg.perception.rim_sample_count,
            grasp_clearance=cfg.perception.grasp_clearance_m,
            reachability_fn=None,
        )
        detection = detect_bowl_rim(rgb, depth, camera.K)
        assert len(detection["grasp_candidates"]) >= 6
        assert 0.05 <= detection["radius_m"] <= 0.1
    finally:
        sim.disconnect()
