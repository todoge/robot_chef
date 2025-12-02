"""Perception-in-the-loop task pipeline for pouring from bowl into pan."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import struct
import zlib

from ..camera import Camera, CameraNoiseModel
from ..config import Pose6D, PourTaskConfig
from ..controller_vision import VisionRefineController
from ..perception.bowl_rim import detect_bowl_rim
from ..simulation import RobotChefSimulation

LOGGER = logging.getLogger(__name__)


def _pose_to_quaternion(pose: Pose6D) -> Tuple[float, float, float, float]:
    return p.getQuaternionFromEuler([pose.roll, pose.pitch, pose.yaw])


def _apply_world_yaw(pose: Pose6D, yaw_deg: float) -> Pose6D:
    yaw = math.radians(yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = pose.x * cos_y - pose.y * sin_y
    y = pose.x * sin_y + pose.y * cos_y
    return Pose6D(
        x=x,
        y=y,
        z=pose.z,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw + yaw,
    )


def _project_point(world_point: Sequence[float], camera_from_world: np.ndarray, K: np.ndarray) -> Tuple[Tuple[float, float], float]:
    point = np.array(list(world_point) + [1.0], dtype=float)
    cam = camera_from_world @ point
    z = cam[2]
    if abs(z) < 1e-6:
        z = 1e-6
    u = (cam[0] / z) * K[0, 0] + K[0, 2]
    v = (cam[1] / z) * K[1, 1] + K[1, 2]
    return (float(u), float(v)), float(z)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB image to disk without external dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("Overlay writer expects RGB images")
    raw = bytearray()
    for row in image:
        raw.append(0)  # no filter
        raw.extend(row.tobytes())
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    with path.open("wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", compressed))
        fh.write(chunk(b"IEND", b""))


class PourBowlIntoPanAndReturn:
    """High-level state machine orchestrating perception, grasp, pour, and replace."""

    def __init__(self) -> None:
        self.camera: Optional[Camera] = None
        self.controller: Optional[VisionRefineController] = None
        self.sim: Optional[RobotChefSimulation] = None
        self.cfg: Optional[PourTaskConfig] = None
        self._metrics: Dict[str, float] = {"transfer_ratio": 0.0, "in_pan": 0, "total": 0}

    # ------------------------------------------------------------------ #
    # Task lifecycle

    def setup(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> None:
        print(sim.objects.keys())
        self.sim = sim
        self.cfg = cfg
        view_cfg = cfg.camera.get_active_view()
        noise_cfg = cfg.camera.noise
        self.camera = Camera(
            client_id=sim.client_id,
            view_xyz=view_cfg.xyz,
            view_rpy_deg=view_cfg.rpy_deg,
            fov_deg=view_cfg.fov_deg,
            near=cfg.camera.near,
            far=cfg.camera.far,
            resolution=view_cfg.resolution,
            noise=CameraNoiseModel(depth_std=noise_cfg.depth_std, drop_prob=noise_cfg.drop_prob),
        )
        
        # Mount camera to the Right Arm End-Effector for IBVS eye-in-hand demo
        # Positions camera 5cm "above" (z) and 5cm "forward" (x) of the wrist, looking down (-y)
        self.camera.mount_to_link(
            parent_body_id=sim.right_arm.body_id,
            parent_link_id=sim.right_arm.ee_link,
            rel_xyz=[0.05, 0.0, 0.05],
            rel_rpy_deg=[0.0, -90.0, 0.0]
        )

        self.controller = VisionRefineController(
            client_id=sim.client_id,
            arm_id=sim.right_arm.body_id,
            ee_link=sim.right_arm.ee_link,
            arm_joints=sim.right_arm.joint_indices,
            dt=sim.dt,
            camera=self.camera,
            gripper_open=lambda width=0.08: sim.gripper_open(width=width),
            gripper_close=lambda force=60.0: sim.gripper_close(force=force),
            camera_fixed=False  # Eye-in-hand setup: camera moves with robot
        )
        sim.gripper_open()
        if cfg.particles > 0:
            sim.spawn_rice_particles(cfg.particles, seed=cfg.seed)
        bowl_obj = sim.objects.get("bowl")
        if bowl_obj is None:
            raise RuntimeError("bowl not found in sim.objects; available keys: "
                            + ", ".join(sim.objects.keys()))
        self._bowl_uid = bowl_obj['body_id']
        
        pan_obj = sim.objects.get("pan")
        if pan_obj is None:
            raise RuntimeError("pan_obj not found in sim.objects; available keys: " + ", ".join(sim.objects.keys()))
        self._pan_uid = pan_obj['body_id']
        
        stove_obj = sim.objects.get("stove_block")
        if stove_obj is None:
            raise RuntimeError("stove_obj not found in sim.objects; available keys: " + ", ".join(sim.objects.keys()))
        self._stove_block_uid = stove_obj['body_id']
        

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        # The dynamic plan depends on perception; defer heavy lifting to execute().
        LOGGER.info("Planning completed (perception-driven execution).")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        if not self.camera or not self.controller:
            raise RuntimeError("Task not set up")
        bowl_entry = sim.objects["bowl"]
        
        # --- INITIAL LOOK MOTION ---
        # Move the arm to a "Look" pose above the expected bowl position
        # This is critical for Eye-in-Hand so the camera sees the target initially.
        bowl_pose_initial: Pose6D = bowl_entry["pose"]
        look_pos = [bowl_pose_initial.x, bowl_pose_initial.y, bowl_pose_initial.z + 0.45]
        # Look straight down (approximate)
        look_quat = p.getQuaternionFromEuler([math.pi, 0, 0]) 
        
        LOGGER.info("Moving to initial observation pose...")
        self.controller.move_waypoint(look_pos, look_quat, max_steps=180)
        # ---------------------------

        detection = self._run_perception(cfg, sim, bowl_entry)
        if detection is None:
            LOGGER.error("Perception failed after retries; aborting task")
            self._metrics = {"transfer_ratio": 0.0, "in_pan": 0, "total": 0}
            return False

        success = self._perform_pour_sequence(sim, cfg, detection)
        self._metrics = sim.count_particles_in_pan()
        return success

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return dict(self._metrics)

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _run_perception(
        self,
        cfg: PourTaskConfig,
        sim: RobotChefSimulation,
        bowl_entry: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        assert self.camera is not None
        attempt = 0
        last_error: Optional[str] = None
        while attempt < 3:
            attempt += 1
            
            # Camera captures from current wrist pose
            rgb, depth, K = self.camera.get_rgbd()
            depth_valid = depth[depth > 0.0]
            depth_range = (
                float(depth_valid.min()) if depth_valid.size else 0.0,
                float(depth_valid.max()) if depth_valid.size else 0.0,
            )
            LOGGER.info(
                "Using RGB-D rim detection (depth range %.3f m – %.3f m, attempt %d)",
                depth_range[0],
                depth_range[1],
                attempt,
            )
            seg = {
                "client_id": sim.client_id,
                "camera_from_world": self.camera.camera_from_world,
                "world_from_camera": self.camera.world_from_camera,
                "perception_cfg": cfg.perception,
                "bowl_properties": bowl_entry.get("properties", {}),
            }
            try:
                detection = detect_bowl_rim(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    seg=seg,
                    bowl_uid=bowl_entry["body_id"],
                )
            except Exception as exc:
                last_error = str(exc)
                LOGGER.warning("Rim detection attempt %d failed: %s", attempt, exc)
                continue

            overlay = rgb.copy()
            for uv in detection["features_px"]["uv_pair"]:
                u, v = int(round(uv[0])), int(round(uv[1]))
                if 0 <= v < overlay.shape[0] and 0 <= u < overlay.shape[1]:
                    overlay[max(0, v - 2) : min(overlay.shape[0], v + 3), max(0, u - 2) : min(overlay.shape[1], u + 3)] = [255, 64, 64]
            rim_pts = detection["rim_pts_3d"]
            LOGGER.info(
                "Detected rim: pts>=200, candidates>=8 (found %d pts, %d candidates)",
                len(rim_pts),
                len(detection["grasp_candidates"]),
            )
            _write_png(Path(f"attempt{attempt}_overlay.png"), overlay)
            if len(detection["grasp_candidates"]) == 0:
                last_error = "No grasp candidates"
                continue
            return detection
        if last_error:
            LOGGER.error("Perception retries exhausted: %s", last_error)
        return None

    def _perform_pour_sequence(
        self,
        sim: RobotChefSimulation,
        cfg: PourTaskConfig,
        detection: Dict[str, object],
    ) -> bool:
        assert self.camera and self.controller
        candidates = sorted(detection["grasp_candidates"], key=lambda c: c.get("quality", 0.0), reverse=True)
        K = self.camera.intrinsics

        for idx, candidate in enumerate(candidates[:3]):
            LOGGER.info("Attempting grasp candidate %d with quality %.2f", idx + 1, candidate["quality"])
            position = np.array(candidate["pose_world"]["position"], dtype=float)
            quat = np.array(candidate["pose_world"]["quaternion"], dtype=float)
            clearance = cfg.perception.grasp_clearance_m

            pregrasp = position.copy()
            pregrasp[2] += clearance + 0.08
            if not self.controller.move_waypoint(pregrasp, quat):
                LOGGER.warning("Failed to reach pre-grasp waypoint, trying next candidate")
                continue

            approach = position.copy()
            approach[2] += clearance
            if not self.controller.move_waypoint(approach, quat):
                LOGGER.warning("Failed to reach approach waypoint, trying next candidate")
                continue

            target_uv = detection["features_px"]["uv_pair"]
            target_Z = detection["features_px"]["Z_pair"]
            finger_links = sim.right_arm.finger_joints

            def get_features():
                # No manual aiming; camera is mounted and moves with arm
                rgb, depth, K = self.camera.get_rgbd()
                det = detect_bowl_rim(
                    rgb=rgb,
                    depth=depth,
                    K=K,
                    seg=None,
                    bowl_uid=self._bowl_uid,   # <- guaranteed non-None now
                )
                if not det or "features_px" not in det:
                    return None, None
                uv_pair = det["features_px"].get("uv_pair")
                Z_pair  = det["features_px"].get("Z_pair")
                if uv_pair is None or Z_pair is None or len(uv_pair) != 2 or len(Z_pair) != 2:
                    return None, None
                (u1, v1), (u2, v2) = uv_pair
                uv = np.array([float(u1), float(v1), float(u2), float(v2)], dtype=float)
                Z  = np.array([float(Z_pair[0]), float(Z_pair[1])], dtype=float)
                return uv, Z


            self.controller.open_gripper()
            refinement_ok = self.controller.refine_to_features_ibvs(
                get_features=get_features,
                target_uv=target_uv,
                target_Z=target_Z,
                pixel_tol=cfg.perception.grasp_clearance_m * 100.0,
                depth_tol=0.01,
                max_time_s=3.0,
                gain=0.4,
                max_joint_vel=0.4,
            )
            if not refinement_ok:
                LOGGER.warning("IBVS refinement failed for candidate %d", idx + 1)
                continue

            # Descend slightly to engage the rim.
            grasp_pose = position.copy()
            grasp_pose[2] += max(0.0, cfg.perception.rim_thickness_m * 0.5)
            if not self.controller.move_waypoint(grasp_pose, quat):
                LOGGER.warning("Could not descend to grasp pose, retrying with next candidate")
                continue

            self.controller.close_gripper()
            sim.step_simulation(steps=120)

            # Lift bowl.
            lift_pose = grasp_pose.copy()
            lift_pose[2] += 0.12
            self.controller.move_waypoint(lift_pose, quat)

            # Move above pan.
            pan_pose_world = sim.objects["pan"]["pose"]
            pan_quat = _pose_to_quaternion(pan_pose_world)
            above_pan = np.array([pan_pose_world.x, pan_pose_world.y, pan_pose_world.z + 0.25])
            self.controller.move_waypoint(above_pan, pan_quat)

            # Pour tilt.
            pan_pour_pose = _apply_world_yaw(cfg.pan_pour_pose, cfg.scene.world_yaw_deg)
            pour_quat = p.getQuaternionFromEuler([pan_pour_pose.roll, pan_pour_pose.pitch, pan_pour_pose.yaw])
            pour_pos = np.array([pan_pour_pose.x, pan_pour_pose.y, pan_pour_pose.z])
            self.controller.move_waypoint(pour_pos, pour_quat)
            hold_steps = max(1, int(cfg.hold_sec / sim.dt))
            for _ in range(hold_steps):
                sim.step_simulation(steps=1)

            # Return towards bowl and place.
            above_bowl = np.array([lift_pose[0], lift_pose[1], lift_pose[2]])
            self.controller.move_waypoint(above_bowl, quat)
            place_pose = grasp_pose.copy()
            place_pose[2] += clearance + 0.03
            self.controller.move_waypoint(place_pose, quat)
            self.controller.open_gripper()
            sim.step_simulation(steps=60)
            return True

        LOGGER.error("All grasp candidates failed")
        return False