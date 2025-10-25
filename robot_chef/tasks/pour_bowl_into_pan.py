"""Perception-in-the-loop pipeline for pouring with a fixed external camera."""

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


# --------------------------------------------------------------------------- #
# Small geometry helpers

def _matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    """Convert a rotation matrix into a quaternion (x, y, z, w)."""
    m = R
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    else:
        
        if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat)
    return tuple(float(v) for v in quat)


def _camera_look_at(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    """Return world rotation matrix that points camera +Z axis at target."""
    forward = target_pos - camera_pos
    norm = np.linalg.norm(forward)
    if norm < 1e-8:
        forward = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        forward /= norm
    z_cam = forward
    up_guess = np.array([0.0, 1.0, 0.0], dtype=float)
    x_cam = np.cross(up_guess, z_cam)
    if np.linalg.norm(x_cam) < 1e-6:
        up_guess = np.array([1.0, 0.0, 0.0], dtype=float)
        x_cam = np.cross(up_guess, z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return np.column_stack([x_cam, y_cam, z_cam])


def _project_world_points(points: np.ndarray, camera_from_world: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D world points into pixels using the camera extrinsics."""
    pts_world = np.asarray(points, dtype=float)
    if pts_world.ndim != 2 or pts_world.shape[1] != 3:
        raise ValueError("points must be shaped (N, 3)")
    homog = np.concatenate([pts_world, np.ones((pts_world.shape[0], 1), dtype=float)], axis=1)
    cam = (camera_from_world @ homog.T).T
    z = cam[:, 2]
    if np.any(z <= 1e-4):
        return None, None
    x = cam[:, 0] / z
    y = cam[:, 1] / z
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = fx * x + cx
    v = fy * y + cy
    uv = np.stack([u, v], axis=1)
    return uv.astype(float), z.astype(float)


def _write_png(path: Path, image: np.ndarray) -> None:
    """Write an RGB image to disk without external dependencies."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img = image
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    h, w, c = img.shape
    if c != 3:
        raise ValueError("Overlay writer expects RGB images")
    raw = bytearray()
    for row in img:
        raw.append(0)
        raw.extend(row.tobytes())
    compressed = zlib.compress(bytes(raw), level=6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    with path.open("wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
        fh.write(chunk(b"IHDR", ihdr))
        fh.write(chunk(b"IDAT", compressed))
        fh.write(chunk(b"IEND", b""))


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


# --------------------------------------------------------------------------- #
# Task implementation

class PourBowlIntoPanAndReturn:
    """Perception-driven pouring task with a fixed external camera."""

    def __init__(self) -> None:
        self.sim: Optional[RobotChefSimulation] = None
        self.cfg: Optional[PourTaskConfig] = None

        self.camera: Optional[Camera] = None
        self.controller: Optional[VisionRefineController] = None
        self.active_arm_name: str = "right"

        self._T_eef_cam = np.eye(4, dtype=float)
        self._T_cam_eef = np.eye(4, dtype=float)
        
        self._feature_points_world: Optional[np.ndarray] = None
        self._target_uv: Optional[np.ndarray] = None
        self._target_Z: Optional[np.ndarray] = None

        self._bowl_uid: Optional[int] = None
        self._pan_uid: Optional[int] = None

        self._metrics: Dict[str, float] = {"transfer_ratio": 0.0, "in_pan": 0, "total": 0}

    # ------------------------------------------------------------------ #
    # Lifecycle

    def setup(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> None:
        self.sim = sim
        self.cfg = cfg

        view_cfg = cfg.camera.get_active_view()
        noise_cfg = cfg.camera.noise

        # --- Setup a FIXED camera ---
        LOGGER.info("Setting up fixed camera view: '%s'", cfg.camera.active_view)
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
        
        # NOTE: We now aim the camera in _run_perception() using the *snapped* pose

        # Use the "left" arm for consistency
        active = sim.left_arm
        self.active_arm_name = "left"
        self._T_eef_cam = np.eye(4, dtype=float)
        self._T_cam_eef = np.eye(4, dtype=float)

        self.controller = VisionRefineController(
            client_id=sim.client_id,
            arm_id=active.body_id,
            ee_link=active.ee_link,
            arm_joints=active.joint_indices,
            dt=sim.dt,
            camera=self.camera,
            handeye_T_cam_in_eef=self._T_cam_eef, # Pass identity, it won't be used
            gripper_open=lambda width=0.08: sim.gripper_open(width=width, arm=self.active_arm_name),
            gripper_close=lambda force=60.0: sim.gripper_close(force=force, arm=self.active_arm_name),
        )

        sim.gripper_open(arm=self.active_arm_name)
        if cfg.particles > 0:
            sim.spawn_rice_particles(cfg.particles, seed=cfg.seed)

        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        pan_entry = sim.objects.get("pan")
        if bowl_entry is None or pan_entry is None:
            raise RuntimeError("Missing bowl or pan objects in simulation.")
        self._bowl_uid = int(bowl_entry["body_id"])
        self._pan_uid = int(pan_entry["body_id"])

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning completed (fixed camera, no IBVS).")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        if not self.camera or not self.controller:
            raise RuntimeError("Task not set up.")
        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        if bowl_entry is None:
            LOGGER.error("Bowl not found in simulation objects.")
            return False

        # Run perception ONCE from the fixed location
        detection = self._run_perception(sim, cfg, bowl_entry)
        if detection is None:
            LOGGER.error("Perception failed from fixed camera pose.")
            self._metrics = {"transfer_ratio": 0.0, "in_pan": 0, "total": 0}
            return False

        success = self._perform_pour_sequence(sim, cfg, detection)
        self._metrics = sim.count_particles_in_pan()
        return success

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return dict(self._metrics)

    # ------------------------------------------------------------------ #
    # Perception

    def _run_perception(
        self,
        sim: RobotChefSimulation,
        cfg: PourTaskConfig,
        bowl_entry: Dict[str, object],
    ) -> Optional[Dict[str, object]]:
        """
        Captures a single detection from the fixed camera pose.
        Aims camera at the correct, snapped pose first.
        """
        assert self.camera is not None
        assert self.controller is not None

        # Get the (now correct) snapped pose from the sim object
        bowl_pose: Pose6D = bowl_entry["pose"]
        if bowl_pose is None:
            LOGGER.error("Bowl pose is missing from simulation object entry.")
            return None
            
        # Aim the camera at the correct, snapped pose RIGHT before capture
        self.camera.aim_at_world(bowl_pose.position)
        LOGGER.info(
            "Aimed camera at final bowl pose: (%.3f, %.3f, %.3f)",
            bowl_pose.x, bowl_pose.y, bowl_pose.z
        )

        LOGGER.info("Running perception from fixed camera.")
        detection = self._capture_detection(sim, cfg, bowl_entry, bowl_pose, attempt=1)
        if detection is not None:
            return detection
        
        LOGGER.error("Failed to detect bowl rim from fixed camera.")
        return None

    def _capture_detection(
        self,
        sim: RobotChefSimulation,
        cfg: PourTaskConfig,
        bowl_entry: Dict[str, object],
        bowl_pose: Pose6D,
        attempt: int,
    ) -> Optional[Dict[str, object]]:
        assert self.camera is not None
        rgb, depth, K, seg_mask = self.camera.get_rgbd(with_segmentation=True)
        depth_valid = depth[depth > 0.0]
        dmin = float(depth_valid.min()) if depth_valid.size else 0.0
        dmax = float(depth_valid.max()) if depth_valid.size else 0.0

        cam_pose = self.camera.world_from_camera[:3, 3]
        LOGGER.info(
            "Camera at (%.3f, %.3f, %.3f) looking at bowl (%.3f, %.3f, %.3f)",
            cam_pose[0],
            cam_pose[1],
            cam_pose[2],
            bowl_pose.x,
            bowl_pose.y,
            bowl_pose.z,
        )

        LOGGER.info(
            "Using RGB-D rim detection (depth range %.3f m – %.3f m, attempt %d)",
            dmin,
            dmax,
            attempt,
        )
        seg = {
            "client_id": sim.client_id,
            "camera_from_world": self.camera.camera_from_world,
            "world_from_camera": self.camera.world_from_camera,
            "perception_cfg": cfg.perception,
            "bowl_properties": bowl_entry.get("properties", {}),
            "bowl_pose": bowl_pose,
            "segmentation": seg_mask,
        }
        try:
            detection = detect_bowl_rim(
                rgb=rgb,
                depth=depth,
                K=K,
                seg=seg,
                bowl_uid=int(bowl_entry["body_id"]),
            )
        except Exception as exc:
            LOGGER.warning("Rim detection attempt %d failed: %s", attempt, exc)
            return None

        candidates = detection.get("grasp_candidates", [])
        rim_pts = np.asarray(detection.get("rim_pts_3d", []), dtype=float)
        if rim_pts.shape[0] < 2 or len(candidates) == 0:
            LOGGER.warning("No valid rim detection on attempt %d", attempt)
            return None

        overlay = rgb.copy()
        for uv in detection["features_px"]["uv_pair"]:
            u, v = int(round(uv[0])), int(round(uv[1]))
            if 0 <= v < overlay.shape[0] and 0 <= u < overlay.shape[1]:
                overlay[max(0, v - 2) : min(overlay.shape[0], v + 3), max(0, u - 2) : min(overlay.shape[1], u + 3)] = [255, 64, 64]
        
        # Save overlay to the CWD
        out_path = Path(f"attempt{attempt:02d}_overlay.png")
        _write_png(out_path, overlay)
        LOGGER.info("Saved detection overlay to %s", out_path.absolute())


        LOGGER.info(
            "Detected rim: pts=%d, candidates=%d",
            rim_pts.shape[0],
            len(candidates),
        )
        
        feature_pts_world = np.asarray(detection["features_px"]["points_world"], dtype=float)
        self._feature_points_world = feature_pts_world
        self._target_uv = np.asarray(detection["features_px"]["uv_pair"], dtype=float)
        self._target_Z = np.asarray(detection["features_px"]["Z_pair"], dtype=float)
        return detection

    # ------------------------------------------------------------------ #
    # Execution

    def _perform_pour_sequence(
        self,
        sim: RobotChefSimulation,
        cfg: PourTaskConfig,
        detection: Dict[str, object],
    ) -> bool:
        assert self.controller is not None
        assert self.camera is not None

        candidates = sorted(detection["grasp_candidates"], key=lambda c: c.get("quality", 0.0), reverse=True)
        if not candidates:
            LOGGER.error("No grasp candidates found in detection.")
            return False

        # Define a threshold for successful grasp.
        # The bowl wall is ~8mm thick. 5mm is a safe threshold.
        MIN_GRASP_WIDTH = 0.005 # 5mm

        for idx, candidate in enumerate(candidates[:3]):
            LOGGER.info("Attempting grasp candidate %d with quality %.2f", idx + 1, candidate["quality"])
            position = np.array(candidate["pose_world"]["position"], dtype=float)
            quat = np.array(candidate["pose_world"]["quaternion"], dtype=float)
            clearance = cfg.perception.grasp_clearance_m

            # 1. Move to pre-grasp (Offset in World Z)
            pregrasp = position.copy()
            pregrasp[2] += clearance + 0.08 # Simple World Z offset
            
            self.controller.open_gripper()
            if not self.controller.move_waypoint(pregrasp, quat):
                LOGGER.warning("Failed to reach pre-grasp waypoint.")
                continue

            # 2. Move to approach (Offset in World Z)
            approach = position.copy()
            approach[2] += clearance # Simple World Z offset
            if not self.controller.move_waypoint(approach, quat):
                LOGGER.warning("Failed to reach approach waypoint.")
                continue

            # 3. --- IBVS is Skipped ---
            LOGGER.info("Skipping IBVS refinement (using fixed camera).")

            # 4. Move to final grasp pose (Offset in World Z)
            grasp_pose = position.copy()
            grasp_pose[2] += max(0.0, getattr(cfg.perception, "rim_thickness_m", 0.006) * 0.5)
            if not self.controller.move_waypoint(grasp_pose, quat):
                LOGGER.warning("Could not descend to final grasp pose.")
                continue

            # 5. Grasp and VERIFY
            self.controller.close_gripper()
            sim.step_simulation(steps=120) # Wait for grasp to settle

            # --- Grasp verification (Force Closure Check) ---
            current_width = sim.get_gripper_width(arm=self.active_arm_name)
            
            if current_width < MIN_GRASP_WIDTH:
                LOGGER.warning(
                    "Grasp verification FAILED for candidate %d (width %.4f m < %.4f m). Retrying.",
                    idx + 1, current_width, MIN_GRASP_WIDTH
                )
                # Open gripper and back away before trying next candidate
                self.controller.open_gripper()
                self.controller.move_waypoint(pregrasp, quat) # Move back to pregrasp
                continue # Try next candidate
            
            LOGGER.info(
                "Grasp verification SUCCEEDED for candidate %d (width %.4f m).",
                idx + 1, current_width
            )
            # --- END Grasp verification ---

            # 6. Lift (Offset in World Z)
            lift_pose = grasp_pose.copy()
            lift_pose[2] += 0.35 # Simple World Z offset
            self.controller.move_waypoint(lift_pose, quat)

            # 7. Move to Pan and Pour
            
            # Get pan pose from simulation (it's already yawed)
            pan_pose_sim: Pose6D = sim.objects["pan"]["pose"]
            pan_quat = _pose_to_quaternion(pan_pose_sim)
            above_pan = np.array([pan_pose_sim.x, pan_pose_sim.y, pan_pose_sim.z + 0.25], dtype=float)
            self.controller.move_waypoint(above_pan, pan_quat)

            # Get pour pose from config (MUST be yawed)
            pan_pour_pose = _apply_world_yaw(cfg.pan_pour_pose, cfg.scene.world_yaw_deg)
            pour_quat = p.getQuaternionFromEuler([pan_pour_pose.roll, pan_pour_pose.pitch, pan_pour_pose.yaw])
            pour_pos = np.array([pan_pour_pose.x, pan_pour_pose.y, pan_pour_pose.z], dtype=float)
            self.controller.move_waypoint(pour_pos, np.array(pour_quat, dtype=float))

            hold_steps = max(1, int(cfg.hold_sec / sim.dt))
            for _ in range(hold_steps):
                sim.step_simulation(steps=1)

            # 8. Return bowl
            self.controller.move_waypoint(lift_pose, quat) # Use the same lift pose
            
            place_pose = grasp_pose.copy()
            place_pose[2] += clearance + 0.03 # Simple World Z offset
            self.controller.move_waypoint(place_pose, quat)
            
            self.controller.open_gripper()
            sim.step_simulation(steps=60)
            
            # Lift arm up after placing
            lift_pose[2] += 0.3 # Even higher
            self.controller.move_waypoint(lift_pose, quat)
            
            LOGGER.info("Grasp and pour sequence successful.")
            return True # <-- Note: We return True inside the loop on success

        # If we exit the loop, all candidates failed verification
        LOGGER.error("All grasp candidates failed (or failed verification).")
        return False


__all__ = ["PourBowlIntoPanAndReturn"]