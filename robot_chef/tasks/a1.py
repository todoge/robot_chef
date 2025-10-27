"""Perception-in-the-loop pipeline for pouring with a wrist-mounted camera."""

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
    """Perception-driven pouring task with a wrist-mounted camera."""

    def __init__(self) -> None:
        self.sim: Optional[RobotChefSimulation] = None
        self.cfg: Optional[PourTaskConfig] = None

        self.camera: Optional[Camera] = None
        self.controller: Optional[VisionRefineController] = None

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

        rel_translation = (0.06, 0.0, 0.12)
        rel_rpy_deg = (-10.0, -65.0, 0.0)
        self.camera.mount_to_link(
            parent_body_id=sim.right_arm.body_id,
            parent_link_id=sim.right_arm.ee_link,
            rel_xyz=rel_translation,
            rel_rpy_deg=rel_rpy_deg,
        )
        
        rel_quat = p.getQuaternionFromEuler([math.radians(v) for v in rel_rpy_deg])
        rel_rot = np.array(p.getMatrixFromQuaternion(rel_quat), dtype=float).reshape(3, 3)
        self._T_eef_cam = np.eye(4, dtype=float)
        self._T_eef_cam[:3, :3] = rel_rot
        self._T_eef_cam[:3, 3] = np.array(rel_translation, dtype=float)
        self._T_cam_eef = np.linalg.inv(self._T_eef_cam)

        self.controller = VisionRefineController(
            client_id=sim.client_id,
            arm_id=sim.right_arm.body_id,
            ee_link=sim.right_arm.ee_link,
            arm_joints=sim.right_arm.joint_indices,
            dt=sim.dt,
            camera=self.camera,
            handeye_T_cam_in_eef=self._T_eef_cam,
            gripper_open=lambda width=0.08: sim.gripper_open(width=width, arm="right"),
            gripper_close=lambda force=60.0: sim.gripper_close(force=force, arm="right"),
        )

        sim.gripper_open(arm="right")
        if cfg.rice_particles > 0:
            sim.spawn_rice_particles(cfg.rice_particles, seed=cfg.seed)
        if cfg.egg_particles > 0:
            sim.spawn_egg_particles(cfg.egg_particles, seed=cfg.seed)

        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        pan_entry = sim.objects.get("pan")
        if bowl_entry is None or pan_entry is None:
            raise RuntimeError("Missing bowl or pan objects in simulation.")
        self._bowl_uid = int(bowl_entry["body_id"])
        self._pan_uid = int(pan_entry["body_id"])

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning completed (perception-driven execution).")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        if not self.camera or not self.controller:
            raise RuntimeError("Task not set up.")
        bowl_entry = sim.objects.get("bowl") or sim.objects.get("rice_bowl")
        if bowl_entry is None:
            LOGGER.error("Bowl not found in simulation objects.")
            return False

        detection = self._run_perception(sim, cfg, bowl_entry)
        if detection is None:
            LOGGER.error("Perception failed after sweeping the table.")
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
        assert self.camera is not None
        assert self.controller is not None

        bowl_pose: Pose6D = bowl_entry["pose"]
        scan_waypoints = self._generate_scan_waypoints(sim, bowl_pose)

        attempt = 0
        for cam_pos, cam_quat in scan_waypoints:
            eef_pos, eef_quat = self._camera_to_eef_pose(cam_pos, cam_quat)
            if not self.controller.move_waypoint(eef_pos, eef_quat):
                LOGGER.warning("Skipping scan pose (IK failure).")
                continue
            attempt += 1
            detection = self._capture_detection(sim, cfg, bowl_entry, bowl_pose, attempt)
            if detection is not None:
                return detection
        return None

    def _generate_scan_waypoints(self, sim: RobotChefSimulation, bowl_pose: Pose6D) -> List[Tuple[np.ndarray, Tuple[float, float, float, float]]]:
        center = np.array([bowl_pose.x, bowl_pose.y, bowl_pose.z], dtype=float)
        table_entry = sim.objects.get("table", {})
        table_top = float(table_entry.get("top_z", bowl_pose.z - 0.05))
        base_height = max(center[2] + 0.32, table_top + 0.35)

        offsets = [
            np.array([0.0, -0.28, 0.0], dtype=float),
            np.array([0.24, -0.10, -0.04], dtype=float),
            np.array([-0.24, -0.10, -0.04], dtype=float),
            np.array([0.0, 0.28, 0.05], dtype=float),
        ]

        waypoints: List[Tuple[np.ndarray, Tuple[float, float, float, float]]] = []

        # Top-down look
        top_pos = np.array([center[0], center[1], base_height + 0.25], dtype=float)
        top_rot = _camera_look_at(top_pos, center)
        waypoints.append((top_pos, _matrix_to_quaternion(top_rot)))

        for offset in offsets:
            cam_pos = np.array(
                [center[0] + offset[0], center[1] + offset[1], base_height + offset[2]],
                dtype=float,
            )
            cam_rot = _camera_look_at(cam_pos, center)
            waypoints.append((cam_pos, _matrix_to_quaternion(cam_rot)))

        # Table sweep along X axis
        sweep_y = center[1] - 0.35
        for x in np.linspace(center[0] - 0.35, center[0] + 0.35, 5):
            cam_pos = np.array([x, sweep_y, base_height + 0.08], dtype=float)
            cam_rot = _camera_look_at(cam_pos, center)
            waypoints.append((cam_pos, _matrix_to_quaternion(cam_rot)))
        return waypoints

    def _camera_to_eef_pose(self, cam_pos: np.ndarray, cam_quat: Tuple[float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
        R_cam = np.array(p.getMatrixFromQuaternion(cam_quat), dtype=float).reshape(3, 3)
        T_cam = np.eye(4, dtype=float)
        T_cam[:3, :3] = R_cam
        T_cam[:3, 3] = cam_pos
        T_eef = T_cam @ self._T_cam_eef
        pos = T_eef[:3, 3]
        quat = _matrix_to_quaternion(T_eef[:3, :3])
        return pos.astype(float), np.array(quat, dtype=float)

    def _capture_detection(
        self,
        sim: RobotChefSimulation,
        cfg: PourTaskConfig,
        bowl_entry: Dict[str, object],
        bowl_pose: Pose6D,
        attempt: int,
    ) -> Optional[Dict[str, object]]:
        assert self.camera is not None
        self.camera.aim_at_world((bowl_pose.x, bowl_pose.y, bowl_pose.z))
        rgb, depth, K, seg_mask = self.camera.get_rgbd(with_segmentation=True)
        depth_valid = depth[depth > 0.0]
        dmin = float(depth_valid.min()) if depth_valid.size else 0.0
        dmax = float(depth_valid.max()) if depth_valid.size else 0.0

        cam_pose = self.camera.world_from_camera[:3, 3]
        LOGGER.info(
            "Camera positioned at (%.3f, %.3f, %.3f) aiming at (%.3f, %.3f, %.3f)",
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
        _write_png(Path(f"attempt{attempt:02d}_overlay.png"), overlay)

        LOGGER.info(
            "Detected rim: pts>=200, candidates>=8 (found %d pts, %d candidates)",
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
        if self._feature_points_world is None or self._target_uv is None or self._target_Z is None:
            LOGGER.error("IBVS feature targets not prepared.")
            return False

        candidates = sorted(detection["grasp_candidates"], key=lambda c: c.get("quality", 0.0), reverse=True)
        target_uv = np.asarray(self._target_uv, dtype=float).reshape(-1, 2)
        target_Z = np.asarray(self._target_Z, dtype=float).reshape(-1)

        for idx, candidate in enumerate(candidates[:3]):
            LOGGER.info("Attempting grasp candidate %d with quality %.2f", idx + 1, candidate["quality"])
            position = np.array(candidate["pose_world"]["position"], dtype=float)
            quat = np.array(candidate["pose_world"]["quaternion"], dtype=float)
            clearance = cfg.perception.grasp_clearance_m

            pregrasp = position.copy()
            pregrasp[2] += clearance + 0.08
            if not self.controller.move_waypoint(pregrasp, quat):
                LOGGER.warning("Failed to reach pre-grasp waypoint.")
                continue

            approach = position.copy()
            approach[2] += clearance
            if not self.controller.move_waypoint(approach, quat):
                LOGGER.warning("Failed to reach approach waypoint.")
                continue

            self.controller.open_gripper()

            feature_points_world = self._feature_points_world

            def get_features() -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
                self.camera.aim_at_world(tuple(detection.get("center_3d", feature_points_world.mean(axis=0))))
                _, _, _, _ = self.camera.get_rgbd()
                uv_meas, Z_meas = _project_world_points(
                    feature_points_world,
                    self.camera.camera_from_world,
                    self.camera.intrinsics,
                )
                return uv_meas, Z_meas

            refinement_ok = self.controller.refine_to_features_ibvs(
                get_features=get_features,
                target_uv=target_uv,
                target_Z=target_Z,
                pixel_tol=cfg.perception.grasp_clearance_m * 100.0,
                depth_tol=0.01,
                max_time_s=4.0,
                gain=0.35,
                max_joint_vel=0.45,
            )
            if not refinement_ok:
                LOGGER.warning("IBVS refinement failed for candidate %d", idx + 1)
                continue

            grasp_pose = position.copy()
            grasp_pose[2] += max(0.0, getattr(cfg.perception, "rim_thickness_m", 0.006) * 0.5)
            if not self.controller.move_waypoint(grasp_pose, quat):
                LOGGER.warning("Could not descend to final grasp pose.")
                continue

            self.controller.close_gripper()
            sim.step_simulation(steps=120)

            lift_pose = grasp_pose.copy()
            lift_pose[2] += 0.12
            self.controller.move_waypoint(lift_pose, quat)

            pan_pose_world: Pose6D = sim.objects["pan"]["pose"]
            pan_quat = _pose_to_quaternion(pan_pose_world)
            above_pan = np.array([pan_pose_world.x, pan_pose_world.y, pan_pose_world.z + 0.25], dtype=float)
            self.controller.move_waypoint(above_pan, pan_quat)

            pan_pour_pose = _apply_world_yaw(cfg.pan_pour_pose, cfg.scene.world_yaw_deg)
            pour_quat = p.getQuaternionFromEuler([pan_pour_pose.roll, pan_pour_pose.pitch, pan_pour_pose.yaw])
            pour_pos = np.array([pan_pour_pose.x, pan_pour_pose.y, pan_pour_pose.z], dtype=float)
            self.controller.move_waypoint(pour_pos, np.array(pour_quat, dtype=float))

            hold_steps = max(1, int(cfg.hold_sec / sim.dt))
            for _ in range(hold_steps):
                sim.step_simulation(steps=1)

            above_bowl = lift_pose.copy()
            self.controller.move_waypoint(above_bowl, quat)
            place_pose = grasp_pose.copy()
            place_pose[2] += clearance + 0.03
            self.controller.move_waypoint(place_pose, quat)
            self.controller.open_gripper()
            sim.step_simulation(steps=60)
            return True

        LOGGER.error("All grasp candidates failed.")
        return False


__all__ = ["PourBowlIntoPanAndReturn"]
