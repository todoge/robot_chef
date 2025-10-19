"""Perception-driven task: grasp bowl rim, pour into pan, replace bowl."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

from .. import config
from ..perception import Camera, configure_scene_context, detect_bowl_rim
from ..tasks import Task

LOGGER = logging.getLogger(__name__)


@dataclass
class GraspCandidate:
    position: np.ndarray  # world xyz
    quaternion: Tuple[float, float, float, float]  # xyzw
    quality: float
    span_m: float
    contacts: Dict[str, Tuple[float, float, float]]


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(vec)
    return vec if norm < 1e-9 else vec / norm


def _quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    q = np.array(quat_xyzw, dtype=float)
    m = np.array(p.getMatrixFromQuaternion(q.tolist()), dtype=float)
    return m.reshape(3, 3)

def _rotation_matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif (m00 > m11) and (m00 > m22):
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=float)
    quat /= np.linalg.norm(quat) + 1e-12
    return tuple(float(v) for v in quat)


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    quat = p.getQuaternionFromEuler((roll, pitch, yaw))
    return tuple(float(v) for v in quat)


def _apply_world_yaw(pos: Sequence[float], yaw_rad: float) -> np.ndarray:
    rot = np.array(
        [
            [math.cos(yaw_rad), -math.sin(yaw_rad), 0.0],
            [math.sin(yaw_rad), math.cos(yaw_rad), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return rot @ np.array(pos, dtype=float)


def _pose6d_to_world(cfg: config.PourTaskConfig, pose: config.Pose6D) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
    yaw_world = math.radians(cfg.scene.world_yaw_deg)
    pos = _apply_world_yaw((pose.x, pose.y, pose.z), yaw_world)
    quat = _euler_to_quat(pose.roll, pose.pitch, pose.yaw + yaw_world)
    return pos, quat


class PourBowlIntoPanAndReturn(Task):
    """Pour rice into pan using perception-based rim grasp."""

    name = "pour_bowl_into_pan"

    def __init__(self) -> None:
        self._camera: Optional[Camera] = None
        self._candidates: List[GraspCandidate] = []
        self._attempt_overlay_paths: List[Path] = []
        self._detection: Optional[Dict[str, object]] = None
        self._last_metrics: Dict[str, float] = {}
        self._cfg: Optional[config.PourTaskConfig] = None
        self._controller_available = False
        self._plan_success = False

    # ------------------------------------------------------------------ lifecycle
    def setup(self, world, cfg: config.PourTaskConfig) -> None:
        LOGGER.info("Setting up pour task with perception.")
        self._cfg = cfg
        self._controller_available = hasattr(world, "controller") and world.controller is not None

        bowl_pose = cfg.camera.get_view(cfg.camera.active_view)
        self._camera = Camera(
            client_id=world.client_id,
            view_name=cfg.camera.active_view,
            position=bowl_pose.xyz,
            rpy_rad=bowl_pose.rpy_rad,
            fov_deg=bowl_pose.fov_deg,
            resolution=bowl_pose.resolution,
            near=cfg.camera.near,
            far=cfg.camera.far,
            depth_noise_std=cfg.camera.noise.depth_std,
            depth_drop_prob=cfg.camera.noise.drop_prob,
            seed=cfg.seed,
        )

        # Initial aim: look at bowl center from configured view
        bowl_obj = world.objects.get("rice_bowl")
        if bowl_obj:
            target = np.array(bowl_obj.pose.position, dtype=float)
            self._camera.aim_at(target, distance=0.65, height_delta=0.10)

        Path("runs").mkdir(exist_ok=True)
        self._attempt_overlay_paths.clear()
        self._plan_success = False

    def plan(self, world, cfg: config.PourTaskConfig) -> bool:
        if not cfg.perception.enable:
            LOGGER.warning("Perception disabled; no fallback planner provided.")
            self._candidates = []
            return False

        bowl_body = getattr(world, "bowl_body", None)
        if bowl_body is None:
            LOGGER.error("Bowl body ID not available; cannot configure perception.")
            self._candidates = []
            return False

        aabb = p.getAABB(bowl_body, physicsClientId=world.client_id)
        bowl_min, bowl_max = np.array(aabb[0], dtype=float), np.array(aabb[1], dtype=float)

        self._candidates = []
        self._detection = None
        self._attempt_overlay_paths.clear()

        yaw_offsets = [0.0, 10.0, -10.0]
        for attempt_idx, yaw_delta in enumerate(yaw_offsets, start=1):
            if self._camera is None:
                break
            if attempt_idx > 1:
                self._camera.rotate_world_yaw(yaw_delta)

            rgb, depth, seg = self._camera.get_rgbd()
            configure_scene_context(
                bowl_aabb_min=tuple(bowl_min.tolist()),
                bowl_aabb_max=tuple(bowl_max.tolist()),
                world_from_cam=self._camera.world_from_cam,
                roi_margin=cfg.perception.bowl_roi_margin_m,
                sample_count=cfg.perception.rim_sample_count,
                grasp_clearance=cfg.perception.grasp_clearance_m,
                reachability_fn=None,
            )
            detection = detect_bowl_rim(
                rgb=rgb,
                depth=depth,
                K=self._camera.K,
                seg=seg,
                bowl_uid=bowl_body,
            )
            num_pts = detection["rim_pts_3d"].shape[0]
            num_candidates = len(detection["grasp_candidates"])
            LOGGER.info(
                "Perception attempt %d: rim_pts=%d candidates=%d",
                attempt_idx,
                num_pts,
                num_candidates,
            )

        filtered = self._filter_candidates(detection, cfg)
        overlay_path = self._write_overlay(
            attempt_idx,
            rgb,
            detection,
            highlight=filtered[0] if filtered else None,
        )
        self._attempt_overlay_paths.append(overlay_path)

        if filtered:
            self._detection = detection
            self._candidates = filtered
            LOGGER.info(
                "Using detection from attempt %d (rim_pts=%d, best_quality=%.3f)",
                    attempt_idx,
                    num_pts,
                    filtered[0].quality,
                )
      

        if not self._candidates:
            LOGGER.error("Perception failed to yield viable grasp candidates.")
        self._plan_success = bool(self._candidates)
        return self._plan_success

    def execute(self, world, cfg: config.PourTaskConfig) -> bool:
        if not self._controller_available or not self._candidates:
            LOGGER.error("Controller or grasp candidates missing; aborting execution.")
            self._plan_success = False
            return False

        controller = world.controller
        grasp_clearance = cfg.perception.grasp_clearance_m
        max_candidates = min(3, len(self._candidates))

        for idx in range(max_candidates):
            cand = self._candidates[idx]
            LOGGER.info("Attempting grasp candidate %d quality=%.3f span=%.3f", idx, cand.quality, cand.span_m)
            waypoints = self._build_waypoints(world, cfg, cand, grasp_clearance, self._detection)
            controller.clear()
            for wp in waypoints:
                controller.add_waypoint(
                    wp["position"],
                    wp["quaternion"],
                    gripper=wp.get("gripper"),
                    dwell=wp.get("dwell", 0.0),
                )
            success = controller.execute_planned()
            if success:
                LOGGER.info("Controller executed candidate %d successfully.", idx)
                self._plan_success = True
                break
            LOGGER.error(
                "Controller execution failed for candidate %d (quality %.3f). Waypoints: %s",
                idx,
                cand.quality,
                waypoints,
            )

        if not self._plan_success:
            LOGGER.error("All candidate executions failed.")
        return self._plan_success

    def metrics(self, world) -> Dict[str, float]:
        total, in_pan, ratio = world.count_particles_in_pan()
        self._last_metrics = {
            "total": float(total),
            "in_pan": float(in_pan),
            "transfer_ratio": float(ratio),
        }
        return self._last_metrics

    # ------------------------------------------------------------------ helpers
    def _filter_candidates(
        self,
        detection: Dict[str, object],
        cfg: config.PourTaskConfig,
    ) -> List[GraspCandidate]:
        raw: List[dict] = detection.get("grasp_candidates", [])
        filtered: List[GraspCandidate] = []
        for entry in raw:
            quality = float(entry.get("quality", 0.0))
            if quality < cfg.perception.min_grasp_quality:
                continue
            pose = entry.get("pose_world", {})
            pos = np.array(pose.get("position", (0.0, 0.0, 0.0)), dtype=float)
            quat = tuple(float(v) for v in pose.get("quaternion", (0.0, 0.0, 0.0, 1.0)))
            span = float(entry.get("span_m", 0.02))
            contacts = {
                key: tuple(float(v) for v in value)
                for key, value in entry.get("contacts", {}).items()
            }
            filtered.append(GraspCandidate(position=pos, quaternion=quat, quality=quality, span_m=span, contacts=contacts))

        filtered.sort(key=lambda c: c.quality, reverse=True)
        return filtered

    def _build_waypoints(
        self,
        world,
        cfg: config.PourTaskConfig,
        cand: GraspCandidate,
        grasp_clearance: float,
        detection: Optional[Dict[str, object]],
    ) -> List[Dict[str, object]]:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
        rim_pos = cand.position
        center = (
            np.array(detection.get("center_3d"), dtype=float)
            if detection and "center_3d" in detection
            else rim_pos - np.array([0.0, 0.0, 0.05], dtype=float)
        )
        outward = _normalize(rim_pos - center)
        if np.linalg.norm(outward) < 1e-6:
            outward = np.array([1.0, 0.0, 0.0], dtype=float)
        tangent = _normalize(np.cross(up, outward))
        if np.linalg.norm(tangent) < 1e-6:
            yaw_guess = math.atan2(outward[1], outward[0]) + math.pi / 2.0
            grasp_quat = _euler_to_quat(math.pi, 0.0, yaw_guess)
        else:
            yaw = math.atan2(tangent[1], tangent[0])
            grasp_quat = _euler_to_quat(math.pi, 0.0, yaw)

        approach_dir = up
        pre_clearance = min(0.25, grasp_clearance + 0.15)
        approach_clearance = max(0.06, grasp_clearance + 0.04)

        pregrasp_pos = rim_pos + approach_dir * pre_clearance
        approach_pos = rim_pos + approach_dir * approach_clearance
        grasp_pos = rim_pos + approach_dir * 0.01
        bite_pos = rim_pos - approach_dir * 0.02

        pan_obj = world.objects.get("pan")
        if pan_obj:
            pan_pos = np.array(pan_obj.pose.position, dtype=float)
            pan_yaw = p.getEulerFromQuaternion(p.getQuaternionFromEuler(pan_obj.pose.orientation_rpy))[2]
        else:
            pan_pos = _apply_world_yaw((cfg.pan_pose.x, cfg.pan_pose.y, cfg.pan_pose.z), math.radians(cfg.scene.world_yaw_deg))
            pan_yaw = cfg.pan_pose.yaw + math.radians(cfg.scene.world_yaw_deg)

        above_pan = pan_pos + np.array([0.0, 0.0, 0.25], dtype=float)
        pan_quat = _euler_to_quat(cfg.pan_pose.roll, cfg.pan_pose.pitch, pan_yaw)
        pour_pos, pour_quat = _pose6d_to_world(cfg, cfg.pan_pour_pose)
        tilt_axis = np.array([1.0, 0.0, 0.0], dtype=float)
        tilt_quat = p.getQuaternionFromAxisAngle(
            tilt_axis.tolist(), math.radians(cfg.tilt_angle_deg)
        )
        tilt_quat = p.multiplyTransforms(
            [0, 0, 0],
            pour_quat,
            [0, 0, 0],
            tilt_quat,
        )[1]

        place_pos, place_quat = _pose6d_to_world(cfg, cfg.bowl_pose)
        place_pos = place_pos + np.array([0.0, 0.0, 0.03], dtype=float)
        retreat_pos = pregrasp_pos + up * 0.05

        return [
            {"position": tuple(pregrasp_pos), "quaternion": grasp_quat, "gripper": "open", "dwell": 0.0},
            {"position": tuple(approach_pos), "quaternion": grasp_quat, "gripper": "open", "dwell": 0.0},
            {"position": tuple(grasp_pos), "quaternion": grasp_quat, "gripper": "open", "dwell": 0.0},
            {"position": tuple(bite_pos), "quaternion": grasp_quat, "gripper": "close", "dwell": 0.2},
            {"position": tuple(pregrasp_pos), "quaternion": grasp_quat, "gripper": "close", "dwell": 0.0},
            {"position": tuple(above_pan), "quaternion": pan_quat, "gripper": "close", "dwell": 0.0},
            {"position": tuple(pour_pos), "quaternion": pour_quat, "gripper": "close", "dwell": 0.0},
            {"position": tuple(pour_pos), "quaternion": tuple(tilt_quat), "gripper": "close", "dwell": cfg.hold_sec},
            {"position": tuple(pour_pos), "quaternion": pour_quat, "gripper": "close", "dwell": 0.2},
            {"position": tuple(above_pan), "quaternion": pan_quat, "gripper": "close", "dwell": 0.0},
            {"position": tuple(place_pos), "quaternion": place_quat, "gripper": "close", "dwell": 0.0},
            {"position": tuple(place_pos), "quaternion": place_quat, "gripper": "open", "dwell": 0.2},
            {"position": tuple(retreat_pos), "quaternion": grasp_quat, "gripper": "open", "dwell": 0.0},
        ]

    def _write_overlay(
        self,
        attempt_idx: int,
        rgb: np.ndarray,
        detection: Dict[str, object],
        highlight: Optional[GraspCandidate] = None,
    ) -> Path:
        if self._camera is None:
            return Path()
        overlay = rgb.copy()
        rim_pts = detection["rim_pts_3d"]
        uv, _ = self._camera.project(rim_pts)
        for u, v in uv:
            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < self._camera.width and 0 <= vi < self._camera.height:
                overlay[max(vi - 1, 0) : min(vi + 2, overlay.shape[0]), max(ui - 1, 0) : min(ui + 2, overlay.shape[1])] = (
                    60,
                    60,
                    255,
                )
        if highlight is not None:
            cand_uv, _ = self._camera.project(np.asarray([highlight.position], dtype=float))
            u, v = cand_uv[0]
            ui = int(round(u))
            vi = int(round(v))
            if 0 <= ui < self._camera.width and 0 <= vi < self._camera.height:
                overlay[max(vi - 2, 0) : min(vi + 3, overlay.shape[0]), max(ui - 2, 0) : min(ui + 3, overlay.shape[1])] = (
                    20,
                    255,
                    20,
                )
        out_dir = Path("runs") / "images"
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"attempt{attempt_idx}_"
        self._camera.save_overlay(out_dir, overlay, prefix=prefix)
        return out_dir / f"{prefix}overlay.png"
