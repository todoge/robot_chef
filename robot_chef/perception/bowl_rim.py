"""Robust bowl rim detection using depth, segmentation, and analytic fallback."""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


def _matrix_to_quaternion(R: np.ndarray) -> Tuple[float, float, float, float]:
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


def _project_points(world_pts: np.ndarray, camera_from_world: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pts_h = np.concatenate([world_pts, np.ones((world_pts.shape[0], 1), dtype=float)], axis=1)
    cam_pts = (camera_from_world @ pts_h.T).T
    Z = cam_pts[:, 2]
    if np.any(Z <= 1e-6):
        return None, None
    x = cam_pts[:, 0] / Z
    y = cam_pts[:, 1] / Z
    u = K[0, 0] * x + K[0, 2]
    v = K[1, 1] * y + K[1, 2]
    uv = np.stack([u, v], axis=1)
    return uv, Z


def _analytic_rim(bowl_pose, bowl_props) -> Tuple[np.ndarray, np.ndarray, float]:
    inner_radius = float(bowl_props.get("inner_radius", 0.07))
    inner_height = float(bowl_props.get("inner_height", 0.05))
    pos = np.array([bowl_pose.x, bowl_pose.y, bowl_pose.z], dtype=float)
    quat = p.getQuaternionFromEuler(bowl_pose.orientation_rpy)
    R_wb = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
    circle_local = np.stack(
        [
            inner_radius * np.cos(angles),
            inner_radius * np.sin(angles),
            np.full_like(angles, inner_height),
        ],
        axis=1,
    )
    rim_world = (R_wb @ circle_local.T).T + pos
    center_world = pos + R_wb @ np.array([0.0, 0.0, inner_height], dtype=float)
    return rim_world, center_world, inner_radius


def detect_bowl_rim(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    seg: Optional[Dict[str, object]] = None,
    bowl_uid: Optional[int] = None,
) -> Dict[str, object]:
    if seg is None:
        raise ValueError("seg dictionary must be provided")
    if bowl_uid is None:
        raise ValueError("bowl_uid must be provided")

    camera_from_world = np.asarray(seg["camera_from_world"], dtype=float)
    world_from_camera = np.asarray(seg["world_from_camera"], dtype=float)
    perception_cfg = seg.get("perception_cfg", None)
    bowl_props = seg.get("bowl_properties", {})
    bowl_pose = seg.get("bowl_pose", None)
    
    LOGGER.debug("JUNSIANG DEBUG: bowl_pose = %s", bowl_pose)
    
    if bowl_pose is None:
        raise ValueError("seg must include 'bowl_pose'")
    seg_mask = seg.get("segmentation", None)

    inner_radius = float(bowl_props.get("inner_radius", 0.07))
    inner_height = float(bowl_props.get("inner_height", 0.05))
    rim_thickness = float(getattr(perception_cfg, "rim_thickness_m", 0.006) if perception_cfg else 0.006)

    rim_points_world: Optional[np.ndarray] = None
    center_world: Optional[np.ndarray] = None
    radius_est: float = inner_radius

    if seg_mask is not None:
        obj_ids = seg_mask.astype(np.int32) & ((1 << 24) - 1)
        bowl_pixels = obj_ids == int(bowl_uid)
        bowl_pixels &= depth > 0.0
        if np.count_nonzero(bowl_pixels) > 80:
            v_idx, u_idx = np.nonzero(bowl_pixels)
            z = depth[v_idx, u_idx]
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
            x = (u_idx - cx) * z / fx
            y = (v_idx - cy) * z / fy
            pts_cam = np.stack([x, y, z], axis=1)
            pts_cam_h = np.concatenate([pts_cam, np.ones((pts_cam.shape[0], 1), dtype=float)], axis=1)
            pts_world = (world_from_camera @ pts_cam_h.T).T[:, :3]

            bowl_pos = np.array([bowl_pose.x, bowl_pose.y, bowl_pose.z], dtype=float)
            R_wb = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(bowl_pose.orientation_rpy)), dtype=float).reshape(3, 3)
            R_bw = R_wb.T
            pts_local = (R_bw @ (pts_world - bowl_pos).T).T

            height_tol = max(0.015, rim_thickness * 3.0)
            radius_tol = max(0.02, rim_thickness * 6.0)
            radial = np.linalg.norm(pts_local[:, :2], axis=1)
            mask = (np.abs(pts_local[:, 2] - inner_height) <= height_tol) & (np.abs(radial - inner_radius) <= radius_tol)
            rim_local = pts_local[mask]

            if rim_local.shape[0] >= 40:
                XY = rim_local[:, :2]
                A = np.column_stack([2.0 * XY[:, 0], 2.0 * XY[:, 1], np.ones(XY.shape[0])])
                b_vec = np.sum(XY ** 2, axis=1)
                sol, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
                cx_local, cy_local, c = sol
                radius_est = math.sqrt(max(cx_local * cx_local + cy_local * cy_local + c, 1e-6))
                center_local = np.array([cx_local, cy_local, inner_height], dtype=float)
                angles = np.linspace(0.0, 2.0 * math.pi, 360, endpoint=False)
                circle_local = np.stack(
                    [
                        radius_est * np.cos(angles),
                        radius_est * np.sin(angles),
                        np.full_like(angles, inner_height),
                    ],
                    axis=1,
                )
                rim_points_world = (R_wb @ circle_local.T).T + bowl_pos
                center_world = bowl_pos + R_wb @ center_local

    if rim_points_world is None or center_world is None:
        rim_points_world, center_world, radius_est = _analytic_rim(bowl_pose, bowl_props)

    feature_angles = [0.0, math.pi]
    R_wb = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(bowl_pose.orientation_rpy)), dtype=float).reshape(3, 3)
    features_world = []
    for ang in feature_angles:
        local = np.array(
            [
                radius_est * math.cos(ang),
                radius_est * math.sin(ang),
                inner_height,
            ],
            dtype=float,
        )
        feat_world = np.array([bowl_pose.x, bowl_pose.y, bowl_pose.z], dtype=float) + R_wb @ local
        features_world.append(feat_world)
    features_world = np.asarray(features_world, dtype=float)
    uv_target, Z_target = _project_points(features_world, camera_from_world, K)
    if uv_target is None or Z_target is None or uv_target.shape[0] < 2:
        cx, cy = float(K[0, 2]), float(K[1, 2])
        uv_target = np.array([[cx, cy], [cx, cy]], dtype=float)
        Z_target = np.array([1.0, 1.0], dtype=float)

    grasp_candidates: List[Dict[str, object]] = []
    sample_count = int(getattr(perception_cfg, "rim_sample_count", 24) if perception_cfg else 24)
    bowl_origin = np.array([bowl_pose.x, bowl_pose.y, bowl_pose.z], dtype=float)
    
    for ang in np.linspace(0.0, 2.0 * math.pi, max(sample_count, 8), endpoint=False):
        radial_local = np.array([math.cos(ang), math.sin(ang), 0.0], dtype=float)
        tangent_local = np.array([-math.sin(ang), math.cos(ang), 0.0], dtype=float)

        R_wb = np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler(bowl_pose.orientation_rpy)), dtype=float).reshape(3, 3)
        grasp_local = np.array(
            [
                radius_est * radial_local[0],
                radius_est * radial_local[1],
                inner_height + rim_thickness, 
            ],
            dtype=float,
        )
        grasp_world = bowl_origin + R_wb @ grasp_local
        tangent_world = R_wb @ tangent_local

        z_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        x_axis = tangent_world
        if np.linalg.norm(x_axis) < 1e-6: continue
        x_axis[2] = 0.0
        if np.linalg.norm(x_axis) < 1e-6: continue 
        x_axis /= np.linalg.norm(x_axis)
        
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)

        R_world = np.column_stack([x_axis, y_axis, z_axis])
        quat = _matrix_to_quaternion(R_world)
        
        quality = float(np.clip(0.95 - 0.1 * abs(math.sin(ang * 2.0)), 0.0, 1.0))
        grasp_candidates.append(
            {
                "pose_world": {
                    "position": tuple(float(v) for v in grasp_world),
                    "quaternion": quat,
                },
                "quality": quality,
            }
        )

    result = {
        "rim_pts_3d": rim_points_world.astype(np.float32),
        "center_3d": center_world.astype(np.float32),
        "radius_m": float(radius_est),
        "grasp_candidates": grasp_candidates,
        "features_px": {
            "uv_pair": [(float(uv_target[i, 0]), float(uv_target[i, 1])) for i in range(uv_target.shape[0])],
            "Z_pair": [float(Z_target[i]) for i in range(Z_target.shape[0])],
            "points_world": features_world.astype(np.float32),
        },
    }
    return result


__all__ = ["detect_bowl_rim"]