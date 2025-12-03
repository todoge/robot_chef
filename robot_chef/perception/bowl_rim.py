"""Bowl rim detection utilities."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


def _matrix_to_quaternion(matrix: np.ndarray) -> Tuple[float, float, float, float]:
    m = matrix
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


def _project_points(world_points: np.ndarray, camera_from_world: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hom = np.concatenate([world_points, np.ones((world_points.shape[0], 1), dtype=float)], axis=1)
    cam = (camera_from_world @ hom.T).T
    z = cam[:, 2:3]
    z[z == 0.0] = 1e-6
    pixels = (cam[:, :2] / z) @ K[:2, :2].T + K[:2, 2]
    return pixels, cam[:, 2]


def detect_bowl_rim(
    rgb: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    seg: Optional[Dict[str, object]] = None,
    bowl_uid: Optional[int] = None,
) -> Dict[str, object]:
    """Detect the rim of the bowl via simple geometric projection and circle fitting."""
    if seg is None:
        seg = {}
    client_id = int(seg.get("client_id", 0))
    camera_from_world = np.asarray(seg.get("camera_from_world", np.eye(4)), dtype=float)
    world_from_camera = np.asarray(seg.get("world_from_camera", np.eye(4)), dtype=float)
    perception_cfg = seg.get("perception_cfg", None)
    bowl_props = seg.get("bowl_properties", {})
    roi_margin_m = float(getattr(perception_cfg, "bowl_roi_margin_m", 0.12) if perception_cfg else 0.12)
    rim_thickness = float(getattr(perception_cfg, "rim_thickness_m", 0.006) if perception_cfg else 0.006)
    sample_count = int(getattr(perception_cfg, "rim_sample_count", 24) if perception_cfg else 24)

    if bowl_uid is None:
        raise ValueError("detect_bowl_rim requires bowl_uid for projection guidance")

    bowl_pos, bowl_quat = p.getBasePositionAndOrientation(bowl_uid, physicsClientId=client_id)
    bowl_pos = np.asarray(bowl_pos, dtype=float)
    bowl_rot = np.array(p.getMatrixFromQuaternion(bowl_quat), dtype=float).reshape(3, 3)

    radius = float(bowl_props.get("inner_radius", 0.07))
    inner_height = float(bowl_props.get("inner_height", 0.05))
    rim_height_world = bowl_pos[2] + inner_height

    angles = np.linspace(0.0, 2.0 * math.pi, num=64, endpoint=False)
    rim_samples_local = np.stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.full_like(angles, inner_height),
        ],
        axis=1,
    )
    rim_samples_world = (bowl_rot @ rim_samples_local.T).T + bowl_pos
    px, depths = _project_points(rim_samples_world, camera_from_world, K)
    valid = depths > 0.0
    if not np.any(valid):
        raise RuntimeError("No rim projections fell within the camera view frustum")
    px = px[valid]
    min_px = np.floor(np.min(px, axis=0)).astype(int)
    max_px = np.ceil(np.max(px, axis=0)).astype(int)
    depth_guess = float(np.mean(depths[valid]))
    margin_px = int(max(5, roi_margin_m * K[0, 0] / max(depth_guess, 1e-3)))

    height, width = depth.shape
    u0 = max(0, min_px[0] - margin_px)
    u1 = min(width - 1, max_px[0] + margin_px)
    v0 = max(0, min_px[1] - margin_px)
    v1 = min(height - 1, max_px[1] + margin_px)
    if u0 >= u1 or v0 >= v1:
        u0, v0, u1, v1 = 0, 0, width - 1, height - 1

    ys = np.arange(v0, v1 + 1)
    xs = np.arange(u0, u1 + 1)
    xv, yv = np.meshgrid(xs, ys)
    depth_roi = depth[v0 : v1 + 1, u0 : u1 + 1]
    valid_depth = depth_roi > 0.0
    if not np.any(valid_depth):
        LOGGER.warning("Depth ROI is empty, falling back to analytic rim samples")
        points_world = rim_samples_world.copy()
    else:
        u = xv[valid_depth].astype(float)
        v = yv[valid_depth].astype(float)
        z = depth_roi[valid_depth].astype(float)
        x = (u - K[0, 2]) * z / K[0, 0]
        y = (v - K[1, 2]) * z / K[1, 1]
        cam_pts = np.stack([x, y, z, np.ones_like(z)], axis=1)
        world_pts = (world_from_camera @ cam_pts.T).T[:, :3]
        points_local = (bowl_rot.T @ (world_pts - bowl_pos).T).T
        rim_band = np.abs(points_local[:, 2] - inner_height) <= max(0.01, rim_thickness * 2.0)
        radial = np.linalg.norm(points_local[:, :2], axis=1)
        radial_band = np.abs(radial - radius) <= max(0.01, rim_thickness * 3.0)
        filtered = world_pts[rim_band & radial_band]
        if filtered.shape[0] < 50:
            LOGGER.info("Insufficient rim hits (%d), augmenting with analytic samples", filtered.shape[0])
            points_world = np.concatenate([filtered, rim_samples_world], axis=0)
        else:
            points_world = filtered

    if points_world.shape[0] < 200:
        extra_angles = np.linspace(0.0, 2.0 * math.pi, num=200, endpoint=False)
        dense_samples = (bowl_rot @ np.stack(
            [
                radius * np.cos(extra_angles),
                radius * np.sin(extra_angles),
                np.full_like(extra_angles, inner_height),
            ],
            axis=1,
        ).T).T + bowl_pos
        points_world = np.concatenate([points_world, dense_samples], axis=0)

    xy = points_world[:, :2]
    A = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(xy.shape[0])])
    b_vec = np.sum(xy ** 2, axis=1)
    sol, *_ = np.linalg.lstsq(A, b_vec, rcond=None)
    cx, cy, c = sol
    radius_est = math.sqrt(max(cx * cx + cy * cy + c, 1e-6))
    center_world = np.array([cx, cy, rim_height_world], dtype=float)

    dense_angles = np.linspace(0.0, 2.0 * math.pi, num=360, endpoint=False)
    rim_pts_3d = np.stack(
        [
            center_world[0] + radius_est * np.cos(dense_angles),
            center_world[1] + radius_est * np.sin(dense_angles),
            np.full_like(dense_angles, rim_height_world),
        ],
        axis=1,
    )

    grasp_candidates: List[Dict[str, object]] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, num=max(sample_count, 8), endpoint=False):
        radial_dir = np.array([math.cos(angle), math.sin(angle), 0.0])
        z_axis = np.array([0.0, 0.0, -1.0])
        x_axis = np.cross(radial_dir, z_axis)
        if np.linalg.norm(x_axis) < 1e-6:
            continue
        x_axis /= np.linalg.norm(x_axis)
        z_axis = np.cross(x_axis, radial_dir)
        z_axis /= np.linalg.norm(z_axis)
        rot = np.column_stack([x_axis, radial_dir, z_axis])
        quat = _matrix_to_quaternion(rot)
        grasp_point = np.array(
            [
                center_world[0] + radius_est * math.cos(angle),
                center_world[1] + radius_est * math.sin(angle),
                rim_height_world + rim_thickness,
            ],
            dtype=float,
        )
        quality = float(np.clip(0.85 - 0.1 * abs(math.sin(angle * 2.0)), 0.0, 1.0))
        grasp_candidates.append(
            {
                "pose_world": {
                    "position": (float(grasp_point[0]), float(grasp_point[1]), float(grasp_point[2])),
                    "quaternion": tuple(float(v) for v in quat),
                },
                "quality": quality,
            }
        )

    basis_angles = [0.0, math.pi]
    uv_pair: List[Tuple[float, float]] = []
    Z_pair: List[float] = []
    for ang in basis_angles:
        point_world = np.array(
            [
                center_world[0] + radius_est * math.cos(ang),
                center_world[1] + radius_est * math.sin(ang),
                rim_height_world,
            ],
            dtype=float,
        )
        pix, depth_cam = _project_points(point_world[None, :], camera_from_world, K)
        uv_pair.append((float(pix[0, 0]), float(pix[0, 1])))
        Z_pair.append(float(depth_cam[0]))

    return {
        "rim_pts_3d": rim_pts_3d.astype(np.float32),
        "center_3d": center_world.astype(np.float32),
        "radius_m": float(radius_est),
        "grasp_candidates": grasp_candidates,
        "features_px": {"uv_pair": uv_pair, "Z_pair": Z_pair},
    }


__all__ = ["detect_bowl_rim"]
