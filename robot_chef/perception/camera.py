"""RGB-D camera utilities for PyBullet renders."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import pybullet as p


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def matrix(self) -> np.ndarray:
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


def _rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    quat = p.getQuaternionFromEuler(tuple(rpy))
    return np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)


def _fov_to_intrinsics(fov_deg: float, width: int, height: int) -> CameraIntrinsics:
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    f = cx / math.tan(math.radians(fov_deg) * 0.5)
    return CameraIntrinsics(fx=f, fy=f, cx=cx, cy=cy)


def _projection_matrix(fov_deg: float, width: int, height: int, near: float, far: float) -> np.ndarray:
    aspect = float(width) / float(height)
    proj = p.computeProjectionMatrixFOV(fov=fov_deg, aspect=aspect, nearVal=near, farVal=far)
    return np.array(proj, dtype=float).reshape(4, 4)


class Camera:
    """Minimal RGB-D camera helper with view control and mild depth noise."""

    def __init__(
        self,
        *,
        client_id: int,
        view_name: str,
        position: Iterable[float],
        rpy_rad: Iterable[float],
        fov_deg: float,
        resolution: Tuple[int, int],
        near: float,
        far: float,
        depth_noise_std: float = 0.0,
        depth_drop_prob: float = 0.0,
        seed: int = 7,
        renderer: Optional[int] = None,
    ) -> None:
        self.cid = client_id
        self.view_name = view_name
        self._pos = np.array(position, dtype=float)
        self._rpy = tuple(float(v) for v in rpy_rad)
        self.fov_deg = float(fov_deg)
        self.width, self.height = int(resolution[0]), int(resolution[1])
        self.near = float(near)
        self.far = float(far)
        self._noise_std = float(depth_noise_std)
        self._drop_prob = float(depth_drop_prob)
        self._rng = np.random.default_rng(seed)

        if renderer is None:
            info = p.getConnectionInfo(self.cid)
            is_gui = info.get("connectionMethod", p.DIRECT) != p.DIRECT
            self.renderer = p.ER_BULLET_HARDWARE_OPENGL if is_gui else p.ER_TINY_RENDERER
        else:
            self.renderer = renderer

        self._intr = _fov_to_intrinsics(self.fov_deg, self.width, self.height)
        self._proj = _projection_matrix(self.fov_deg, self.width, self.height, self.near, self.far)
        self._rebuild_view(self._pos, self._target_from_pose())

    # ------------------------------------------------------------------ public API
    @property
    def intrinsics(self) -> CameraIntrinsics:
        return self._intr

    @property
    def K(self) -> np.ndarray:
        return self._intr.matrix

    @property
    def world_from_cam(self) -> np.ndarray:
        return self._world_from_cam.copy()

    @property
    def cam_from_world(self) -> np.ndarray:
        return self._cam_from_world.copy()

    def aim_at(self, target_world: Sequence[float], distance: float, height_delta: float = 0.0) -> None:
        """
        Position the camera so that it looks at `target_world` from a given distance and height offset.
        The ray direction keeps the current optical axis; distance is measured along that ray.
        """
        target = np.array(target_world, dtype=float)
        rot = _rpy_to_matrix(self._rpy)
        forward = rot @ np.array([0.0, 0.0, -1.0], dtype=float)
        forward /= np.linalg.norm(forward) + 1e-12

        eye = target - forward * float(distance)
        eye[2] += float(height_delta)
        self._pos = eye
        self._rebuild_view(self._pos, target)

    def rotate_world_yaw(self, delta_deg: float) -> None:
        roll, pitch, yaw = self._rpy
        yaw += math.radians(float(delta_deg))
        self._rpy = (roll, pitch, yaw)
        self._rebuild_view(self._pos, self._target_from_pose())

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        vm = self._view_matrix_ogl.T.flatten().tolist()
        pm = self._proj.flatten().tolist()
        _, _, rgb_buf, depth_buf, seg_buf = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=vm,
            projectionMatrix=pm,
            renderer=self.renderer,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.cid,
        )
        rgb = np.reshape(np.asarray(rgb_buf, dtype=np.uint8), (self.height, self.width, 4))[..., :3]
        depth = np.asarray(depth_buf, dtype=np.float32).reshape(self.height, self.width)
        if depth.max() <= 1.0 + 1e-6:
            depth = self._depth_from_buffer(depth)
        seg = np.asarray(seg_buf, dtype=np.int32).reshape(self.height, self.width)

        if self._drop_prob > 0.0:
            drop = self._rng.random(depth.shape) < self._drop_prob
            depth = depth.copy()
            depth[drop] = 0.0
        if self._noise_std > 0.0:
            noise = self._rng.normal(0.0, self._noise_std, size=depth.shape).astype(np.float32)
            depth = np.maximum(0.0, depth + noise)
        return rgb, depth, seg

    def project(self, points_world: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pts = np.asarray(points_world, dtype=float).reshape(-1, 3)
        homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
        cam = (self._cam_from_world @ homo.T).T[:, :3]
        z = cam[:, 2:3]
        uv = cam[:, :2] / np.maximum(z, 1e-8)
        uv[:, 0] = self._intr.fx * uv[:, 0] + self._intr.cx
        uv[:, 1] = self._intr.fy * uv[:, 1] + self._intr.cy
        return uv, z[:, 0]

    def save_images(self, out_dir: Path, rgb: np.ndarray, depth: np.ndarray, prefix: str = "") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio  # lazy import

        imageio.imwrite(out_dir / f"{prefix}rgb.png", rgb)
        depth_vis = np.clip(depth / max(depth.max(), 1e-6), 0.0, 1.0)
        imageio.imwrite(out_dir / f"{prefix}depth.png", (depth_vis * 65535).astype(np.uint16))

    def save_overlay(self, out_dir: Path, overlay: np.ndarray, prefix: str = "") -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        import imageio.v2 as imageio

        imageio.imwrite(out_dir / f"{prefix}overlay.png", overlay)

    # ------------------------------------------------------------------ internals
    def _target_from_pose(self) -> np.ndarray:
        rot = _rpy_to_matrix(self._rpy)
        forward = rot @ np.array([0.0, 0.0, -1.0], dtype=float)
        return self._pos + forward

    def _rebuild_view(self, eye: np.ndarray, target: np.ndarray) -> None:
        raw = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=[0.0, 0.0, 1.0],
        )
        view_rm = np.array(raw, dtype=float).reshape(4, 4).T  # OpenGL row-major
        self._view_matrix_ogl = view_rm

        # Map OpenGL convention (+Y up, -Z forward) to pinhole convention (+Z forward, +Y down)
        flip = np.diag([1.0, -1.0, -1.0, 1.0])
        self._cam_from_world = flip @ view_rm
        self._world_from_cam = np.linalg.inv(self._cam_from_world)

    def _depth_from_buffer(self, zbuf: np.ndarray) -> np.ndarray:
        near, far = self.near, self.far
        z = np.clip(zbuf, 0.0, 1.0).astype(np.float32)
        denom = np.maximum(far + near - (2.0 * z - 1.0) * (far - near), 1e-6)
        return (2.0 * near * far) / denom
