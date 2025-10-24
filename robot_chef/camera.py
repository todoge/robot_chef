"""RGB-D camera utilities for the Robot Chef simulation."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


@dataclass
class CameraNoiseModel:
    depth_std: float = 0.0
    drop_prob: float = 0.0


class Camera:
    """Thin wrapper over PyBullet's virtual camera with intrinsic bookkeeping."""

    def __init__(
        self,
        client_id: int,
        view_xyz: Sequence[float],
        view_rpy_deg: Sequence[float],
        table_height: float,
        fov_deg: float = 60,
        near: float = 0.1,
        far: float = 3.0,
        resolution: Tuple[int, int] = (224,224), #(640, 480),
        noise: Optional[CameraNoiseModel] = None,
    ) -> None:
        self.client_id = int(client_id)
        self._position = np.array(view_xyz, dtype=float)
        self._rpy_deg = np.array(view_rpy_deg, dtype=float)
        self._fov_deg = float(fov_deg)
        self._near = float(near)
        self._far = float(far)
        self._width = int(resolution[0])
        self._height = int(resolution[1])
        self._noise = noise or CameraNoiseModel()
        self._table_height = table_height

        self._orientation_quat = p.getQuaternionFromEuler(np.radians(self._rpy_deg))
        self._target = self._position + self._orientation_matrix() @ np.array([0.0, 0.0, -1.0])

        self._update_matrices()

    # ------------------------------------------------------------------ #
    # Public API

    def aim_at(self, target_xyz: Sequence[float], distance: float, height_delta: float) -> None:
        """Reposition camera so that the supplied target sits near the image center."""
        target = np.asarray(target_xyz, dtype=float)
        forward = self._orientation_matrix() @ np.array([0.0, 0.0, 1.0])
        forward_norm = math.hypot(forward[0], forward[1])
        if forward_norm < 1e-5:
            forward = np.array([1.0, 0.0, -0.2])
            forward_norm = math.hypot(forward[0], forward[1])
        direction_xy = forward[:2] / forward_norm
        new_position = np.array(
            [
                target[0] - distance * direction_xy[0],
                target[1] - distance * direction_xy[1],
                target[2] + float(height_delta),
            ],
            dtype=float,
        )
        new_forward = target - new_position
        new_forward /= np.linalg.norm(new_forward)
        yaw = math.atan2(new_forward[1], new_forward[0])
        pitch = math.asin(-new_forward[2])
        self._rpy_deg = np.degrees([0.0, pitch, yaw])
        self._orientation_quat = p.getQuaternionFromEuler(np.radians(self._rpy_deg))
        self._position = new_position
        self._target = target
        self._update_matrices()
        LOGGER.info(
            "Camera positioned at (%.3f, %.3f, %.3f) aiming at (%.3f, %.3f, %.3f) (distance=%.2f)",
            new_position[0],
            new_position[1],
            new_position[2],
            target[0],
            target[1],
            target[2],
            float(distance),
        )

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        width, height = self._width, self._height
        img = p.getCameraImage(
            width,
            height,
            viewMatrix=self._view_matrix.flatten().tolist(),
            projectionMatrix=self._projection_matrix.flatten().tolist(),
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )
        rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (height, width, 4))[..., :3].copy()
        depth_buffer = np.reshape(np.asarray(img[3], dtype=np.float32), (height, width))
        depth_buffer[np.isinf(depth_buffer)] = np.nan
        depth_buffer = np.nan_to_num(depth_buffer, nan=np.nanmedian(depth_buffer))
        depth = self._depth_buffer_to_meters(depth_buffer).astype(np.float32)
        depth_normalized = depth - np.median(depth)

        if self._noise.depth_std > 0.0:
            depth += np.random.normal(0.0, self._noise.depth_std, size=depth.shape).astype(np.float32)
        if self._noise.drop_prob > 0.0:
            mask = np.random.random(size=depth.shape) < float(self._noise.drop_prob)
            depth[mask] = 0.0
        return rgb, depth_buffer, depth_normalized, self._intrinsics.copy()

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics.copy()

    @property
    def pose_world(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._position.copy(), np.array(self._orientation_quat, dtype=float)

    @property
    def world_from_camera(self) -> np.ndarray:
        return self._world_from_camera.copy()

    @property
    def camera_from_world(self) -> np.ndarray:
        return self._camera_from_world.copy()

    # ------------------------------------------------------------------ #
    # Internals

    def _orientation_matrix(self) -> np.ndarray:
        rot = np.array(p.getMatrixFromQuaternion(self._orientation_quat), dtype=float).reshape(3, 3)
        return rot

    def _update_matrices(self) -> None:
        rot = self._orientation_matrix()
        forward = rot @ np.array([0.0, 0.0, 1.0])
        up = rot @ np.array([0.0, 1.0, 0.0])
        target = self._position + forward
        view = p.computeViewMatrix(
            cameraEyePosition=self._position.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        self._view_matrix = np.asarray(view, dtype=float).reshape(4, 4)
        self._camera_from_world = self._view_matrix.copy()
        self._world_from_camera = np.linalg.inv(self._camera_from_world)
        aspect = self._width / self._height
        proj = p.computeProjectionMatrixFOV(
            fov=self._fov_deg,
            aspect=aspect,
            nearVal=self._near,
            farVal=self._far,
        )
        self._projection_matrix = np.asarray(proj, dtype=float).reshape(4, 4)
        self._intrinsics = self._compute_intrinsics(self._fov_deg, self._width, self._height, self._near, self._far)

    @staticmethod
    def _compute_intrinsics(fov_deg: float, width: int, height: int, near: float, far: float) -> np.ndarray:
        fov_rad = math.radians(fov_deg)
        fy = (height / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy * (width / height)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        K = np.array(
            [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        return K

    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        near, far = self._near, self._far
        depth = (2.0 * near * far) / (far + near - (2.0 * depth_buffer - 1.0) * (far - near))
        camera_height = self._position[2]
        table_depth = camera_height - self._table_height
        depth_augmented = np.minimum(depth, table_depth)
        return depth_augmented


__all__ = ["Camera", "CameraNoiseModel"]
