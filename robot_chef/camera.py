"""RGB-D camera abstraction with wrist-mount support and reliable extrinsics."""

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


def _rotation_from_euler(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [math.radians(v) for v in rpy_deg]
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])), dtype=float).reshape(3, 3)


def _quaternion_from_matrix(R: np.ndarray) -> Tuple[float, float, float, float]:
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


def _make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


class Camera:
    """Wrapper over PyBullet's virtual camera that keeps consistent CV extrinsics."""

    def __init__(
        self,
        client_id: int,
        *,
        fov_deg: float,
        near: float,
        far: float,
        resolution: Tuple[int, int],
        noise: Optional[CameraNoiseModel] = None,
        view_xyz: Sequence[float] = (0.0, 0.0, 1.0),
        view_rpy_deg: Sequence[float] = (0.0, -45.0, 0.0),
    ) -> None:
        self.client_id = int(client_id)
        self._width, self._height = int(resolution[0]), int(resolution[1])
        self._fov_deg = float(fov_deg)
        self._near = float(near)
        self._far = float(far)
        self._noise = noise or CameraNoiseModel()

        # Intrinsic matrix (pinhole, origin at top-left, y downward)
        f = (self._height / 2.0) / math.tan(math.radians(self._fov_deg) / 2.0)
        cx = (self._width - 1) / 2.0
        cy = (self._height - 1) / 2.0
        self._intrinsics = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=float)

        # Free-camera pose (used when not mounted)
        self._world_position = np.array(view_xyz, dtype=float)
        self._world_rotation = _rotation_from_euler(view_rpy_deg)  # maps camera frame -> world

        # Mount configuration
        self._mounted = False
        self._parent_body: Optional[int] = None
        self._parent_link: Optional[int] = None
        self._rel_translation = np.zeros(3, dtype=float)
        self._rel_rotation = np.eye(3, dtype=float)

        # Extrinsic caches (OpenGL frame and CV frame)
        self._T_world_cam_gl = np.eye(4, dtype=float)
        self._T_cam_world_gl = np.eye(4, dtype=float)
        self._T_world_cam_cv = np.eye(4, dtype=float)
        self._T_cam_world_cv = np.eye(4, dtype=float)

        self._update_from_free_pose()

    # ------------------------------------------------------------------ #
    # Mounting / configuration

    def mount_to_link(
        self,
        *,
        parent_body_id: int,
        parent_link_id: int,
        rel_xyz: Sequence[float],
        rel_rpy_deg: Sequence[float],
    ) -> None:
        """Attach camera rigidly to a robot link."""
        self._parent_body = int(parent_body_id)
        self._parent_link = int(parent_link_id)
        self._rel_translation = np.asarray(rel_xyz, dtype=float)
        self._rel_rotation = _rotation_from_euler(rel_rpy_deg)
        self._mounted = True
        self._update_from_mount()
        LOGGER.info(
            "Camera mounted to body=%d link=%d with rel xyz=%s rpy_deg=%s",
            self._parent_body,
            self._parent_link,
            np.array(rel_xyz, dtype=float),
            np.array(rel_rpy_deg, dtype=float),
        )

    def aim_at_world(self, target_xyz: Sequence[float]) -> None:
        """Rotate the relative mount so the camera optical axis (+Z) points at the target."""
        if not self._mounted:
            self._update_free_orientation(target_xyz)
            return

        assert self._parent_body is not None and self._parent_link is not None
        ls = p.getLinkState(
            self._parent_body,
            self._parent_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_pos = np.asarray(ls[4], dtype=float)
        link_rot = np.array(p.getMatrixFromQuaternion(ls[5]), dtype=float).reshape(3, 3)

        cam_pos = link_pos + link_rot @ self._rel_translation
        forward = np.asarray(target_xyz, dtype=float) - cam_pos
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return
        forward /= norm

        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(forward, up_hint)) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=float)

        z_cam = forward  # +Z forward
        x_cam = np.cross(up_hint, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            x_cam = np.cross(np.array([1.0, 0.0, 0.0], dtype=float), z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        R_world_cam = np.column_stack([x_cam, y_cam, z_cam])
        self._rel_rotation = link_rot.T @ R_world_cam
        self._update_from_mount()

    # ------------------------------------------------------------------ #
    # Capture

    def get_rgbd(self, with_segmentation: bool = False):
        """Return RGB, depth (meters), intrinsics, and optional segmentation mask."""
        if self._mounted:
            self._update_from_mount()
        else:
            self._update_from_free_pose()

        eye = self._T_world_cam_gl[:3, 3]
        R_wc = self._T_world_cam_gl[:3, :3]
        forward = R_wc[:, 2]  # +Z forward
        up = R_wc[:, 1]
        target = eye + forward

        flags = p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX if with_segmentation else 0
        img = p.getCameraImage(
            self._width,
            self._height,
            viewMatrix=p.computeViewMatrix(eye, target, up),
            projectionMatrix=p.computeProjectionMatrixFOV(
                fov=self._fov_deg,
                aspect=self._width / float(self._height),
                nearVal=self._near,
                farVal=self._far,
            ),
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            flags=flags,
            physicsClientId=self.client_id,
        )

        rgba = np.reshape(np.asarray(img[2], dtype=np.uint8), (self._height, self._width, 4))
        rgb = rgba[..., :3].copy()

        depth_buf = np.reshape(np.asarray(img[3], dtype=np.float32), (self._height, self._width))
        depth = self._depth_buffer_to_meters(depth_buf)

        if self._noise.depth_std > 0.0:
            depth = depth + np.random.normal(0.0, self._noise.depth_std, depth.shape).astype(np.float32)
        if self._noise.drop_prob > 0.0:
            drop = np.random.rand(*depth.shape) < float(self._noise.drop_prob)
            depth = depth.copy()
            depth[drop] = 0.0

        seg_mask = None
        if with_segmentation:
            seg_mask = np.reshape(np.asarray(img[4], dtype=np.int32), (self._height, self._width))

        return rgb, depth.astype(np.float32), self._intrinsics.copy(), seg_mask

    # ------------------------------------------------------------------ #
    # Accessors

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics.copy()

    @property
    def world_from_camera(self) -> np.ndarray:
        """4x4 transform from camera CV frame (+X right, +Y down, +Z forward) to world."""
        return self._T_world_cam_cv.copy()

    @property
    def camera_from_world(self) -> np.ndarray:
        """4x4 transform from world to camera CV frame."""
        return self._T_cam_world_cv.copy()

    @property
    def pose_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return camera position and quaternion in world coordinates (OpenGL frame)."""
        pos = self._T_world_cam_gl[:3, 3].copy()
        quat = _quaternion_from_matrix(self._T_world_cam_gl[:3, :3])
        return pos, np.asarray(quat, dtype=float)

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _update_free_orientation(self, target_xyz: Sequence[float]) -> None:
        eye = self._world_position
        forward = np.asarray(target_xyz, dtype=float) - eye
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            return
        forward /= norm
        up_hint = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(np.dot(forward, up_hint)) > 0.95:
            up_hint = np.array([0.0, 1.0, 0.0], dtype=float)
        z_cam = forward
        x_cam = np.cross(up_hint, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        self._world_rotation = np.column_stack([x_cam, y_cam, z_cam])
        self._update_from_free_pose()

    def _update_from_mount(self) -> None:
        assert self._parent_body is not None and self._parent_link is not None
        ls = p.getLinkState(
            self._parent_body,
            self._parent_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_pos = np.asarray(ls[4], dtype=float)
        link_rot = np.array(p.getMatrixFromQuaternion(ls[5]), dtype=float).reshape(3, 3)
        R_wc = link_rot @ self._rel_rotation
        t_wc = link_pos + link_rot @ self._rel_translation
        self._update_matrices(R_wc, t_wc)

    def _update_from_free_pose(self) -> None:
        self._update_matrices(self._world_rotation, self._world_position)

    def _update_matrices(self, R_wc: np.ndarray, t_wc: np.ndarray) -> None:
        self._T_world_cam_gl = _make_T(R_wc, t_wc)
        self._T_cam_world_gl = np.linalg.inv(self._T_world_cam_gl)
        self._update_cv_frames()

    def _update_cv_frames(self) -> None:
        # Convert OpenGL camera frame (x right, y up, z forward) to CV convention (x right, y down, z forward)
        T_gl_to_cv = np.eye(4, dtype=float)
        T_gl_to_cv[1, 1] = -1.0
        T_cv_to_gl = T_gl_to_cv
        self._T_cam_world_cv = T_gl_to_cv @ self._T_cam_world_gl
        self._T_world_cam_cv = self._T_world_cam_gl @ T_cv_to_gl

    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        n, f = self._near, self._far
        return (2.0 * n * f) / (f + n - (2.0 * depth_buffer - 1.0) * (f - n))


__all__ = ["Camera", "CameraNoiseModel"]
