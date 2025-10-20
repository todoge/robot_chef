"""RGB-D camera utilities for the Robot Chef simulation.

This camera supports both:
  1) Free (world-placed) viewing
  2) Wrist-mounted viewing: the camera extrinsics are recomputed from the
     parent link state on every get_rgbd() call so it follows the arm.

Coordinate convention:
- The camera looks along its local -Z axis.
- The camera's "up" is its local +Y axis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Small linear algebra helpers

def _rpy_deg_to_quat(rpy_deg: Sequence[float]) -> Tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(rpy_deg[0]),
                        math.radians(rpy_deg[1]),
                        math.radians(rpy_deg[2]))
    return p.getQuaternionFromEuler([roll, pitch, yaw])

def _quat_to_R(quat: Sequence[float]) -> np.ndarray:
    return np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)

def _make_T(R: np.ndarray, t: Sequence[float]) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T

def _invert_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    Rt = R.T
    t = T[:3, 3]
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = Rt
    Ti[:3, 3] = -Rt @ t
    return Ti


# --------------------------------------------------------------------------- #
# API

@dataclass
class CameraNoiseModel:
    depth_std: float = 0.0   # meters (Gaussian)
    drop_prob: float = 0.0   # probability of dropping a depth pixel (set to 0)

class Camera:
    """Thin wrapper around PyBullet's virtual camera with intrinsic bookkeeping."""

    def __init__(
        self,
        client_id: int,
        *,
        # World placement (used only when not mounted)
        view_xyz: Sequence[float] = (0.0, 0.0, 1.0),
        view_rpy_deg: Sequence[float] = (0.0, -45.0, 0.0),
        fov_deg: float = 60.0,
        near: float = 0.05,
        far: float = 5.0,
        resolution: Tuple[int, int] = (640, 480),
        noise: Optional[CameraNoiseModel] = None,
    ) -> None:
        self.client_id = int(client_id)
        self._width, self._height = int(resolution[0]), int(resolution[1])
        self._fov_deg = float(fov_deg)
        self._near, self._far = float(near), float(far)
        self._noise = noise or CameraNoiseModel()

        # --- Intrinsics
        self._K = self._compute_intrinsics(self._fov_deg, self._width, self._height)

        # --- World (free) placement defaults
        self._world_pos = np.array(view_xyz, dtype=float)
        self._world_quat = _rpy_deg_to_quat(view_rpy_deg)

        # --- Mount state (None if not mounted)
        self._mounted = False
        self._parent_body: Optional[int] = None
        self._parent_link: Optional[int] = None
        self._rel_R = np.eye(3, dtype=float)   # link->cam rotation
        self._rel_t = np.array([0.0, 0.0, 0.0], dtype=float)  # link->cam translation

        # Buffers for extrinsics
        self._T_world_cam = np.eye(4, dtype=float)
        self._T_cam_world = np.eye(4, dtype=float)
        self._update_extrinsics_world()

        # Projection matrix (depends only on intrinsics + near/far)
        self._update_projection()

    # ------------------------------------------------------------------ #
    # Mounting / Aiming

    def mount_to_link(
        self,
        *,
        parent_body_id: int,
        parent_link_id: int,
        rel_xyz: Sequence[float] = (0.05, 0.0, 0.10),
        rel_rpy_deg: Sequence[float] = (0.0, -55.0, 0.0),
    ) -> None:
        """Attach camera to a robot link. Camera looks along -Z."""
        self._parent_body = int(parent_body_id)
        self._parent_link = int(parent_link_id)
        self._rel_t = np.asarray(rel_xyz, dtype=float)
        self._rel_R = _quat_to_R(_rpy_deg_to_quat(rel_rpy_deg))
        self._mounted = True
        # Compute initial extrinsics from current link pose
        self._update_extrinsics_from_mount()
        LOGGER.info(
            "Camera mounted to body=%d link=%d with rel (xyz=%s, rpy_deg=%s)",
            self._parent_body, self._parent_link,
            np.array(self._rel_t), np.degrees(np.array([0.0, 0.0, 0.0])) if False else np.array(rel_rpy_deg)
        )

    def aim_at_world(self, target_xyz: Sequence[float]) -> None:
        """Rotate ONLY the relative mount so -Z_cam points at target_xyz."""
        if not self._mounted:
            # For free camera, just set world rotation to look at target
            self._look_at_free(target_xyz)
            return
        # Get current link world pose
        ls = p.getLinkState(
            self._parent_body, self._parent_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_pos = np.array(ls[4], dtype=float)
        link_quat = np.array(ls[5], dtype=float)
        R_link = _quat_to_R(link_quat)

        # Current cam world position with existing rel transform
        cam_pos = link_pos + R_link @ self._rel_t
        to_tgt = np.asarray(target_xyz, dtype=float) - cam_pos
        n = np.linalg.norm(to_tgt)
        if n < 1e-8:
            return
        to_tgt /= n

        # Build camera frame with -Z along to_tgt and +Y roughly upright
        z_cam = -to_tgt
        up_guess = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(up_guess, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            up_guess = np.array([1.0, 0.0, 0.0])
            x_cam = np.cross(up_guess, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        R_world_cam_des = np.column_stack([x_cam, y_cam, z_cam])
        # Convert to new relative rotation: R_rel = R_link^T * R_world_cam
        self._rel_R = R_link.T @ R_world_cam_des
        self._update_extrinsics_from_mount()

    # ------------------------------------------------------------------ #
    # Capture

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (rgb, depth_meters, K). Recomputes extrinsics if mounted."""
        if self._mounted:
            self._update_extrinsics_from_mount()

        # Build a proper OpenGL view matrix from world_from_camera
        eye = self._T_world_cam[:3, 3]
        Rwc = self._T_world_cam[:3, :3]
        forward = Rwc @ np.array([0.0, 0.0, -1.0])   # camera looks along -Z
        up = Rwc @ np.array([0.0, 1.0,  0.0])        # +Y is camera up
        target = eye + forward

        view = p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )

        # Projection straight from FOV (PyBullet handles layout)
        proj = p.computeProjectionMatrixFOV(
            fov=self._fov_deg,
            aspect=self._width / float(self._height),
            nearVal=self._near,
            farVal=self._far,
        )

        img = p.getCameraImage(
            self._width,
            self._height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
            physicsClientId=self.client_id,
        )

        # RGBA -> RGB
        rgb = np.reshape(np.asarray(img[2], dtype=np.uint8), (self._height, self._width, 4))[..., :3].copy()

        # Depth buffer -> meters
        depth_buf = np.reshape(np.asarray(img[3], dtype=np.float32), (self._height, self._width))
        depth = self._depth_buffer_to_meters(depth_buf)

        # Optional noise
        if self._noise.depth_std > 0.0:
            depth = depth + np.random.normal(0.0, self._noise.depth_std, depth.shape).astype(np.float32)
        if self._noise.drop_prob > 0.0:
            mask = np.random.random(size=depth.shape) < float(self._noise.drop_prob)
            depth = depth.copy()
            depth[mask] = 0.0

        return rgb, depth.astype(np.float32), self._K.copy()
    # ------------------------------------------------------------------ #
    # Properties

    @property
    def intrinsics(self) -> np.ndarray:
        return self._K.copy()

    @property
    def world_from_camera(self) -> np.ndarray:
        return self._T_world_cam.copy()

    @property
    def camera_from_world(self) -> np.ndarray:
        return self._T_cam_world.copy()

    # ------------------------------------------------------------------ #
    # Internals

    def _look_at_free(self, target_xyz: Sequence[float]) -> None:
        eye = np.array(self._world_pos, dtype=float)
        tgt = np.asarray(target_xyz, dtype=float)
        fwd = tgt - eye
        n = np.linalg.norm(fwd)
        if n < 1e-8:
            return
        fwd /= n
        z_cam = -fwd
        up_guess = np.array([0.0, 1.0, 0.0])
        x_cam = np.cross(up_guess, z_cam)
        if np.linalg.norm(x_cam) < 1e-6:
            up_guess = np.array([1.0, 0.0, 0.0])
            x_cam = np.cross(up_guess, z_cam)
        x_cam /= np.linalg.norm(x_cam)
        y_cam = np.cross(z_cam, x_cam)
        R_world_cam = np.column_stack([x_cam, y_cam, z_cam])
        self._T_world_cam = _make_T(R_world_cam, eye)
        self._T_cam_world = _invert_T(self._T_world_cam)

    def _update_extrinsics_from_mount(self) -> None:
        """Compute world_from_camera from current link state + relative mount."""
        assert self._parent_body is not None and self._parent_link is not None
        ls = p.getLinkState(
            self._parent_body, self._parent_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_pos = np.array(ls[4], dtype=float)  # worldLinkFramePosition
        link_quat = np.array(ls[5], dtype=float) # worldLinkFrameOrientation
        R_link = _quat_to_R(link_quat)
        t_link = link_pos

        # T_world_cam = T_world_link * T_link_cam
        T_world_link = _make_T(R_link, t_link)
        T_link_cam = _make_T(self._rel_R, self._rel_t)
        self._T_world_cam = T_world_link @ T_link_cam
        self._T_cam_world = _invert_T(self._T_world_cam)

    def _update_extrinsics_world(self) -> None:
        """Use free (non-mounted) world pose."""
        R_world = _quat_to_R(self._world_quat)
        self._T_world_cam = _make_T(R_world, self._world_pos)
        self._T_cam_world = _invert_T(self._T_world_cam)

    def _update_projection(self) -> None:
        # Build a true OpenGL-style projection that matches our intrinsics.
        aspect = self._width / float(self._height)
        proj = p.computeProjectionMatrixFOV(
            fov=self._fov_deg,
            aspect=aspect,
            nearVal=self._near,
            farVal=self._far,
        )
        self._projection_matrix = np.asarray(proj, dtype=float).reshape(4, 4)

    @staticmethod
    def _compute_intrinsics(fov_deg: float, width: int, height: int) -> np.ndarray:
        fov_rad = math.radians(fov_deg)
        fy = (height / 2.0) / math.tan(fov_rad / 2.0)
        fx = fy * (width / height)
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        return np.array([[fx, 0.0, cx],
                         [0.0, fy, cy],
                         [0.0, 0.0, 1.0]], dtype=float)

    def _depth_buffer_to_meters(self, depth_buffer: np.ndarray) -> np.ndarray:
        n, f = self._near, self._far
        # Standard OpenGL depth deprojection
        return (2.0 * n * f) / (f + n - (2.0 * depth_buffer - 1.0) * (f - n))
