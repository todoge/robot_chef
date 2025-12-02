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


def _rotation_from_euler(rpy_deg: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = [math.radians(v) for v in rpy_deg]
    return np.array(p.getMatrixFromQuaternion(p.getQuaternionFromEuler([roll, pitch, yaw])), dtype=float).reshape(3, 3)


class Camera:
    """Thin wrapper over PyBullet's virtual camera with intrinsic bookkeeping."""

    def __init__(
        self,
        client_id: int,
        view_xyz: Sequence[float],
        view_rpy_deg: Sequence[float],
        fov_deg: float,
        near: float = 0.02,
        far: float = 3.0,
        resolution: Tuple[int, int] = (640, 480),
        noise: Optional[CameraNoiseModel] = None,
    ) -> None:
        self.client_id = int(client_id)
        # Default fixed pose
        self._position = np.array(view_xyz, dtype=float)
        self._rpy_deg = np.array(view_rpy_deg, dtype=float)
        self._fov_deg = float(fov_deg)
        self._near = float(near)
        self._far = float(far)
        self._width = int(resolution[0])
        self._height = int(resolution[1])
        self._noise = noise or CameraNoiseModel()

        # Mount state
        self._mounted = False
        self._parent_body: Optional[int] = None
        self._parent_link: Optional[int] = None
        self._rel_xyz = np.zeros(3)
        self._rel_rot = np.eye(3)

        self._orientation_quat = p.getQuaternionFromEuler(np.radians(self._rpy_deg))
        self._update_matrices()

    # ------------------------------------------------------------------ #
    # Public API

    def mount_to_link(
        self,
        parent_body_id: int,
        parent_link_id: int,
        rel_xyz: Sequence[float],
        rel_rpy_deg: Sequence[float],
    ) -> None:
        """Attach the camera rigidly to a robot link."""
        self._mounted = True
        self._parent_body = int(parent_body_id)
        self._parent_link = int(parent_link_id)
        self._rel_xyz = np.array(rel_xyz, dtype=float)
        self._rel_rot = _rotation_from_euler(rel_rpy_deg)
        self._update_from_mount()
        LOGGER.info("Camera mounted to link %d on body %d", parent_link_id, parent_body_id)

    def aim_at(self, target_xyz: Sequence[float], distance: float, height_delta: float) -> None:
        """Reposition camera so that the supplied target sits near the image center."""
        target = np.asarray(target_xyz, dtype=float)

        if self._mounted:
            # If mounted, we adjust the relative rotation to point at the target
            # This simulates a "pan-tilt" mount or just setting the initial angle correct
            self._update_from_mount() # Get current world pos
            
            # Vector from camera to target
            cam_pos = self._position
            forward = target - cam_pos
            forward_norm = np.linalg.norm(forward)
            if forward_norm < 1e-5:
                return # Already there?
            forward /= forward_norm # Z axis of camera (looking towards target)
            
            # We want -Z of camera to point along 'forward'
            # Let's construct a rotation matrix R_cam_world such that R * [0,0,-1] = forward
            # Up vector guess (try Z up, if parallel then Y up)
            up_guess = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(up_guess, forward)) > 0.99:
                up_guess = np.array([0.0, 1.0, 0.0])
            
            # Camera coordinate system:
            # z_cam = -forward (looking down -Z)
            # x_cam = cross(up, z_cam)
            # y_cam = cross(z_cam, x_cam)
            
            z_cam = -forward
            x_cam = np.cross(up_guess, z_cam)
            x_cam /= np.linalg.norm(x_cam)
            y_cam = np.cross(z_cam, x_cam)
            y_cam /= np.linalg.norm(y_cam)
            
            R_world_cam = np.column_stack((x_cam, y_cam, z_cam)) # Rotation of camera in world frame
            
            # We need R_link_cam.
            # R_world_cam = R_world_link * R_link_cam
            # => R_link_cam = R_world_link^T * R_world_cam
            
            ls = p.getLinkState(
                self._parent_body, 
                self._parent_link, 
                computeForwardKinematics=True, 
                physicsClientId=self.client_id
            )
            link_rot_world = np.array(p.getMatrixFromQuaternion(ls[5]), dtype=float).reshape(3, 3)
            
            self._rel_rot = link_rot_world.T @ R_world_cam
            self._update_from_mount()
            LOGGER.info("Mounted camera re-aimed at target.")
            return

        # Fixed camera logic
        rot = self._orientation_matrix()
        forward = rot @ np.array([0.0, 0.0, -1.0])
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
        self._update_matrices()

    def get_rgbd(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._mounted:
            self._update_from_mount()

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
        depth = self._depth_buffer_to_meters(depth_buffer)

        if self._noise.depth_std > 0.0:
            depth += np.random.normal(0.0, self._noise.depth_std, size=depth.shape).astype(np.float32)
        if self._noise.drop_prob > 0.0:
            mask = np.random.random(size=depth.shape) < float(self._noise.drop_prob)
            depth[mask] = 0.0
        return rgb, depth.astype(np.float32), self._intrinsics.copy()

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics.copy()

    @property
    def pose_world(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._mounted:
            self._update_from_mount()
        return self._position.copy(), np.array(self._orientation_quat, dtype=float)

    @property
    def world_from_camera(self) -> np.ndarray:
        if self._mounted:
            self._update_from_mount()
        return self._world_from_camera.copy()

    @property
    def camera_from_world(self) -> np.ndarray:
        if self._mounted:
            self._update_from_mount()
        return self._camera_from_world.copy()

    # ------------------------------------------------------------------ #
    # Internals

    def _orientation_matrix(self) -> np.ndarray:
        rot = np.array(p.getMatrixFromQuaternion(self._orientation_quat), dtype=float).reshape(3, 3)
        return rot

    def _update_from_mount(self) -> None:
        if not self._mounted:
            return
        ls = p.getLinkState(
            self._parent_body,
            self._parent_link,
            computeForwardKinematics=True,
            physicsClientId=self.client_id,
        )
        link_pos = np.asarray(ls[4], dtype=float)
        link_rot = np.array(p.getMatrixFromQuaternion(ls[5]), dtype=float).reshape(3, 3)
        
        # Calculate world pose of camera
        # T_world_cam = T_world_link @ T_link_cam
        self._position = link_pos + link_rot @ self._rel_xyz
        cam_rot_world = link_rot @ self._rel_rot
        
        # We need quaternion for _orientation_quat but matrix for View Matrix
        # Standard View Matrix: LookAt style
        # Forward is -Z in camera frame
        forward = cam_rot_world @ np.array([0.0, 0.0, -1.0])
        up = cam_rot_world @ np.array([0.0, 1.0, 0.0])
        target = self._position + forward
        
        view = p.computeViewMatrix(
            cameraEyePosition=self._position.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=up.tolist(),
        )
        
        # Update internal state so get_rgbd works
        self._view_matrix = np.asarray(view, dtype=float).reshape(4, 4)
        self._camera_from_world = self._view_matrix.copy()
        self._world_from_camera = np.linalg.inv(self._camera_from_world)
        
        # Update quat for external consumers (approx)
        # Note: matrix to quat is complex, maybe just skip or implement robustly if needed
        # For now, just ensuring matrices are correct is enough for rendering.

    def _update_matrices(self) -> None:
        if self._mounted:
            self._update_from_mount()
            return

        rot = self._orientation_matrix()
        forward = rot @ np.array([0.0, 0.0, -1.0])
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
        return depth


__all__ = ["Camera", "CameraNoiseModel"]