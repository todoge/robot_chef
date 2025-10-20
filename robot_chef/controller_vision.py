# robot_chef/controller_vision.py
"""Image-based visual servo controller for the wrist-mounted camera."""

from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


def _skew(v: Sequence[float]) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _adjoint(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    A = np.zeros((6, 6), dtype=float)
    A[:3, :3] = R
    A[3:, 3:] = R
    A[3:, :3] = _skew(t) @ R
    return A


def _damped_pinv(J: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    Sd = S / (S * S + lam * lam)
    return Vt.T @ np.diag(Sd) @ U.T


class VisionRefineController:
    """Closed-loop IBVS controller that drives the Panda wrist toward rim features."""

    def __init__(
        self,
        *,
        client_id: int,
        arm_id: int,
        ee_link: int,
        arm_joints: Sequence[int],
        dt: float,
        camera,
        handeye_T_cam_in_eef: Optional[np.ndarray] = None,
        gripper_open: Optional[Callable[[float], None]] = None,
        gripper_close: Optional[Callable[[float], None]] = None,
    ) -> None:
        self.client_id = int(client_id)
        self.arm_id = int(arm_id)
        self.ee_link = int(ee_link)
        self.arm_joints = [int(j) for j in arm_joints]  # body joint indices for the 7 arm joints
        self.dt = float(dt)
        self.camera = camera
        self._open = gripper_open or (lambda width=0.08: None)
        self._close = gripper_close or (lambda force=60.0: None)

        self._T_cam_eef = np.eye(4, dtype=float) if handeye_T_cam_in_eef is None else np.asarray(handeye_T_cam_in_eef, dtype=float)
        if self._T_cam_eef.shape != (4, 4):
            raise ValueError("handeye_T_cam_in_eef must be 4x4")
        self._Adj_cam_eef = _adjoint(self._T_cam_eef)

        # ---- Build DoF list (movable joints only) and map: joint_index -> dof_col ----
        self._movable_joint_indices: List[int] = []
        self._dof_from_joint: dict[int, int] = {}
        n = p.getNumJoints(self.arm_id, physicsClientId=self.client_id)
        for j in range(n):
            jt = p.getJointInfo(self.arm_id, j, physicsClientId=self.client_id)[2]
            if jt in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC, getattr(p, "JOINT_SPHERICAL", -1), getattr(p, "JOINT_PLANAR", -1)):
                self._dof_from_joint[j] = len(self._movable_joint_indices)
                self._movable_joint_indices.append(j)

        if not self._movable_joint_indices:
            raise RuntimeError("No movable joints detected for arm; cannot compute Jacobian.")

        # Map the 7 arm joints into DoF columns (ignore fingers for IBVS)
        self._arm_dof_cols: List[int] = [self._dof_from_joint[j] for j in self.arm_joints if j in self._dof_from_joint]
        if len(self._arm_dof_cols) != len(self.arm_joints):
            LOGGER.warning("Some arm joints are not movable or not found in DoF map; IBVS may be ill-posed.")

    # ------------------------------------------------------------------ #
    # Gripper helpers

    def open_gripper(self, width: float = 0.08) -> None:
        self._open(width)

    def close_gripper(self, force: float = 60.0) -> None:
        self._close(force)

    # ------------------------------------------------------------------ #
    # Waypoint motion

    def move_waypoint(self, pos: Sequence[float], quat_xyzw: Sequence[float], timeout_s: float = 3.0) -> bool:
        pos = [float(v) for v in pos]
        quat_xyzw = [float(v) for v in quat_xyzw]
        try:
            full_ik = p.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                pos,
                quat_xyzw,
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
        except Exception as exc:
            LOGGER.error("IK failed: %s", exc)
            return False

        if not full_ik or len(full_ik) <= max(self.arm_joints):
            LOGGER.error("IK returned insufficient joint angles")
            return False

        target = [float(full_ik[j]) for j in self.arm_joints]
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            p.setJointMotorControlArray(
                self.arm_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                target,
                positionGains=[0.08] * len(self.arm_joints),
                forces=[200.0] * len(self.arm_joints),
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)
        return True

    # ------------------------------------------------------------------ #
    # IBVS refinement

    def refine_to_features_ibvs(
        self,
        *,
        get_features: Callable[[], Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
        target_uv: np.ndarray,
        target_Z: np.ndarray,
        pixel_tol: float = 3.0,
        depth_tol: float = 0.008,
        max_time_s: float = 4.0,
        gain: float = 0.35,
        max_joint_vel: float = 0.5,
    ) -> bool:
        target_uv = np.asarray(target_uv, dtype=float).reshape(-1, 2)
        target_Z = np.asarray(target_Z, dtype=float).reshape(-1)
        if target_uv.shape[0] == 0:
            LOGGER.error("No features provided for IBVS.")
            return False
        if target_Z.shape[0] != target_uv.shape[0]:
            raise ValueError("target_Z must have the same length as target_uv")

        K = self.camera.intrinsics
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        t0 = time.time()
        while time.time() - t0 < max_time_s:
            uv_meas, Z_meas = get_features()
            if uv_meas is None or Z_meas is None:
                LOGGER.warning("IBVS: feature callback returned None")
                return False
            uv_meas = np.asarray(uv_meas, dtype=float).reshape(-1, 2)
            Z_meas = np.asarray(Z_meas, dtype=float).reshape(-1)
            if uv_meas.shape != target_uv.shape:
                LOGGER.error("IBVS: feature shape mismatch (%s vs %s)", uv_meas.shape, target_uv.shape)
                return False

            pix_err = uv_meas - target_uv
            rms_px = float(np.sqrt(np.mean(pix_err ** 2)))
            depth_err = float(np.mean(np.abs(Z_meas - target_Z)))
            if rms_px < pixel_tol and depth_err < depth_tol:
                LOGGER.info("IBVS refine success (rms=%.2f px, depth=%.4f m)", rms_px, depth_err)
                self._halt_arm()
                return True

            # Interaction matrix for each point
            xn = (uv_meas[:, 0] - cx) / fx
            yn = (uv_meas[:, 1] - cy) / fy
            Z_use = np.where(Z_meas > 1e-3, Z_meas, target_Z)

            L_rows = []
            for xni, yni, Zi in zip(xn, yn, Z_use):
                Zi = max(float(Zi), 1e-3)
                L_rows.append([-1.0 / Zi, 0.0, xni / Zi, xni * yni, -(1.0 + xni * xni), yni])
                L_rows.append([0.0, -1.0 / Zi, yni / Zi, 1.0 + yni * yni, -xni * yni, -xni])
            L = np.asarray(L_rows, dtype=float)

            e = np.empty(2 * target_uv.shape[0], dtype=float)
            e[0::2] = pix_err[:, 0] / fx
            e[1::2] = pix_err[:, 1] / fy

            v_cam = -gain * (_damped_pinv(L, lam=1e-3) @ e)
            v_cam = v_cam.reshape(6)
            v_eef = self._Adj_cam_eef @ v_cam

            # ---- Build DOF-sized vectors (movable joints only) ----
            q_dof, dq_dof = self._dof_vectors()
            zeros = [0.0] * len(q_dof)

            # Use positional args (more portable across PyBullet builds)
            pos_jac, orn_jac = p.calculateJacobian(
                self.arm_id,
                self.ee_link,
                [0.0, 0.0, 0.0],
                q_dof,
                dq_dof,
                zeros,
                physicsClientId=self.client_id,
            )
            J_full = np.vstack([np.asarray(pos_jac, dtype=float), np.asarray(orn_jac, dtype=float)])

            # Keep columns for the 7 arm joints (map joint index -> DoF column)
            cols = self._arm_dof_cols
            if not cols:
                LOGGER.error("No arm DoF columns mapped; aborting IBVS.")
                return False
            J = J_full[:, cols]

            qdot = _damped_pinv(J, lam=1e-3) @ v_eef
            qdot = np.clip(qdot, -max_joint_vel, max_joint_vel)

            p.setJointMotorControlArray(
                self.arm_id,
                self.arm_joints,
                p.VELOCITY_CONTROL,
                qdot.tolist(),
                forces=[160.0] * len(self.arm_joints),
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)

        LOGGER.warning("IBVS timeout after %.2f s", max_time_s)
        self._halt_arm()
        return False

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _dof_vectors(self) -> Tuple[List[float], List[float]]:
        """Joint vectors for MOVABLE joints only, in the robot DoF order."""
        q, dq = [], []
        for j in self._movable_joint_indices:
            s = p.getJointState(self.arm_id, j, physicsClientId=self.client_id)
            q.append(float(s[0]))
            dq.append(float(s[1]))
        return q, dq

    def _halt_arm(self) -> None:
        p.setJointMotorControlArray(
            self.arm_id,
            self.arm_joints,
            p.VELOCITY_CONTROL,
            [0.0] * len(self.arm_joints),
            forces=[0.0] * len(self.arm_joints),
            physicsClientId=self.client_id,
        )