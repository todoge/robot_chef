"""Image-based visual servo controller for the wrist-mounted camera."""

from __future__ import annotations

import logging
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Linear algebra helpers

def _skew(vec: Sequence[float]) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=float,
    )


def _adjoint(T: np.ndarray) -> np.ndarray:
    """Adjoint matrix that maps twists: v_out = Ad_T * v_in."""
    R = T[:3, :3]
    t = T[:3, 3]
    adj = np.zeros((6, 6), dtype=float)
    adj[:3, :3] = R
    adj[3:, 3:] = R
    adj[3:, :3] = _skew(t) @ R
    return adj


def _damped_pseudoinverse(J: np.ndarray, damping: float = 1e-4) -> np.ndarray:
    """Tikhonov-damped pseudoinverse using SVD."""
    U, S, Vt = np.linalg.svd(J, full_matrices=False)
    S_damped = S / (S * S + damping * damping)
    return Vt.T @ np.diag(S_damped) @ U.T


# --------------------------------------------------------------------------- #
# Controller

class VisionRefineController:
    """Maps camera pixel error to joint velocities using IBVS."""

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
        self.arm_joints = list(int(j) for j in arm_joints)
        self.dt = float(dt)
        self.camera = camera
        self._gripper_open = gripper_open or (lambda width=0.08: None)
        self._gripper_close = gripper_close or (lambda force=60.0: None)

        self._T_e_c = np.eye(4, dtype=float)
        if handeye_T_cam_in_eef is not None:
            T = np.asarray(handeye_T_cam_in_eef, dtype=float)
            if T.shape != (4, 4):
                raise ValueError("handeye_T_cam_in_eef must be 4x4")
            self._T_e_c = T
        self._Adj_e_c = _adjoint(self._T_e_c)

        self._num_joints = p.getNumJoints(self.arm_id, physicsClientId=self.client_id)
        self._arm_cols = list(self.arm_joints)

    # ------------------------------------------------------------------ #
    # Convenience wrappers

    def open_gripper(self, width: float = 0.08) -> None:
        self._gripper_open(width)

    def close_gripper(self, force: float = 60.0) -> None:
        self._gripper_close(force)

    # ------------------------------------------------------------------ #
    # Motion primitives

    def move_waypoint(self, pos: Sequence[float], quat_xyzw: Sequence[float], steps: int = 180) -> bool:
        """Move to the desired end-effector pose via nullspace IK."""
        try:
            q_sol = p.calculateInverseKinematics(
                bodyUniqueId=self.arm_id,
                endEffectorLinkIndex=self.ee_link,
                targetPosition=list(float(v) for v in pos),
                targetOrientation=list(float(v) for v in quat_xyzw),
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
        except Exception as exc:
            LOGGER.error("IK failed: %s", exc)
            return False
        if not q_sol or len(q_sol) <= max(self.arm_joints):
            LOGGER.error("IK returned insufficient joint angles")
            return False

        target_positions = [float(q_sol[j]) for j in self.arm_joints]
        p.setJointMotorControlArray(
            bodyUniqueId=self.arm_id,
            jointIndices=self.arm_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_positions,
            positionGains=[0.08] * len(self.arm_joints),
            forces=[200.0] * len(self.arm_joints),
            physicsClientId=self.client_id,
        )
        for _ in range(max(steps, 1)):
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
        target_uv = np.asarray(target_uv, dtype=float)
        target_Z = np.asarray(target_Z, dtype=float)
        if target_uv.ndim != 2 or target_uv.shape[1] != 2:
            raise ValueError("target_uv must be shaped (N, 2)")
        if target_Z.ndim != 1 or target_Z.shape[0] != target_uv.shape[0]:
            raise ValueError("target_Z must be length-N vector")

        intrinsics = self.camera.intrinsics
        fx = float(intrinsics[0, 0])
        fy = float(intrinsics[1, 1])
        cx = float(intrinsics[0, 2])
        cy = float(intrinsics[1, 2])

        max_iters = max(1, int(max_time_s / max(self.dt, 1e-4)))
        for idx in range(max_iters):
            uv_meas, Z_meas = get_features()
            if uv_meas is None or Z_meas is None:
                LOGGER.warning("IBVS step %d: feature callback returned None", idx)
                return False
            uv_meas = np.asarray(uv_meas, dtype=float).reshape(-1, 2)
            Z_meas = np.asarray(Z_meas, dtype=float).reshape(-1)
            if uv_meas.shape != target_uv.shape:
                LOGGER.error("IBVS: feature shape mismatch (%s vs %s)", uv_meas.shape, target_uv.shape)
                return False

            pixel_error = uv_meas - target_uv
            rms_error = math.sqrt(float(np.mean(pixel_error ** 2)))
            depth_error = float(np.mean(np.abs(Z_meas - target_Z)))
            if rms_error < pixel_tol and depth_error < depth_tol:
                LOGGER.info(
                    "VisionRefineController: IBVS grasp refine success (rms=%.2fpx depth=%.4fm)",
                    rms_error,
                    depth_error,
                )
                self._halt_arm()
                return True

            x_norm = (uv_meas[:, 0] - cx) / fx
            y_norm = (uv_meas[:, 1] - cy) / fy
            Z_use = np.where(Z_meas > 1e-3, Z_meas, target_Z)

            L_rows = []
            for xn, yn, Zi in zip(x_norm, y_norm, Z_use):
                Zi = max(float(Zi), 1e-3)
                L_rows.append([-1.0 / Zi, 0.0, xn / Zi, xn * yn, -(1.0 + xn * xn), yn])
                L_rows.append([0.0, -1.0 / Zi, yn / Zi, 1.0 + yn * yn, -xn * yn, -xn])
            L = np.asarray(L_rows, dtype=float)  # shape (2N, 6)

            error_vec = np.empty(2 * target_uv.shape[0], dtype=float)
            error_vec[0::2] = pixel_error[:, 0] / fx
            error_vec[1::2] = pixel_error[:, 1] / fy

            v_cam = -gain * (_damped_pseudoinverse(L, damping=1e-4) @ error_vec)
            v_e = self._Adj_e_c @ v_cam  # twist in end-effector frame

            # Convert to world twist using current EE orientation
            link_state = p.getLinkState(
                self.arm_id,
                self.ee_link,
                computeForwardKinematics=True,
                physicsClientId=self.client_id,
            )
            eef_quat = link_state[5]
            R_we = np.array(p.getMatrixFromQuaternion(eef_quat), dtype=float).reshape(3, 3)
            linear_world = R_we @ v_e[:3]
            angular_world = R_we @ v_e[3:]
            twist_world = np.concatenate([linear_world, angular_world])

            q_full, dq_full, dd_full = self._full_joint_vectors()
            pos_jac, orn_jac = p.calculateJacobian(
                self.arm_id,
                self.ee_link,
                [0.0, 0.0, 0.0],
                q_full,
                dq_full,
                dd_full,
                physicsClientId=self.client_id,
            )
            J_full = np.vstack([np.asarray(pos_jac, dtype=float), np.asarray(orn_jac, dtype=float)])
            J = J_full[:, self._arm_cols]

            qdot = _damped_pseudoinverse(J, damping=1e-4) @ twist_world
            qdot = np.clip(qdot, -max_joint_vel, max_joint_vel)

            p.setJointMotorControlArray(
                bodyUniqueId=self.arm_id,
                jointIndices=self.arm_joints,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocities=qdot.tolist(),
                forces=[160.0] * len(self.arm_joints),
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)

        LOGGER.warning("VisionRefineController: IBVS grasp refine timeout (%.2fs)", max_time_s)
        self._halt_arm()
        return False

    # ------------------------------------------------------------------ #
    # Internals

    def _full_joint_vectors(self) -> Tuple[List[float], List[float], List[float]]:
        q, dq = [], []
        for joint_idx in range(self._num_joints):
            state = p.getJointState(self.arm_id, joint_idx, physicsClientId=self.client_id)
            q.append(float(state[0]))
            dq.append(float(state[1]))
        dd = [0.0] * self._num_joints
        return q, dq, dd

    def _halt_arm(self) -> None:
        p.setJointMotorControlArray(
            bodyUniqueId=self.arm_id,
            jointIndices=self.arm_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[0.0] * len(self.arm_joints),
            forces=[0.0] * len(self.arm_joints),
            physicsClientId=self.client_id,
        )


__all__ = ["VisionRefineController"]
