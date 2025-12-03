"""Vision-based refinement controller using IBVS + PyBullet Jacobians."""

from __future__ import annotations
import logging
import math
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger("robot_chef.controller")

def _damped_pinv(J: np.ndarray, lam: float = 1e-4) -> np.ndarray:
    JTJ = J.T @ J
    n = JTJ.shape[0]
    return np.linalg.solve(JTJ + (lam * lam) * np.eye(n), J.T)

def _quat_mul(q1: Sequence[float], q2: Sequence[float]) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
    ], dtype=float)

def _clamp(v: np.ndarray, lim: float) -> np.ndarray:
    n = np.linalg.norm(v, ord=np.inf)
    return v if n <= lim or n == 0 else (v * (lim / n))

class VisionRefineController:
    """IBVS refinement with correct PyBullet Jacobian usage."""

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
        self.client_id = client_id
        self.arm_id = arm_id
        self.ee_link = ee_link
        self.arm_joints = list(arm_joints)
        self.dt = float(dt)
        self.camera = camera
        self.handeye_T_cam_in_eef = np.eye(4) if handeye_T_cam_in_eef is None else np.array(handeye_T_cam_in_eef, float)
        self._open = gripper_open or (lambda width=0.08: None)
        self._close = gripper_close or (lambda force=60.0: None)

        self._num_joints = p.getNumJoints(self.arm_id, physicsClientId=self.client_id)


    def _full_joint_vectors(self) -> Tuple[List[float], List[float], List[float]]:
        q, dq = [], []
        for j in range(self._num_joints):
            st = p.getJointState(self.arm_id, j, physicsClientId=self.client_id)
            q.append(float(st[0]))
            dq.append(float(st[1]))
        dd = [0.0] * self._num_joints
        return q, dq, dd

    def _arm_column_selector(self) -> List[int]:
        return list(self.arm_joints)

    def open_gripper(self, width: float = 0.08) -> None:
        self._open(width)

    def close_gripper(self, force: float = 60.0) -> None:
        self._close(force)

    def move_waypoint(self, pos: Sequence[float], quat_xyzw: Sequence[float], max_steps: int = 240) -> bool:
        """Robust waypoint move using IK + position control."""
        try:
            q_sol = p.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                targetPosition=list(pos),
                targetOrientation=list(quat_xyzw),
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
        except Exception as exc:
            LOGGER.error("IK exception: %s", exc)
            return False

        if not q_sol or len(q_sol) < max(self.arm_joints) + 1:
            LOGGER.error("IK returned undersized solution.")
            return False

        for _ in range(max_steps):
            for j_idx, j in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.arm_id, j, p.POSITION_CONTROL,
                    targetPosition=float(q_sol[j]),
                    force=200.0,
                    positionGain=0.1, velocityGain=1.0,
                    physicsClientId=self.client_id,
                )
            p.stepSimulation(physicsClientId=self.client_id)
        return True

    def refine_to_features_ibvs(
        self,
        *,
        get_features: Callable[[], Tuple[np.ndarray, np.ndarray]], 
        target_uv: np.ndarray,
        target_Z: np.ndarray,
        pixel_tol: float = 3.0,
        depth_tol: float = 0.008,
        max_time_s: float = 4.0,
        gain: float = 0.35,
        max_joint_vel: float = 0.5,
    ) -> bool:
        """Closed-loop IBVS refinement. Uses full Jacobian then selects arm columns."""
        dt = self.dt
        max_steps = max(1, int(max_time_s / dt))
        arm_cols = self._arm_column_selector()

        for step in range(max_steps):
            uv, Z = get_features()
            if uv is None or Z is None or len(uv) != len(target_uv):
                LOGGER.warning("IBVS: invalid features at step %d", step)
                return False

            e_px = (uv - target_uv).reshape(-1, 1)  # (2N,1)
            rms_px = float(np.sqrt(np.mean(e_px**2)))
            mean_dz = float(np.mean(np.abs(Z - target_Z)))
            if rms_px < pixel_tol and mean_dz < depth_tol:
                LOGGER.info("VisionRefineController: IBVS grasp refine success (rms=%.2f px, dZ=%.4f m)", rms_px, mean_dz)
                return True

            N = len(Z)
            fx = fy = 1.0 
            L_blocks = []
            for i in range(N):
                Zi = max(1e-3, float(Z[i]))
                ui, vi = float(uv[2*i]), float(uv[2*i+1])
                L_i = np.array([
                    [-fx / Zi, 0.0, ui / Zi,  ui*vi / fx, -(fx*fx + ui*ui) / fx,  vi],
                    [0.0, -fy / Zi, vi / Zi,  (fy*fy + vi*vi) / fy, -ui*vi / fy, -ui],
                ], dtype=float)
                L_blocks.append(L_i)
            L = np.vstack(L_blocks) 

            v_cam = -gain * _damped_pinv(L, lam=1e-3) @ e_px 
            v_cam = v_cam.flatten() 

            v_eef = v_cam 

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
            Jp = np.asarray(pos_jac, dtype=float) 
            Jo = np.asarray(orn_jac, dtype=float) 
            J_full = np.vstack([Jp, Jo])          
            J = J_full[:, arm_cols]

            qdot_arm = _damped_pinv(J, lam=1e-3) @ v_eef.reshape(6, 1)
            qdot_arm = _clamp(qdot_arm.flatten(), max_joint_vel)

            for k, j in enumerate(self.arm_joints):
                p.setJointMotorControl2(
                    self.arm_id, j, p.VELOCITY_CONTROL,
                    targetVelocity=float(qdot_arm[k]),
                    force=120.0,
                    physicsClientId=self.client_id,
                )
            p.stepSimulation(physicsClientId=self.client_id)

        LOGGER.warning("VisionRefineController: IBVS timeout (no convergence)")
        return False
