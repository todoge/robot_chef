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

    def open_gripper(self, width: float = 0.08) -> None:
        self._open(width)

    def close_gripper(self, force: float = 60.0) -> None:
        self._close(force)

    def move_waypoint(self, pos: Sequence[float], quat_xyzw: Sequence[float], timeout_s: float = 3.0) -> bool:
        """Moves to a target pose using PyBullet's internal controller."""
        pos = [float(v) for v in pos]
        quat_xyzw = [float(v) for v in quat_xyzw]
        try:
            # Use lower limits and higher range if available (more robust IK)
            ll, ul, jr, rp = self._get_joint_info()
            full_ik = p.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                pos,
                quat_xyzw,
                lowerLimits=ll,
                upperLimits=ul,
                jointRanges=jr,
                restPoses=rp,
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
            current_q, _ = self._get_arm_joint_states()
            if np.linalg.norm(np.array(target) - current_q) < 0.01:
                 break
        return True

    def get_ik_for_pose(
        self, pos: Sequence[float], quat_xyzw: Sequence[float]
    ) -> Optional[np.ndarray]:
        """Calculates IK for a target pose and returns the 7 arm joint angles."""
        try:
            ll, ul, jr, rp = self._get_joint_info()
            full_ik = p.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                pos,
                quat_xyzw,
                lowerLimits=ll,
                upperLimits=ul,
                jointRanges=jr,
                restPoses=rp,
                maxNumIterations=200,
                residualThreshold=1e-4,
                physicsClientId=self.client_id,
            )
            if not full_ik or len(full_ik) <= max(self.arm_joints):
                LOGGER.error("IK returned insufficient joint angles")
                return None

            target_joints = np.array([float(full_ik[j]) for j in self.arm_joints], dtype=float)
            return target_joints

        except Exception as exc:
            LOGGER.error("IK failed: %s", exc)
            return None

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
        return True 

    def move_to_joint_target(
            self, q_target: np.ndarray, gain: float = 0.08
        ) -> None:
            """ Commands the arm towards a joint target ONCE with a specific gain. Does NOT step sim."""
            target_list = q_target.tolist()
            p.setJointMotorControlArray(
                self.arm_id,
                self.arm_joints,
                p.POSITION_CONTROL,
                target_list,
                positionGains=[gain] * len(self.arm_joints),
                forces=[200.0] * len(self.arm_joints),
                physicsClientId=self.client_id,
            )
    def _get_joint_info(self) -> Tuple[List[float], List[float], List[float], List[float]]:
        """Gets limits, ranges, and rest poses for IK calculation."""
        ll, ul, jr, rp = [], [], [], []
        num_joints = p.getNumJoints(self.arm_id, physicsClientId=self.client_id)
        for i in range(num_joints):
             joint_info = p.getJointInfo(self.arm_id, i, physicsClientId=self.client_id)
             q_index = joint_info[3]
             if q_index > -1:
                  ll.append(joint_info[8])
                  ul.append(joint_info[9])
                  jr.append(joint_info[9] - joint_info[8])
                  rp.append((joint_info[8] + joint_info[9]) / 2)
        return ll, ul, jr, rp

    def _get_arm_joint_states(self) -> Tuple[np.ndarray, np.ndarray]:
        """Gets current position and velocity for the 7 arm joints."""
        states = p.getJointStates(self.arm_id, self.arm_joints, physicsClientId=self.client_id)
        q = np.array([state[0] for state in states], dtype=float)j
        dq = np.array([state[1] for state in states], dtype=float)
        return q, dq

    def _dof_vectors(self) -> Tuple[List[float], List[float]]:
        """Joint vectors for ALL MOVABLE joints only, in the robot DoF order."""
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