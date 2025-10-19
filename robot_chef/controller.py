# robot_chef/controller.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

LOGGER = logging.getLogger("robot_chef.controller")


@dataclass
class Waypoint:
    pos: Tuple[float, float, float]
    quat: Tuple[float, float, float, float]
    gripper: Optional[str] = None  # "open" | "close" | None
    dwell: float = 0.0             # seconds to hold after reaching


class WaypointController:
    """
    Waypoint controller for a 7-DoF arm (Franka Panda):
      • Tries null-space IK with limits/ranges/(optional) damping and rest pose.
      • Falls back to simpler IK variants if the solver complains.
      • Interpolates joint targets with a per-step clamp to avoid jumps.
      • Calls user-provided gripper callbacks ("open"/"close").
    """

    def __init__(
        self,
        *,
        arm_id: int,
        ee_link: int,
        arm_joints: Sequence[int],
        dt: float,
        gripper_open: Callable[[float], None],
        gripper_close: Callable[[float], None],
        joint_lower_limits: Optional[Sequence[float]] = None,
        joint_upper_limits: Optional[Sequence[float]] = None,
        joint_ranges: Optional[Sequence[float]] = None,
        joint_damping: Optional[Sequence[float]] = None,
        rest_pose: Optional[Sequence[float]] = None,
        max_ik_iters: int = 100,
        ik_threshold: float = 1e-3,
        max_joint_step: float = 0.02,  # rad per physics step
        hold_force: float = 180.0,
    ) -> None:
        self.pb = p
        self.arm_id = arm_id
        self.ee_link = ee_link
        self.joints = list(arm_joints)
        self.ndof = len(self.joints)
        self.dt = float(dt)

        self._open_cb = gripper_open
        self._close_cb = gripper_close

        self.max_ik_iters = int(max_ik_iters)
        self.ik_threshold = float(ik_threshold)
        self.max_joint_step = float(max_joint_step)
        self.hold_force = float(hold_force)

        # Read limits from the sim if not provided
        if any(x is None for x in (joint_lower_limits, joint_upper_limits, joint_ranges, joint_damping, rest_pose)):
            ll, ul, jr, jd, rp = [], [], [], [], []
            for j in self.joints:
                ji = self.pb.getJointInfo(self.arm_id, j)
                lo, hi = ji[8], ji[9]
                ll.append(float(lo))
                ul.append(float(hi))
                jr.append(float(hi - lo) if hi > lo else 2.0 * np.pi)
                jd.append(float(ji[6]))
                rp.append(float(self.pb.getJointState(self.arm_id, j)[0]))
            joint_lower_limits = ll
            joint_upper_limits = ul
            joint_ranges = jr
            joint_damping = jd
            rest_pose = rp

        # Ensure all arrays are exactly ndof long (PyBullet is strict)
        def _fit(arr: Sequence[float]) -> List[float]:
            arr = list(map(float, arr))
            if len(arr) < self.ndof:
                arr = arr + [arr[-1] if arr else 0.0] * (self.ndof - len(arr))
            if len(arr) > self.ndof:
                arr = arr[: self.ndof]
            return arr

        self.ll = _fit(joint_lower_limits)      # type: ignore[arg-type]
        self.ul = _fit(joint_upper_limits)      # type: ignore[arg-type]
        self.jr = _fit(joint_ranges)            # type: ignore[arg-type]
        self.jd = _fit(joint_damping)           # type: ignore[arg-type]
        self.rp = _fit(rest_pose)               # type: ignore[arg-type]

        # Small safety margin for checking limits after IK
        self._limit_eps = 0.08  # rad

        self._queue: List[Waypoint] = []

    # ------------- Public API -------------

    def clear(self) -> None:
        self._queue.clear()

    def add_waypoint(
        self,
        pos: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
        *,
        gripper: Optional[str] = None,
        dwell: float = 0.0,
    ) -> None:
        self._queue.append(Waypoint(pos=pos, quat=quat, gripper=gripper, dwell=float(dwell)))

    def execute_planned(self) -> bool:
        if not self._queue:
            LOGGER.warning("No waypoints queued.")
            return False

        for idx, wp in enumerate(self._queue):
            # Optional pre-move gripper
            if wp.gripper == "open":
                self._open_cb(0.08)
            elif wp.gripper == "close":
                self._close_cb(60.0)

            ok, q_target, reason = self._solve_ik_resilient(wp.pos, wp.quat)
            if not ok or q_target is None:
                LOGGER.error(
                    "IK failed at waypoint %d -> %s | pos=%s quat=%s",
                    idx,
                    reason or "unspecified",
                    np.round(wp.pos, 4),
                    np.round(wp.quat, 4),
                )
                self._queue.clear()
                return False

            if not self._blend_to_joint_target(q_target, idx):
                self._queue.clear()
                return False

            # Dwell to hold pose
            steps = int(max(0, round(wp.dwell / self.dt)))
            for _ in range(steps):
                self.pb.stepSimulation()

        self._queue.clear()
        return True

    # ------------- IK & motion internals -------------

    def _solve_ik_resilient(
        self,
        pos: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
    ) -> Tuple[bool, Optional[Sequence[float]], str]:
        """
        Try IK in three tiers:
          1) Nullspace IK with damping and restPose.
          2) Nullspace IK without damping (some PyBullet builds are picky).
          3) Vanilla IK (no limits). As last resort.
        Also jitters wrist-yaw slightly if orientation is near-singular.
        """
        # If orientation is almost identity or perfectly aligned, add a tiny yaw to avoid singularities.
        quat = tuple(map(float, quat))
        if abs(quat[3]) > 0.999 or np.linalg.norm(quat[:3]) < 1e-3:
            # rotate about tool-z by ~0.5 deg
            yaw = np.deg2rad(0.5)
            # Convert to euler, add yaw, back to quat
            eul = self.pb.getEulerFromQuaternion(quat)
            quat = self.pb.getQuaternionFromEuler((eul[0], eul[1], eul[2] + yaw))

        # Tier 1: full nullspace with damping
        ok, q, reason = self._ik_nullspace(pos, quat, use_damping=True)
        if ok:
            return True, q, ""

        # Tier 2: nullspace without damping (avoid PyBullet damping-length issues)
        LOGGER.debug("Retrying IK without damping.")
        ok, q, reason2 = self._ik_nullspace(pos, quat, use_damping=False)
        if ok:
            return True, q, ""

        # Tier 3: vanilla IK (no limits) — least robust but sometimes succeeds
        LOGGER.debug("Retrying IK with vanilla solver (no limits).")
        try:
            q = self.pb.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                targetPosition=pos,
                targetOrientation=quat,
                maxNumIterations=self.max_ik_iters,
                residualThreshold=self.ik_threshold,
            )
        except Exception as exc:
            LOGGER.exception("Vanilla IK raised: %s", exc)
            return False, None, f"vanilla_ik_exception:{exc}"

        if q is None:
            return False, None, "vanilla_ik_none"

        q = list(q[: self.ndof])
        if not np.all(np.isfinite(q)):
            return False, None, "vanilla_ik_non_finite"

        return True, q, ""

    def _ik_nullspace(
        self,
        pos: Tuple[float, float, float],
        quat: Tuple[float, float, float, float],
        *,
        use_damping: bool,
    ) -> Tuple[bool, Optional[Sequence[float]], str]:
        kwargs = dict(
            lowerLimits=self.ll,
            upperLimits=self.ul,
            jointRanges=self.jr,
            restPoses=self.rp,
            maxNumIterations=self.max_ik_iters,
            residualThreshold=self.ik_threshold,
        )
        if use_damping:
            kwargs["jointDamping"] = self.jd

        try:
            q = self.pb.calculateInverseKinematics(
                self.arm_id,
                self.ee_link,
                targetPosition=pos,
                targetOrientation=quat,
                **kwargs,
            )
        except Exception as exc:
            LOGGER.debug("Nullspace IK raised (%s).", exc)
            return False, None, f"nullspace_exception:{exc}"

        if q is None:
            return False, None, "nullspace_none"

        q = list(q[: self.ndof])
        if not np.all(np.isfinite(q)):
            return False, None, "nullspace_non_finite"

        # Soft-limit check (allow a little slack)
        for qi, lo, hi in zip(q, self.ll, self.ul):
            if hi > lo and not (lo - self._limit_eps <= qi <= hi + self._limit_eps):
                return False, None, f"joint_out_of_range:{qi:.3f} not in [{lo:.3f},{hi:.3f}]"

        return True, q, ""

    def _blend_to_joint_target(self, target_q: Sequence[float], waypoint_idx: int) -> bool:
        target_q = np.asarray(target_q, dtype=float)
        cur = np.array([self.pb.getJointState(self.arm_id, j)[0] for j in self.joints], dtype=float)

        # Already close?
        if np.linalg.norm(target_q - cur, ord=np.inf) < 1e-3:
            self._apply_joint_positions(target_q)
            for _ in range(2):
                self.pb.stepSimulation()
            return True

        max_delta = float(np.max(np.abs(target_q - cur)))
        n_steps = max(1, int(np.ceil(max_delta / self.max_joint_step)))

        for k in range(1, n_steps + 1):
            alpha = k / float(n_steps)
            q_k = (1.0 - alpha) * cur + alpha * target_q
            self._apply_joint_positions(q_k)
            self.pb.stepSimulation()

        # Allow extra settling steps for the motors to converge
        final_error = target_q - np.array([self.pb.getJointState(self.arm_id, j)[0] for j in self.joints])
        max_err = np.linalg.norm(final_error, ord=np.inf)
        settle_budget = max(5, int(max_err / self.max_joint_step))
        for _ in range(settle_budget):
            if max_err < 5e-2:
                break
            self._apply_joint_positions(target_q)
            self.pb.stepSimulation()
            final_error = target_q - np.array([self.pb.getJointState(self.arm_id, j)[0] for j in self.joints])
            max_err = np.linalg.norm(final_error, ord=np.inf)

        if max_err >= 5e-2:
            LOGGER.error(
                "Waypoint %d joint tolerance exceeded. max_error=%.4f rad",
                waypoint_idx,
                float(max_err),
            )
            return False

        return True

    def _apply_joint_positions(self, q: Sequence[float]) -> None:
        for j, qj in zip(self.joints, q):
            self.pb.setJointMotorControl2(
                self.arm_id,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(qj),
                force=self.hold_force,
            )
