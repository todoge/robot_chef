"""Inverse kinematics helpers."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import pybullet as p

LOGGER = logging.getLogger(__name__)


@dataclass
class IKResult:
    success: bool
    joint_positions: Tuple[float, ...]
    attempts: int


def solve(
    sim,
    arm: Dict[str, object],
    target_position: Sequence[float],
    target_orientation: Sequence[float],
    attempts: int = 5,
    position_jitter: float = 0.005,
) -> IKResult:
    """Solve IK with jitter retries and joint limit clipping."""
    body = arm["body"]
    eef = arm["eef"]
    joint_indices = arm["arm_joints"]
    lower_limits = list(arm["joint_lower_limits"])
    upper_limits = list(arm["joint_upper_limits"])
    joint_ranges = list(arm["joint_ranges"])
    rest_pose = list(arm["rest_pose"])

    last_solution = ()
    for attempt in range(1, attempts + 1):
        offset_position = list(target_position)
        if attempt > 1:
            offset_position = [
                target_position[i] + random.uniform(-position_jitter, position_jitter) for i in range(3)
            ]
        solution = p.calculateInverseKinematics(
            bodyUniqueId=body,
            endEffectorLinkIndex=eef,
            targetPosition=offset_position,
            targetOrientation=target_orientation,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_pose,
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=sim.client_id,
        )
        if not solution:
            continue
        if len(solution) < len(joint_indices):
            continue
        joint_positions = list(solution[: len(joint_indices)])
        joint_positions = tuple(
            max(lower_limits[idx], min(upper_limits[idx], joint_positions[idx])) for idx in range(len(joint_indices))
        )
        last_solution = joint_positions
        if _pose_within_tolerance(sim, arm, joint_positions, target_position, target_orientation):
            return IKResult(True, joint_positions, attempt)
    if last_solution:
        LOGGER.warning("IK fallback used after %d attempts; tolerances not met.", attempts)
        return IKResult(True, tuple(last_solution), attempts)
    return IKResult(False, last_solution, attempts)


def _pose_within_tolerance(sim, arm, joint_positions, target_position, target_orientation) -> bool:
    """Verify IK solution by forwarding without altering the simulation state."""
    if not joint_positions:
        return False
    body = arm["body"]
    joint_indices = arm["arm_joints"]
    current_states = [p.getJointState(body, j, physicsClientId=sim.client_id) for j in joint_indices]
    current_positions = [state[0] for state in current_states]
    current_velocities = [state[1] for state in current_states]
    try:
        for idx, joint_index in enumerate(joint_indices):
            p.resetJointState(
                bodyUniqueId=body,
                jointIndex=joint_index,
                targetValue=joint_positions[idx],
                targetVelocity=0.0,
                physicsClientId=sim.client_id,
            )
        link_state = p.getLinkState(
            body,
            arm["eef"],
            computeForwardKinematics=True,
            physicsClientId=sim.client_id,
        )
        if not link_state:
            return False
        position = link_state[4]
        orientation = link_state[5]
        pos_error = math.dist(position, target_position)
        dot = sum(a * b for a, b in zip(orientation, target_orientation))
        if dot < 0.0:
            dot = -dot
        dot = max(min(dot, 1.0), -1.0)
        ori_error = math.acos(dot) * 2.0
        pos_ok = pos_error <= 0.01
        ori_ok = ori_error <= math.radians(12.0)
        return pos_ok and ori_ok
    finally:
        # Restore previous joint states to avoid perturbing the simulation.
        for idx, joint_index in enumerate(joint_indices):
            p.resetJointState(
                bodyUniqueId=body,
                jointIndex=joint_index,
                targetValue=current_positions[idx],
                targetVelocity=current_velocities[idx],
                physicsClientId=sim.client_id,
            )
