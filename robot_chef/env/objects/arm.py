import logging
import math
from dataclasses import dataclass
from typing import List, Tuple

import pybullet as p

from ...config import Pose6D

LOGGER = logging.getLogger(__name__)


@dataclass
class ArmState:
    body_id: int
    joint_indices: Tuple[int, ...]
    ee_link: int
    finger_joints: Tuple[int, int]
    joint_lower: Tuple[float, ...]
    joint_upper: Tuple[float, ...]
    joint_ranges: Tuple[float, ...]
    joint_rest: Tuple[float, ...]


def create_arm(
    client_id: int,
    pose: Pose6D,
) -> int:
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
    base_orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    arm_id = p.loadURDF(
        "franka_panda/panda.urdf",
        basePosition=pose.position,
        baseOrientation=base_orientation,
        useFixedBase=True,
        flags=flags,
        physicsClientId=client_id,
    )
    joint_indices: List[int] = []
    joint_lower: List[float] = []
    joint_upper: List[float] = []
    joint_rest: List[float] = []
    joint_ranges: List[float] = []

    for j in range(p.getNumJoints(arm_id, physicsClientId=client_id)):
        info = p.getJointInfo(arm_id, j, physicsClientId=client_id)
        joint_type = info[2]
        if joint_type != p.JOINT_REVOLUTE:
            continue
        lower = float(info[8])
        upper = float(info[9])
        joint_indices.append(j)
        joint_lower.append(lower)
        joint_upper.append(upper)
        joint_ranges.append((upper - lower) if upper > lower else 2.0 * math.pi)
        joint_rest.append((lower + upper) * 0.5 if upper > lower else 0.0)
        if len(joint_indices) == 7:
            break

    if len(joint_indices) != 7:
        raise RuntimeError("Expected Franka Panda arm with 7 revolute joints")

    # Find Panda end-effector link "panda_hand".
    ee_link_index = None
    for j in range(p.getNumJoints(arm_id, physicsClientId=client_id)):
        info = p.getJointInfo(arm_id, j, physicsClientId=client_id)
        if info[12].decode() == "panda_hand":
            ee_link_index = j
            break
    if ee_link_index is None:
        LOGGER.warning("panda_hand link not found; defaulting to final arm joint link")
        ee_link_index = joint_indices[-1]

    # Finger joints.
    finger_joint_names = {"panda_finger_joint1", "panda_finger_joint2"}
    fingers: List[int] = []
    for j in range(p.getNumJoints(arm_id, physicsClientId=client_id)):
        info = p.getJointInfo(arm_id, j, physicsClientId=client_id)
        if info[1].decode() in finger_joint_names:
            fingers.append(j)
    if len(fingers) != 2:
        raise RuntimeError("Failed to locate Panda finger joints")

    # Initialize arm joints.
    for idx in joint_indices:
        p.resetJointState(arm_id, idx, targetValue=0.0, targetVelocity=0.0, physicsClientId=client_id)
        p.setJointMotorControl2(
            arm_id,
            idx,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            force=240.0,
            physicsClientId=client_id,
        )

    return ArmState(
        body_id=arm_id,
        joint_indices=tuple(joint_indices),
        ee_link=ee_link_index,
        finger_joints=(fingers[0], fingers[1]),
        joint_lower=tuple(joint_lower),
        joint_upper=tuple(joint_upper),
        joint_ranges=tuple(joint_ranges),
        joint_rest=tuple(joint_rest),
    )