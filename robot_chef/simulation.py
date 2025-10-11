"""PyBullet simulation utilities for the robot chef demo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pybullet as p
import pybullet_data

from . import config


@dataclass
class LoadedObject:
    """Container storing metadata for spawned objects."""

    body_id: int
    name: str
    pose: config.ObjectPose


class DualArmRobot:
    """Wrapper around two single-arm manipulators in PyBullet."""

    def __init__(self, client_id: int) -> None:
        self.client_id = client_id
        self.left_arm = self._load_arm(config.LEFT_ARM_BASE, base_name="left")
        self.right_arm = self._load_arm(config.RIGHT_ARM_BASE, base_name="right")

    def _load_arm(self, pose: config.ObjectPose, base_name: str) -> Dict[str, int]:
        position = pose.position
        orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
        arm = p.loadURDF(
            fileName="franka_panda/panda.urdf",
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        end_effector_link = 11  # panda_hand
        return {"body": arm, "eef": end_effector_link, "name": base_name}

    def reset_arm(self, arm: Dict[str, int]) -> None:
        for joint_index in range(p.getNumJoints(arm["body"], physicsClientId=self.client_id)):
            p.resetJointState(arm["body"], joint_index, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client_id)

    def move_eef_to_pose(
        self,
        arm: Dict[str, int],
        position: Tuple[float, float, float],
        orientation_rpy: Tuple[float, float, float],
        num_steps: int = 240,
    ) -> None:
        orientation = p.getQuaternionFromEuler(orientation_rpy)
        joint_positions = p.calculateInverseKinematics(
            arm["body"],
            arm["eef"],
            targetPosition=position,
            targetOrientation=orientation,
            physicsClientId=self.client_id,
        )
        controllable_joints = range(7)
        for step in range(num_steps):
            for joint_index in controllable_joints:
                p.setJointMotorControl2(
                    arm["body"],
                    joint_index,
                    p.POSITION_CONTROL,
                    targetPosition=joint_positions[joint_index],
                    force=200,
                    physicsClientId=self.client_id,
                )
            p.stepSimulation(physicsClientId=self.client_id)

    def open_gripper(self, arm: Dict[str, int], width: float = 0.08) -> None:
        finger_joints = [9, 10]
        for joint_index in finger_joints:
            p.setJointMotorControl2(
                arm["body"],
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=width,
                force=100,
                physicsClientId=self.client_id,
            )

    def close_gripper(self, arm: Dict[str, int], force: float = 60) -> None:
        finger_joints = [9, 10]
        for joint_index in finger_joints:
            p.setJointMotorControl2(
                arm["body"],
                joint_index,
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=force,
                physicsClientId=self.client_id,
            )


class RobotChefSimulation:
    """High-level helper that loads the environment and robots."""

    def __init__(self, gui: bool = True) -> None:
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        self.robots = DualArmRobot(self.client_id)
        self.objects: Dict[str, LoadedObject] = {}
        self._load_environment()

    def _load_environment(self) -> None:
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        table_height = config.TABLE_HEIGHT
        table_thickness = 0.05
        table_half_extents = [0.8, 0.8, table_thickness / 2.0]
        table_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents, physicsClientId=self.client_id)
        table_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=table_half_extents,
            rgbaColor=[0.8, 0.7, 0.6, 1.0],
            physicsClientId=self.client_id,
        )
        table_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col_shape,
            baseVisualShapeIndex=table_visual_shape,
            basePosition=[0, 0, table_height - table_thickness / 2.0],
            physicsClientId=self.client_id,
        )
        self.objects["table"] = LoadedObject(table_body, "table", config.ObjectPose((0, 0, table_height), (0, 0, 0)))

        # Load kitchenware as simple cylinders/boxes for visualization.
        self._spawn_bowls()
        self._spawn_pan()
        self._spawn_plate()
        self._spawn_sauce_bottle()

    def _spawn_bowls(self) -> None:
        for name, pose in config.BOWL_POSES.items():
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.07, height=0.05, physicsClientId=self.client_id)
            vis = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER,
                radius=0.07,
                length=0.05,
                rgbaColor=[0.9, 0.9, 0.9, 1.0],
                physicsClientId=self.client_id,
            )
            body = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=list(pose.position),
                physicsClientId=self.client_id,
            )
            self.objects[f"bowl_{name}"] = LoadedObject(body, f"bowl_{name}", pose)

    def _spawn_pan(self) -> None:
        pose = config.PAN_POSE
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.04, physicsClientId=self.client_id)
        vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.12,
            length=0.04,
            rgbaColor=[0.2, 0.2, 0.2, 1.0],
            physicsClientId=self.client_id,
        )
        body = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pose.position),
            physicsClientId=self.client_id,
        )
        self.objects["pan"] = LoadedObject(body, "pan", pose)

    def _spawn_plate(self) -> None:
        pose = config.PLATE_POSE
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.02, physicsClientId=self.client_id)
        vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.12,
            length=0.02,
            rgbaColor=[1.0, 1.0, 1.0, 1.0],
            physicsClientId=self.client_id,
        )
        body = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pose.position),
            physicsClientId=self.client_id,
        )
        self.objects["plate"] = LoadedObject(body, "plate", pose)

    def _spawn_sauce_bottle(self) -> None:
        pose = config.SAUCE_BOTTLE_POSE
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.035, height=0.18, physicsClientId=self.client_id)
        vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.035,
            length=0.18,
            rgbaColor=[0.9, 0.2, 0.1, 1.0],
            physicsClientId=self.client_id,
        )
        body = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pose.position),
            physicsClientId=self.client_id,
        )
        self.objects["sauce_bottle"] = LoadedObject(body, "sauce_bottle", pose)

    def step_simulation(self, steps: int = 120) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)

    def disconnect(self) -> None:
        p.disconnect(self.client_id)


def interpolate_circle(center: Tuple[float, float, float], radius: float, angle: float) -> Tuple[float, float, float]:
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    return (x, y, center[2])
