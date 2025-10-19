"""Simulation harness for the Robot Chef pouring task."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from .config import Pose6D, PourTaskConfig
from .env.objects import pan as pan_factory
from .env.objects import rice_bowl as bowl_factory
from .env.particles import ParticleSet, spawn_spheres

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


class RobotChefSimulation:
    """Encapsulates the physics world, robot arms, and task objects."""

    def __init__(self, gui: bool, recipe: PourTaskConfig):
        self.recipe = recipe
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        LOGGER.info("Connected to PyBullet with client_id=%s (gui=%s)", self.client_id, gui)

        self.dt = 1.0 / 240.0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(numSolverIterations=120, fixedTimeStep=self.dt, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        self.objects: Dict[str, Dict[str, object]] = {}
        self.particles: Optional[ParticleSet] = None

        self._setup_environment()
        self.left_arm = self._spawn_arm(base_position=[-0.25, 0.55, 0.0], base_orientation=p.getQuaternionFromEuler([0.0, 0.0, math.pi / 2.0]))
        self.right_arm = self._spawn_arm(base_position=[-0.25, -0.55, 0.0], base_orientation=p.getQuaternionFromEuler([0.0, 0.0, -math.pi / 2.0]))

        # Default active arm is the right arm for pouring motions.
        self._active_arm_name = "right"
        self.gripper_open()
        self.step_simulation(steps=60)

    # ------------------------------------------------------------------ #
    # Environment & robot helpers

    def _setup_environment(self) -> None:
        LOGGER.info("Setting up environment objects")
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.objects["plane"] = {"body_id": plane_id}

        # Position the table so that its surface aligns with z ~ 0.75 (matching recipe poses).
        table_height_offset = 0.0
        table_pos = [0.5, 0.0, table_height_offset]
        table_orientation = p.getQuaternionFromEuler([0.0, 0.0, math.radians(self.recipe.scene.world_yaw_deg)])
        table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=table_pos,
            baseOrientation=table_orientation,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.objects["table"] = {"body_id": table_id, "base_position": table_pos}

        bowl_pose = self._apply_world_yaw(self.recipe.bowl_pose)
        bowl_id, bowl_props = bowl_factory.create_rice_bowl(self.client_id, pose=bowl_pose)
        self.objects["bowl"] = {"body_id": bowl_id, "properties": bowl_props, "pose": bowl_pose}

        pan_pose = self._apply_world_yaw(self.recipe.pan_pose)
        pan_id, pan_props = pan_factory.create_pan(self.client_id, pose=pan_pose)
        self.objects["pan"] = {"body_id": pan_id, "properties": pan_props, "pose": pan_pose}

    def _spawn_arm(self, base_position: Sequence[float], base_orientation: Sequence[float]) -> ArmState:
        flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS
        arm_id = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=base_position,
            baseOrientation=base_orientation,
            useFixedBase=True,
            flags=flags,
            physicsClientId=self.client_id,
        )
        joint_indices: List[int] = []
        joint_lower: List[float] = []
        joint_upper: List[float] = []
        joint_rest: List[float] = []
        joint_ranges: List[float] = []

        for j in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(arm_id, j, physicsClientId=self.client_id)
            joint_type = info[2]
            if joint_type != p.JOINT_REVOLUTE:
                continue
            joint_indices.append(j)
            joint_lower.append(info[8])
            joint_upper.append(info[9])
            joint_ranges.append(info[9] - info[8])
            joint_rest.append((info[8] + info[9]) * 0.5 if info[8] < info[9] else 0.0)
            if len(joint_indices) == 7:
                break

        if len(joint_indices) != 7:
            raise RuntimeError("Expected Franka Panda arm with 7 revolute joints")

        ee_link = p.getBodyInfo(arm_id, physicsClientId=self.client_id)[0].decode()
        # Panda end-effector link is named "panda_hand". Find its index.
        ee_link_index = None
        for j in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(arm_id, j, physicsClientId=self.client_id)
            if info[12].decode() == "panda_hand":
                ee_link_index = j
                break
        if ee_link_index is None:
            LOGGER.warning("panda_hand link not found; defaulting to final arm joint link")
            ee_link_index = joint_indices[-1]

        finger_joint_names = {"panda_finger_joint1", "panda_finger_joint2"}
        fingers: List[int] = []
        for j in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(arm_id, j, physicsClientId=self.client_id)
            if info[1].decode() in finger_joint_names:
                fingers.append(j)
        if len(fingers) != 2:
            raise RuntimeError("Failed to locate Panda finger joints")

        for idx in joint_indices:
            p.resetJointState(arm_id, idx, targetValue=0.0, targetVelocity=0.0, physicsClientId=self.client_id)
            p.setJointMotorControl2(
                arm_id,
                idx,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=240.0,
                physicsClientId=self.client_id,
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

    # ------------------------------------------------------------------ #
    # Public API

    def step_simulation(self, steps: int) -> None:
        for _ in range(max(1, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_joint_state(self, arm: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        arm_state = self._get_arm(arm)
        q = []
        dq = []
        for idx in arm_state.joint_indices:
            state = p.getJointState(arm_state.body_id, idx, physicsClientId=self.client_id)
            q.append(state[0])
            dq.append(state[1])
        return np.asarray(q, dtype=float), np.asarray(dq, dtype=float)

    def set_joint_positions(self, q_target: Sequence[float], arm: Optional[str] = None, kp: float = 0.3) -> None:
        arm_state = self._get_arm(arm)
        if len(q_target) != len(arm_state.joint_indices):
            raise ValueError("q_target must have length 7 for the Panda arm")
        p.setJointMotorControlArray(
            bodyUniqueId=arm_state.body_id,
            jointIndices=list(arm_state.joint_indices),
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(q_target),
            positionGains=[kp] * len(arm_state.joint_indices),
            forces=[240.0] * len(arm_state.joint_indices),
            physicsClientId=self.client_id,
        )

    def set_joint_velocities(
        self,
        qdot_target: Sequence[float],
        arm: Optional[str] = None,
        max_force: float = 180.0,
    ) -> None:
        arm_state = self._get_arm(arm)
        if len(qdot_target) != len(arm_state.joint_indices):
            raise ValueError("qdot_target must have length 7 for the Panda arm")
        p.setJointMotorControlArray(
            bodyUniqueId=arm_state.body_id,
            jointIndices=list(arm_state.joint_indices),
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=list(qdot_target),
            forces=[max_force] * len(arm_state.joint_indices),
            physicsClientId=self.client_id,
        )

    def gripper_open(self, width: float = 0.08, arm: Optional[str] = None) -> None:
        arm_state = self._get_arm(arm)
        target = max(width * 0.5, 0.0)
        for joint_id in arm_state.finger_joints:
            p.setJointMotorControl2(
                arm_state.body_id,
                joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=100.0,
                physicsClientId=self.client_id,
            )

    def gripper_close(self, force: float = 60.0, arm: Optional[str] = None) -> None:
        arm_state = self._get_arm(arm)
        for joint_id in arm_state.finger_joints:
            p.setJointMotorControl2(
                arm_state.body_id,
                joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=force,
                physicsClientId=self.client_id,
            )

    def spawn_rice_particles(
        self,
        count: int,
        radius: float = 0.005,
        seed: int = 7,
    ) -> Optional[ParticleSet]:
        bowl_entry = self.objects.get("bowl")
        if not bowl_entry:
            LOGGER.warning("Cannot spawn particles before bowl is created")
            return None
        bowl_id = bowl_entry["body_id"]
        position, orientation = p.getBasePositionAndOrientation(bowl_id, physicsClientId=self.client_id)
        bowl_props = bowl_entry["properties"]
        particle_set = spawn_spheres(
            client_id=self.client_id,
            count=count,
            radius=radius,
            center=position,
            bowl_radius=bowl_props["inner_radius"],
            bowl_height=bowl_props["inner_height"],
            spawn_height=bowl_props["spawn_height"],
            seed=seed,
        )
        self.particles = particle_set
        LOGGER.info("Spawned %d rice particles (radius=%.4f)", particle_set.count, radius)
        return particle_set

    def count_particles_in_pan(self) -> Dict[str, float]:
        pan_entry = self.objects.get("pan")
        if not pan_entry or not self.particles:
            return {"total": 0, "in_pan": 0, "transfer_ratio": 0.0}
        pan_id = pan_entry["body_id"]
        center, _ = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.client_id)
        props = pan_entry["properties"]
        total, inside, ratio = self.particles.count_in_pan(
            client_id=self.client_id,
            center=center,
            inner_radius=props["inner_radius"],
            base_height=props["base_height"],
            lip_height=props["lip_height"],
        )
        return {"total": total, "in_pan": inside, "transfer_ratio": ratio}

    def disconnect(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # ------------------------------------------------------------------ #
    # Internal helpers

    def _get_arm(self, arm: Optional[str]) -> ArmState:
        arm_name = arm or self._active_arm_name
        if arm_name == "left":
            return self.left_arm
        if arm_name == "right":
            return self.right_arm
        raise ValueError(f"Unknown arm '{arm_name}'")

    def _apply_world_yaw(self, pose: Pose6D) -> Pose6D:
        yaw = math.radians(self.recipe.scene.world_yaw_deg)
        cos_y = math.cos(yaw)
        sin_y = math.sin(yaw)
        x = pose.x * cos_y - pose.y * sin_y
        y = pose.x * sin_y + pose.y * cos_y
        return Pose6D(
            x=x,
            y=y,
            z=pose.z,
            roll=pose.roll,
            pitch=pose.pitch,
            yaw=pose.yaw + yaw,
        )


__all__ = ["RobotChefSimulation", "ArmState"]
