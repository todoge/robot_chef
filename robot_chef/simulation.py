# robot_chef/simulation.py
"""PyBullet simulation utilities for the robot chef demo."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pybullet as p
import pybullet_data

from .controller import WaypointController
from . import config
from .env.objects.pan import create_pan
from .env.objects.rice_bowl import create_rice_bowl
from .env.particles import ParticleSet, spawn_spheres


def pose6d_to_object_pose(pose: config.Pose6D) -> config.ObjectPose:
    return config.ObjectPose(position=pose.position, orientation_rpy=pose.orientation_rpy)


@dataclass
class LoadedObject:
    """Container storing metadata for spawned objects."""
    body_id: int
    name: str
    pose: config.ObjectPose
    properties: Dict[str, float] = field(default_factory=dict)


class DualArmRobot:
    """Wrapper around two single-arm manipulators in PyBullet (Franka Panda)."""

    def __init__(
        self,
        client_id: int,
        *,
        left_base: config.ObjectPose = config.LEFT_ARM_BASE,
        right_base: config.ObjectPose = config.RIGHT_ARM_BASE,
    ) -> None:
        self.client_id = client_id
        self.left_arm = self._load_arm(left_base, base_name="left")
        self.right_arm = self._load_arm(right_base, base_name="right")

    def _load_arm(self, pose: config.ObjectPose, base_name: str) -> Dict[str, object]:
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
        arm_joint_indices: List[int] = []
        lower_limits: List[float] = []
        upper_limits: List[float] = []
        joint_ranges: List[float] = []
        joint_damping: List[float] = []
        rest_pose: List[float] = []
        finger_joints: List[int] = []

        num_joints = p.getNumJoints(arm, physicsClientId=self.client_id)
        for joint_index in range(num_joints):
            info = p.getJointInfo(arm, joint_index, physicsClientId=self.client_id)
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]

            if joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
                finger_joints.append(joint_index)

            if joint_name.startswith("panda_joint") and joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                arm_joint_indices.append(joint_index)
                lower_limits.append(info[8])
                upper_limits.append(info[9])
                joint_ranges.append(info[9] - info[8] if info[9] > info[8] else 2.0 * math.pi)
                joint_damping.append(info[6])
                rest_pose.append(p.getJointState(arm, joint_index, physicsClientId=self.client_id)[0])

        if len(arm_joint_indices) != 7:
            raise RuntimeError(f"Expected 7 arm joints for Franka, got {len(arm_joint_indices)}")

        return {
            "body": arm,
            "eef": end_effector_link,
            "name": base_name,
            "arm_joints": arm_joint_indices,
            "joint_lower_limits": lower_limits,
            "joint_upper_limits": upper_limits,
            "joint_ranges": joint_ranges,
            "joint_damping": joint_damping,
            "finger_joints": finger_joints,  # usually [9,10]
            "rest_pose": rest_pose,
        }

    # Simple gripper helpers for the Panda
    def open_gripper(self, arm: Dict[str, object], width: float = 0.08) -> None:
        finger_joints = arm.get("finger_joints", [9, 10])  # type: ignore[assignment]
        for j in finger_joints:
            p.setJointMotorControl2(
                arm["body"],  # type: ignore[index]
                int(j),
                p.POSITION_CONTROL,
                targetPosition=width,
                force=100,
                physicsClientId=self.client_id,
            )

    def close_gripper(self, arm: Dict[str, object], force: float = 60.0) -> None:
        finger_joints = arm.get("finger_joints", [9, 10])  # type: ignore[assignment]
        for j in finger_joints:
            p.setJointMotorControl2(
                arm["body"],  # type: ignore[index]
                int(j),
                p.POSITION_CONTROL,
                targetPosition=0.0,
                force=force,
                physicsClientId=self.client_id,
            )


class RobotChefSimulation:
    """High-level helper that loads the environment and robots."""

    def __init__(self, gui: bool = True, recipe: Optional[config.PourTaskConfig] = None) -> None:
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation(physicsClientId=self.client_id)
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)

        # Time step & convenience dt for controllers
        p.setTimeStep(config.SIMULATION_STEP, physicsClientId=self.client_id)
        self.dt = config.SIMULATION_STEP
        self.sim_time = 0.0

        # Scene rotation
        self.recipe_config = recipe
        self.world_yaw_rad = math.radians(recipe.scene.world_yaw_deg) if recipe is not None else 0.0
        self._cos_yaw = math.cos(self.world_yaw_rad)
        self._sin_yaw = math.sin(self.world_yaw_rad)

        # Robots
        left_base = self._rotate_object_pose(config.LEFT_ARM_BASE)
        right_base = self._rotate_object_pose(config.RIGHT_ARM_BASE)
        self.robots = DualArmRobot(self.client_id, left_base=left_base, right_base=right_base)
        # Expose common handles for convenience
        self.right_arm = self.robots.right_arm
        self.left_arm = self.robots.left_arm

        # Objects & particles
        self.objects: Dict[str, LoadedObject] = {}
        self.particles: Optional[ParticleSet] = None

        # IDs the task expects
        self.bowl_body: Optional[int] = None  # rice bowl body id
        self.pan_body: Optional[int] = None   # pan body id

        # Load the world
        self._load_environment()

        # ---- Controller that the task can use ----
        self.controller = WaypointController(
            arm_id=int(self.right_arm["body"]),
            ee_link=int(self.right_arm["eef"]),
            arm_joints=[int(j) for j in self.right_arm["arm_joints"]],  # type: ignore[index]
            dt=self.dt,
            gripper_open=self.gripper_open,
            gripper_close=self.gripper_close,
        )

    # ---------- scene transforms ----------
    def _rotate_point(self, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        if abs(self.world_yaw_rad) < 1e-8:
            return point
        x, y, z = point
        rx = self._cos_yaw * x - self._sin_yaw * y
        ry = self._sin_yaw * x + self._cos_yaw * y
        return (rx, ry, z)

    def _rotate_object_pose(self, pose: config.ObjectPose) -> config.ObjectPose:
        position = self._rotate_point(pose.position)
        roll, pitch, yaw = pose.orientation_rpy
        return config.ObjectPose(position=position, orientation_rpy=(roll, pitch, yaw + self.world_yaw_rad))

    def _rotate_pose6d(self, pose: config.Pose6D) -> config.Pose6D:
        x, y, z = self._rotate_point((pose.x, pose.y, pose.z))
        return config.Pose6D(x=x, y=y, z=z, roll=pose.roll, pitch=pose.pitch, yaw=pose.yaw + self.world_yaw_rad)

    # ---------- world building ----------
    def _load_environment(self) -> None:
        p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        # Table (simple box + 4 legs)
        table_height = config.TABLE_HEIGHT
        table_thickness = 0.05
        table_half_extents = [1.2, 0.9, table_thickness / 2.0]
        top_height = table_height - table_thickness / 2.0

        table_col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=table_half_extents, physicsClientId=self.client_id)
        table_visual_shape = p.createVisualShape(
            shapeType=p.GEOM_BOX, halfExtents=table_half_extents, rgbaColor=[0.7, 0.6, 0.5, 1.0], physicsClientId=self.client_id
        )
        table_body = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col_shape,
            baseVisualShapeIndex=table_visual_shape,
            basePosition=[0, 0, top_height],
            physicsClientId=self.client_id,
        )
        self.objects["table"] = LoadedObject(table_body, "table", config.ObjectPose((0, 0, table_height), (0, 0, 0)))

        leg_half_extents = [0.05, 0.05, table_height / 2.0]
        leg_positions = [
            (table_half_extents[0] - leg_half_extents[0], table_half_extents[1] - leg_half_extents[1]),
            (table_half_extents[0] - leg_half_extents[0], -(table_half_extents[1] - leg_half_extents[1])),
            (-(table_half_extents[0] - leg_half_extents[0]), table_half_extents[1] - leg_half_extents[1]),
            (-(table_half_extents[0] - leg_half_extents[0]), -(table_half_extents[1] - leg_half_extents[1])),
        ]
        leg_color = [0.4, 0.3, 0.2, 1.0]
        leg_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=leg_half_extents, physicsClientId=self.client_id)
        leg_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=leg_half_extents, rgbaColor=leg_color, physicsClientId=self.client_id)
        for idx, (px, py) in enumerate(leg_positions):
            leg_body = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=leg_collision,
                baseVisualShapeIndex=leg_visual,
                basePosition=[px, py, leg_half_extents[2]],
                physicsClientId=self.client_id,
            )
            self.objects[f"table_leg_{idx}"] = LoadedObject(
                leg_body, f"table_leg_{idx}", config.ObjectPose((px, py, leg_half_extents[2]), (0.0, 0.0, 0.0))
            )

        # From recipe config if present; otherwise spawn simple primitives
        if self.recipe_config is not None:
            self._spawn_recipe_objects(self.recipe_config)
        else:
            self._spawn_default_bowls()
            self._spawn_default_pan()

        # Quick-access ids
        if "rice_bowl" in self.objects:
            self.bowl_body = self.objects["rice_bowl"].body_id
        if "pan" in self.objects:
            self.pan_body = self.objects["pan"].body_id

    def _spawn_default_bowls(self) -> None:
        for name, pose in config.BOWL_POSES.items():
            rotated = self._rotate_object_pose(pose)
            col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.07, height=0.05, physicsClientId=self.client_id)
            vis = p.createVisualShape(
                shapeType=p.GEOM_CYLINDER, radius=0.07, length=0.05, rgbaColor=[0.9, 0.9, 0.9, 1.0], physicsClientId=self.client_id
            )
            body = p.createMultiBody(
                baseMass=0.1,
                baseCollisionShapeIndex=col,
                baseVisualShapeIndex=vis,
                basePosition=list(rotated.position),
                baseOrientation=p.getQuaternionFromEuler(rotated.orientation_rpy),
                physicsClientId=self.client_id,
            )
            self.objects[f"bowl_{name}"] = LoadedObject(body, f"bowl_{name}", rotated)

    def _spawn_default_pan(self) -> None:
        pose = self._rotate_object_pose(config.PAN_POSE)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.12, height=0.04, physicsClientId=self.client_id)
        vis = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER, radius=0.12, length=0.04, rgbaColor=[0.2, 0.2, 0.2, 1.0], physicsClientId=self.client_id
        )
        body = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(pose.position),
            baseOrientation=p.getQuaternionFromEuler(pose.orientation_rpy),
            physicsClientId=self.client_id,
        )
        self.objects["pan"] = LoadedObject(body, "pan", pose)

    def _spawn_recipe_objects(self, recipe_cfg: config.PourTaskConfig) -> None:
        # Rice bowl
        bowl_pose = self._rotate_pose6d(recipe_cfg.bowl_pose)
        bowl_body, bowl_props = create_rice_bowl(self.client_id, bowl_pose)
        bowl_obj = LoadedObject(bowl_body, "rice_bowl", pose6d_to_object_pose(bowl_pose), properties=bowl_props)
        self.objects["rice_bowl"] = bowl_obj
        self.bowl_body = bowl_body

        # Pan
        pan_pose = self._rotate_pose6d(recipe_cfg.pan_pose)
        pan_body, pan_props = create_pan(self.client_id, pan_pose)
        pan_obj = LoadedObject(pan_body, "pan", pose6d_to_object_pose(pan_pose), properties=pan_props)
        self.objects["pan"] = pan_obj
        self.pan_body = pan_body

    # ---------- simulation & particles ----------
    def step_simulation(self, steps: int = 120) -> None:
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self.client_id)
            self.sim_time += self.dt

    def spawn_rice_particles(self, count: int, radius: float = 0.005, seed: int = 7) -> ParticleSet:
        bowl = self.objects.get("rice_bowl")
        if bowl is None:
            raise RuntimeError("Rice bowl object not available for particle spawning.")
        bowl_radius = bowl.properties.get("inner_radius", bowl.properties.get("radius", 0.07))
        bowl_height = bowl.properties.get("inner_height", bowl.properties.get("height", 0.05))
        spawn_height = bowl.properties.get("spawn_height", 0.02)
        self.particles = spawn_spheres(
            client_id=self.client_id,
            count=count,
            radius=radius,
            center=bowl.pose.position,
            bowl_radius=bowl_radius,
            bowl_height=bowl_height,
            spawn_height=spawn_height,
            seed=seed,
        )
        return self.particles

    def count_particles_in_pan(self) -> Tuple[int, int, float]:
        if self.particles is None:
            return (0, 0, 0.0)
        pan = self.objects.get("pan")
        if pan is None:
            return (self.particles.count, 0, 0.0)
        center = pan.pose.position
        inner_radius = pan.properties.get("inner_radius", 0.14)
        base_height = pan.properties.get("base_height", center[2])
        lip_height = pan.properties.get("lip_height", center[2] + 0.05)
        return self.particles.count_in_pan(
            client_id=self.client_id,
            center=center,
            inner_radius=inner_radius,
            base_height=base_height,
            lip_height=lip_height,
        )

    # ---------- gripper hooks used by WaypointController ----------
    def gripper_open(self, width: float = 0.08) -> None:
        self.robots.open_gripper(self.right_arm, width=width)

    def gripper_close(self, force: float = 60.0) -> None:
        self.robots.close_gripper(self.right_arm, force=force)

    def disconnect(self) -> None:
        p.disconnect(self.client_id)


# ------------ helper used by StirFry action -------------
def interpolate_circle(center: Tuple[float, float, float], radius: float, angle: float) -> Tuple[float, float, float]:
    """Return a point on a horizontal circle (z stays constant)."""
    x = center[0] + radius * math.cos(angle)
    y = center[1] + radius * math.sin(angle)
    return (x, y, center[2])
