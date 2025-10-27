"""Simulation harness for the Robot Chef pouring task.

- Builds table, bowl, pan and a stove support block.
- Uses AABB-based snapping so the stove sits on the table, the pan sits on the stove,
  and the bowl sits on the table (deterministic contact layout).
- Adds two fixed pedestals ("mounts") and places the Panda arms on their tops at
  the table's top-Z, with XY positions computed from the table AABB + a safety margin,
  so mounts never intersect the table, regardless of scene.world_yaw_deg.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from .config import Pose6D, MainConfig
from .env.objects import pan as pan_factory
from .env.objects import rice_bowl as bowl_factory
from .env.objects.particles import ParticleSet, spawn_particles

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data containers


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


# --------------------------------------------------------------------------- #
# Simulation


class RobotChefSimulation:
    """Encapsulates the physics world, robot arms, and task objects."""

    def __init__(self, gui: bool, recipe: MainConfig):
        self.recipe = recipe
        self.gui = gui

        # -----------------------------
        # 1. Connect to PyBullet
        # -----------------------------
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        LOGGER.info("Connected to PyBullet (client_id=%s, gui=%s)", self.client_id, gui)

        # -----------------------------
        # 2. Configure physics
        # -----------------------------
        self.dt = 1.0 / 240.0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(numSolverIterations=120, fixedTimeStep=self.dt, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)

        # -----------------------------
        # 3. Initialize internal state
        # -----------------------------
        self.objects: Dict[str, Dict[str, object]] = {}
        self.rice_particles: Optional[ParticleSet] = None
        self.egg_particles: Optional[ParticleSet] = None

        # -----------------------------
        # 4. Build the environment
        # -----------------------------
        self._setup_environment()

        # -----------------------------
        # 5. Place pedestals for robot arms
        # -----------------------------
        left_mount_id, right_mount_id, z_table = self._place_pedestals_clear_of_table(
            side=0.20,   # 20 cm square column
            margin=0.08  # 8 cm clearance from table edges
        )
        self.objects["left_mount"] = {"body_id": left_mount_id, "top_z": z_table}
        self.objects["right_mount"] = {"body_id": right_mount_id, "top_z": z_table}

        # -----------------------------
        # 6. Spawn Panda arms on pedestals
        # -----------------------------
        yaw = math.radians(self.recipe.scene.world_yaw_deg)
        left_base_pos, _ = p.getBasePositionAndOrientation(left_mount_id, physicsClientId=self.client_id)
        right_base_pos, _ = p.getBasePositionAndOrientation(right_mount_id, physicsClientId=self.client_id)

        self.left_arm = self._spawn_arm(
            base_position=[left_base_pos[0], left_base_pos[1], z_table],
            base_orientation=p.getQuaternionFromEuler([0.0, 0.0, yaw + math.pi / 2.0]),
        )
        self.right_arm = self._spawn_arm(
            base_position=[right_base_pos[0], right_base_pos[1], z_table],
            base_orientation=p.getQuaternionFromEuler([0.0, 0.0, yaw - math.pi / 2.0]),
        )

        # -----------------------------
        # 7. Attach spatula to right hand
        # -----------------------------
        self._attach_spatula_to_hand(
            arm_state=self.right_arm,
            spatula_sdf_path="robot_chef/env/spatula/model.sdf",
        )
        # -----------------------------
        # 8. Default settings and simulation stabilization
        # -----------------------------
        self._active_arm_name = "right"
        self.gripper_open()
        self.step_simulation(steps=60)

    # ------------------------------------------------------------------ #
    # Environment & object setup

    def _setup_environment(self) -> None:
        """Load and place environment objects (plane, table, bowl, pan, stove)."""
        LOGGER.info("Setting up environment objects")

        # 1. Plane
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.objects["plane"] = {"body_id": plane_id}

        # 2. Table
        table_height_offset = -0.05
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

        # 3. Bowl and pan
        bowl_pose = self._apply_world_yaw(self.recipe.bowl_pose)
        bowl_id, bowl_props = bowl_factory.create_rice_bowl(self.client_id, pose=bowl_pose)
        self.objects["bowl"] = {"body_id": bowl_id, "properties": bowl_props, "pose": bowl_pose}

        pan_pose = self._apply_world_yaw(self.recipe.pan_pose)
        pan_id, pan_props = pan_factory.create_pan(self.client_id, pose=pan_pose)
        self.objects["pan"] = {"body_id": pan_id, "properties": pan_props, "pose": pan_pose}

        # 4. Stove support block
        stove_half_xy = float(pan_props["half_side"]) + float(pan_props["wall_thickness"]) + 0.02
        stove_height = max(0.01, float(pan_props["depth"]) * 0.9)
        stove_half_extent = (stove_half_xy, stove_half_xy, stove_height / 2.0)

        stove_pose = Pose6D(
            x=pan_pose.x,
            y=pan_pose.y,
            z=pan_pose.z - stove_height,
            roll=0.0,
            pitch=0.0,
            yaw=pan_pose.yaw,
        )
        stove_orientation = p.getQuaternionFromEuler(stove_pose.orientation_rpy)
        stove_collision = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=stove_half_extent, physicsClientId=self.client_id
        )
        stove_visual = p.createVisualShape(
            p.GEOM_BOX, halfExtents=stove_half_extent, rgbaColor=[0.96, 0.96, 0.96, 1.0], physicsClientId=self.client_id
        )
        stove_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=stove_collision,
            baseVisualShapeIndex=stove_visual,
            basePosition=[stove_pose.x, stove_pose.y, stove_pose.z],
            baseOrientation=stove_orientation,
            physicsClientId=self.client_id,
        )
        p.changeDynamics(stove_id, -1, lateralFriction=1.1, spinningFriction=0.9, rollingFriction=0.6, physicsClientId=self.client_id)
        self.objects["stove_block"] = {"body_id": stove_id, "half_extents": stove_half_extent, "pose": stove_pose}

        # -----------------------------
        # 5. Snap objects to supports
        # -----------------------------
        z_table = self._get_table_top_z()
        self._place_on_support(stove_id, support_top_z=z_table)
        stove_top = self._get_body_top_z(stove_id)
        self._place_on_support(pan_id, support_top_z=stove_top)
        self._place_on_support(bowl_id, support_top_z=z_table)

        # Update pan base height for later metrics
        pan_aabb_min, _ = p.getAABB(pan_id, physicsClientId=self.client_id)
        pan_props["base_height"] = float(pan_aabb_min[2])

        # Let contacts settle for one frame
        p.stepSimulation(physicsClientId=self.client_id)

    # ---------- Mounts / pedestals with clearance ---------- #

    def _get_table_footprint(self) -> Tuple[Tuple[float, float], float, float, float, float]:
        """
        Returns (center_xy, half_len_x, half_len_y, top_z, yaw_rad) for the table.
        half_len_x/half_len_y are from the world-aligned AABB (good enough for clearance).
        """
        tbl = self.objects["table"]["body_id"]
        (aabb_min, aabb_max) = p.getAABB(tbl, physicsClientId=self.client_id)
        cx = 0.5 * (aabb_min[0] + aabb_max[0])
        cy = 0.5 * (aabb_min[1] + aabb_max[1])
        half_len_x = 0.5 * (aabb_max[0] - aabb_min[0])
        half_len_y = 0.5 * (aabb_max[1] - aabb_min[1])
        top_z = float(aabb_max[2])
        yaw = math.radians(self.recipe.scene.world_yaw_deg)
        return (cx, cy), half_len_x, half_len_y, top_z, yaw

    def _table_local_axes(self, yaw: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Unit axes of the table frame projected in world XY.
        +x_table points 'forward', +y_table to the table's left.
        """
        ux = (math.cos(yaw), math.sin(yaw))           # +x_table in world
        uy = (-math.sin(yaw), math.cos(yaw))          # +y_table in world
        return ux, uy

    def _place_pedestals_clear_of_table(self, side: float = 0.20, margin: float = 0.08) -> Tuple[int, int, float]:
        """
        Compute two mount positions 'behind' the table (−x_table direction)
        with clearance = margin + pedestal_half. Builds pedestals and returns
        (left_mount_id, right_mount_id, top_z).
        """
        (cx, cy), half_x, half_y, top_z, yaw = self._get_table_footprint()
        ux, uy = self._table_local_axes(yaw)

        ped_half = side * 0.5
        # Distance behind back edge (−x_table) to avoid collisions
        back_clear = half_x + margin + ped_half

        # Lateral (±y_table) placement near the table edges with clearance
        lat_clear = max(0.0, half_y - margin - ped_half)
        # World XY for left/right mounts
        # left = +y_table; right = −y_table
        left_xy = (cx - ux[0] * back_clear + uy[0] * lat_clear,
                   cy - ux[1] * back_clear + uy[1] * lat_clear)
        right_xy = (cx - ux[0] * back_clear - uy[0] * lat_clear,
                    cy - ux[1] * back_clear - uy[1] * lat_clear)

        left_id = self._spawn_pedestal("left_mount", xy=left_xy, top_z=top_z, side=side)
        right_id = self._spawn_pedestal("right_mount", xy=right_xy, top_z=top_z, side=side)
        return left_id, right_id, top_z

    def _spawn_pedestal(self, name: str, xy: Tuple[float, float], top_z: float, side: float = 0.24) -> int:
        """
        Create a fixed mount (square column) from floor to top_z at world XY.
        The column is a box of size (side, side, height). Uses collision + visual.
        """
        height = max(0.05, float(top_z))  # from z=0 (floor) up to top_z
        half_extents = (side / 2.0, side / 2.0, height / 2.0)
        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents, physicsClientId=self.client_id)
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.35, 0.35, 0.38, 1.0],  # neutral “metallic” tone
            physicsClientId=self.client_id,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=[float(xy[0]), float(xy[1]), half_extents[2]],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            physicsClientId=self.client_id,
        )
        p.changeDynamics(body_id, -1, lateralFriction=1.0, spinningFriction=0.6, physicsClientId=self.client_id)
        return body_id
    
    def _attach_spatula_to_hand(self, arm_state: ArmState, spatula_sdf_path: str) -> None:
        """
        Load the spatula SDF model and attach it to the Panda hand via a fixed constraint.
        """
        # Load the spatula from SDF
        spatula_ids = p.loadSDF(spatula_sdf_path, physicsClientId=self.client_id)
        if not spatula_ids:
            raise RuntimeError(f"Failed to load spatula from {spatula_sdf_path}")
        spatula_id = spatula_ids[0]  # SDF can return multiple bodies; usually first is main

        # Get Panda hand pose
        hand_pos, hand_orn = p.getLinkState(arm_state.body_id, arm_state.ee_link, physicsClientId=self.client_id)[:2]

        # Optionally adjust offset so spatula handle sits nicely in the gripper
        # Example: translate along local Z of hand (you may need to tweak)
        offset_pos = [0.0, 0.0, 0.1]  # 10 cm along hand Z
        spatula_pos = [hand_pos[0] + offset_pos[0], hand_pos[1] + offset_pos[1], hand_pos[2] + offset_pos[2]]
        spatula_orn = hand_orn

        # Move spatula to hand
        p.resetBasePositionAndOrientation(spatula_id, spatula_pos, spatula_orn, physicsClientId=self.client_id)

        # Create fixed constraint to attach spatula to hand
        p.createConstraint(
            parentBodyUniqueId=arm_state.body_id,
            parentLinkIndex=arm_state.ee_link,
            childBodyUniqueId=spatula_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],  # you can tweak to align
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=self.client_id,
        )

    # ---------- Arm spawning ---------- #

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
        for j in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(arm_id, j, physicsClientId=self.client_id)
            if info[12].decode() == "panda_hand":
                ee_link_index = j
                break
        if ee_link_index is None:
            LOGGER.warning("panda_hand link not found; defaulting to final arm joint link")
            ee_link_index = joint_indices[-1]

        # Finger joints.
        finger_joint_names = {"panda_finger_joint1", "panda_finger_joint2"}
        fingers: List[int] = []
        for j in range(p.getNumJoints(arm_id, physicsClientId=self.client_id)):
            info = p.getJointInfo(arm_id, j, physicsClientId=self.client_id)
            if info[1].decode() in finger_joint_names:
                fingers.append(j)
        if len(fingers) != 2:
            raise RuntimeError("Failed to locate Panda finger joints")

        # Initialize arm joints.
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
        pan_entry = self.objects.get("pan")
        if not pan_entry:
            LOGGER.warning("Cannot spawn rice particles before pan is created")
            return None
        pan_id = pan_entry["body_id"]
        position, _ = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.client_id)
        pan_props = pan_entry["properties"]
        print("pan_props", pan_props)
        particle_set = spawn_particles(
            client_id=self.client_id,
            count=count,
            radius=radius,
            center=position,
            pan_radius=float(pan_props["inner_radius"]),
            pan_height=float(pan_props["depth"]),
            spawn_height=float(pan_props["spawn_height"]),
            seed=seed,
        )
        self.rice_particles = particle_set
        LOGGER.info("Spawned %d rice particles (radius=%.4f)", particle_set.count, radius)
        return particle_set
    
    def spawn_egg_particles(
        self,
        count: int,
        radius: float = 0.005,
        seed: int = 7,
    ) -> Optional[ParticleSet]:
        pan_entry = self.objects.get("pan")
        if not pan_entry:
            LOGGER.warning("Cannot spawn egg particles before pan is created")
            return None
        pan_id = pan_entry["body_id"]
        position, _ = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.client_id)
        pan_props = pan_entry["properties"]
        particle_set = spawn_particles(
            client_id=self.client_id,
            count=count,
            radius=radius,
            center=position,
            pan_radius=float(pan_props["inner_radius"]),
            pan_height=float(pan_props["depth"]),
            spawn_height=float(pan_props["spawn_height"]),
            seed=seed,
        )
        self.egg_particles = particle_set
        LOGGER.info("Spawned %d egg particles (radius=%.4f)", particle_set.count, radius)
        return particle_set

    def count_particles_in_pan(self) -> Dict[str, float]:
        pan_entry = self.objects.get("pan")
        if not pan_entry or not self.rice_particles:
            return {"total": 0, "in_pan": 0, "transfer_ratio": 0.0}
        pan_id = pan_entry["body_id"]
        center, _ = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.client_id)
        props = pan_entry["properties"]
        total, inside, ratio = self.rice_particles.count_in_pan(
            client_id=self.client_id,
            center=center,
            inner_radius=float(props["inner_radius"]),
            base_height=float(props["base_height"]),
            lip_height=float(props["lip_height"]),
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

    def _get_table_top_z(self) -> float:
        """Return the world Z of the table's top using its AABB."""
        tbl = self.objects.get("table")
        if not tbl:
            return 0.0
        aabb_min, aabb_max = p.getAABB(tbl["body_id"], physicsClientId=self.client_id)
        return float(aabb_max[2])

    def _get_body_top_z(self, body_id: int) -> float:
        aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=self.client_id)
        return float(aabb_max[2])

    def _place_on_support(self, body_id: int, support_top_z: float) -> None:
        """Translate body along +Z so its bottom AABB touches support_top_z."""
        aabb_min, _ = p.getAABB(body_id, physicsClientId=self.client_id)
        bottom_z = float(aabb_min[2])
        dz = support_top_z - bottom_z
        if abs(dz) < 1e-4:
            return
        base_pos, base_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=self.client_id)
        p.resetBasePositionAndOrientation(
            body_id,
            (float(base_pos[0]), float(base_pos[1]), float(base_pos[2]) + dz),
            base_orn,
            physicsClientId=self.client_id,
        )

    # Backwards-compatible alias (used by older code)
    def _place_on_table(self, body_id: int, z_table: float) -> None:
        self._place_on_support(body_id, support_top_z=z_table)


__all__ = ["RobotChefSimulation", "ArmState"]
