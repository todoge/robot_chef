import logging
import math
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p
import pybullet_data

from robot_chef.config import MainConfig, Pose6D
from robot_chef.env.objects import arm as arm_factory
from robot_chef.env.objects import pan as pan_factory
from robot_chef.env.objects import particles as particle_factory
from robot_chef.env.objects import pedestal as pedestal_factory
from robot_chef.env.objects import rice_bowl as bowl_factory
from robot_chef.env.objects import stove as stove_factory
from robot_chef.utils import apply_world_yaw

logger = logging.getLogger(__name__)


class StirSimulator:

    def __init__(self, gui: bool, cfg: MainConfig):
        self.gui = gui
        self.cfg = cfg
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        logger.info("Connected to PyBullet (client_id=%s, gui=%s)", self.client_id, gui)

    # region : Public APIs

    def setup(self) -> None:
        logger.info("Setting up simulator")
        self._active_arm_name = "right"
        self._setup_physics()
        self._spawn_objects()
        self._snap_objects_to_supports()
        self.open_grippers()
        self.step_simulation(steps=60)
        logger.info("Completed simulator setup")

    def step_simulation(self, steps: int) -> None:
        for _ in range(max(1, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)

    def open_grippers(self, width: float = 0.08, arm: Optional[str] = None) -> None:
        arm_state = self.get_arm(arm)
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
        arm_state = self.get_arm(arm)
        for joint_id in arm_state.finger_joints:
            p.setJointMotorControl2(
                arm_state.body_id,
                joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                force=force,
                physicsClientId=self.client_id,
            )

    def get_arm(self, arm: Optional[str]) -> arm_factory.ArmState:
        arm_name = arm or self._active_arm_name
        if arm_name == "left":
            return self.left_arm
        if arm_name == "right":
            return self.right_arm
        raise ValueError(f"Unknown arm '{arm_name}'")

    def get_joint_state(
        self, arm: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        arm_state = self.get_arm(arm)
        q = []
        dq = []
        for idx in arm_state.joint_indices:
            state = p.getJointState(
                arm_state.body_id, idx, physicsClientId=self.client_id
            )
            q.append(state[0])
            dq.append(state[1])
        return np.asarray(q, dtype=float), np.asarray(dq, dtype=float)

    def set_joint_positions(
        self,
        target_pos: Sequence[float],
        arm: Optional[str] = None,
        kp: float = 0.3,
    ) -> None:
        arm_state = self.get_arm(arm)
        if len(target_pos) != len(arm_state.joint_indices):
            raise ValueError(
                (
                    f"q_target must have length 7 for the Panda arm : "
                    f"{len(target_pos)} : {str(target_pos)}"
                )
            )
        p.setJointMotorControlArray(
            bodyUniqueId=arm_state.body_id,
            jointIndices=list(arm_state.joint_indices),
            controlMode=p.POSITION_CONTROL,
            targetPositions=list(target_pos),
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
        arm_state = self.get_arm(arm)
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

    def disconnect(self) -> None:
        if p.isConnected(self.client_id):
            p.disconnect(self.client_id)

    # endregion : Public APIs

    # region : Object Spawners

    def _spawn_objects(self) -> None:
        self.objects: dict[str, dict[str, object]] = {}
        self._spawn_plane()
        self._spawn_table()
        self._spawn_bowl()
        self._spawn_pan()
        self._spawn_stove()
        self._spawn_spatula()
        self._spawn_pedestals()
        self._spawn_panda_arms()
        self._spawn_rice_particles()
        self._spawn_egg_particles()

    def _spawn_plane(self) -> None:
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
        self.objects["plane"] = {"body_id": plane_id}

    def _spawn_table(self) -> None:
        table_height_offset = -0.05
        table_pos = [0.5, 0.0, table_height_offset]
        table_orientation = p.getQuaternionFromEuler(
            [0.0, 0.0, math.radians(self.cfg.scene.world_yaw_deg)]
        )
        table_id = p.loadURDF(
            "table/table.urdf",
            basePosition=table_pos,
            baseOrientation=table_orientation,
            useFixedBase=True,
            physicsClientId=self.client_id,
        )
        self.objects["table"] = {"body_id": table_id, "base_position": table_pos}

    def _spawn_bowl(self) -> None:
        bowl_pose = apply_world_yaw(self.cfg.bowl_pose, self.cfg.scene.world_yaw_deg)
        bowl_id, bowl_props = bowl_factory.create_rice_bowl(
            self.client_id,
            pose=bowl_pose,
        )
        self.objects["bowl"] = {
            "body_id": bowl_id,
            "properties": bowl_props,
            "pose": bowl_pose,
        }

    def _spawn_pan(self) -> None:
        pan_pose = apply_world_yaw(self.cfg.pan_pose, self.cfg.scene.world_yaw_deg)
        pan_id, pan_props = pan_factory.create_pan(
            self.client_id,
            pose=pan_pose,
        )
        self.objects["pan"] = {
            "body_id": pan_id,
            "properties": pan_props,
            "pose": pan_pose,
        }

    def _spawn_stove(self) -> None:
        pan = self.objects["pan"]
        pan_pose, pan_props = pan["pose"], pan["properties"]
        stove_half_xy = (
            float(pan_props["half_side"]) + float(pan_props["wall_thickness"]) + 0.02
        )
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
        stove_id = stove_factory.create_stove(
            self.client_id,
            stove_pose,
            stove_half_extent,
        )
        self.objects["stove"] = {
            "body_id": stove_id,
            "pose": stove_pose,
            "half_extents": stove_half_extent,
        }

    def _spawn_spatula(self) -> None:
        spatula_id = p.loadSDF(
            "robot_chef/env/spatula/model.sdf",
            physicsClientId=self.client_id,
        )[0]
        base_orientation = p.getQuaternionFromEuler(
            self.cfg.spatula_pose.orientation_rpy
        )
        p.resetBasePositionAndOrientation(
            spatula_id,
            self.cfg.spatula_pose.position,
            base_orientation,
            physicsClientId=self.client_id,
        )
        self.objects["spatula"] = {"body_id": spatula_id}

    def _spawn_pedestals(self) -> None:
        aabb_min, aabb_max = p.getAABB(
            self.objects["table"]["body_id"],
            physicsClientId=self.client_id,
        )
        cx = 0.5 * (aabb_min[0] + aabb_max[0])
        cy = 0.5 * (aabb_min[1] + aabb_max[1])
        half_x = 0.5 * (aabb_max[0] - aabb_min[0])
        half_y = 0.5 * (aabb_max[1] - aabb_min[1])
        top_z = float(aabb_max[2])
        yaw = math.radians(self.cfg.scene.world_yaw_deg)
        ux = math.cos(yaw), math.sin(yaw)
        uy = -math.sin(yaw), math.cos(yaw)
        side, margin = 0.2, 0.08
        ped_half = side * 0.5
        back_clear = half_x + margin + ped_half
        lat_clear = max(0.0, half_y - margin - ped_half)
        left_xy = (
            cx - ux[0] * back_clear + uy[0] * lat_clear,
            cy - ux[1] * back_clear + uy[1] * lat_clear,
        )
        right_xy = (
            cx - ux[0] * back_clear - uy[0] * lat_clear,
            cy - ux[1] * back_clear - uy[1] * lat_clear,
        )
        height = max(0.05, float(top_z))
        half_extents = (side / 2.0, side / 2.0, height / 2.0)
        left_pedestal_id = pedestal_factory.create_pedestal(
            self.client_id,
            Pose6D(left_xy[0], left_xy[1], half_extents[2], 0.0, 0.0, 0.0),
            half_extents,
        )
        right_pedestal_id = pedestal_factory.create_pedestal(
            self.client_id,
            Pose6D(right_xy[0], right_xy[1], half_extents[2], 0.0, 0.0, 0.0),
            half_extents,
        )
        self.objects["left_pedestal"] = {"body_id": left_pedestal_id, "top_z": top_z}
        self.objects["right_pedestal"] = {"body_id": right_pedestal_id, "top_z": top_z}

    def _spawn_panda_arms(self) -> None:
        yaw = math.radians(self.cfg.scene.world_yaw_deg)
        left_base_pos, _ = p.getBasePositionAndOrientation(
            self.objects["left_pedestal"]["body_id"],
            physicsClientId=self.client_id,
        )
        right_base_pos, _ = p.getBasePositionAndOrientation(
            self.objects["right_pedestal"]["body_id"],
            physicsClientId=self.client_id,
        )
        table_top = self._get_body_top_z("table")
        self.left_arm = arm_factory.create_arm(
            self.client_id,
            Pose6D(
                left_base_pos[0],
                left_base_pos[1],
                table_top,
                0.0,
                0.0,
                yaw + math.pi / 2.0,
            ),
        )
        self.right_arm = arm_factory.create_arm(
            self.client_id,
            Pose6D(
                right_base_pos[0],
                right_base_pos[1],
                table_top,
                0.0,
                0.0,
                yaw - math.pi / 2.0,
            ),
        )
        self.objects["left_arm"] = {"body_id": self.left_arm.body_id}
        self.objects["right_arm"] = {"body_id": self.right_arm.body_id}

    def _spawn_rice_particles(self) -> None:
        if self.cfg.rice_particles <= 0:
            logger.info(f"No rice particles to spawn")
            return
        pan = self.objects["pan"]
        position = p.getBasePositionAndOrientation(
            pan["body_id"],
            physicsClientId=self.client_id,
        )[0]
        self._rice_particles = particle_factory.spawn_particles(
            client_id=self.client_id,
            count=self.cfg.rice_particles,
            radius=0.005,
            center=position,
            pan_radius=float(pan["properties"]["inner_radius"]),
            pan_height=float(pan["properties"]["depth"]),
            spawn_height=float(pan["properties"]["spawn_height"]),
            seed=7,
        )

    def _spawn_egg_particles(self) -> None:
        if self.cfg.egg_particles <= 0:
            logger.info(f"No egg particles to spawn")
            return
        pan = self.objects["pan"]
        position = p.getBasePositionAndOrientation(
            pan["body_id"],
            physicsClientId=self.client_id,
        )[0]
        self._egg_particles = particle_factory.spawn_particles(
            client_id=self.client_id,
            count=self.cfg.egg_particles,
            radius=0.005,
            center=position,
            pan_radius=float(pan["properties"]["inner_radius"]),
            pan_height=float(pan["properties"]["depth"]),
            spawn_height=float(pan["properties"]["spawn_height"]),
            seed=7,
            color=(1.0, 1.0, 0.878, 1.0),
        )

    # endregion : Object Spawners

    # region : Internal State Utils

    def _setup_physics(self) -> None:
        self._dt = 1.0 / 240.0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(
            numSolverIterations=120,
            fixedTimeStep=self._dt,
            physicsClientId=self.client_id,
        )
        p.setTimeStep(self._dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self.client_id,
        )

    # endregion : Internal State Utils

    # region : Internal Pose Utils

    def _snap_objects_to_supports(self) -> None:
        # get support tops
        table_top = self._get_body_top_z("table")
        stove_top = self._get_body_top_z("stove")
        # snap items on supports
        self._place_on_support("stove", support_top_z=table_top)
        self._place_on_support("bowl", support_top_z=table_top)
        self._place_on_support("spatula", support_top_z=table_top)


        stove_top = self._get_body_top_z("stove")
        self._place_on_support("pan", support_top_z=stove_top)

        # update states
        pan_bottom = self._get_body_bottom_z("pan")
        self.objects["pan"]["properties"]["base_height"] = pan_bottom

    def _place_on_support(self, object_name: str, support_top_z: float) -> None:
        if object_name not in self.objects:
            logger.info(f"Object {object_name} does not exist")
            return
        bottom_z = self._get_body_bottom_z(object_name)
        dz = support_top_z - bottom_z
        if abs(dz) < 1e-4:
            return
        base_pos, base_orn = p.getBasePositionAndOrientation(
            self.objects[object_name]["body_id"],
            physicsClientId=self.client_id,
        )
        p.resetBasePositionAndOrientation(
            self.objects[object_name]["body_id"],
            (float(base_pos[0]), float(base_pos[1]), float(base_pos[2]) + dz),
            base_orn,
            physicsClientId=self.client_id,
        )

    def _get_body_top_z(self, object_name: str) -> float:
        if object_name not in self.objects:
            logger.error(f"Object {object_name} does not exist")
            return
        aabb_max = p.getAABB(
            self.objects[object_name]["body_id"],
            physicsClientId=self.client_id,
        )[1]
        return float(aabb_max[2])

    def _get_body_bottom_z(self, object_name: str) -> float:
        if object_name not in self.objects:
            logger.info(f"Object {object_name} does not exist")
            return
        aabb_min, _ = p.getAABB(
            self.objects[object_name]["body_id"],
            physicsClientId=self.client_id,
        )
        return float(aabb_min[2])

    # endregion : Internal Pose Utils
