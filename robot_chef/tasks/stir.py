import json
import logging
import math

import numpy as np
import pybullet as p

from robot_chef.config import MainConfig
from robot_chef.simulator import StirSimulator
from robot_chef.utils import pause

LOGGER = logging.getLogger(__name__)


class TaskStir:
    """Bimanual stirring task using spatula and pan handle"""
    def setup(self, sim: StirSimulator, cfg: MainConfig) -> None:
        self.sim = sim
        self.cfg = cfg
        self.sim.open_grippers(arm="right")
        self.sim.open_grippers(arm="left")
        self.wrist_comp_pitch = 0.0
        self.pan_hold_pos = None

    def plan(self) -> bool:
        return True

    def execute(self) -> None:
        spatula_arm = self._arm_closer_to_spatula()
        pan_arm = "right" if spatula_arm == "left" else "left"
        self._prepare_pan(pan_arm)
        self._prepare_spatula(spatula_arm)
        self._stir(pan_arm, spatula_arm)
        pause()

    def metrics(self, print: bool = False):
        metrics = {
            "retain_ratio": 0.0,
            "beads_inside": 0.0,
            "bead_original": 0.0,
        }
        metrics = self.sim.count_particles_in_pan()
        if print:
            LOGGER.info("\n" + json.dumps({
                "metrics": metrics,
            }, indent=4))
        return metrics

    def _inverse_kinematics(self, arm: str, pos: np.ndarray, quat: tuple) -> np.ndarray:
        arm_state = self.sim.get_arm(arm)
        q = p.calculateInverseKinematics(
            arm_state.body_id,
            arm_state.ee_link,
            pos.tolist(),
            quat,
            maxNumIterations=200,
            residualThreshold=1e-4,
            physicsClientId=self.sim.client_id,
        )
        return np.array(q[:7], dtype=float)

    def _arm_closer_to_spatula(self) -> str:
        spatula_id = self.sim.objects["spatula"]["body_id"]
        spatula_pos = p.getBasePositionAndOrientation(
            bodyUniqueId=spatula_id,
            physicsClientId=self.sim.client_id,
        )[0]
        spatula_pos_xy = np.array(spatula_pos[:2])

        left_pedestal_id = self.sim.objects["left_pedestal"]["body_id"]
        right_pedestal_id = self.sim.objects["right_pedestal"]["body_id"]

        left_base_pos = p.getBasePositionAndOrientation(
            bodyUniqueId=left_pedestal_id,
            physicsClientId=self.sim.client_id,
        )[0]
        left_base_pos_xy = np.array(left_base_pos[:2])
        
        right_base_pos = p.getBasePositionAndOrientation(
            bodyUniqueId=right_pedestal_id,
            physicsClientId=self.sim.client_id,
        )[0]
        right_base_pos_xy = np.array(right_base_pos[:2])

        dist_left = np.sum((spatula_pos_xy - left_base_pos_xy) ** 2)
        dist_right = np.sum((spatula_pos_xy - right_base_pos_xy) ** 2)

        return "left" if dist_left < dist_right else "right"
    
    def _move_arm_damp_velo(self, arm: str, target_joints: np.ndarray, k: float = 5, tol: float = 1e-3):
        target_joints = np.array(target_joints, dtype=float)
        prev_joints = None
        while True:
            current_joints, _ = self.sim.get_joint_state(arm)
            current_joints = np.array(current_joints, dtype=float)
            delta = target_joints - current_joints
            if np.linalg.norm(delta) < tol:
                break
            if prev_joints is not None and np.linalg.norm(current_joints - prev_joints) < tol:
                break
            prev_joints = current_joints
            velocities = k * delta
            self.sim.set_joint_velocities(velocities, arm=arm)
            self.sim.step_simulation(steps=1)

    def _prepare_pan(self, arm: str) -> None:
        pan_id = self.sim.objects["pan"]["body_id"]
        props = self.sim.objects["pan"]["properties"]

        handle_offset = props["handle_offset"]
        handle_length = props["handle_length"]
        depth = props["depth"]
        wall_thickness = props["wall_thickness"]

        pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
        rot = np.array(p.getMatrixFromQuaternion(pan_orn)).reshape(3,3)

        grasp_fraction = 0.25
        base_local_x = handle_offset - handle_length/2
        
        handle_top_local_z = depth / 2 + wall_thickness 

        grasp_local = np.array([
            base_local_x + grasp_fraction * handle_length,
            0,
            handle_top_local_z
        ])

        grasp_world = np.array(pan_pos) + rot @ grasp_local
        approach_world = grasp_world + np.array([0,0,0.15]) 

        LOGGER.info(f"Target Grasp World Position: {grasp_world}")

        pan_yaw = p.getEulerFromQuaternion(pan_orn)[2]
        
        # point fingers down
        grip_quat = p.getQuaternionFromEuler([math.pi, 0, pan_yaw]) 

        LOGGER.info("Arm approaching pan handle")
        j_approach = self._inverse_kinematics(arm, approach_world, grip_quat)
        self._move_arm_damp_velo(arm, j_approach)

        LOGGER.info("Arm to grasp pan handle")
        j_grasp = self._inverse_kinematics(arm, grasp_world, grip_quat)
        self.sim.set_joint_positions(j_grasp, arm)

        LOGGER.info("Arm grasping pan handle")
        self.sim.gripper_close(arm=arm)
        self.sim.step_simulation(steps=240)

        LOGGER.info("Creating constraint to lock pan to gripper")
        arm_id = self.sim.arm[arm].body_id
        gripper_link = self.sim.arm[arm].ee_link
        
        ee_state = p.getLinkState(arm_id, gripper_link, physicsClientId=self.sim.client_id)
        ee_world_pos, ee_world_orn = ee_state[0], ee_state[1]
        
        pan_world_pos, pan_world_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
        inv_ee_pos, inv_ee_orn = p.invertTransform(ee_world_pos, ee_world_orn)
        parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
            inv_ee_pos, inv_ee_orn, 
            pan_world_pos, pan_world_orn
        )
        c_id = p.createConstraint(
            parentBodyUniqueId=arm_id,
            parentLinkIndex=gripper_link,
            childBodyUniqueId=pan_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=parent_frame_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=parent_frame_orn,
            physicsClientId=self.sim.client_id,
        )
        p.changeConstraint(c_id, maxForce=2000)

    def _prepare_spatula(self, arm: str) -> None:
        spatula_id = self.sim.objects["spatula"]["body_id"]
        pan_id = self.sim.objects["pan"]["body_id"]

        # move arm to spatula handle
        aabb_min, aabb_max = p.getAABB(spatula_id)
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)
        dimensions = aabb_max - aabb_min
        handle_length = dimensions[0]
        grasp_fraction = 0.15  # closer to the base
        local_grasp = np.array([-grasp_fraction * handle_length, 0, 0])
        spatula_pos, spatula_orn = p.getBasePositionAndOrientation(spatula_id)
        print(f"{spatula_pos=} {spatula_orn=}")
        rot_matrix = np.array(p.getMatrixFromQuaternion(spatula_orn)).reshape(3, 3)
        grasp_shift = rot_matrix @ local_grasp
        grasp_pos = np.array(spatula_pos) + grasp_shift
        grasp_pos_lift = grasp_pos + np.array([0, 0, spatula_pos[2] * 0.5])
        print(f"{grasp_pos=} {grasp_pos_lift=}")
        spatula_yaw = p.getEulerFromQuaternion(spatula_orn)[2]
        ee_down_quat = p.getQuaternionFromEuler([math.pi, 0, spatula_yaw])
        joint_pos = self._inverse_kinematics(arm, grasp_pos_lift, ee_down_quat)
        print(f"{joint_pos=}")
        self._move_arm_damp_velo(arm, joint_pos)
        self._move_arm_damp_velo(arm, self._inverse_kinematics(arm, grasp_pos, ee_down_quat), k=1.5)

        # grasp handle
        self.sim.gripper_close(arm=arm)
        self.sim.step_simulation(steps=240)
        gripper_link = self.sim.arm[arm].ee_link
        p.createConstraint( # TODO : grasp at lower handle
            parentBodyUniqueId=self.sim.arm[arm].body_id,
            parentLinkIndex=gripper_link,
            childBodyUniqueId=spatula_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],  # gripper origin
            childFramePosition=[0, 0, 0],  # offset from spatula CoM to grasp point
            physicsClientId=self.sim.client_id,
        )

        # lift spatula slightly
        ee_link = self.sim.arm[arm].ee_link
        ee_pos, ee_orn = p.getLinkState(self.sim.arm[arm].body_id, ee_link, physicsClientId=self.sim.client_id)[:2]
        ee_pos = np.array(ee_pos)
        lift_vector = np.array([0, 0, 0.5])
        lift_target = ee_pos + lift_vector
        joint_pos = self._inverse_kinematics(arm, lift_target, ee_orn)
        self._move_arm_damp_velo(arm, joint_pos)

        # move spatula to above pan
        pan_pos, _ = p.getBasePositionAndOrientation(pan_id)
        pan_pos = np.array(pan_pos)
        target_above_pan = pan_pos + np.array([0.15, 0, 0.5])
        joint_pos = self._inverse_kinematics(arm, target_above_pan, ee_orn)
        self._move_arm_damp_velo(arm, joint_pos)

        # tilt spatula
        _, current_orn = p.getLinkState(
            self.sim.arm[arm].body_id, 
            ee_link, 
            physicsClientId=self.sim.client_id
        )[:2]
        ee_euler = list(p.getEulerFromQuaternion(current_orn))
        ee_euler[1] += math.pi / 2
        target_rotated_orn = p.getQuaternionFromEuler(ee_euler)
        joint_pos = self._inverse_kinematics(arm, target_above_pan, target_rotated_orn)
        self._move_arm_damp_velo(arm, joint_pos)

    def _stir(self, pan_arm: str, spatula_arm: str) -> None:
        LOGGER.info("Beginning linear stirring sequence...")

        pan_id = self.sim.objects["pan"]["body_id"]
        pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
        pan_pos = np.array(pan_pos)
        
        rot_matrix = np.array(p.getMatrixFromQuaternion(pan_orn)).reshape(3, 3)
        pan_axis_x = rot_matrix[:, 0] 

        stir_z_height = pan_pos[2] + 0.03 
        
        amplitude = 0.24
        
        target_a_pos = pan_pos - (pan_axis_x * amplitude)
        target_a_pos[2] = stir_z_height # Override Z

        target_b_pos = pan_pos + (pan_axis_x * amplitude)
        target_b_pos[2] = stir_z_height # Override Z

        ee_link = self.sim.arm[spatula_arm].ee_link
        arm_body = self.sim.arm[spatula_arm].body_id
        _, current_orn = p.getLinkState(arm_body, ee_link, physicsClientId=self.sim.client_id)[:2]

        LOGGER.info("Descend into pan (Point A)...")
        j_start = self._inverse_kinematics(spatula_arm, target_a_pos, current_orn)
        self._move_arm_damp_velo(spatula_arm, j_start, k=2.0)

        num_cycles = 4
        stir_speed_k = 3.5
        
        for i in range(num_cycles):
            LOGGER.info(f"Stir cycle {i+1}/{num_cycles}")
            j_b = self._inverse_kinematics(spatula_arm, target_b_pos, current_orn)
            self._move_arm_damp_velo(spatula_arm, j_b, k=stir_speed_k, tol=1e-3)
            j_a = self._inverse_kinematics(spatula_arm, target_a_pos, current_orn)
            self._move_arm_damp_velo(spatula_arm, j_a, k=stir_speed_k, tol=1e-3)

        LOGGER.info("Retracting spatula...")
        curr_pos = p.getLinkState(arm_body, ee_link, physicsClientId=self.sim.client_id)[0]
        retract_target = np.array(curr_pos) + np.array([0, 0, 0.15])
        
        j_retract = self._inverse_kinematics(spatula_arm, retract_target, current_orn)
        self._move_arm_damp_velo(spatula_arm, j_retract)
