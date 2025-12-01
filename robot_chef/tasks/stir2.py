import json
import logging
import math
from typing import Dict, Optional

import numpy as np
import pybullet as p

from robot_chef.camera import Camera, CameraNoiseModel
from robot_chef.config import MainConfig, Pose6D
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
        cam_view_cfg = cfg.camera.get_active_view()
        cam_noise_cfg = cfg.camera.noise
        self.camera = Camera(
            client_id=self.sim.client_id,
            fov_deg=cam_view_cfg.fov_deg,
            near=cfg.camera.near,
            far=cfg.camera.far,
            resolution=cam_view_cfg.resolution,
            noise=CameraNoiseModel(
                depth_std=cam_noise_cfg.depth_std,
                drop_prob=cam_noise_cfg.drop_prob,
            ),
            view_xyz=cam_view_cfg.xyz,
            view_rpy_deg=cam_view_cfg.rpy_deg,
        )

        self.wrist_comp_pitch = 0.0  # Current compensation angle
        self.pan_hold_pos = None     # Where we want the pan to stay

    def plan(self) -> bool:
        return True

    def execute(self) -> None:
        spatula_arm = self._arm_closer_to_spatula()
        pan_arm = "right" if spatula_arm == "left" else "left"
        self._prepare_pan(pan_arm)
        self._prepare_spatula(spatula_arm)
        # self._stir(pan_arm, spatula_arm)
        # for i in range(10000): # Run for some time
            
        #     # --- TASK A: BALANCE PAN ---
        #     # This continuously corrects the pan angle every step
        #     self._maintain_balance(pan_arm)
            
        #     # --- TASK B: DO OTHER STUFF (e.g. Stir) ---
        #     # You can add stirring logic here later. 
        #     # For now, we just let the spatula sit, or move it slightly.
            
        #     # --- TASK C: STEP PHYSICS ---
        #     self.sim.step_simulation(steps=1)
            
        #     # Optional: Sleep to visualize in real-time
        #     # time.sleep(1./240.)
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

    # def _prepare_pan(self, arm: str) -> None:
    #     # === GET PAN GEOMETRY ===
    #     pan_id = self.sim.objects["pan"]["body_id"]
    #     props = self.sim.objects["pan"]["properties"]

    #     handle_offset = props["handle_offset"]
    #     handle_length = props["handle_length"]
    #     depth = props["depth"]
    #     wall_thickness = props["wall_thickness"]

    #     slider_map = self.sim.setup_debug_sliders(arm)
    #     self.sim.interactive_mode_loop(arm, slider_map)

    #     # # === PAN POSE ===
    #     # pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id)
    #     # rot = np.array(p.getMatrixFromQuaternion(pan_orn)).reshape(3,3)

    #     # # === COMPUTE GRASP POINT ALONG HANDLE ===
    #     # grasp_fraction = 0.7   # 0 = very base, 1 = very tip

    #     # # handle center base position in local frame
    #     base_local_x = handle_offset - handle_length/2

    #     # grasp_local = np.array([
    #     #     base_local_x + grasp_fraction * handle_length,
    #     #     0,
    #     #     depth/2 + wall_thickness  # handle height
    #     # ])

    #     # # convert to world coordinates
    #     # grasp_world = np.array(pan_pos) + rot @ grasp_local

    #     # # move slightly above before descending
    #     # approach_world = grasp_world + np.array([0,0,0.10])

    #     # LOGGER.info(f"Target Grasp World Position: {grasp_world}")
    #     # LOGGER.info(f"Target Approach World Position: {approach_world}")

    #     # # === ORIENT END EFFECTOR TO GRASP HANDLE ===
    #     # yaw = p.getEulerFromQuaternion(pan_orn)[2]
    #     # grip_quat = p.getQuaternionFromEuler([math.pi/2, 0, yaw])

    #     # # === MOVE TO APPROACH POSE ===
    #     # j_approach = self._inverse_kinematics(arm, approach_world, grip_quat)
    #     # self._move_arm_damp_velo(arm, j_approach)

    #     # # === MOVE DOWN TO GRASP ===
    #     # j_grasp = self._inverse_kinematics(arm, grasp_world, grip_quat)
    #     # self._move_arm_damp_velo(arm, j_grasp)

    #     # # === CLOSE GRIPPER ===
    #     # self.sim.gripper_close(arm=arm)
    #     # self.sim.step_simulation(steps=240)

    #     # # === FIX CONSTRAINT SO PAN MOVES WITH ROBOT ===
    #     # gripper_link = self.sim.arm[arm].ee_link
    #     # c_id = p.createConstraint(
    #     #     parentBodyUniqueId=self.sim.arm[arm].body_id,
    #     #     parentLinkIndex=gripper_link,
    #     #     childBodyUniqueId=pan_id,
    #     #     childLinkIndex=-1,
    #     #     jointType=p.JOINT_FIXED,
    #     #     jointAxis=[0,0,0],
    #     #     parentFramePosition=[0,0,0],
    #     #     childFramePosition=[0,0,0],
    #     #     physicsClientId=self.sim.client_id,
    #     # )

    #     # print("Grasped pan at world:", grasp_world)

    # def _prepare_pan(self, arm: str) -> None:
    #     # === GET PAN GEOMETRY ===
    #     pan_id = self.sim.objects["pan"]["body_id"]
    #     props = self.sim.objects["pan"]["properties"]

    #     handle_offset = props["handle_offset"]
    #     handle_length = props["handle_length"]
    #     depth = props["depth"]
    #     wall_thickness = props["wall_thickness"]

    #     # === PAN POSE ===
    #     pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
    #     rot = np.array(p.getMatrixFromQuaternion(pan_orn)).reshape(3,3)

    #     grasp_fraction = 0.7    # 0 = very base, 1 = very tip
    #     base_local_x = handle_offset - handle_length/2
        
    #     # Local Z position is the handle's top surface
    #     handle_top_local_z = depth/2 + wall_thickness 

    #     grasp_local = np.array([
    #         base_local_x + grasp_fraction * handle_length,
    #         0,
    #         handle_top_local_z
    #     ])

    #     # convert to world coordinates
    #     grasp_world = np.array(pan_pos) + rot @ grasp_local

    #     # move slightly above before descending
    #     # APPROACH POSE: 10 cm above the grasp point
    #     approach_world = grasp_world + np.array([0,0,0.10]) 

    #     LOGGER.info(f"Target Grasp World Position: {grasp_world}")
    #     LOGGER.info(f"Target Approach World Position: {approach_world}")

    #     # === ORIENT END EFFECTOR FOR TOP-DOWN GRASP ===
    #     pan_yaw = p.getEulerFromQuaternion(pan_orn)[2]
        
    #     # This rotation points the gripper fingers down (pi roll)
    #     # and aligns the gripper's opening with the handle's axis (pan_yaw)
    #     grip_quat = p.getQuaternionFromEuler([math.pi, 0, pan_yaw]) 

    #     # === MOVE TO APPROACH POSE ===
    #     LOGGER.info("Arm approaching pan handle")
    #     j_approach = self._inverse_kinematics(arm, approach_world, grip_quat)
    #     self._move_arm_damp_velo(arm, j_approach)

    #     # === MOVE DOWN TO GRASP ===
    #     LOGGER.info("Arm to grasp pan handle")
    #     j_grasp = self._inverse_kinematics(arm, grasp_world, grip_quat)
    #     self.sim.set_joint_positions(j_grasp, arm)

    #     # === CLOSE GRIPPER ===
    #     LOGGER.info("Arm grasping pan handle")
    #     self.sim.gripper_close(arm=arm)
    #     self.sim.step_simulation(steps=240)

    #     # === FIX CONSTRAINT SO PAN MOVES WITH ROBOT ===
    #     LOGGER.info("Arm to pan handle constraint")
    #     gripper_link = self.sim.arm[arm].ee_link
    #     # p.createConstraint(
    #     #     parentBodyUniqueId=self.sim.arm[arm].body_id,
    #     #     parentLinkIndex=gripper_link,
    #     #     childBodyUniqueId=pan_id,
    #     #     childLinkIndex=-1,
    #     #     jointType=p.JOINT_FIXED,
    #     #     jointAxis=[0,0,0],
    #     #     parentFramePosition=[0,0,0],
    #     #     childFramePosition=[0,0,0],
    #     #     physicsClientId=self.sim.client_id,
    #     # )
    #     arm_id = self.sim.arm[arm].body_id
        
    #     # 1. Get the exact current world positions
    #     #    (Use physicsClientId to ensure we get the sim state, not rendered state)
    #     ee_state = p.getLinkState(arm_id, gripper_link, physicsClientId=self.sim.client_id)
    #     ee_world_pos, ee_world_orn = ee_state[0], ee_state[1]
        
    #     pan_world_pos, pan_world_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)

    #     # 2. Compute the Pan's position/rotation relative to the Gripper Frame
    #     #    Mathematically: T_gripper_to_pan = inv(T_world_to_gripper) * T_world_to_pan
    #     inv_ee_pos, inv_ee_orn = p.invertTransform(ee_world_pos, ee_world_orn)
        
    #     # This calculates where the pan is, looking from the perspective of the gripper
    #     parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
    #         inv_ee_pos, inv_ee_orn, 
    #         pan_world_pos, pan_world_orn
    #     )

    #     # 3. Create the constraint using these calculated offsets
    #     #    We lock the parent frame at the calculated offset, and the child frame at its own origin.
    #     #    This tells PyBullet: "Keep them exactly this far apart and oriented this way."
    #     c_id = p.createConstraint(
    #         parentBodyUniqueId=arm_id,
    #         parentLinkIndex=gripper_link,
    #         childBodyUniqueId=pan_id,
    #         childLinkIndex=-1,
    #         jointType=p.JOINT_FIXED,
    #         jointAxis=[0, 0, 0],
    #         parentFramePosition=parent_frame_pos,      # <--- The calculated offset
    #         childFramePosition=[0, 0, 0],              # <--- Pan origin
    #         parentFrameOrientation=parent_frame_orn,   # <--- The calculated rotation
    #         physicsClientId=self.sim.client_id,
    #     )
        
    #     # 4. (Optional but recommended) Allow the constraint to be slightly "soft" 
    #     #    to absorb numerical errors rather than exploding.
    #     p.changeConstraint(c_id, maxForce=2000)

    #     print("Grasped pan at world:", grasp_world)

    #     # --------------------------------------------------------------
    #     # CALCULATE LIFT & PULL VECTORS
    #     # --------------------------------------------------------------
    #     robot_id = self.sim.arm[arm].body_id
    #     gripper_link = self.sim.arm[arm].ee_link
        
    #     # 1. Get current Gripper Position (World Frame)
    #     ee_state = p.getLinkState(robot_id, gripper_link, physicsClientId=self.sim.client_id)
    #     current_pos = np.array(ee_state[0])
    #     current_orn = ee_state[1] # Keep the current orientation so we don't spill!

    #     # 2. Get Robot Base Position
    #     base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.sim.client_id)
    #     base_pos = np.array(base_pos)

    #     # 3. Calculate "Pull" Vector (Direction from Gripper -> Base)
    #     # We ignore Z because we only want to pull horizontally
    #     direction_vector = base_pos - current_pos
    #     direction_vector[2] = 0.0  # Zero out Z
        
    #     # Normalize to get a unit vector (length of 1.0)
    #     dist = np.linalg.norm(direction_vector)
    #     if dist > 0:
    #         direction_unit = direction_vector / dist
    #     else:
    #         direction_unit = np.array([0, 0, 0]) # Already at base (unlikely)

    #     # 4. Define Movements
    #     lift_amount = 0.1   # Lift 5cm up
    #     pull_amount = 0.20   # Pull 10cm closer to body
        
    #     # 5. Compute Final Target
    #     # Target = Current + (Up * amount) + (Direction * amount)
    #     target_pos = current_pos + np.array([0, 0, lift_amount]) + (direction_unit * pull_amount)

    #     LOGGER.info(f"Moving pan to hover: {target_pos}")

    #     # --------------------------------------------------------------
    #     # EXECUTE MOVE
    #     # --------------------------------------------------------------
        
    #     # Calculate IK
    #     j_hover = self._inverse_kinematics(arm, target_pos, current_orn)
        
    #     # Execute movement
    #     # Since the pan is heavy (1.0kg), we might need a tighter tolerance or more steps
    #     self._move_arm_damp_velo(arm, j_hover, tol=1e-3)
        
    #     # IMPORTANT: Lock the joints at the end to hold the weight
    #     self.sim.set_joint_positions(j_hover, arm)

    def _prepare_pan(self, arm: str) -> None:
        # === GET PAN GEOMETRY ===
        pan_id = self.sim.objects["pan"]["body_id"]
        props = self.sim.objects["pan"]["properties"]

        handle_offset = props["handle_offset"]
        handle_length = props["handle_length"]
        depth = props["depth"]
        wall_thickness = props["wall_thickness"]

        # === PAN POSE ===
        pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
        rot = np.array(p.getMatrixFromQuaternion(pan_orn)).reshape(3,3)

        grasp_fraction = 0.25
        base_local_x = handle_offset - handle_length/2
        
        # Local Z position is the handle's top surface
        handle_top_local_z = depth/2 + wall_thickness 

        grasp_local = np.array([
            base_local_x + grasp_fraction * handle_length,
            0,
            handle_top_local_z
        ])

        # convert to world coordinates
        grasp_world = np.array(pan_pos) + rot @ grasp_local

        # APPROACH POSE: 10 cm above the grasp point
        approach_world = grasp_world + np.array([0,0,0.10]) 

        LOGGER.info(f"Target Grasp World Position: {grasp_world}")

        # === ORIENT END EFFECTOR FOR TOP-DOWN GRASP ===
        pan_yaw = p.getEulerFromQuaternion(pan_orn)[2]
        
        # Point fingers down (pi) and align with handle (pan_yaw)
        grip_quat = p.getQuaternionFromEuler([math.pi, 0, pan_yaw]) 

        # === MOVE TO APPROACH POSE ===
        LOGGER.info("Arm approaching pan handle")
        j_approach = self._inverse_kinematics(arm, approach_world, grip_quat)
        self._move_arm_damp_velo(arm, j_approach)

        # === MOVE DOWN TO GRASP ===
        LOGGER.info("Arm to grasp pan handle")
        j_grasp = self._inverse_kinematics(arm, grasp_world, grip_quat)
        self.sim.set_joint_positions(j_grasp, arm)

        # === CLOSE GRIPPER ===
        LOGGER.info("Arm grasping pan handle")
        self.sim.gripper_close(arm=arm)
        self.sim.step_simulation(steps=120)

        # # ==============================================================
        # # CRITICAL PHYSICS FIXES (Mass, Collision, Stiffness)
        # # ==============================================================
        
        # # 1. Update Mass so the pan can actually move (Dynamic)
        # p.changeDynamics(pan_id, -1, mass=1.0, physicsClientId=self.sim.client_id)

        # # 2. Disable Collision between Robot and Pan to prevent spasms
        # arm_id = self.sim.arm[arm].body_id
        # num_joints = p.getNumJoints(arm_id, physicsClientId=self.sim.client_id)
        # for i in range(num_joints):
        #     p.setCollisionFilterPair(arm_id, pan_id, i, -1, enableCollision=0, physicsClientId=self.sim.client_id)

        # # 3. Stiffen the arm so it doesn't sag under the 1.0kg weight
        # self._stiffen_arm_for_heavy_object(arm)

        # # ==============================================================
        # # ZERO-ENERGY CONSTRAINT (Prevents Flying)
        # # ==============================================================
        # LOGGER.info("Creating constraint...")
        # gripper_link = self.sim.arm[arm].ee_link
        
        # # Get exact current poses
        # ee_state = p.getLinkState(arm_id, gripper_link, physicsClientId=self.sim.client_id)
        # ee_world_pos, ee_world_orn = ee_state[0], ee_state[1]
        # pan_world_pos, pan_world_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)

        # # Calculate Pan relative to Gripper
        # inv_ee_pos, inv_ee_orn = p.invertTransform(ee_world_pos, ee_world_orn)
        # parent_frame_pos, parent_frame_orn = p.multiplyTransforms(
        #     inv_ee_pos, inv_ee_orn, 
        #     pan_world_pos, pan_world_orn
        # )

        # c_id = p.createConstraint(
        #     parentBodyUniqueId=arm_id,
        #     parentLinkIndex=gripper_link,
        #     childBodyUniqueId=pan_id,
        #     childLinkIndex=-1,
        #     jointType=p.JOINT_FIXED,
        #     jointAxis=[0, 0, 0],
        #     parentFramePosition=parent_frame_pos,      # Calculated offset
        #     childFramePosition=[0, 0, 0],              # Pan origin
        #     parentFrameOrientation=parent_frame_orn,   # Calculated rotation
        #     physicsClientId=self.sim.client_id,
        # )
        # p.changeConstraint(c_id, maxForce=5000)

        # print("Grasped pan at world:", grasp_world)

        # # --------------------------------------------------------------
        # # CALCULATE LIFT & PULL (FIXED FOR SAG)
        # # --------------------------------------------------------------
        
        # # 1. Get Coordinates
        # robot_id = self.sim.arm[arm].body_id
        # gripper_link = self.sim.arm[arm].ee_link
        
        # ee_state = p.getLinkState(robot_id, gripper_link, physicsClientId=self.sim.client_id)
        # current_pos = np.array(ee_state[0])
        # current_orn_messy = ee_state[1]

        # base_pos, _ = p.getBasePositionAndOrientation(robot_id, physicsClientId=self.sim.client_id)
        # base_pos = np.array(base_pos)

        # # 2. Calculate Pull Direction (Horizontal only)
        # direction_vector = base_pos - current_pos
        # direction_vector[2] = 0.0
        # dist = np.linalg.norm(direction_vector)
        # direction_unit = direction_vector / dist if dist > 0 else np.zeros(3)

        # # 3. DEFINE ABSOLUTE TARGETS
        # SAFE_HOVER_HEIGHT = 0.60 
        # target_xy = current_pos[:2] + (direction_unit[:2] * 0.20)
        # target_pos = np.array([target_xy[0], target_xy[1], SAFE_HOVER_HEIGHT])
        
        # # === SAVE TARGET FOR THE LOOP ===
        # self.pan_hold_pos = target_pos

        # LOGGER.info(f"Moving to hold position: {target_pos}")

        # # Move roughly there to start (using your strong move)
        # # We start with 0 compensation
        # neutral_orn = p.getQuaternionFromEuler([math.pi, 0, 0]) 
        # j_start = self._inverse_kinematics(arm, target_pos, neutral_orn)
        # self._move_arm_strong(arm, j_start, duration_sec=2.0)

        # # 3. DEFINE ABSOLUTE TARGETS (Crucial for Anti-Sag)
        # #    We ignore current Z and go to a safe absolute height.
        # SAFE_HOVER_HEIGHT = 0.60  # High enough to clear table even if it droops
        
        # #    Target X,Y = Current X,Y + (Pull Direction * 20cm)
        # target_xy = current_pos[:2] + (direction_unit[:2] * 0.20)
        
        # target_pos = np.array([target_xy[0], target_xy[1], SAFE_HOVER_HEIGHT])

        # # 4. FIX ORIENTATION & COMPENSATE FOR WEIGHT
        # current_rpy = p.getEulerFromQuaternion(current_orn_messy)
        # current_yaw = current_rpy[2]

        # # === PITCH COMPENSATION ===
        # # We tilt the wrist BACK (-0.25 rad) to counteract the heavy pan pulling it down.
        # # Like a fisherman pulling up a rod.
        # compensation_angle = -0.25 
        
        # # [Roll=PI, Pitch=Compensation, Yaw=Current]
        # compensated_orn = p.getQuaternionFromEuler([math.pi, compensation_angle, current_yaw])

        # LOGGER.info(f"Power lifting pan to: {target_pos}")

        # # 5. EXECUTE POWER LIFT
        # j_hover = self._inverse_kinematics(arm, target_pos, compensated_orn)
        
        # # === USE STRONG MOVE INSTEAD OF DAMP VELO ===
        # self._move_arm_strong(arm, j_hover, duration_sec=3.0)
        
        # # 6. Final Lock
        # self.sim.set_joint_positions(j_hover, arm)

        # # ==============================================================
        # # LIFT & PULL (With Anti-Tilt Logic)
        # # ==============================================================
        
        # # 1. Get current Gripper Position
        # ee_state = p.getLinkState(arm_id, gripper_link, physicsClientId=self.sim.client_id)
        # current_pos = np.array(ee_state[0])
        # current_orn_messy = ee_state[1] # This might be slightly tilted/sagging!

        # # 2. Calculate Direction to Robot Base
        # base_pos, _ = p.getBasePositionAndOrientation(arm_id, physicsClientId=self.sim.client_id)
        # base_pos = np.array(base_pos)

        # direction_vector = base_pos - current_pos
        # direction_vector[2] = 0.0  # Zero out Z (horizontal pull only)
        
        # dist = np.linalg.norm(direction_vector)
        # if dist > 0:
        #     direction_unit = direction_vector / dist
        # else:
        #     direction_unit = np.array([0, 0, 0])

        # # 3. Define Offsets
        # lift_amount = 0.10   # Lift 10cm up
        # pull_amount = 0.20   # Pull 20cm closer

        # target_pos = current_pos + np.array([0, 0, lift_amount]) + (direction_unit * pull_amount)
        
        # # 4. FIX TILT: Force a perfectly flat orientation
        # #    We preserve the Yaw (direction) but force Pitch/Roll to be flat.
        # current_rpy = p.getEulerFromQuaternion(current_orn_messy)
        # current_yaw = current_rpy[2]
        
        # # [Roll=PI (fingers down), Pitch=0 (flat), Yaw=Current]
        # flat_orn = p.getQuaternionFromEuler([math.pi, 0, current_yaw])

        # LOGGER.info(f"Moving pan to hover: {target_pos}")

        # # 5. Execute with Stiff Joints
        # j_hover = self._inverse_kinematics(arm, target_pos, flat_orn)

        # # Use a tighter tolerance (1e-3) because we are carrying a heavy load
        # self._move_arm_damp_velo(arm, j_hover, k=5, tol=1e-3)
        
        # # Lock final position
        # self.sim.set_joint_positions(j_hover, arm)

    def _stiffen_arm_for_heavy_object(self, arm: str):
        """Increases the max force of the arm joints to hold the heavy pan."""
        robot_id = self.sim.arm[arm].body_id
        
        # === FIX: HARDCODE ARM INDICES ===
        arm_indices = [0, 1, 2, 3, 4, 5, 6]
        
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=arm_indices,
            controlMode=p.POSITION_CONTROL,
            forces=[1000.0] * 7, # Apply to exactly 7 joints
            physicsClientId=self.sim.client_id
        )

    def _prepare_spatula(self, arm: str) -> None:
        spatula_id = self.sim.objects["spatula"]["body_id"]
        pan_id = self.sim.objects["pan"]["body_id"]

        # move arm to spatula handle
        aabb_min, aabb_max = p.getAABB(spatula_id)
        aabb_min = np.array(aabb_min)
        aabb_max = np.array(aabb_max)
        dimensions = aabb_max - aabb_min
        print("Spatula dimensions (X, Y, Z):", dimensions)
        handle_length = dimensions[0]
        print("Approx handle length along X:", handle_length)
        grasp_fraction = 0.15  # closer to the base
        local_grasp = np.array([-grasp_fraction * handle_length, 0, 0.3])  # along local X
        spatula_pos, spatula_orn = p.getBasePositionAndOrientation(spatula_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(spatula_orn)).reshape(3, 3)
        print(f"{spatula_pos=}")
        print(f"{spatula_orn=}")
        print(f"{rot_matrix=}")

        # grasp_pos = np.array(spatula_pos) + np.array([0, 0, 0.12]) # define grasp slightly above handle
        grasp_shift = rot_matrix @ local_grasp
        print(f"{grasp_shift=}")
        grasp_pos = np.array(spatula_pos) + grasp_shift
        print(f"{grasp_pos=}")
        spatula_yaw = p.getEulerFromQuaternion(spatula_orn)[2]
        ee_down_quat = p.getQuaternionFromEuler([math.pi, 0, spatula_yaw])
        joint_pos = self._inverse_kinematics(arm, grasp_pos, ee_down_quat)
        self._move_arm_damp_velo(arm, joint_pos)
        # pause()

        # grasp
        self.sim.gripper_close(arm=arm)
        self.sim.step_simulation(steps=240)
        gripper_link = self.sim.arm[arm].ee_link
        c_id = p.createConstraint( # TODO : grasp at lower handle
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

        ee_link = self.sim.arm[arm].ee_link
        ee_pos, ee_orn = p.getLinkState(self.sim.arm[arm].body_id, ee_link, physicsClientId=self.sim.client_id)[:2]
        ee_pos = np.array(ee_pos)
        lift_vector = np.array([0, 0, 0.5])
        lift_target = ee_pos + lift_vector
        joint_pos = self._inverse_kinematics(arm, lift_target, ee_orn)
        self._move_arm_damp_velo(arm, joint_pos)
        # pause()

        # --------------------------------------------------------------
        # Move spatula ABOVE PAN
        # --------------------------------------------------------------
        pan_pos, pan_orn = p.getBasePositionAndOrientation(pan_id)
        pan_pos = np.array(pan_pos)

        # Position the spatula 25 cm above pan center
        target_above_pan = pan_pos + np.array([0, 0, 0.25])
        joint_pos = self._inverse_kinematics(arm, target_above_pan, ee_orn)
        self._move_arm_damp_velo(arm, joint_pos)

        self.sim.step_simulation(steps=120)

        # --------------------------------------------------------------
        # ROTATE SPATULA 90 DEGREES (around world Z axis)
        # --------------------------------------------------------------
        # Get current end effector orientation
        _, current_orn = p.getLinkState(
            self.sim.arm[arm].body_id, 
            ee_link, 
            physicsClientId=self.sim.client_id
        )[:2]

        # Convert to Euler → apply 90° rotation → back to quaternion
        ee_euler = list(p.getEulerFromQuaternion(current_orn))
        ee_euler[1] += math.pi / 2
        target_rotated_orn = p.getQuaternionFromEuler(ee_euler)

        # Move arm with rotated orientation
        joint_pos = self._inverse_kinematics(arm, target_above_pan, target_rotated_orn)
        self._move_arm_damp_velo(arm, joint_pos)

        # --------------------------------------------------------------
        # PRINT FINAL SPATULA POSE
        # --------------------------------------------------------------
        final_pos, final_orn = p.getBasePositionAndOrientation(spatula_id)
        final_euler = p.getEulerFromQuaternion(final_orn)

        print("\n=== FINAL SPATULA POSE ===")
        print(f"Position:  {np.round(final_pos, 4)}")
        print(f"Quaternion:{np.round(final_orn, 4)}")
        print(f"Euler XYZ: {np.round(final_euler, 4)}")
        print("==========================\n")

        

    def _stir(self, pan_arm: str, spatula_arm: str) -> None:
        # TODO : stirring
        # # --------------------------------------------------------------
        # # SIMPLE BACK-AND-FORTH STIRRING (linear motion)
        # # --------------------------------------------------------------

        # stir_amplitude = 0.12      # how far to move left/right (meters)
        # stir_height = 0.12         # vertical height above the pan
        # stir_cycles = 5            # number of back-and-forth passes
        # stir_steps = 90            # steps per half-stroke
        # stir_speed = 1.0           # scaling factor for arm velocity

        # print("\nStarting back-and-forth stirring...")

        # pan_center = np.array(pan_pos)
        # base_height = pan_center[2] + stir_height

        # # The direction of motion (X axis of pan frame works fine)
        # # You can switch to Y axis by swapping indices.
        # forward_dir = np.array([1.0, 0.0, 0.0])

        # for cycle in range(stir_cycles):

        #     # ---- FORWARD stroke ----
        #     print("forward")
        #     for i in range(stir_steps):
        #         frac = i / stir_steps
        #         delta = forward_dir * (frac * stir_amplitude)
        #         target_pos = pan_center + np.array([delta[0], delta[1], stir_height])

        #         joint_pos = self._inverse_kinematics(arm, target_pos, target_rotated_orn)
        #         self.sim.set_joint_positions(joint_pos, arm)
        #         # self._move_arm_damp_velo(arm, joint_pos)
        #         self.sim.step_simulation(steps=20)

        #     # ---- BACKWARD stroke ----
        #     print("backward")
        #     for i in range(stir_steps):
        #         frac = i / stir_steps
        #         delta = forward_dir * ((1 - frac) * stir_amplitude)
        #         target_pos = pan_center + np.array([delta[0], delta[1], stir_height])

        #         joint_pos = self._inverse_kinematics(arm, target_pos, target_rotated_orn)
        #         # self._move_arm_damp_velo(arm, joint_pos, speed_scale=stir_speed)
        #         self.sim.set_joint_positions(joint_pos, arm)
        #         self.sim.step_simulation(steps=20)

        #     print(f"Completed cycle {cycle + 1}/{stir_cycles}")

        # print("Back-and-forth stirring complete.\n")
        pass

    def _move_arm_strong(self, arm: str, target_joints: np.ndarray, duration_sec: float = 2.0):
        robot_id = self.sim.arm[arm].body_id
        
        # === FIX: HARDCODE ARM INDICES ===
        # Franka Panda arm joints are indices 0, 1, 2, 3, 4, 5, 6
        # We ignore indices 7+ (fingers, fixed frames)
        arm_indices = [0, 1, 2, 3, 4, 5, 6]
        
        # Ensure target_joints matches this length (7)
        targets = target_joints[:7]
        
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=arm_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targets,
            forces=[500.0] * 7,       # Apply to exactly 7 joints
            positionGains=[0.2] * 7, # Apply to exactly 7 joints
            velocityGains=[1.0] * 7,  # Apply to exactly 7 joints
            physicsClientId=self.sim.client_id
        )

        steps = int(duration_sec * 240)
        for _ in range(steps):
            self.sim.step_simulation(steps=1)

    def _maintain_balance(self, arm: str):
        """
        Reads pan tilt and updates arm target. Call this inside your main loop.
        """
        if self.pan_hold_pos is None:
            return

        pan_id = self.sim.objects["pan"]["body_id"]
        robot_id = self.sim.arm[arm].body_id
        
        # 1. MEASURE TILT (Error Signal)
        _, pan_orn = p.getBasePositionAndOrientation(pan_id, physicsClientId=self.sim.client_id)
        pan_rpy = p.getEulerFromQuaternion(pan_orn)
        
        # Check Pitch (Y-axis) for sag. 
        # (If your pan is rotated 90deg, you might need pan_rpy[0] instead)
        current_sag = pan_rpy[1]

        # 2. UPDATE COMPENSATION (P-Controller)
        # If sag is positive (tilting forward), we decrease pitch (tilt back)
        # Low gain (0.1) for smooth stability
        self.wrist_comp_pitch -= (current_sag * 0.1)
        
        # Clamp to prevent wrist breaking (+/- 1.0 radian)
        self.wrist_comp_pitch = np.clip(self.wrist_comp_pitch, -1.0, 1.0)

        # 3. CALCULATE NEW TARGET
        # Get current Yaw to maintain direction
        ee_link = self.sim.arm[arm].ee_link
        _, current_orn = p.getLinkState(robot_id, ee_link, physicsClientId=self.sim.client_id)[:2]
        current_yaw = p.getEulerFromQuaternion(current_orn)[2]

        # Combine: Roll=180 (down), Pitch=Comp, Yaw=Current
        target_orn = p.getQuaternionFromEuler([math.pi, self.wrist_comp_pitch, current_yaw])
        
        # 4. SOLVE IK (Fast)
        j_target = self._inverse_kinematics(arm, self.pan_hold_pos, target_orn)
        
        # 5. COMMAND MOTORS (Non-Blocking)
        arm_indices = [0, 1, 2, 3, 4, 5, 6]
        p.setJointMotorControlArray(
            bodyUniqueId=robot_id,
            jointIndices=arm_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=j_target[:7],
            forces=[1000.0] * 7,      # High Stiffness
            positionGains=[0.1] * 7,  # High Stiffness
            physicsClientId=self.sim.client_id
        )