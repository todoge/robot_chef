from __future__ import annotations

import os
import logging
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

from .camera import Camera

from .tasks.detect_object import Object_Detector
from .tasks.predict_grasp import Grasp_Predictor

from .config import Pose6D, PourTaskConfig
from .env.objects import pan as pan_factory
from .env.objects import rice_bowl as bowl_factory
from .env.objects import spatula as spatula
from .env.particles import ParticleSet, spawn_spheres

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data containers


@dataclass
class ArmState:
    body_id: int
    joint_indices: Tuple[int, ...]
    ee_link: int
    eef: int
    finger_joints: Tuple[int, int]
    joint_lower: Tuple[float, ...]
    joint_upper: Tuple[float, ...]
    joint_ranges: Tuple[float, ...]
    joint_rest: Tuple[float, ...]

class Pose:
    def __init__(self, x, y, z, orientation_rpy):
        self.x = x
        self.y = y
        self.z = z
        self.orientation_rpy = orientation_rpy

# --------------------------------------------------------------------------- #
# Simulation


class RobotChefSimulation(gym.Env):
    """Encapsulates the physics world, robot arms, and task objects."""

    def __init__(self, gui: bool, recipe: PourTaskConfig):
        self.recipe = recipe
        self.gui = gui
        self.client_id = p.connect(p.GUI if gui else p.DIRECT)

        #p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)
        LOGGER.info("Connected to PyBullet with client_id=%s (gui=%s)", self.client_id, gui)

        self.dt = 1.0 / 240.0
        p.resetSimulation(physicsClientId=self.client_id)
        p.setPhysicsEngineParameter(numSolverIterations=120, fixedTimeStep=self.dt, physicsClientId=self.client_id)
        p.setTimeStep(self.dt, physicsClientId=self.client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=self.client_id)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client_id)
        
        self.max_steps = 1000
        self.current_step = 0

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        space = 150
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(space,), dtype=np.float32)

        self.objects: Dict[str, Dict[str, object]] = {}
        self.particles: Optional[ParticleSet] = None

        # Default active arm is the right arm for pouring motions.
        self._active_arm_name = "left"
        #self.gripper_open()
        self.step_simulation(steps=60)

        self.IMG_WIDTH = 224
        self.IMG_HEIGHT = 224
        self.obj_detector = Object_Detector((self.IMG_WIDTH, self.IMG_HEIGHT))
        self.grasping_predictor = Grasp_Predictor()
        self.camera = None
        self.keyposes = None

        self.saved_state_id = None
        
        # Grasp execution parameters
        self.RECALIBRATE_HEIGHT = 0.55  # Height above for predicting grasp processing
        self.APPROACH_HEIGHT = 0.35  # Height above object for pre-grasp
        self.GRASP_CLEARANCE = 0.1  # Clearance when grasping
        self.LIFT_HEIGHT = 0.25      # Height to lift object to
        self.FRAME_MARKER_URDF = os.path.join(os.path.dirname(os.path.realpath(__file__)), "env/frame_marker.urdf")
        self.eef_marker = p.loadURDF(self.FRAME_MARKER_URDF, [0, 0, 0], useFixedBase=True, globalScaling=0.18)

    def reset(self, *, options=None, **kwargs) -> None:
        print("Resetting...")
        if self.saved_state_id is None:
            LOGGER.info("Setting up environment objects")
            plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)
            self.objects["plane"] = {"body_id": plane_id}

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
            # Pedestals with collision-safe placement and arm mounting
            left_mount_id, right_mount_id, z_table = self._place_pedestals_clear_of_table(
                side=0.20,   # 20 cm square column
                margin=0.08  # 8 cm clearance from table edges
            )

            self.objects["left_mount"] = {"body_id": left_mount_id, "top_z": z_table}
            self.objects["right_mount"] = {"body_id": right_mount_id, "top_z": z_table}

            yaw = math.radians(self.recipe.scene.world_yaw_deg)
            left_base_pos, _ = p.getBasePositionAndOrientation(left_mount_id, physicsClientId=self.client_id)
            right_base_pos, _ = p.getBasePositionAndOrientation(right_mount_id, physicsClientId=self.client_id)

            self.left_arm = self._spawn_arm(
                base_position=[left_base_pos[0], left_base_pos[1], z_table],
                base_orientation=p.getQuaternionFromEuler([0.0, 0.0, yaw + math.pi / 2.0]),
            )
            
            z_table = self._get_table_top_z()

            bowl_1_pose = Pose(0,0,self._get_table_top_z(),[0,0,0])
            bowl_1_id, bowl_1_params = bowl_factory.create_rounded_bowl(self.client_id, bowl_1_pose)
            self.objects["bowl"] = {"body_id": bowl_1_id, "properties": bowl_1_params, "pose": bowl_1_pose}
            self.spawn_rice_particles(20)
            bowl_2_pose = Pose(0,0.3,self._get_table_top_z(),[0,0,0])
            bowl_2_id, bowl_2_params = bowl_factory.create_rounded_bowl(self.client_id, bowl_2_pose)
            self.objects["pouring_bowl"] = {"body_id": bowl_2_id, "properties": bowl_2_params, "pose": bowl_2_pose}

            p.stepSimulation(physicsClientId=self.client_id)

            down_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
            self.move_arm_to_pose("left", [0,0,1.0], down_orn, max_secs=5.0)
            ee_pos, ee_orn = self.get_eef_state("left")
            self.setup_camera(ee_pos, ee_orn)
            self.gripper_open(arm="left")
            ee_pos, ee_orn = self.get_eef_state("left")
            self.setup_camera(ee_pos, ee_orn)
            rgb, depth_buffer, depth_norm, _= self.camera.get_rgbd()
            pixel_row, pixel_col, grasp_angle, grasp_width, quality, quality_map, angle_map, width_map = self.grasping_predictor.predict_grasp(depth_norm)
            #self.grasping_predictor.visualize_grasp_predictions(depth_norm, quality_map, angle_map, width_map, pixel_row, pixel_col, "grasp_visualisation_recali.png")
            coord = self.pixel_to_world(pixel_row, pixel_col, depth_buffer)
            self.set_grasping_keyposes(coord)
            self.move_arm_to_pose("left", self.keyposes["pregrasp"], down_orn, max_secs=3.0)
            grasp_euler = [math.pi, 0, grasp_angle]
            grasp_orn = p.getQuaternionFromEuler(grasp_euler)
            self.move_gripper_straight_down(self.keyposes["pregrasp"], self.keyposes["grasp"], grasp_orn, "left")
            self.gripper_close(arm="left", force=200.0)
            self.move_arm_to_pose("left", self.keyposes["lift"], down_orn, max_secs=3.0)
            self.spawn_one_keypose_markers([0,0,self._get_table_top_z() + 0.05], "Poured thresh")
            self.saved_state_id = p.saveState()
        else:
            p.restoreState(self.saved_state_id)
            self.current_step = 0

        obs = self.get_obs()
        info = {}
        return obs, info

    def step(self, action):
        arm = self._get_arm("left")
        joint_limits = [
            (-2.8973, 2.8973),
            (-1.7628, 1.7628),
            (-2.8973, 2.8973),
            (-3.0718, -0.0698),
            (-2.8973, 2.8973),
            (-0.0175, 3.7525),
            (-2.8973, 2.8973),
        ]
        denorm_action = []
        for i, a in enumerate(action):
            low, high = joint_limits[i]
            # Scale from [-1,1] to [low, high]
            scaled = low + (a + 1.0) * 0.5 * (high - low)
            denorm_action.append(scaled)
        for idx in range(7):
            # Map action[idx] from [-1,1] to joint limit range (example)
            # You need to define real joint limits here for safety
            target_pos = denorm_action[idx]  # example scaling
            p.setJointMotorControl2(arm.body_id, idx, p.POSITION_CONTROL, targetPosition=target_pos, maxVelocity=1.0, force=80)

        p.stepSimulation()
        self.current_step += 1

        reward = 0.0
        terminated = False
        truncated =False

        obs = self.get_obs()
        balls_poured_correctly = self._count_balls_in_target_bowl(self.objects["pouring_bowl"]["properties"], 2)
        balls_poured = self._count_balls_poured()
        balls_poured_wrongly = balls_poured - balls_poured_correctly
        reward += balls_poured_correctly * 2.0
        reward -= balls_poured_wrongly * 1.0
        if balls_poured_correctly == 20:
            reward += 50.0
        
        if balls_poured == 20:
            terminated = True

        truncated = self.current_step >= self.max_steps 
        if truncated:
            print("[Truncated] Steps used up")
        if terminated:
            print("[Terminated] All balls are poured")

        ee_pos, _ = self.get_eef_state("left")
        bowl_one_pos, bowl_one_orn = p.getBasePositionAndOrientation(self.objects["bowl"]["body_id"])
        bowl_two_pos, _ = p.getBasePositionAndOrientation(self.objects["pouring_bowl"]["body_id"])
        dist_between_bowls_xy = np.sqrt((bowl_one_pos[0] - bowl_two_pos[0])**2 + (bowl_one_pos[1] - bowl_two_pos[1])**2)
        
        if dist_between_bowls_xy < 0.2:
            reward += (0.2 - dist_between_bowls_xy) / 0.2
        else:
            reward -= min((dist_between_bowls_xy - 0.2) * 2.0, 5.0)
        
        height_diff = abs(bowl_one_pos[2] - bowl_two_pos[2])
        if height_diff > 0.3 :
            reward -= min(height_diff * 2.0, 3.0)
        else:
            reward += 0.5

        bowl_euler = p.getEulerFromQuaternion(bowl_one_orn)
        tilt_angle = abs(bowl_euler[0]) + abs(bowl_euler[1])
        if dist_between_bowls_xy < 0.2:
            if tilt_angle > 0.3:
                reward += min(tilt_angle * 2.0, 2.0)
            else:
                reward -= 0.2

        if dist_between_bowls_xy > 0.3 and tilt_angle > 0.5:
            reward -= 1.0

        distance = np.linalg.norm(np.array(ee_pos) - np.array(bowl_one_pos))
        if distance > 0.15:
            terminated = True
            reward -= 20
            print("[Terminate] Not grasping bowl anymore")

        reward -=  0.2

        if terminated or truncated:
            print(f"[Result for episode] {balls_poured_correctly} in target bowl")
        
        info = {}
        return obs, reward, terminated, truncated, info

    def get_obs(self, arm="left"):
        arm = self._get_arm(arm)
        joint_positions = [p.getJointState(arm.body_id, i)[0] for i in range(7)]
        joint_velocities = [p.getJointState(arm.body_id, i)[1] for i in range(7)]
        bowl_one_pos, bowl_one_orn = p.getBasePositionAndOrientation(self.objects["bowl"]["body_id"])
        bowl_two_pos, bowl_two_orn = p.getBasePositionAndOrientation(self.objects["pouring_bowl"]["body_id"])
        balls_states = []
        balls = self.particles.body_ids
        for i in range(20):
            if i < len(balls):
                ball_id = balls[i]
                pos, _ = p.getBasePositionAndOrientation(ball_id)
                lin_vel, _ = p.getBaseVelocity(ball_id)
                balls_states.extend(list(pos) + list(lin_vel))
            else:
                balls_states.extend([0.0] * 6)
        balls_poured = self._count_balls_in_target_bowl(self.objects["pouring_bowl"]["properties"], 2)
        elapsed_steps = self.current_step
        obs = np.array(
            joint_positions +
            joint_velocities +
            list(bowl_one_pos) + list(bowl_one_orn) +
            list(bowl_two_pos) + list(bowl_two_orn) +
            balls_states +
            [balls_poured, elapsed_steps],
            dtype=np.float32
        )

        return obs
        
    def _count_balls_poured(self):
        count = 0
        table_height = self._get_table_top_z()
        for ball_id in self.particles.body_ids:
            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
            if ball_pos[2] < table_height + 0.05:
                count += 1
        return count


    def _count_balls_in_target_bowl(self, bowl_params, bowl):
        count = 0
        bowl_pos = None
        if bowl == 1:
            bowl_pos, _ = p.getBasePositionAndOrientation(self.objects["bowl"]["body_id"])
        else:
            bowl_pos, _ = p.getBasePositionAndOrientation(self.objects["pouring_bowl"]["body_id"])
            
        bowl_pos = np.array(bowl_pos)

        # Unpack bowl parameters (all assumed scaled and in world units)
        total_height = bowl_params["inner_height"] + bowl_params["base_thickness"]
        inner_radius = bowl_params["inner_radius"]

        # Calculate vertical boundaries relative to bowl center position
        bowl_bottom_z = bowl_pos[2] - (total_height / 2)
        bowl_top_z = bowl_pos[2] + (total_height / 2)

        # Define a radius threshold (inner radius of bowl) for horizontal check
        radius_threshold = inner_radius

        for ball_id in self.particles.body_ids:
            ball_pos, _ = p.getBasePositionAndOrientation(ball_id)
            ball_pos = np.array(ball_pos)

            # Horizontal distance in XY plane from bowl center
            dist_xy = np.linalg.norm(ball_pos[:2] - bowl_pos[:2])

            # Check ball inside bowl horizontal radius and between bottom and top Z
            if dist_xy <= radius_threshold and bowl_bottom_z <= ball_pos[2] <= bowl_top_z:
                count += 1

        return count

    
    def render(self):
        # Optional slow down for visualization
        time.sleep(1./240.)  # Sleep for one simulation timestep (default 240 Hz)
    
    def close(self):
        p.disconnect(self.physics_client)

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

        n = p.getNumJoints(arm_id)
        link_names = [p.getJointInfo(arm_id, j)[12].decode("utf-8") for j in range(n)]
        eef_link_name = "panda_hand"
        eef = link_names.index(eef_link_name) #returns eef coordinate, ee_link_index is joint index

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
            eef=eef,
            finger_joints=(fingers[0], fingers[1]),
            joint_lower=tuple(joint_lower),
            joint_upper=tuple(joint_upper),
            joint_ranges=tuple(joint_ranges),
            joint_rest=tuple(joint_rest),
        )

    def wait_steps(self, steps=240, arm=None):
      arm = self._get_arm(arm)
      for _ in range(steps):
        if self.eef_marker and arm.body_id and arm.eef is not None:
            pos, orn = p.getLinkState(arm.body_id, arm.eef)[4:6]
            p.resetBasePositionAndOrientation(self.eef_marker, pos, orn)
        p.stepSimulation()
        #time.sleep(1./240.) #removed for rl

    # ------------------------------------------------------------------ #
    # Public API

    def setup_camera(self, eef_pos, eef_orn):
      cam_offset = [0, 0, 0.05]
      cam_pos, cam_orn = p.multiplyTransforms(eef_pos, eef_orn, cam_offset, [0, 0, 0, 1])
      forward_vec = [0, 0, 0.1] 
      cam_pos, cam_orn = p.multiplyTransforms(cam_pos, cam_orn, forward_vec, [0, 0, 0, 1])
      cam_rpy_rad = p.getEulerFromQuaternion(cam_orn)
      cam_rpy_deg = np.degrees(cam_rpy_rad)
      self.camera = Camera(self.client_id, cam_pos, cam_rpy_deg, self._get_table_top_z())


    def step_simulation(self, steps: int) -> None:
        for _ in range(max(1, int(steps))):
            p.stepSimulation(physicsClientId=self.client_id)

    def get_eef_state(self, arm):
        arm = self._get_arm(arm)
        ee_pos, ee_orn = p.getLinkState(arm.body_id, arm.eef)[:2]
        return ee_pos, ee_orn

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

    def set_joint_velocities( #Unused
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
                positionGain=0.01,
                velocityGain=0.1,
                force=100.0,
                physicsClientId=self.client_id,
            )
        self.wait_steps(60, arm)

    def gripper_close(self, force: float = 60.0, arm: Optional[str] = None) -> None:
        arm_state = self._get_arm(arm)
        for joint_id in arm_state.finger_joints:
            p.setJointMotorControl2(
                arm_state.body_id,
                joint_id,
                controlMode=p.POSITION_CONTROL,
                targetPosition=0.0,
                positionGain=0.01,
                velocityGain=0.1,
                force=force,
                physicsClientId=self.client_id,
            )
        self.wait_steps(120, arm)

    def spawn_rice_particles(
        self,
        count: int,
        radius: float = 0.0075,
        seed: int = 7,
    ) -> Optional[ParticleSet]:
        bowl_entry = self.objects.get("bowl")
        if not bowl_entry:
            LOGGER.warning("Cannot spawn particles before bowl is created")
            return None
        bowl_id = bowl_entry["body_id"]
        position, _ = p.getBasePositionAndOrientation(bowl_id, physicsClientId=self.client_id)
        bowl_props = bowl_entry["properties"]
        particle_set = spawn_spheres(
            client_id=self.client_id,
            count=count,
            radius=radius,
            center=position,
            bowl_radius=float(bowl_props["inner_radius"]),
            bowl_height=float(bowl_props["inner_height"]),
            spawn_height=float(bowl_props["spawn_height"]),
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

    def spawn_one_keypose_markers(self, pos, name, scale=0.1):
      down_orn = p.getQuaternionFromEuler([math.pi, 0, 0]) #gripper to look down
      p.loadURDF(self.FRAME_MARKER_URDF, pos, down_orn, useFixedBase=True, globalScaling=scale)
      # Add text above the frame
      text_pos = [pos[0], pos[1], pos[2] + 0.01]
      p.addUserDebugText(name, text_pos, textColorRGB=[1, 1, 0], textSize=1.2)

    # Tutorial code to spawn markers for visualisation
    def _spawn_keypose_markers(self, scale=0.1):
      for name, pos in self.keyposes.items():
        down_orn = p.getQuaternionFromEuler([math.pi, 0, 0]) #gripper to look down
        p.loadURDF(self.FRAME_MARKER_URDF, pos, down_orn, useFixedBase=True, globalScaling=scale)
        # Add text above the frame
        text_pos = [pos[0], pos[1], pos[2] + 0.01]
        p.addUserDebugText(name, text_pos, textColorRGB=[1, 1, 0], textSize=1.2)

    # Tutorial code
    def _smooth_joint_targets(self, body_id: int, joint_indices: List[int], ik_targets: List[float], step_size=0.02):
        out = []
        for k, j in enumerate(joint_indices):
            curr = p.getJointState(body_id, j)[0]
            tgt  = float(ik_targets[k])
            delta = tgt - curr
            if delta > step_size: new = curr + step_size
            elif delta < -step_size: new = curr - step_size
            else: new = tgt
            out.append(new)
        return out
    
    def set_recalibration_keypose(self, pos):
      recalibrate_pose = [pos[0], pos[1], pos[2] + self.RECALIBRATE_HEIGHT]
      self.keyposes = {
          "recalibration": recalibrate_pose
      }
      self._spawn_keypose_markers()
    
    def set_grasping_keyposes(self, grasp_pos_world):
      pregrasp = [grasp_pos_world[0], grasp_pos_world[1], grasp_pos_world[2] + self.APPROACH_HEIGHT]
      grasp_pose = [grasp_pos_world[0], grasp_pos_world[1], grasp_pos_world[2] + self.GRASP_CLEARANCE]
      lift_pose = [-0.05, 0.4, grasp_pos_world[2] + self.LIFT_HEIGHT]

      self.keyposes = {
          "pregrasp": pregrasp,
          "grasp": grasp_pose,
          "lift": lift_pose,
      }

      self._spawn_keypose_markers()

    # Tutorial code (slightly modified to fit class)
    def move_arm_to_pose(self, arm,
                     target_pos: List[float],
                     target_orn: Tuple[float, float, float, float],
                     max_secs=2.0,
                     pos_gain=0.3,
                     vel_gain=1.0,
                     torque=200):
        """IK + PD stepper until EE is near target or timeout."""
        arm = self._get_arm(arm)
        max_steps = int(max_secs / self.dt)
        for _ in range(max_steps):
            rest = [p.getJointState(arm.body_id, j)[0] for j in arm.joint_indices]
            ik = p.calculateInverseKinematics(
                arm.body_id, arm.eef,
                target_pos, target_orn,
                lowerLimits=arm.joint_lower, upperLimits=arm.joint_upper, jointRanges=arm.joint_ranges, restPoses=rest
            )
            ik = list(ik[:len(arm.joint_indices)])
            smoothed = self._smooth_joint_targets(arm.body_id, arm.joint_indices, ik, step_size=0.015)
            for idx, j in enumerate(arm.joint_indices):
                p.setJointMotorControl2(arm.body_id, j, p.POSITION_CONTROL,
                                        targetPosition=smoothed[idx],
                                        positionGain=pos_gain, velocityGain=vel_gain, force=torque)
                
            if self.eef_marker:
                pos, orn = p.getLinkState(arm.body_id, arm.eef)[4:6]
                p.resetBasePositionAndOrientation(self.eef_marker, pos, orn)

            p.stepSimulation()
            #time.sleep(self.dt) #removed for rl

            pos, orn = p.getLinkState(arm.body_id, arm.eef)[4:6]
            pos_err = np.linalg.norm(np.array(pos) - np.array(target_pos))
            orn_err = 1 - abs(np.dot(orn, target_orn))
            if pos_err < 0.001 and orn_err < 0.0001:
                return True
        return False

    def pixel_to_world(self, pixel_row, pixel_col, depth_image,  
                           img_width=224, img_height=224):
      depth_buffer = depth_image

      # Convert pixel to NDC coordinates
      x_ndc = (2.0 * pixel_col / img_width) - 1.0
      y_ndc = 1.0 - (2.0 * pixel_row / img_height)
      z_ndc = 2.0 * depth_buffer[pixel_row, pixel_col] - 1.0
      
      # Create homogeneous NDC point
      ndc_point = np.array([x_ndc, y_ndc, z_ndc, 1.0])
      
      # Get inverse matrices
      view_matrix_np = self.camera._view_matrix.T
      proj_matrix_np = self.camera._projection_matrix.T
      
      view_inv = np.linalg.inv(view_matrix_np)
      proj_inv = np.linalg.inv(proj_matrix_np)
      
      # Unproject: NDC -> Clip -> Camera -> World
      clip_point = proj_inv @ ndc_point
      
      # Perspective divide
      if clip_point[3] != 0:
          clip_point = clip_point / clip_point[3]
      
      # Transform to world space
      world_point = view_inv @ clip_point
      
      return [world_point[0], world_point[1], world_point[2]]    

    def move_gripper_straight_down(self, pregrasp_pose, grasp_pose, grasp_orn, arm):
        arm = self._get_arm(arm)
        start_pos = np.array(pregrasp_pose)
        end_pos = np.array(grasp_pose)
        down_orn = p.getQuaternionFromEuler([math.pi, 0, 0]) #gripper to look down
        steps = 500
        half_steps = steps // 2 # rotation of gripper should reach endgoal halfway down
        trajectory = [start_pos + (end_pos - start_pos) * t/steps for t in range(steps+1)]
        orn_traj = []
        for i in range(half_steps):
            t = i / (half_steps - 1)   # normalized [0,1] for slerp smoothness
            q = p.getQuaternionSlerp(down_orn, grasp_orn, t)
            orn_traj.append(q)
        for i in range(steps - half_steps):
            orn_traj.append(grasp_orn) # fix gripper angled position for the rest of the journey

        for pos, orn in zip(trajectory, orn_traj):
          joint_pos = p.calculateInverseKinematics(arm.body_id, arm.eef, pos, orn)[:-2]
          p.setJointMotorControlArray(arm.body_id, arm.joint_indices, p.POSITION_CONTROL, joint_pos)
          p.stepSimulation()
          #time.sleep(self.dt) # [IMPT]: reduce jumping of joints for real life time simulation, removed for rl


__all__ = ["RobotChefSimulation", "ArmState"]
