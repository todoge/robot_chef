"""Task scenario for path planning: Right arm must reach under Left arm."""

from __future__ import annotations

import logging
import time
import math
import random
import pybullet as p
import numpy as np

from ..config import PourTaskConfig, Pose6D
from ..simulation import RobotChefSimulation
from ..env.objects import create_specula

LOGGER = logging.getLogger(__name__)

class UnderArmReachingTask:
    """
    Scenario:
    1. Left Arm: Holds a specula and hovers rigidly over the table.
    2. Target: A bowl placed underneath/behind the Left Arm's bridge.
    3. Goal: Right Arm starts at home, must path plan to grasp the bowl without hitting the Left Arm.
    """

    def __init__(self) -> None:
        self.sim = None
        self.cfg = None
        self._grasped_object_id = None # Track what we are holding

    def setup(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> None:
        self.sim = sim
        self.cfg = cfg
        
        # --- 0. SAFETY FIRST: Move Right Arm to Safe Home ---
        safe_right_joints = [-1.5, -1.0, 0.0, -2.2, 0.0, 1.8, 0.785]
        sim.set_joint_positions(safe_right_joints, arm="right")
        sim.step_simulation(20)

        # 1. Spawn Specula
        specula_pose = Pose6D(0.0, 0.0, 0.5, 0, 0, 0) 
        self.specula_id, _ = create_specula(sim.client_id, specula_pose)
        
        # 2. Setup Left Arm (The Obstacle)
        left_arm = sim.left_arm
        
        # Move Left Arm to a "Blocking" configuration
        # UPDATED: Rotated base to -1.2 to point arm towards the right side
        blocking_joints = [-1.2, 0.3, 0.0, -1.4, 0.0, 2.0, 0.785]
        
        sim.set_joint_positions(blocking_joints, arm="left")
        sim.step_simulation(50) 
        
        # 3. Attach Specula to Left Hand
        ls = p.getLinkState(left_arm.body_id, left_arm.ee_link, physicsClientId=sim.client_id)
        eef_pos = ls[4]
        eef_orn = ls[5]
        
        p.resetBasePositionAndOrientation(self.specula_id, eef_pos, eef_orn, physicsClientId=sim.client_id)
        
        # Create Constraint (Rigid Lock)
        p.createConstraint(
            parentBodyUniqueId=left_arm.body_id,
            parentLinkIndex=left_arm.ee_link,
            childBodyUniqueId=self.specula_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, -0.05], 
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=[0, 0, 0, 1],
            physicsClientId=sim.client_id
        )
        
        sim.gripper_close(arm="left")
        sim.step_simulation(50)

        LOGGER.info("Waiting for specula to render...")
        for _ in range(20):
            sim.step_simulation(1)
            time.sleep(0.05)
            
        if not sim.particles:
             LOGGER.info("Spawning particles manually...")
             sim.spawn_rice_particles(80, seed=cfg.seed)
        
        LOGGER.info("Scenario Setup: Left Arm is holding specula in blocking pose.")

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning phase: The environment is static. The Left Arm is the obstacle.")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        """
        Execute RRT Reach -> Grasp -> Safe Neutral -> Move to Pan -> Pour.
        """
        LOGGER.info("Execution: Calculating RRT Path...")
        
        bowl = sim.objects.get("bowl")
        pan = sim.objects.get("pan")
        if not bowl or not pan: return False

        target_pose = bowl["pose"]
        
        # --- CALCULATE RIM GRASP ---
        # Bowl properties from create_rice_bowl
        bowl_radius = 0.07 
        bowl_rim_height = 0.08 # wall_height
        bowl_base_z = target_pose.z
        
        # Grasp Target: The TOP edge of the rim (closest to robot)
        # y - radius puts us at the front edge
        grasp_x = target_pose.x
        grasp_y = target_pose.y - bowl_radius 
        
        # IMPORTANT FIX: Account for Hand Length (Palm to Finger center ~10.3cm)
        # If we send the "Hand Link" to the rim height, the fingers smash into the floor.
        # We must offset the Z target UP by the finger length so the TIPS land on the rim.
        HAND_OFFSET_Z = 0.103 
        
        # Target Tip Position: 1.5cm below rim top
        tip_z = bowl_base_z + bowl_rim_height - 0.015
        
        # Target Link Position: Tip Z + Hand Length
        grasp_z = tip_z + HAND_OFFSET_Z
        
        grasp_target_pos = [grasp_x, grasp_y, grasp_z]
        grasp_orn = p.getQuaternionFromEuler([3.14, 0, 0]) # Face down

        LOGGER.info(f"Bowl Center: {target_pose.position}")
        LOGGER.info(f"Calculated Rim Grasp Target (Link): {grasp_target_pos} (Tips at {tip_z:.3f})")

        q_start, _ = sim.get_joint_state(arm="right")
        
        # --- 1. RRT: Home -> Pre-Grasp (Hover) ---
        # Search for valid goal
        q_pre_grasp = None
        for hover_gap in np.linspace(0.30, 0.10, 5): 
            # Hover directly above the RIM grasp point, not center
            test_pos = [grasp_x, grasp_y, grasp_z + hover_gap]
            q_pre_grasp = self._find_valid_goal(sim, test_pos, grasp_orn, trials=100)
            if q_pre_grasp is not None:
                LOGGER.info(f"Found valid Pre-Grasp at hover gap +{hover_gap:.2f}m")
                break
        
        if q_pre_grasp is None:
            LOGGER.error("CRITICAL: Could not find ANY valid goal configuration!")
            return False

        LOGGER.info("Planning Path to Pre-Grasp...")
        path_in = self._rrt_connect(sim, q_start, q_pre_grasp, max_iter=5000) 
        
        if not path_in:
            LOGGER.error("RRT Failed to find path IN!")
            return False

        LOGGER.info("Smoothing Path IN...")
        path_in = self._smooth_path(sim, path_in)

        # Execute Path IN
        LOGGER.info(f"Moving to hover ({len(path_in)} steps)...")
        self._execute_path(sim, path_in)
        sim.gripper_open(arm="right")
        time.sleep(0.5)

        # --- 2. Cartesian: Grasp Sequence ---
        LOGGER.info("Descending to grasp rim...")
        self._linear_move(sim, grasp_target_pos, grasp_orn, steps=40)
        
        # --- MAGIC GRASP ---
        # Freeze Bowl temporarily to prevent spin during grasp
        bowl_id = int(bowl["body_id"])
        p.changeDynamics(bowl_id, -1, mass=0) # Make static
        
        sim.gripper_close(arm="right", force=60.0)
        for _ in range(50): sim.step_simulation(1)
        
        # Restore mass and Lock bowl to gripper
        p.changeDynamics(bowl_id, -1, mass=0.3)
        self._create_grasp_constraint(sim, bowl_id)
        self._grasped_object_id = bowl_id

        LOGGER.info("Lifting bowl (retracting slightly)...")
        # Lift UP and slightly AWAY (back towards robot) to clear the arch
        lift_pos = [grasp_x, grasp_y - 0.1, grasp_z + 0.20]
        self._linear_move(sim, lift_pos, grasp_orn, steps=40)

        # --- 3. RRT: Lift -> Safe Neutral (Retract) ---
        LOGGER.info("Planning Path to Safe Neutral...")
        
        # Neutral Pose
        q_neutral = np.array([-0.8, -0.2, 0.0, -2.0, 0.0, 1.8, 0.785])
        
        # CRITICAL FIX: Re-read state fresh from physics engine
        q_lifted, _ = sim.get_joint_state(arm="right")
        
        # UPDATED: Use start_safety_override=True because we are holding the bowl
        path_out = self._rrt_connect(sim, q_lifted, q_neutral, max_iter=5000, start_safety_override=True)
        
        if not path_out:
            LOGGER.error("RRT Failed to find path OUT to neutral!")
            return False
            
        LOGGER.info("Smoothing Path OUT...")
        path_out = self._smooth_path(sim, path_out)
        
        LOGGER.info(f"Retracting to Neutral ({len(path_out)} steps)...")
        self._execute_path(sim, path_out)
        time.sleep(0.5)

        # --- 4. Cartesian: Neutral -> Pan ---
        LOGGER.info("Moving to Pan...")
        pan_pose = pan["pose"]
        # Pour Pos: Center of pan, Z high enough.
        # Add HAND_OFFSET_Z here too so tips are at correct height!
        pour_z_clearance = 0.30
        pour_pos = [pan_pose.x, pan_pose.y, pan_pose.z + pour_z_clearance + HAND_OFFSET_Z]
        self._linear_move(sim, pour_pos, grasp_orn, steps=60)
        
        # --- 5. Pour (Tilt) ---
        LOGGER.info("Pouring...")
        tilt_orn = p.getQuaternionFromEuler([3.14, -2.1, 0.0]) 
        self._linear_move(sim, pour_pos, tilt_orn, steps=80)
        
        for _ in range(120): sim.step_simulation(1)
        
        LOGGER.info("Untilting...")
        self._linear_move(sim, pour_pos, grasp_orn, steps=50)

        metrics = sim.count_particles_in_pan()
        LOGGER.info(f"Task Complete. Metrics: {metrics}")
        
        return True
    
    # ------------------------------------------------------------------ #
    # Helpers

    def _create_grasp_constraint(self, sim, object_id):
        """Creates a fixed constraint between the gripper palm and the object."""
        # Find Panda Hand link index
        arm = sim.right_arm
        # 11 is usually panda_hand, but use ee_link just in case
        parent_link = arm.ee_link 
        
        # Get current poses
        parent_state = p.getLinkState(arm.body_id, parent_link, computeForwardKinematics=True, physicsClientId=sim.client_id)
        parent_pos = np.array(parent_state[4])
        parent_orn = np.array(parent_state[5])
        
        child_state = p.getBasePositionAndOrientation(object_id, physicsClientId=sim.client_id)
        child_pos = np.array(child_state[0])
        child_orn = np.array(child_state[1])
        
        # Calculate relative pose: T_parent_inv * T_child
        # This gives child frame in parent frame
        inv_parent_pos, inv_parent_orn = p.invertTransform(parent_pos, parent_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_parent_pos, inv_parent_orn, child_pos, child_orn)
        
        # Create constraint using the computed relative pose
        cid = p.createConstraint(
            parentBodyUniqueId=arm.body_id,
            parentLinkIndex=parent_link,
            childBodyUniqueId=object_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=rel_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=rel_orn,
            childFrameOrientation=[0, 0, 0, 1], # Identity
            physicsClientId=sim.client_id
        )
        
        p.changeConstraint(cid, maxForce=2000)
        LOGGER.info(f"Grasp constraint created (ID {cid}) with relative lock")
        return cid

    def _find_valid_goal(self, sim, target_pos, target_orn, trials=100):
        """Attempts to find a valid IK solution by trying random IK seeds (rest poses)."""
        
        # 1. Try Default
        q = self._solve_ik_safe(sim, target_pos, target_orn, margin=0.002) 
        if q is not None: 
            return q
            
        LOGGER.warning(f"Exact Pre-Grasp at z={target_pos[2]:.2f} in collision. Attempting IK with random seeds...")
        
        arm = sim.right_arm
        
        # 2. Randomize Posture (Null Space Search)
        for i in range(trials):
            random_rest = []
            for j in range(len(arm.joint_lower)):
                val = random.uniform(arm.joint_lower[j], arm.joint_upper[j])
                random_rest.append(val)
            
            noise = np.random.uniform(-0.05, 0.05, 3)
            perturbed_pos = np.array(target_pos) + noise
            
            q = self._solve_ik_safe(sim, perturbed_pos, target_orn, rest_pose=random_rest, margin=0.002)
            if q is not None:
                LOGGER.info(f"Found valid goal after {i+1} random seeds.")
                return q
                
        return None

    def _solve_ik_safe(self, sim, pos, orn, rest_pose=None, debug=False, margin=0.005):
        """Attempts to find a collision-free IK solution."""
        arm = sim.right_arm
        args = {
            "bodyUniqueId": arm.body_id,
            "endEffectorLinkIndex": arm.ee_link,
            "targetPosition": pos,
            "targetOrientation": orn,
            "maxNumIterations": 100,
            "residualThreshold": 1e-4,
            "physicsClientId": sim.client_id
        }
        if rest_pose is not None:
            args["lowerLimits"] = list(arm.joint_lower)
            args["upperLimits"] = list(arm.joint_upper)
            args["jointRanges"] = list(arm.joint_ranges)
            args["restPoses"] = list(rest_pose)
        
        full_ik = p.calculateInverseKinematics(**args)
        q = np.array(full_ik[:7])
        
        if not self._check_collision(sim, q, debug=debug, margin=margin):
            return q
        return None

    def _execute_path(self, sim, path):
        for q in path:
            sim.set_joint_positions(q, arm="right")
            for _ in range(5): sim.step_simulation(1)

    def _linear_move(self, sim, target_pos, target_orn, steps=60):
        current_pos = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[4]
        current_orn = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[5]
        
        for t in np.linspace(0, 1, steps):
            pos = np.array(current_pos) * (1 - t) + np.array(target_pos) * t
            orn = target_orn 
            
            full_ik = p.calculateInverseKinematics(
                sim.right_arm.body_id, sim.right_arm.ee_link, pos, orn,
                physicsClientId=sim.client_id
            )
            sim.set_joint_positions(full_ik[:7], arm="right")
            sim.step_simulation(1)
            time.sleep(0.01)

    # ------------------------------------------------------------------ #
    # RRT Implementation

    def _check_collision(self, sim: RobotChefSimulation, q: np.ndarray, debug: bool = False, margin: float = 0.005) -> bool:
        for i, idx in enumerate(sim.right_arm.joint_indices):
            p.resetJointState(sim.right_arm.body_id, idx, q[i], physicsClientId=sim.client_id)
        
        p.performCollisionDetection(physicsClientId=sim.client_id)
        
        contacts = p.getContactPoints(bodyA=sim.right_arm.body_id, physicsClientId=sim.client_id)
        
        right_mount_id = sim.objects.get("right_mount", {}).get("body_id")
        
        for c in contacts:
            body_b = c[2]
            if body_b == sim.right_arm.body_id: continue
            if right_mount_id is not None and body_b == right_mount_id: continue
            
            # IGNORE GRASPED OBJECT
            if self._grasped_object_id is not None and body_b == self._grasped_object_id:
                continue

            if debug:
                body_name = p.getBodyInfo(body_b, physicsClientId=sim.client_id)[1].decode()
                LOGGER.debug(f"Collision detected with Body ID {body_b} ({body_name})")
            return True

        obstacles = [sim.left_arm.body_id, self.specula_id]
        
        for obs_id in obstacles:
            closest = p.getClosestPoints(
                bodyA=sim.right_arm.body_id, 
                bodyB=obs_id, 
                distance=margin, 
                physicsClientId=sim.client_id
            )
            if len(closest) > 0:
                if debug: LOGGER.debug(f"Safety Margin violation with obstacle {obs_id}")
                return True
                
        return False

    def _check_segment_collision(self, sim: RobotChefSimulation, q1: np.ndarray, q2: np.ndarray, step_rad: float = 0.02) -> bool:
        dist = np.linalg.norm(q2 - q1)
        steps = int(math.ceil(dist / step_rad))
        
        if steps <= 1:
            return self._check_collision(sim, q2)
            
        for k in range(1, steps + 1):
            t = k / steps
            q_interp = q1 + (q2 - q1) * t
            if self._check_collision(sim, q_interp):
                return True
        return False
    
    def _restore_state(self, sim: RobotChefSimulation, q: np.ndarray):
        for i, idx in enumerate(sim.right_arm.joint_indices):
            p.resetJointState(sim.right_arm.body_id, idx, q[i], physicsClientId=sim.client_id)
            # CRITICAL: Also reset velocities to zero to stop "flinging"
            p.resetJointState(sim.right_arm.body_id, idx, q[i], targetVelocity=0.0, physicsClientId=sim.client_id)

    def _rrt_connect(self, sim: RobotChefSimulation, q_start: np.ndarray, q_goal: np.ndarray, max_iter=5000, start_safety_override=False) -> list:
        # Check endpoints first
        # Use override to skip start check if requested (e.g. holding object)
        if not start_safety_override and self._check_collision(sim, q_start):
             LOGGER.error("RRT Start Configuration is in collision.")
             return []
        if self._check_collision(sim, q_goal, debug=True): 
             LOGGER.error("RRT Goal Configuration is in collision.")
             return []

        J_MIN = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
        J_MAX = np.array([ 2.8,  1.7,  2.8, -0.06, 2.8,  3.7,  2.8])
        
        q_real_start, _ = sim.get_joint_state(arm="right")

        class Node:
            def __init__(self, q, parent):
                self.q = q
                self.parent = parent

        start_node = Node(q_start, None)
        goal_node = Node(q_goal, None)
        
        tree_a = [start_node]
        tree_b = [goal_node]
        
        step_size = 0.08 

        def get_nearest(tree, q_rand):
            dists = [np.linalg.norm(node.q - q_rand) for node in tree]
            idx = np.argmin(dists)
            return tree[idx]

        def extend(tree, q_target):
            nearest = get_nearest(tree, q_target)
            direction = q_target - nearest.q
            dist = np.linalg.norm(direction)
            
            if dist < step_size:
                q_new = q_target
            else:
                q_new = nearest.q + (direction / dist) * step_size
            
            if not self._check_segment_collision(sim, nearest.q, q_new):
                new_node = Node(q_new, nearest)
                tree.append(new_node)
                if np.linalg.norm(q_new - q_target) < 1e-3:
                    return "Reached", new_node
                return "Advanced", new_node
            
            return "Trapped", None

        path = []
        
        for i in range(max_iter):
            q_rand = np.random.uniform(J_MIN, J_MAX)
            status, node_a = extend(tree_a, q_rand)
            
            if status != "Trapped":
                status_b, node_b = extend(tree_b, node_a.q)
                while status_b == "Advanced":
                    status_b, node_b = extend(tree_b, node_a.q)
                
                if status_b == "Reached":
                    LOGGER.info(f"RRT Connected at iter {i}")
                    
                    curr = node_a
                    while curr:
                        path.append(curr.q)
                        curr = curr.parent
                    path.reverse()
                    
                    curr = node_b.parent
                    while curr:
                        path.append(curr.q)
                        curr = curr.parent
                        
                    break
            
            tree_a, tree_b = tree_b, tree_a

        self._restore_state(sim, q_real_start)
        return path

    def _smooth_path(self, sim, path, iterations=50):
        if len(path) < 3: return path
        
        q_real_start, _ = sim.get_joint_state(arm="right")
        
        for _ in range(iterations):
            if len(path) < 3: break
            
            i, j = sorted(random.sample(range(len(path)), 2))
            if j - i <= 1: continue 
            
            q1 = path[i]
            q2 = path[j]
            
            if not self._check_segment_collision(sim, q1, q2):
                path = path[:i+1] + path[j:]
                
        self._restore_state(sim, q_real_start)
        return path

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return sim.count_particles_in_pan()

__all__ = ["UnderArmReachingTask"]