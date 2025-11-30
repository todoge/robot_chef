"""Task scenario for path planning: Right arm must reach under Left arm."""

from __future__ import annotations

import logging
import time
import math
import random
import pybullet as p
import numpy as np

from ..config import PourTaskConfig, Pose6D
from ..simulation import RobotChefSimulation # Explicit import for type checking
from ..env.objects import create_specula

LOGGER = logging.getLogger(__name__)

class UnderArmReachingTask:
    """
    Scenario:
    1. Left Arm: Holds a specula and hovers rigidly over the table (Obstacle).
    2. Target: A bowl placed underneath/behind the Left Arm's bridge.
    3. Task: Right Arm navigates from Start -> Bowl (Phase 1), then Bowl -> Pan (Phase 2).
    """

    def __init__(self) -> None:
        self.sim = None
        self.cfg = None
        self._grasped_object_id = None 
        self._active_constraint_id = None

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
        # Rotated base to -1.8 to point arm towards the front-right sector (obstacle mode)
        blocking_joints = [-1.85, 0.51, 0.0, -1.4, 0.0, 2.0, 0.785]
        
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
             sim.spawn_rice_particles(20, radius=0.01, seed=cfg.seed)
        
        LOGGER.info("Scenario Setup: Left Arm is holding specula in blocking pose.")

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning phase: The environment is static. The Left Arm is the obstacle.")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        """
        Execute RRT: Start -> Bowl Hover -> Pan Hover (Simplified).
        """
        LOGGER.info("Execution: Calculating RRT Path...")
        
        bowl = sim.objects.get("bowl")
        pan = sim.objects.get("pan")
        if not bowl or not pan: return False

        target_pose = bowl["pose"]
        bowl_id = int(bowl["body_id"])
        bowl_radius = 0.07 
        
        # --- Define Targets ---
        # 1. Bowl Edge Hover Position
        grasp_x = target_pose.x
        # Adjusted Y to center grasp on the wall thickness (radius 0.07 - half_wall 0.004)
        grasp_y = target_pose.y - (bowl_radius - 0.004)
        
        # Target Z: Wrist/Palm position.
        # Fingers are ~10cm long. Bowl is ~8cm high.
        # Adjusted Z for deeper grasp (0.105) to ensure visual contact
        grasp_z = target_pose.z + 0.105 
        
        grasp_target_pos = [grasp_x, grasp_y, grasp_z]
        grasp_orn = p.getQuaternionFromEuler([3.14, 0, 0]) # Face down

        # Hover 10cm above grasp
        bowl_hover_pos = [grasp_x, grasp_y, grasp_z + 0.10]

        # 2. Pan Hover Position
        pan_pose = pan["pose"]
        pan_hover_pos = [pan_pose.x, pan_pose.y, pan_pose.z + 0.35]
        
        # Lift Position (Start of Phase 2)
        lift_pos = [grasp_x, grasp_y - 0.05, grasp_z + 0.20]

        q_start, _ = sim.get_joint_state(arm="right")
        
        # --- PRE-COMPUTE PLANS ---
        LOGGER.info("Pre-planning: Computing all paths before execution...")

        # 1. Path to Bowl
        # Force an intermediate waypoint to keep arm over the table initially
        # This prevents wide swings that go out of bounds
        q_intermediate_guess = [0.0, -0.5, 0.0, -2.0, 0.0, 2.0, 0.785] # A generic "high center" pose
        
        # Increased trials for finding valid IK (was 100)
        q_bowl_hover = self._find_valid_goal(sim, bowl_hover_pos, grasp_orn, trials=300)
        if q_bowl_hover is None:
            LOGGER.error("CRITICAL: Could not find valid configuration at Bowl!")
            return False

        # Plan in two segments: Start -> High Center -> Bowl Hover
        # Increased RRT iterations (was 2000 -> 3000)
        path_to_center = self._rrt_connect(sim, q_start, np.array(q_intermediate_guess), max_iter=3000)
        if not path_to_center:
             # Fallback to direct plan if intermediate fails (unlikely)
             path_to_bowl = self._rrt_connect(sim, q_start, q_bowl_hover, max_iter=7000)
        else:
             # Increased RRT iterations (was 3000 -> 5000)
             path_from_center = self._rrt_connect(sim, np.array(q_intermediate_guess), q_bowl_hover, max_iter=5000)
             if not path_from_center:
                 LOGGER.error("RRT Failed: Center -> Bowl")
                 return False
             path_to_bowl = path_to_center + path_from_center[1:]

        if not path_to_bowl:
            LOGGER.error("RRT Failed to find path to Bowl!")
            return False
        path_to_bowl = self._smooth_path(sim, path_to_bowl)
        
        # 2. Path to Pan
        # We need the configuration AFTER the lift to start the second RRT plan.
        # We estimate q_lift by solving IK for the lift_pos, using q_bowl_hover as a seed.
        q_lift = self._solve_ik_safe(sim, lift_pos, grasp_orn, rest_pose=q_bowl_hover, margin=0.002)
        if q_lift is None:
             # Fallback, increased trials (was 50 -> 150)
             q_lift = self._find_valid_goal(sim, lift_pos, grasp_orn, trials=150)
             
        if q_lift is None:
            LOGGER.error("CRITICAL: Could not find valid configuration at Lift Pose!")
            return False
            
        # Increased trials (was 50 -> 150)
        q_pan_hover = self._find_valid_goal(sim, pan_hover_pos, grasp_orn, trials=150)
        if q_pan_hover is None:
             LOGGER.error("CRITICAL: Could not find valid configuration at Pan!")
             return False
        
        # Plan path assuming arm-only collisions (bowl collisions handled by safety margins/luck in this simplified scope)
        # Increased RRT iterations (was 5000 -> 8000)
        path_to_pan = self._rrt_connect(sim, q_lift, q_pan_hover, max_iter=8000)
        if not path_to_pan:
            LOGGER.error("RRT Failed to find path to Pan!")
            return False
        path_to_pan = self._smooth_path(sim, path_to_pan)

        LOGGER.info("Planning complete. Starting Execution...")

        # --- EXECUTION ---

        # --- PHASE 1: Start -> Bowl Hover ---
        LOGGER.info(f"Moving to Bowl Hover ({len(path_to_bowl)} steps)...")
        # Faster approach (2.0s)
        self._execute_path_interpolated(sim, path_to_bowl, total_time=2.0)
        sim.gripper_open(arm="right")
        time.sleep(0.5)
        
        # --- PHASE 1b: Descend & Grasp ---
        LOGGER.info("Descending to grasp...")
        # Disable collision to prevent premature stop
        self._set_gripper_collision(sim, bowl_id, enable=False)
        # Faster descent
        self._linear_move(sim, grasp_target_pos, grasp_orn, steps=60)
        
        # Re-enable collision BEFORE closing so fingers physically press the wall
        self._set_gripper_collision(sim, bowl_id, enable=True)
        
        # Grasp
        sim.gripper_close(arm="right", force=200.0)
        for _ in range(50): sim.step_simulation(1)
        
        # Settle
        for _ in range(20): sim.step_simulation(1)
        
        # Constrain
        self._active_constraint_id = self._create_grasp_constraint(sim, bowl_id)
        self._grasped_object_id = bowl_id
        
        # Lift slightly
        # Faster lift
        self._linear_move(sim, lift_pos, grasp_orn, steps=60)
        
        # BRIDGE: Smooth transition to exact RRT start config to avoid jerk
        LOGGER.info("Adjusting grip for transport...")
        self._move_joints_smooth(sim, q_lift, duration=0.5)

        # --- PHASE 2: Bowl Hover -> Pan Hover ---
        LOGGER.info(f"Moving to Pan ({len(path_to_pan)} steps)...")
        # Faster transport (1.5s)
        self._execute_path_interpolated(sim, path_to_pan, total_time=1.5)
        time.sleep(0.5)

        # --- PHASE 3: Pour ---
        LOGGER.info("Pouring...")
        # Steeper angle (-3.1 rad) to fully invert container
        tilt_orn = p.getQuaternionFromEuler([3.14, -3.1, 0.0]) 
        
        # Slower, smoother pour (600 steps)
        self._linear_move(sim, pan_hover_pos, tilt_orn, steps=600)
        for _ in range(50): sim.step_simulation(1)
        # Faster return (200 steps)
        self._linear_move(sim, pan_hover_pos, grasp_orn, steps=200)

        metrics = sim.count_particles_in_pan()
        LOGGER.info(f"Task Complete. Metrics: {metrics}")
        return True
    
    # ------------------------------------------------------------------ #
    # Helpers

    def _move_joints_smooth(self, sim, q_target, duration=1.0):
        """Linearly interpolates joint positions to a target configuration."""
        q_current, _ = sim.get_joint_state(arm="right")
        steps = int(duration * 240) # 240Hz physics
        if steps < 1: steps = 1
        for t in np.linspace(0, 1, steps):
            q = q_current + (q_target - q_current) * t
            sim.set_joint_positions(q, arm="right")
            sim.step_simulation(1)
            if self._grasped_object_id is not None:
                sim.gripper_close(arm="right", force=60.0)

    def _set_gripper_collision(self, sim, object_id, enable=True):
        """Enables or disables collision between gripper fingers and object."""
        arm = sim.right_arm
        for i in range(p.getNumJoints(arm.body_id, physicsClientId=sim.client_id)):
            info = p.getJointInfo(arm.body_id, i, physicsClientId=sim.client_id)
            link_name = info[12].decode()
            if "finger" in link_name or "hand" in link_name:
                p.setCollisionFilterPair(arm.body_id, object_id, i, -1, int(enable), physicsClientId=sim.client_id)

    def _create_grasp_constraint(self, sim, object_id):
        """Creates a fixed constraint between the gripper palm and the object."""
        arm = sim.right_arm
        parent_link = arm.ee_link 
        
        parent_state = p.getLinkState(arm.body_id, parent_link, computeForwardKinematics=True, physicsClientId=sim.client_id)
        parent_pos = np.array(parent_state[4])
        parent_orn = np.array(parent_state[5])
        
        child_state = p.getBasePositionAndOrientation(object_id, physicsClientId=sim.client_id)
        child_pos = np.array(child_state[0])
        child_orn = np.array(child_state[1])
        
        inv_parent_pos, inv_parent_orn = p.invertTransform(parent_pos, parent_orn)
        rel_pos, rel_orn = p.multiplyTransforms(inv_parent_pos, inv_parent_orn, child_pos, child_orn)
        
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
        q = self._solve_ik_safe(sim, target_pos, target_orn, margin=0.002) 
        if q is not None: 
            return q
        LOGGER.warning(f"Exact Pose at z={target_pos[2]:.2f} in collision. Attempting IK with random seeds...")
        arm = sim.right_arm
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

    def _execute_path(self, sim, path, steps_per_point=5):
        """Executes a joint path with specified slowness."""
        for q in path:
            sim.set_joint_positions(q, arm="right")
            for _ in range(steps_per_point): sim.step_simulation(1)
    
    def _execute_path_interpolated(self, sim, path, total_time=2.0):
        """Linearly interpolates through path waypoints to ensure slow, smooth motion."""
        if len(path) < 2: return
        steps_per_sec = 240
        total_steps = int(total_time * steps_per_sec)
        dists = [np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) for i in range(len(path)-1)]
        total_dist = sum(dists)
        if total_dist == 0: return
        for i in range(len(path)-1):
            q1 = np.array(path[i])
            q2 = np.array(path[i+1])
            segment_dist = dists[i]
            segment_steps = int((segment_dist / total_dist) * total_steps)
            segment_steps = max(1, segment_steps)
            for t in np.linspace(0, 1, segment_steps):
                q_interp = q1 + (q2 - q1) * t
                sim.set_joint_positions(q_interp, arm="right")
                sim.step_simulation(1)
                if self._grasped_object_id is not None:
                    sim.gripper_close(arm="right", force=60.0)

    def _linear_move(self, sim, target_pos, target_orn, steps=60):
        current_pos = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[4]
        current_orn = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[5]
        for t in np.linspace(0, 1, steps):
            pos = np.array(current_pos) * (1 - t) + np.array(target_pos) * t
            
            # Fix: Interpolate orientation using Slerp to avoid aggressive snap
            orn = p.getQuaternionSlerp(current_orn, target_orn, t)
            
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
            if self._grasped_object_id is not None and body_b == self._grasped_object_id: continue
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

    def _check_segment_collision(self, sim: RobotChefSimulation, q1: np.ndarray, q2: np.ndarray, step_rad: float = 0.02, margin: float = 0.005) -> bool:
        dist = np.linalg.norm(q2 - q1)
        steps = int(math.ceil(dist / step_rad))
        
        if steps <= 1:
            return self._check_collision(sim, q2, margin=margin)
            
        for k in range(1, steps + 1):
            t = k / steps
            q_interp = q1 + (q2 - q1) * t
            if self._check_collision(sim, q_interp, margin=margin):
                return True
        return False
    
    def _restore_state(self, sim: RobotChefSimulation, q: np.ndarray):
        for i, idx in enumerate(sim.right_arm.joint_indices):
            p.resetJointState(sim.right_arm.body_id, idx, q[i], physicsClientId=sim.client_id)
            p.resetJointState(sim.right_arm.body_id, idx, q[i], targetVelocity=0.0, physicsClientId=sim.client_id)

    def _rrt_connect(self, sim: RobotChefSimulation, q_start: np.ndarray, q_goal: np.ndarray, max_iter=5000, start_safety_override=False, collision_margin=0.005) -> list:
        
        # Disable rendering during planning to prevent "ghost" flickering
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=sim.client_id)
        
        # Capture current state before any manipulation (especially useful if q_start is not exactly current state)
        q_real_start, _ = sim.get_joint_state(arm="right")

        try:
            # Check endpoints first
            if not start_safety_override and self._check_collision(sim, q_start, margin=collision_margin):
                 LOGGER.error("RRT Start Configuration is in collision.")
                 return []
            if self._check_collision(sim, q_goal, debug=True, margin=collision_margin): 
                 LOGGER.error("RRT Goal Configuration is in collision.")
                 return []
    
            J_MIN = np.array([-2.8, -1.7, -2.8, -3.0, -2.8, -0.01, -2.8])
            J_MAX = np.array([ 2.8,  1.7,  2.8, -0.06, 2.8,  3.7,  2.8])
    
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
                
                if not self._check_segment_collision(sim, nearest.q, q_new, margin=collision_margin):
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
        
        finally:
            # ALWAYS restore state and rendering, even on error
            self._restore_state(sim, q_real_start)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=sim.client_id)
            
        return path

    def _smooth_path(self, sim, path, iterations=50, collision_margin=0.005):
        # Disable rendering for smoother
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=sim.client_id)
        
        # Capture start state for restore
        q_real_start, _ = sim.get_joint_state(arm="right")

        try:
            if len(path) < 3: return path
            
            for _ in range(iterations):
                if len(path) < 3: break
                
                i, j = sorted(random.sample(range(len(path)), 2))
                if j - i <= 1: continue 
                
                q1 = path[i]
                q2 = path[j]
                
                if not self._check_segment_collision(sim, q1, q2, margin=collision_margin):
                    path = path[:i+1] + path[j:]
        finally:
            self._restore_state(sim, q_real_start)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=sim.client_id)
            
        return path

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return sim.count_particles_in_pan()

__all__ = ["UnderArmReachingTask"]