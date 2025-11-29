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

    def setup(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> None:
        self.sim = sim
        self.cfg = cfg
        
        # --- 0. SAFETY FIRST: Move Right Arm to Safe Home ---
        # Moving explicitly to a safe "Retracted Right" pose to avoid startup collisions
        # Joints: [pan, lift, pan, lift, twist, bend, twist]
        safe_right_joints = [-1.5, -1.0, 0.0, -2.2, 0.0, 1.8, 0.785]
        sim.set_joint_positions(safe_right_joints, arm="right")
        sim.step_simulation(20)

        # 1. Spawn Specula
        specula_pose = Pose6D(0.0, 0.0, 0.5, 0, 0, 0) 
        self.specula_id, _ = create_specula(sim.client_id, specula_pose)
        
        # 2. Setup Left Arm (The Obstacle)
        left_arm = sim.left_arm
        
        # Move Left Arm to a "Blocking" configuration
        # J1 (1.3): Increased shoulder lean to flatten the arm (lower elbow).
        # J3 (-1.4): Bent elbow to bring hand closer to table level.
        blocking_joints = [-2.2, 1.3, 0.0, -1.4, 0.0, 1.6, 0.785]
        
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

        # Wait for specula to render
        LOGGER.info("Waiting for specula to render...")
        for _ in range(20):
            sim.step_simulation(1)
            time.sleep(0.05)
            
        # Ensure particles are there if config missed them (though we updated yaml)
        if not sim.particles:
             LOGGER.info("Spawning particles manually...")
             sim.spawn_rice_particles(80, seed=cfg.seed)
        
        LOGGER.info("Scenario Setup: Left Arm is holding specula in blocking pose.")

    def plan(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        LOGGER.info("Planning phase: The environment is static. The Left Arm is the obstacle.")
        return True

    def execute(self, sim: RobotChefSimulation, cfg: PourTaskConfig) -> bool:
        """
        Execute RRT Reach -> Grasp -> Lift -> Move to Pan -> Pour.
        """
        LOGGER.info("Execution: Calculating RRT Path...")
        
        bowl = sim.objects.get("bowl")
        pan = sim.objects.get("pan")
        if not bowl or not pan: return False

        # --- 1. RRT to Pre-Grasp (Hover) ---
        target_pose = bowl["pose"]
        # Hover slightly above bowl (0.15m up)
        pre_grasp_pos = [target_pose.x, target_pose.y, target_pose.z + 0.15] 
        grasp_orn = p.getQuaternionFromEuler([3.14, 0, 0]) # Face down

        q_start, _ = sim.get_joint_state(arm="right")
        
        # Calculate Goal Config (Pre-Grasp)
        full_ik = p.calculateInverseKinematics(
            sim.right_arm.body_id, sim.right_arm.ee_link, pre_grasp_pos, grasp_orn,
            maxNumIterations=100, residualThreshold=1e-4, physicsClientId=sim.client_id
        )
        q_pre_grasp = np.array(full_ik[:7])

        # Run RRT
        LOGGER.info("Planning Path to Pre-Grasp...")
        path = self._rrt_connect(sim, q_start, q_pre_grasp)
        
        if not path:
            LOGGER.error("RRT Failed to find a path!")
            return False

        # --- Advanced Technique: Path Smoothing ---
        LOGGER.info(f"Raw Path found ({len(path)} steps). Smoothing...")
        path = self._smooth_path(sim, path)

        # Execute RRT Path
        LOGGER.info(f"Smoothed Path ({len(path)} steps). Moving to hover...")
        self._execute_path(sim, path)
        sim.gripper_open(arm="right")
        time.sleep(0.5)

        # --- 2. Cartesian: Down to Grasp ---
        LOGGER.info("Descending to grasp...")
        # Target Z: Bowl rim is ~0.05m high. We want gripper fingers around it.
        grasp_pos = [target_pose.x, target_pose.y, target_pose.z + 0.04]
        self._linear_move(sim, grasp_pos, grasp_orn, steps=40)
        
        # --- 3. Close Gripper ---
        sim.gripper_close(arm="right", force=100)
        for _ in range(50): sim.step_simulation(1)

        # --- 4. Cartesian: Lift Bowl ---
        LOGGER.info("Lifting bowl...")
        # Lift high enough to clear pan walls but assume obstacle is avoided by x/y separation
        lift_pos = [target_pose.x, target_pose.y, target_pose.z + 0.25]
        self._linear_move(sim, lift_pos, grasp_orn, steps=40)

        # --- 5. Cartesian: Move Above Pan ---
        LOGGER.info("Moving to Pan...")
        pan_pose = pan["pose"]
        pour_pos = [pan_pose.x, pan_pose.y, pan_pose.z + 0.30]
        self._linear_move(sim, pour_pos, grasp_orn, steps=60)
        
        # --- 6. Pour (Tilt) ---
        LOGGER.info("Pouring...")
        # Tilt 120 degrees around Y axis relative to gripper
        # Simple approximation: rotate target orientation
        tilt_orn = p.getQuaternionFromEuler([3.14, -2.1, 0.0]) 
        self._linear_move(sim, pour_pos, tilt_orn, steps=80)
        
        # Shake / Wait for particles
        for _ in range(120): sim.step_simulation(1)
        
        # Untilt
        LOGGER.info("Untilting...")
        self._linear_move(sim, pour_pos, grasp_orn, steps=50)

        # Metrics
        metrics = sim.count_particles_in_pan()
        LOGGER.info(f"Task Complete. Metrics: {metrics}")
        
        return True
    
    # ------------------------------------------------------------------ #
    # Helpers

    def _execute_path(self, sim, path):
        for q in path:
            sim.set_joint_positions(q, arm="right")
            for _ in range(5): sim.step_simulation(1)
            # time.sleep(0.02) # Optional: speed up

    def _linear_move(self, sim, target_pos, target_orn, steps=60):
        """Linearly interpolate EEF pose (Cartesian control)."""
        current_pos = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[4]
        current_orn = p.getLinkState(sim.right_arm.body_id, sim.right_arm.ee_link, physicsClientId=sim.client_id)[5]
        
        # Very simple lerp
        for t in np.linspace(0, 1, steps):
            pos = np.array(current_pos) * (1 - t) + np.array(target_pos) * t
            # Slerp for orientation is better, but simple lerp/nlerp or just fixed target is often okay for small moves.
            # Here we just switch to target orn for simplicity or keep fixed if same
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

    def _check_collision(self, sim: RobotChefSimulation, q: np.ndarray) -> bool:
        """
        Checks collision for the Right Arm at configuration q with a SAFETY MARGIN.
        CRITICAL FIX: Uses p.resetJointState to INSTANTLY teleport collision shapes.
        sim.set_joint_positions does not move shapes until stepSimulation is called.
        """
        # Teleport right arm to test configuration
        for i, idx in enumerate(sim.right_arm.joint_indices):
            p.resetJointState(sim.right_arm.body_id, idx, q[i], physicsClientId=sim.client_id)
        
        p.performCollisionDetection(physicsClientId=sim.client_id)
        
        # 1. Standard Contact Check (Actual Touching)
        contacts = p.getContactPoints(bodyA=sim.right_arm.body_id, physicsClientId=sim.client_id)
        
        # Filter contacts: We must ignore the mount (pedestal) the arm is standing on.
        right_mount_id = sim.objects.get("right_mount", {}).get("body_id")
        
        for c in contacts:
            body_b = c[2]
            # Ignore self (handled by URDF usually) and mount
            if body_b == sim.right_arm.body_id: continue
            if right_mount_id is not None and body_b == right_mount_id: continue
            
            # If we hit anything else (Left Arm, Table, Bowl, Specula), it's a collision
            return True

        # 2. Safety Margin Check (Proximity)
        # We enforce a 2.5cm buffer zone around the obstacles.
        margin = 0.025
        obstacles = [sim.left_arm.body_id, self.specula_id]
        
        for obs_id in obstacles:
            closest = p.getClosestPoints(
                bodyA=sim.right_arm.body_id, 
                bodyB=obs_id, 
                distance=margin, 
                physicsClientId=sim.client_id
            )
            if len(closest) > 0:
                return True
                
        return False

    def _check_segment_collision(self, sim: RobotChefSimulation, q1: np.ndarray, q2: np.ndarray, step_rad: float = 0.02) -> bool:
        """
        Checks collision densely along the linear path segment between q1 and q2.
        Interpolates every 'step_rad' radians to prevent tunneling through thin obstacles.
        """
        dist = np.linalg.norm(q2 - q1)
        steps = int(math.ceil(dist / step_rad))
        
        # Always check at least the endpoint if close
        if steps <= 1:
            return self._check_collision(sim, q2)
            
        for k in range(1, steps + 1):
            t = k / steps
            q_interp = q1 + (q2 - q1) * t
            if self._check_collision(sim, q_interp):
                return True
        return False
    
    def _restore_state(self, sim: RobotChefSimulation, q: np.ndarray):
        """Restores the robot physics state after planning teleportation."""
        for i, idx in enumerate(sim.right_arm.joint_indices):
            p.resetJointState(sim.right_arm.body_id, idx, q[i], physicsClientId=sim.client_id)

    def _rrt_connect(self, sim: RobotChefSimulation, q_start: np.ndarray, q_goal: np.ndarray, max_iter=2000) -> list:
        """Bi-Directional RRT with Strict Segment Checking."""
        
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
        
        step_size = 0.05 

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
            
            # UPDATED: Use Segment Check instead of Point Check
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

        # Restore simulation state cleanly
        self._restore_state(sim, q_real_start)
        return path

    def _smooth_path(self, sim, path, iterations=50):
        """
        Shortcut Smoothing with Segment Checking.
        """
        if len(path) < 3: return path
        
        q_real_start, _ = sim.get_joint_state(arm="right")
        
        for _ in range(iterations):
            if len(path) < 3: break
            
            # Pick two random indices
            i, j = sorted(random.sample(range(len(path)), 2))
            if j - i <= 1: continue # Adjacent points
            
            q1 = path[i]
            q2 = path[j]
            
            # UPDATED: Use helper to check the entire shortcut segment
            if not self._check_segment_collision(sim, q1, q2):
                # Shortcut successful! Remove intermediate points
                path = path[:i+1] + path[j:]
                
        # Restore simulation state cleanly
        self._restore_state(sim, q_real_start)
        return path

    def metrics(self, sim: RobotChefSimulation) -> Dict[str, float]:
        return sim.count_particles_in_pan()

__all__ = ["UnderArmReachingTask"]