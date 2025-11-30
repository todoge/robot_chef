import json
import logging
import math
from typing import Dict, Optional

import numpy as np
import pybullet as p

from robot_chef.camera import Camera, CameraNoiseModel
from robot_chef.config import MainConfig, Pose6D
from robot_chef.simulator import StirSimulator

LOGGER = logging.getLogger(__name__)


class TaskStir:
    """Bimanual stirring task using spatula and pan handle"""
    def setup(self, sim: StirSimulator, cfg: MainConfig) -> None:
        self.sim = sim
        self.cfg = cfg
        self.sim.open_grippers(arm="right")
        self.sim.open_grippers(arm="left")
        cam_view_cfg = cfg.camera.get_view()
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

    def plan(self) -> bool:
        return True

    def execute(self) -> None:

        # define actionable objects
        spatula_id = self.sim.objects["spatula"]["body_id"]
        pan_id = self.sim.objects["pan"]["body_id"]
        arm = "right"

        # move arm to spatula handle
        spatula_pos, spatula_orn = p.getBasePositionAndOrientation(spatula_id)
        grasp_pos = np.array(spatula_pos) + np.array([0, 0, 0.2]) # define grasp slightly above handle
        _, _, spatula_yaw = p.getEulerFromQuaternion(spatula_orn)
        ee_down_quat = p.getQuaternionFromEuler([math.pi, 0, spatula_yaw])
        joint_pos = self._inverse_kinematics(arm, grasp_pos, ee_down_quat)
        self.sim.set_joint_positions(joint_pos, arm=arm)
        self.sim.step_simulation(steps=240)

        # grasp
        self.sim.gripper_close(arm="right")
        self.sim.step_simulation(steps=120)
        gripper_link = self.sim.right_arm.ee_link
        c_id = p.createConstraint(
            parentBodyUniqueId=self.sim.right_arm.body_id,
            parentLinkIndex=gripper_link,
            childBodyUniqueId=spatula_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=[0, 0, 0],
            physicsClientId=self.sim.client_id,
        )

        # # stirring motion
        # pan_pos = p.getBasePositionAndOrientation(pan_id)[0]
        # radius = 0.07
        # height = pan_pos[2] + 0.03
        # for theta in np.linspace(0, 2 * math.pi, 50):
        #     target_pos = [
        #         pan_pos[0] + radius * math.cos(theta),
        #         pan_pos[1] + radius * math.sin(theta),
        #         height,
        #     ]
        #     joint_pos = p.calculateInverseKinematics(
        #         self.sim.right_arm.body_id,
        #         gripper_link,
        #         target_pos,
        #         targetOrientation=spatula_orn,
        #         physicsClientId=self.sim.client_id,
        #     )
        #     self.sim.set_joint_positions(joint_pos, arm="right")
        #     self.sim.step_simulation(steps=10)


    def metrics(self, print: bool = False):
        metrics = {
            "retain_ratio": 0.0,
            "beads_inside": 0.0,
            "bead_original": 0.0,
        }
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
