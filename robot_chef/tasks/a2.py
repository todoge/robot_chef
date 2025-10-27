import logging
import math
from typing import Dict, Optional

import numpy as np
import pybullet as p

from ..config import Pose6D, MainConfig
from ..sim import Simulator

LOGGER = logging.getLogger(__name__)


class A2:
    """Bimanual stirring task using spatula and pan handle"""
    def __init__(self):
        self.sim: Simulator | None = None
        self.right_arm_state = None
        self.left_arm_state = None
        self.spatula_id: int | None = None

    def setup(self, sim: Simulator, cfg: MainConfig):
        self.sim = sim
        self.cfg = cfg
        self.sim.setup()
        self.sim.open_grippers(arm="right")
        self.sim.open_grippers(arm="left")

    def plan(self) -> bool:
        return True

    def execute(self):

        # define actionable objects
        spatula_id = self.sim.objects["spatula"]["body_id"]
        pan_id = self.sim.objects["pan"]["body_id"]
        arm = "right"

        # move arm to spatula handle
        spatula_pos, spatula_orn = p.getBasePositionAndOrientation(spatula_id)
        grasp_pos = np.array(spatula_pos) + np.array([0, 0, 0.05]) # define grasp slightly above handle
        joint_pos = self._inverse_kinematics(arm, grasp_pos, spatula_orn)
        self.sim.set_joint_positions(joint_pos, arm=arm)
        self.sim.step_simulation(steps=240)

        # grasp
        self.sim.gripper_close(arm="right")
        self.sim.step_simulation(steps=120)

        


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
