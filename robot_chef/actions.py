"""Action primitives implementing the fried rice recipe."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from . import config
from .simulation import RobotChefSimulation, interpolate_circle


@dataclass
class ActionResult:
    name: str
    success: bool
    details: str = ""


class ActionPrimitive:
    name: str

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        raise NotImplementedError


class PourBowlToPan(ActionPrimitive):
    """Pick up a bowl and pour it into the pan."""

    name = "A1_pour_bowl"

    def __init__(self, ingredient: str) -> None:
        self.ingredient = ingredient

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        bowl = sim.objects[f"bowl_{self.ingredient}"]
        pan = sim.objects["pan"]
        right_arm = sim.robots.right_arm
        sim.robots.open_gripper(right_arm)

        approach_height = config.TABLE_HEIGHT + 0.25
        pre_grasp = (bowl.pose.position[0], bowl.pose.position[1], approach_height)
        grasp_pose = (bowl.pose.position[0], bowl.pose.position[1], bowl.pose.position[2] + 0.05)

        sim.robots.move_eef_to_pose(right_arm, pre_grasp, (math.pi, 0, 0))
        sim.robots.move_eef_to_pose(right_arm, grasp_pose, (math.pi, 0, 0))
        sim.robots.close_gripper(right_arm)
        sim.step_simulation(60)

        pour_position = (pan.pose.position[0], pan.pose.position[1], pan.pose.position[2] + 0.25)
        sim.robots.move_eef_to_pose(right_arm, pour_position, (math.pi, 0, 0))
        sim.step_simulation(60)

        tilt_orientation = (math.pi - config.POUR_TILT_ANGLE, 0, 0)
        sim.robots.move_eef_to_pose(right_arm, pour_position, tilt_orientation)
        sim.step_simulation(240)

        sim.robots.move_eef_to_pose(right_arm, pre_grasp, (math.pi, 0, 0))
        sim.robots.open_gripper(right_arm)
        sim.step_simulation(30)
        return ActionResult(self.name, True, f"Poured {self.ingredient} into pan")


class StirFry(ActionPrimitive):
    """Two-arm stirring with synchronized motion."""

    name = "A2_stir_fry"

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        left_arm = sim.robots.left_arm
        right_arm = sim.robots.right_arm
        pan = sim.objects["pan"]

        # Left arm grasps the pan rim
        grasp_height = pan.pose.position[2] + 0.05
        left_pre_grasp = (pan.pose.position[0] - 0.05, pan.pose.position[1] + 0.1, grasp_height + 0.1)
        left_grasp = (pan.pose.position[0] - 0.05, pan.pose.position[1] + 0.1, grasp_height)
        sim.robots.move_eef_to_pose(left_arm, left_pre_grasp, (math.pi, 0, math.pi / 2))
        sim.robots.move_eef_to_pose(left_arm, left_grasp, (math.pi, 0, math.pi / 2))
        sim.robots.close_gripper(left_arm)
        sim.step_simulation(60)

        # Right arm grasps spatula (simplified as hovering above pan center)
        center = (pan.pose.position[0], pan.pose.position[1], config.STIR_HEIGHT)
        sim.robots.move_eef_to_pose(right_arm, (center[0], center[1], center[2] + 0.1), (math.pi, 0, 0))
        sim.robots.close_gripper(right_arm)
        sim.step_simulation(30)

        angle = 0.0
        for _ in range(6):
            for _ in range(60):
                point = interpolate_circle(center, config.STIR_RADIUS, angle)
                sim.robots.move_eef_to_pose(right_arm, point, (math.pi, 0, 0), num_steps=1)
                angle += config.STIR_SPEED * config.SIMULATION_STEP
            sim.step_simulation(1)

        sim.robots.open_gripper(right_arm)
        sim.step_simulation(30)
        sim.robots.open_gripper(left_arm)
        sim.step_simulation(30)
        return ActionResult(self.name, True, "Completed stir frying")


class OpenBottle(ActionPrimitive):
    """Two-arm bottle opening."""

    name = "A3_open_bottle"

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        left_arm = sim.robots.left_arm
        right_arm = sim.robots.right_arm
        bottle = sim.objects["sauce_bottle"]

        top_height = bottle.pose.position[2] + 0.18 / 2.0
        body_grasp = (bottle.pose.position[0], bottle.pose.position[1], top_height - 0.05)
        cap_grasp = (bottle.pose.position[0], bottle.pose.position[1], top_height + 0.02)

        sim.robots.move_eef_to_pose(left_arm, (body_grasp[0], body_grasp[1] + 0.05, body_grasp[2] + 0.1), (math.pi, 0, 0))
        sim.robots.move_eef_to_pose(left_arm, body_grasp, (math.pi, 0, 0))
        sim.robots.close_gripper(left_arm)
        sim.step_simulation(60)

        sim.robots.move_eef_to_pose(right_arm, (cap_grasp[0], cap_grasp[1] - 0.05, cap_grasp[2] + 0.1), (math.pi, 0, 0))
        sim.robots.move_eef_to_pose(right_arm, cap_grasp, (math.pi, 0, 0))
        sim.robots.close_gripper(right_arm)
        sim.step_simulation(60)

        # Simulate twisting motion by rotating around z-axis
        for step in range(120):
            rotation = (math.pi, 0, step * 0.01)
            sim.robots.move_eef_to_pose(right_arm, cap_grasp, rotation, num_steps=1)
            sim.step_simulation(1)

        sim.robots.open_gripper(right_arm)
        sim.step_simulation(30)
        sim.robots.open_gripper(left_arm)
        sim.step_simulation(30)
        return ActionResult(self.name, True, "Opened sauce bottle")


class PourBottleToPan(ActionPrimitive):
    name = "A4_pour_bottle"

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        right_arm = sim.robots.right_arm
        bottle = sim.objects["sauce_bottle"]
        pan = sim.objects["pan"]

        grasp_height = bottle.pose.position[2] + 0.15
        approach = (bottle.pose.position[0], bottle.pose.position[1], grasp_height + 0.1)
        grasp = (bottle.pose.position[0], bottle.pose.position[1], grasp_height)

        sim.robots.move_eef_to_pose(right_arm, approach, (math.pi, 0, 0))
        sim.robots.move_eef_to_pose(right_arm, grasp, (math.pi, 0, 0))
        sim.robots.close_gripper(right_arm)
        sim.step_simulation(60)

        pour_position = (pan.pose.position[0], pan.pose.position[1] - 0.05, pan.pose.position[2] + 0.3)
        sim.robots.move_eef_to_pose(right_arm, pour_position, (math.pi, 0, 0))
        sim.step_simulation(30)

        tilt_orientation = (math.pi - config.POUR_TILT_ANGLE, 0, 0)
        sim.robots.move_eef_to_pose(right_arm, pour_position, tilt_orientation)
        sim.step_simulation(240)

        sim.robots.move_eef_to_pose(right_arm, approach, (math.pi, 0, 0))
        sim.robots.open_gripper(right_arm)
        sim.step_simulation(30)
        return ActionResult(self.name, True, "Poured sauce into pan")


class PanToPlate(ActionPrimitive):
    name = "A5_pan_to_plate"

    def run(self, sim: RobotChefSimulation) -> ActionResult:
        left_arm = sim.robots.left_arm
        pan = sim.objects["pan"]
        plate = sim.objects["plate"]

        grasp = (pan.pose.position[0] - 0.05, pan.pose.position[1], pan.pose.position[2] + 0.05)
        sim.robots.move_eef_to_pose(left_arm, (grasp[0], grasp[1], grasp[2] + 0.1), (math.pi, 0, math.pi / 2))
        sim.robots.move_eef_to_pose(left_arm, grasp, (math.pi, 0, math.pi / 2))
        sim.robots.close_gripper(left_arm)
        sim.step_simulation(60)

        pour_position = (plate.pose.position[0], plate.pose.position[1], plate.pose.position[2] + 0.25)
        sim.robots.move_eef_to_pose(left_arm, pour_position, (math.pi, 0, math.pi / 2))
        sim.step_simulation(30)

        tilt_orientation = (math.pi - config.PAN_TILT_ANGLE, 0, math.pi / 2)
        sim.robots.move_eef_to_pose(left_arm, pour_position, tilt_orientation)
        sim.step_simulation(240)

        sim.robots.open_gripper(left_arm)
        sim.step_simulation(30)
        return ActionResult(self.name, True, "Transferred contents to plate")


RECIPE_ACTIONS: Sequence[ActionPrimitive] = [
    PourBowlToPan("rice"),
    PourBowlToPan("meat"),
    PourBowlToPan("onion"),
    StirFry(),
    OpenBottle(),
    PourBottleToPan(),
    PanToPlate(),
]
