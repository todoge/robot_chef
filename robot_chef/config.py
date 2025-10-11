"""Configuration constants for the robot chef simulation."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ObjectPose:
    position: Tuple[float, float, float]
    orientation_rpy: Tuple[float, float, float]


# Workspace layout (in world coordinates, meters)
TABLE_HEIGHT = 0.75

BOWL_POSES = {
    "rice": ObjectPose(position=(0.4, 0.35, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.57)),
    "meat": ObjectPose(position=(0.5, 0.15, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.2)),
    "onion": ObjectPose(position=(0.4, -0.05, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, -0.6)),
}

PAN_POSE = ObjectPose(position=(0.2, 0.0, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))
PLATE_POSE = ObjectPose(position=(-0.2, 0.35, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))
SAUCE_BOTTLE_POSE = ObjectPose(position=(0.55, -0.25, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))

# Robot base poses relative to table center.
LEFT_ARM_BASE = ObjectPose(position=(-0.6, 0.3, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.57))
RIGHT_ARM_BASE = ObjectPose(position=(-0.6, -0.3, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.57))

# Motion tuning parameters
POUR_TILT_ANGLE = 1.2  # radians
PAN_TILT_ANGLE = 1.0
STIR_RADIUS = 0.08
STIR_HEIGHT = TABLE_HEIGHT + 0.05
STIR_SPEED = 0.6

SIMULATION_STEP = 1.0 / 240.0
