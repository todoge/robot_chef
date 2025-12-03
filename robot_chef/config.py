"""Configuration utilities for the robot chef simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import yaml


@dataclass(frozen=True)
class ObjectPose:
    position: Tuple[float, float, float]
    orientation_rpy: Tuple[float, float, float]


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float

    @property
    def position(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    @property
    def orientation_rpy(self) -> Tuple[float, float, float]:
        return (self.roll, self.pitch, self.yaw)


@dataclass(frozen=True)
class TolerancesConfig:
    place_back_xy: float
    place_back_yaw_deg: float


@dataclass(frozen=True)
class SceneConfig:
    world_yaw_deg: float = 0.0

def _tuple3(values: Iterable[float]) -> Tuple[float, float, float]:
    seq = list(values)
    if len(seq) != 3:
        raise ValueError(f"Expected three values, got {values!r}")
    return tuple(float(v) for v in seq)

@dataclass(frozen=True)
class MainConfig:
    bowl_pose: Pose6D
    pan_pose: Pose6D
    spatula_stir_pose: Pose6D
    spatula_pose: Pose6D
    tilt_angle_deg: float
    hold_sec: float
    rice_particles: int
    egg_particles: int
    tolerances: TolerancesConfig
    scene: SceneConfig = field(default_factory=SceneConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MainConfig":
        def pose_from_list(values):
            if not isinstance(values, (list, tuple)) or len(values) != 6:
                raise ValueError(f"Expected pose list of length 6, got {values!r}")
            return Pose6D(*map(float, values))
        return cls(
            bowl_pose=pose_from_list(data.get("bowl_pose")),
            pan_pose=pose_from_list(data.get("pan_pose")),
            spatula_stir_pose=pose_from_list(data.get("spatula_stir_pose")),
            spatula_pose=pose_from_list(data.get("spatula_pose")),
            tilt_angle_deg=float(data.get("tilt_angle_deg")),
            hold_sec=float(data.get("hold_sec")),
            rice_particles=int(data.get("rice_particles")),
            egg_particles=int(data.get("egg_particles")),
            tolerances=TolerancesConfig(
                place_back_xy=float(data.get("tolerances").get("place_back_xy")),
                place_back_yaw_deg=float(data.get("tolerances").get("place_back_yaw_deg")),
            ),
            scene=SceneConfig(
                world_yaw_deg=float(data.get("scene").get("world_yaw_deg")),
            ),
            seed=int(data.get("seed")),
        )


def load_main_config(path: Union[str, Path], overrides: Optional[Dict[str, str]] = None) -> MainConfig:
    """Load the stir demo configuration from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {path}")
    if overrides:
        _apply_overrides(data, overrides)
    return MainConfig.from_dict(data)


TABLE_HEIGHT = 0.75

BOWL_POSES = {
    "rice": ObjectPose(position=(0.4, 0.35, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.57)),
    "meat": ObjectPose(position=(0.5, 0.15, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.2)),
    "onion": ObjectPose(position=(0.4, -0.05, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, -0.6)),
}

PAN_POSE = ObjectPose(position=(0.2, 0.0, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))
PLATE_POSE = ObjectPose(position=(-0.2, 0.35, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))
SAUCE_BOTTLE_POSE = ObjectPose(position=(0.55, -0.25, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 0.0))

LEFT_ARM_BASE = ObjectPose(position=(-0.45, 0.28, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, 1.45))
RIGHT_ARM_BASE = ObjectPose(position=(-0.45, -0.28, TABLE_HEIGHT), orientation_rpy=(0.0, 0.0, -1.45))

SPATULA_TILT_ANGLE = 1.2  # radians
STIR_RADIUS = 0.08
STIR_HEIGHT = TABLE_HEIGHT + 0.05
STIR_SPEED = 0.6

SIMULATION_STEP = 1.0 / 240.0

def _apply_overrides(root: Dict[str, object], overrides: Dict[str, str]) -> None:
    for dotted_key, raw_value in overrides.items():
        keys = dotted_key.split(".")
        target = root
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                target[key] = {}
            target = target[key]  # type: ignore[assignment]
        target[keys[-1]] = _parse_override_value(raw_value)


def _parse_override_value(raw: str) -> object:
    lowered = raw.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        if "." in raw or "e" in lowered:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw
