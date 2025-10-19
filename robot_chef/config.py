"""Configuration utilities for the robot chef simulation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Union

import yaml


# -----------------------------
# Basic pose containers
# -----------------------------

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


# -----------------------------
# Task-level parameters
# -----------------------------

@dataclass(frozen=True)
class PourTaskTolerances:
    place_back_xy: float
    place_back_yaw_deg: float


@dataclass(frozen=True)
class SceneConfig:
    world_yaw_deg: float = 0.0


# -----------------------------
# Camera configuration
# -----------------------------

@dataclass(frozen=True)
class CameraNoiseConfig:
    # Canonical field name
    depth_std: float = 0.0
    drop_prob: float = 0.0

    # Backward/forward compatibility alias: allow .depth access
    @property
    def depth(self) -> float:
        return self.depth_std

    @depth.setter  # type: ignore[attr-defined]
    def depth(self, value: float) -> None:
        # dataclasses with frozen=True don't allow setting; emulate immutability by raising
        raise AttributeError("CameraNoiseConfig is frozen; set 'depth_std' via config parsing instead.")


@dataclass(frozen=True)
class CameraViewConfig:
    xyz: Tuple[float, float, float]
    # Store degrees; accept both 'rpy_deg' and 'rpy' (deg) from YAML
    rpy_deg: Tuple[float, float, float]
    fov_deg: float
    resolution: Tuple[int, int] = (640, 480)

    @property
    def rpy_rad(self) -> Tuple[float, float, float]:
        factor = 3.141592653589793 / 180.0
        return tuple(float(v) * factor for v in self.rpy_deg)


@dataclass(frozen=True)
class CameraConfig:
    active_view: str
    views: Dict[str, CameraViewConfig] = field(default_factory=dict)
    noise: CameraNoiseConfig = field(default_factory=CameraNoiseConfig)
    near: float = 0.05
    far: float = 5.0

    def get_view(self, name: str) -> CameraViewConfig:
        if name not in self.views:
            available = ", ".join(sorted(self.views))
            raise KeyError(f"Camera view '{name}' not defined (available: {available})")
        return self.views[name]

    @property
    def active(self) -> CameraViewConfig:
        return self.get_view(self.active_view)


# -----------------------------
# Perception configuration
# -----------------------------

@dataclass(frozen=True)
class PerceptionConfig:
    enable: bool = False
    bowl_roi_margin_m: float = 0.08
    rim_sample_count: int = 16
    grasp_clearance_m: float = 0.02
    min_grasp_quality: float = 0.5
    # Optional: expose rim thickness for forced-closure tuning
    rim_thickness_m: float = 0.006


# -----------------------------
# Helpers for parsing
# -----------------------------

def _tuple3(values: Iterable[float]) -> Tuple[float, float, float]:
    seq = list(values)
    if len(seq) != 3:
        raise ValueError(f"Expected three values, got {values!r}")
    return tuple(float(v) for v in seq)


def _coerce_resolution(val) -> Tuple[int, int]:
    if isinstance(val, (list, tuple)) and len(val) == 2:
        return (int(val[0]), int(val[1]))
    raise ValueError(f"Camera view requires resolution as [width, height], got {val!r}")


def _default_camera_views() -> Dict[str, CameraViewConfig]:
    return {
        "top": CameraViewConfig(
            xyz=(0.0, 0.0, 1.6),
            rpy_deg=(-90.0, 0.0, 0.0),
            fov_deg=70.0,
            resolution=(640, 480),
        ),
        "front": CameraViewConfig(
            xyz=(1.0, 0.0, 1.2),
            rpy_deg=(-30.0, 0.0, 180.0),
            fov_deg=70.0,
            resolution=(640, 480),
        ),
        "oblique": CameraViewConfig(
            xyz=(0.8, 0.4, 1.3),
            rpy_deg=(-40.0, 0.0, 160.0),
            fov_deg=70.0,
            resolution=(640, 480),
        ),
    }


def _default_camera_config() -> CameraConfig:
    return CameraConfig(
        active_view="oblique",
        views=_default_camera_views(),
        noise=CameraNoiseConfig(),
        near=0.05,
        far=5.0,
    )


# -----------------------------
# Root config for the pour task
# -----------------------------

@dataclass(frozen=True)
class PourTaskConfig:
    bowl_pose: Pose6D
    pan_pose: Pose6D
    pan_pour_pose: Pose6D
    tilt_angle_deg: float
    hold_sec: float
    particles: int
    tolerances: PourTaskTolerances
    scene: SceneConfig = field(default_factory=SceneConfig)
    camera: CameraConfig = field(default_factory=_default_camera_config)
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    seed: int = 7

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PourTaskConfig":
        def pose_from_list(values):
            if not isinstance(values, (list, tuple)) or len(values) != 6:
                raise ValueError(f"Expected pose list of length 6, got {values!r}")
            return Pose6D(*map(float, values))

        tolerances = data.get("tolerances", {}) or {}
        tol = PourTaskTolerances(
            place_back_xy=float(tolerances.get("place_back_xy", 0.03)),
            place_back_yaw_deg=float(tolerances.get("place_back_yaw_deg", 8.0)),
        )

        # Pan renamed from wok — support both for compatibility
        pan_pose_data = data.get("pan_pose") or data.get("wok_pose")
        pan_pour_pose_data = data.get("pan_pour_pose") or data.get("wok_pour_pose")
        if pan_pose_data is None or pan_pour_pose_data is None:
            raise ValueError("Pan pose configuration missing (expected pan_pose and pan_pour_pose).")

        scene_data = data.get("scene") or {}
        scene = SceneConfig(world_yaw_deg=float(scene_data.get("world_yaw_deg", 0.0)))

        camera_data = data.get("camera") or {}
        view_data = camera_data.get("views") or {}

        if view_data:
            views: Dict[str, CameraViewConfig] = {}
            for name, cfg in view_data.items():
                if not isinstance(cfg, dict):
                    raise ValueError(f"Camera view '{name}' must be a mapping, got {cfg!r}")
                xyz = _tuple3(cfg.get("xyz", (0.0, 0.0, 1.0)))
                # Accept 'rpy' (deg) or 'rpy_deg'
                rpy = _tuple3(cfg.get("rpy_deg", cfg.get("rpy", (-90.0, 0.0, 0.0))))
                fov = float(cfg.get("fov_deg", cfg.get("fov", 70.0)))
                resolution = _coerce_resolution(cfg.get("resolution", (640, 480)))
                views[name] = CameraViewConfig(
                    xyz=xyz,
                    rpy_deg=rpy,
                    fov_deg=fov,
                    resolution=resolution,
                )
        else:
            views = _default_camera_views()

        noise_data = camera_data.get("noise") or {}
        # Accept 'depth' or 'depth_std'
        depth_std = float(noise_data.get("depth_std", noise_data.get("depth", 0.0)))
        noise = CameraNoiseConfig(
            depth_std=depth_std,
            drop_prob=float(noise_data.get("drop_prob", 0.0)),
        )

        camera_cfg = CameraConfig(
            active_view=str(camera_data.get("active_view", "oblique")),
            views=views,
            noise=noise,
            near=float(camera_data.get("near", 0.05)),
            far=float(camera_data.get("far", 5.0)),
        )

        perception_data = data.get("perception") or {}
        perception = PerceptionConfig(
            enable=bool(perception_data.get("enable", False)),
            bowl_roi_margin_m=float(perception_data.get("bowl_roi_margin_m", 0.08)),
            rim_sample_count=int(perception_data.get("rim_sample_count", 16)),
            grasp_clearance_m=float(perception_data.get("grasp_clearance_m", 0.02)),
            min_grasp_quality=float(perception_data.get("min_grasp_quality", 0.5)),
            rim_thickness_m=float(perception_data.get("rim_thickness_m", 0.006)),
        )

        return cls(
            bowl_pose=pose_from_list(data.get("bowl_pose")),
            pan_pose=pose_from_list(pan_pose_data),
            pan_pour_pose=pose_from_list(pan_pour_pose_data),
            tilt_angle_deg=float(data.get("tilt_angle_deg", 140.0)),
            hold_sec=float(data.get("hold_sec", 3.0)),
            particles=int(data.get("particles", 80)),
            tolerances=tol,
            scene=scene,
            camera=camera_cfg,
            perception=perception,
            seed=int(data.get("seed", 7)),
        )


def load_pour_task_config(path: Union[str, Path], overrides: Optional[Dict[str, str]] = None) -> PourTaskConfig:
    """Load the pour demo configuration from a YAML file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at root of {path}")
    if overrides:
        _apply_overrides(data, overrides)
    return PourTaskConfig.from_dict(data)


# -----------------------------
# Additional constants (optional)
# -----------------------------

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

POUR_TILT_ANGLE = 1.2  # radians
PAN_TILT_ANGLE = 1.0
STIR_RADIUS = 0.08
STIR_HEIGHT = TABLE_HEIGHT + 0.05
STIR_SPEED = 0.6

SIMULATION_STEP = 1.0 / 240.0


# -----------------------------
# Overrides helpers
# -----------------------------

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
