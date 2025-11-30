"""Configuration loading utilities for the Robot Chef pour task."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import yaml


Number = Union[int, float]


@dataclass(frozen=True)
class Pose6D:
    """Simple pose representation storing xyz position and rpy orientation."""

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

    def as_list(self) -> Tuple[float, float, float, float, float, float]:
        return (self.x, self.y, self.z, self.roll, self.pitch, self.yaw)

    @classmethod
    def from_iterable(cls, values: Iterable[Number]) -> "Pose6D":
        vals = list(values)
        if len(vals) != 6:
            raise ValueError(f"Pose6D requires 6 values, got {len(vals)}")
        return cls(
            x=float(vals[0]),
            y=float(vals[1]),
            z=float(vals[2]),
            roll=float(vals[3]),
            pitch=float(vals[4]),
            yaw=float(vals[5]),
        )


@dataclass(frozen=True)
class CameraView:
    xyz: Tuple[float, float, float]
    rpy_deg: Tuple[float, float, float]
    fov_deg: float
    resolution: Tuple[int, int]


@dataclass(frozen=True)
class CameraNoise:
    depth_std: float = 0.0
    drop_prob: float = 0.0


@dataclass(frozen=True)
class CameraConfig:
    active_view: str
    near: float
    far: float
    views: Dict[str, CameraView]
    noise: CameraNoise

    def get_active_view(self) -> CameraView:
        if self.active_view not in self.views:
            raise KeyError(f"Camera view '{self.active_view}' not defined in config")
        return self.views[self.active_view]


@dataclass(frozen=True)
class PerceptionConfig:
    enable: bool
    bowl_roi_margin_m: float
    rim_sample_count: int
    grasp_clearance_m: float
    rim_thickness_m: float


@dataclass(frozen=True)
class TolerancesConfig:
    place_back_xy: float
    place_back_yaw_deg: float


@dataclass(frozen=True)
class SceneConfig:
    world_yaw_deg: float


@dataclass(frozen=True)
class TaskNoiseConfig:
    depth_std: float = 0.0
    drop_prob: float = 0.0


@dataclass(frozen=True)
class TaskRuntimeConfig:
    pan_pour_pose: Pose6D
    tilt_angle_deg: float
    hold_sec: float
    noise: TaskNoiseConfig


@dataclass(frozen=True)
class PourTaskConfig:
    bowl_pose: Pose6D
    bowl_pose_1: Pose6D
    pan_pose: Pose6D
    pan_pour_pose: Pose6D
    tilt_angle_deg: float
    hold_sec: float
    particles: int
    tolerances: TolerancesConfig
    scene: SceneConfig
    camera: CameraConfig
    perception: PerceptionConfig
    task: TaskRuntimeConfig
    seed: int


def _as_pose(mapping: Mapping[str, Sequence[Number]], key: str) -> Pose6D:
    if key not in mapping:
        raise KeyError(f"Missing required pose '{key}' in config file")
    value = mapping[key]
    return Pose6D.from_iterable(value)


def _load_camera(config_dict: MutableMapping[str, object]) -> CameraConfig:
    camera_dict = dict(config_dict.get("camera", {}))
    if "views" not in camera_dict or not isinstance(camera_dict["views"], Mapping):
        raise KeyError("camera.views missing or invalid")
    views: Dict[str, CameraView] = {}
    for name, data in camera_dict["views"].items():
        xyz = tuple(float(v) for v in data.get("xyz", (0.0, 0.0, 1.0)))
        rpy_deg = tuple(float(v) for v in data.get("rpy", (0.0, 0.0, 0.0)))
        fov_deg = float(data.get("fov_deg", 60.0))
        resolution = data.get("resolution", (640, 480))
        res_tuple = (int(resolution[0]), int(resolution[1]))
        views[name] = CameraView(xyz=xyz, rpy_deg=rpy_deg, fov_deg=fov_deg, resolution=res_tuple)
    noise_dict = camera_dict.get("noise", {}) or {}
    camera_noise = CameraNoise(
        depth_std=float(noise_dict.get("depth_std", 0.0)),
        drop_prob=float(noise_dict.get("drop_prob", 0.0)),
    )
    active_view = str(camera_dict.get("active_view", "default"))
    near = float(camera_dict.get("near", 0.02))
    far = float(camera_dict.get("far", 3.0))
    return CameraConfig(
        active_view=active_view,
        near=near,
        far=far,
        views=views,
        noise=camera_noise,
    )


def _load_perception(config_dict: MutableMapping[str, object]) -> PerceptionConfig:
    perception_dict = dict(config_dict.get("perception", {}))
    return PerceptionConfig(
        enable=bool(perception_dict.get("enable", True)),
        bowl_roi_margin_m=float(perception_dict.get("bowl_roi_margin_m", 0.12)),
        rim_sample_count=int(perception_dict.get("rim_sample_count", 24)),
        grasp_clearance_m=float(perception_dict.get("grasp_clearance_m", 0.02)),
        rim_thickness_m=float(perception_dict.get("rim_thickness_m", 0.006)),
    )


def _load_tolerances(config_dict: MutableMapping[str, object]) -> TolerancesConfig:
    tol_dict = dict(config_dict.get("tolerances", {}))
    return TolerancesConfig(
        place_back_xy=float(tol_dict.get("place_back_xy", 0.03)),
        place_back_yaw_deg=float(tol_dict.get("place_back_yaw_deg", 8.0)),
    )


def _load_scene(config_dict: MutableMapping[str, object]) -> SceneConfig:
    scene_dict = dict(config_dict.get("scene", {}))
    return SceneConfig(world_yaw_deg=float(scene_dict.get("world_yaw_deg", 0.0)))


def _load_task_runtime(config_dict: MutableMapping[str, object]) -> TaskRuntimeConfig:
    task_dict = dict(config_dict.get("task", {}))
    pan_pour_pose = Pose6D.from_iterable(
        task_dict.get("pan_pour_pose", config_dict.get("pan_pour_pose", (0, 0, 0, 0, 0, 0)))
    )
    tilt_angle_deg = float(task_dict.get("tilt_angle_deg", config_dict.get("tilt_angle_deg", 135.0)))
    hold_sec = float(task_dict.get("hold_sec", config_dict.get("hold_sec", 2.0)))
    noise_dict = task_dict.get("noise", {}) or {}
    noise = TaskNoiseConfig(
        depth_std=float(noise_dict.get("depth_std", 0.0)),
        drop_prob=float(noise_dict.get("drop_prob", 0.0)),
    )
    return TaskRuntimeConfig(
        pan_pour_pose=pan_pour_pose,
        tilt_angle_deg=tilt_angle_deg,
        hold_sec=hold_sec,
        noise=noise,
    )


def load_pour_task_config(path: Union[str, Path]) -> PourTaskConfig:
    """Load the pour task configuration from a YAML file and validate."""
    with open(Path(path), "r", encoding="utf-8") as fh:
        data: MutableMapping[str, object] = yaml.safe_load(fh) or {}

    bowl_pose = Pose6D.from_iterable(data.get("bowl_pose", (0, 0, 0, 0, 0, 0)))
    bowl_pose_1 = Pose6D.from_iterable(data.get("bowl_pose_1", (0, 0, 0, 0, 0, 0)))
    pan_pose = Pose6D.from_iterable(data.get("pan_pose", (0, 0, 0, 0, 0, 0)))
    pan_pour_pose = Pose6D.from_iterable(
        data.get("pan_pour_pose", data.get("pan_pose", (0, 0, 0, 0, 0, 0)))
    )
    task_runtime = _load_task_runtime(data)

    tilt_angle_deg = float(data.get("tilt_angle_deg", task_runtime.tilt_angle_deg))
    hold_sec = float(data.get("hold_sec", task_runtime.hold_sec))
    particles = int(data.get("particles", 0))
    seed = int(data.get("seed", 0))

    camera_cfg = _load_camera(data)
    perception_cfg = _load_perception(data)
    tolerances_cfg = _load_tolerances(data)
    scene_cfg = _load_scene(data)

    return PourTaskConfig(
        bowl_pose=bowl_pose,
        bowl_pose_1=bowl_pose_1,
        pan_pose=pan_pose,
        pan_pour_pose=pan_pour_pose,
        tilt_angle_deg=tilt_angle_deg,
        hold_sec=hold_sec,
        particles=particles,
        tolerances=tolerances_cfg,
        scene=scene_cfg,
        camera=camera_cfg,
        perception=perception_cfg,
        task=task_runtime,
        seed=seed,
    )


__all__ = [
    "Pose6D",
    "CameraView",
    "CameraNoise",
    "CameraConfig",
    "PerceptionConfig",
    "TolerancesConfig",
    "SceneConfig",
    "TaskNoiseConfig",
    "TaskRuntimeConfig",
    "PourTaskConfig",
    "load_pour_task_config",
]
