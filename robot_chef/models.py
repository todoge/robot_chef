from pathlib import Path
from typing import Dict, List

from pydantic import BaseModel
from pydantic_yaml import parse_yaml_raw_as

# --- Basic reusable models ---

class Pose(BaseModel):
    """6-D pose: [x, y, z, roll, pitch, yaw]"""
    __root__: List[float]


class XYZ(BaseModel):
    __root__: List[float]


class RPY(BaseModel):
    __root__: List[float]


class Resolution(BaseModel):
    __root__: List[int]


# --- Config sections ---

class Tolerances(BaseModel):
    place_back_xy: float
    place_back_yaw_deg: float


class Scene(BaseModel):
    world_yaw_deg: float


class CameraView(BaseModel):
    xyz: XYZ
    rpy: RPY
    fov_deg: float
    resolution: Resolution


class CameraNoise(BaseModel):
    depth_std: float
    drop_prob: float


class CameraConfig(BaseModel):
    active_view: str
    near: float
    far: float
    views: Dict[str, CameraView]
    noise: CameraNoise


class Perception(BaseModel):
    enable: bool
    bowl_roi_margin_m: float
    rim_sample_count: int
    grasp_clearance_m: float
    rim_thickness_m: float


class TaskNoise(BaseModel):
    depth_std: float
    drop_prob: float


class Task(BaseModel):
    spatual_stir_pose: Pose
    tilt_angle_deg: float
    hold_sec: float
    noise: TaskNoise


# --- Top-level root model ---

class Config(BaseModel):
    bowl_pose: Pose
    pan_pose: Pose
    spatula_stir_pose: Pose
    spatula_pose: Pose

    tilt_angle_deg: float
    hold_sec: float
    rice_particles: int
    egg_particles: int

    tolerances: Tolerances
    scene: Scene
    camera: CameraConfig
    perception: Perception
    task: Task

    seed: int

yaml_text = Path("config.yml").read_text()
config = parse_yaml_raw_as(Config, yaml_text)