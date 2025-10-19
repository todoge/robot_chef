"""Perception utilities for RGB-D sensing and geometric reasoning."""

from .camera import Camera
from .bowl_rim import (
    configure_scene_context,
    detect_bowl_rim,
)

__all__ = [
    "Camera",
    "configure_scene_context",
    "detect_bowl_rim",
]
