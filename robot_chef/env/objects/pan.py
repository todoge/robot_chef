"""Factory for spawning a shallow pan with a flat base and short walls."""

from __future__ import annotations

import math
from typing import Dict, Tuple, Sequence, List

import pybullet as p

from ... import config


def create_pan(
    client_id: int,
    pose: config.Pose6D,
    inner_radius: float = 0.16,
    depth: float = 0.045,
    wall_thickness: float = 0.012,
    base_thickness: float = 0.012,
    handle_length: float = 0.22,
    handle_width: float = 0.045,
    handle_thickness: float = 0.02,
    friction: float = 1.0,
) -> Tuple[int, Dict[str, float]]:
    """Spawn a shallow pan with a flat base, circular walls, and an extended handle."""
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z + base_thickness / 2.0)
    half_side = inner_radius

    shape_types: List[int] = []
    radii: List[float] = []
    lengths: List[float] = []
    half_extents: List[List[float]] = []
    collision_positions: List[List[float]] = []
    orientations: List[List[float]] = []
    colors: List[List[float]] = []

    def append_shape(
        shape_type: int,
        *,
        radius: float = 0.0,
        length: float = 0.0,
        half: Sequence[float] = (0.0, 0.0, 0.0),
        position: Sequence[float] = (0.0, 0.0, 0.0),
        orientation_local: Sequence[float] = (0.0, 0.0, 0.0, 1.0),
        color: Sequence[float] = (0.18, 0.18, 0.18, 1.0),
    ) -> None:
        shape_types.append(shape_type)
        radii.append(radius)
        lengths.append(length)
        half_extents.append(list(half))
        collision_positions.append(list(position))
        orientations.append(list(orientation_local))
        colors.append(list(color))

    append_shape(
        p.GEOM_BOX,
        half=(half_side, half_side, base_thickness / 2.0),
        color=(0.16, 0.16, 0.16, 1.0),
    )

    wall_z = base_thickness / 2.0 + depth / 2.0
    wall_offset = half_side + wall_thickness / 2.0

    append_shape(
        p.GEOM_BOX,
        half=(wall_thickness / 2.0, half_side, depth / 2.0),
        position=(wall_offset, 0.0, wall_z),
    )
    append_shape(
        p.GEOM_BOX,
        half=(wall_thickness / 2.0, half_side, depth / 2.0),
        position=(-wall_offset, 0.0, wall_z),
    )
    append_shape(
        p.GEOM_BOX,
        half=(half_side + wall_thickness, wall_thickness / 2.0, depth / 2.0),
        position=(0.0, wall_offset, wall_z),
    )
    append_shape(
        p.GEOM_BOX,
        half=(half_side + wall_thickness, wall_thickness / 2.0, depth / 2.0),
        position=(0.0, -wall_offset, wall_z),
    )

    handle_offset = inner_radius + wall_thickness / 2.0 + handle_length / 2.0
    tip_offset = inner_radius + wall_thickness / 2.0 + handle_length + (handle_width * 0.6) / 2.0
    handle_height_local = base_thickness / 2.0 + depth / 2.0

    append_shape(
        p.GEOM_BOX,
        half=(handle_length / 2.0, handle_width / 2.0, handle_thickness / 2.0),
        position=(handle_offset, 0.0, handle_height_local),
        color=(0.12, 0.12, 0.12, 1.0),
    )
    append_shape(
        p.GEOM_BOX,
        half=((handle_width * 0.6) / 2.0, (handle_width * 1.1) / 2.0, handle_thickness / 2.0),
        position=(tip_offset, 0.0, handle_height_local),
        color=(0.10, 0.10, 0.10, 1.0),
    )

    collision = p.createCollisionShapeArray(
        shapeTypes=shape_types,
        radii=radii,
        halfExtents=half_extents,
        lengths=lengths,
        collisionFramePositions=collision_positions,
        collisionFrameOrientations=orientations,
        physicsClientId=client_id,
    )

    visual = p.createVisualShapeArray(
        shapeTypes=shape_types,
        radii=radii,
        halfExtents=half_extents,
        lengths=lengths,
        rgbaColors=colors,
        visualFramePositions=collision_positions,
        visualFrameOrientations=orientations,
        physicsClientId=client_id,
    )

    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=base_position,
        baseOrientation=orientation,
        physicsClientId=client_id,
    )

    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.3,
        restitution=0.0,
        physicsClientId=client_id,
    )

    properties = {
        "inner_radius": inner_radius,
        "half_side": half_side,
        "base_height": pose.z,
        "lip_height": pose.z + depth,
        "depth": depth,
        "wall_thickness": wall_thickness,
        "handle_offset": handle_offset,
        "handle_length": handle_length,
    }
    return body_id, properties
