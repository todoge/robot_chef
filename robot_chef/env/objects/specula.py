"""Factory for spawning a kitchen specula (spatula)."""

from __future__ import annotations

from typing import Dict, Tuple, List, Sequence

import pybullet as p

from ... import config


def create_specula(
    client_id: int,
    pose: config.Pose6D,
    handle_length: float = 0.15,
    handle_radius: float = 0.012,
    blade_length: float = 0.10,
    blade_width: float = 0.08,
    blade_thickness: float = 0.005,
    mass: float = 0.2,
) -> Tuple[int, Dict[str, float]]:
    """Spawn a specula/spatula."""
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z)

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
        color: Sequence[float] = (0.6, 0.6, 0.6, 1.0),
    ) -> None:
        shape_types.append(shape_type)
        radii.append(radius)
        lengths.append(length)
        half_extents.append(list(half))
        collision_positions.append(list(position))
        orientations.append(list(orientation_local))
        colors.append(list(color))

    # 1. Handle (Cylinder-ish via Capsule or Box)
    # Using Box for stability in gripper
    append_shape(
        p.GEOM_BOX,
        half=(handle_radius, handle_radius, handle_length / 2.0),
        position=(0.0, 0.0, 0.0), # Center of handle is origin
        color=(0.1, 0.1, 0.1, 1.0), # Dark handle
    )

    # 2. Blade (Flat Box)
    # Attached to the end of the handle (z = +handle_length/2)
    blade_z = handle_length / 2.0 + blade_length / 2.0
    append_shape(
        p.GEOM_BOX,
        half=(blade_width / 2.0, blade_thickness / 2.0, blade_length / 2.0),
        position=(0.0, 0.0, blade_z),
        color=(0.7, 0.7, 0.8, 1.0), # Metal blade
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
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=base_position,
        baseOrientation=orientation,
        physicsClientId=client_id,
    )

    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=0.8,
        spinningFriction=0.8,
        physicsClientId=client_id,
    )

    return body_id, {"handle_length": handle_length}