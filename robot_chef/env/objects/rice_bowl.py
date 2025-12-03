"""Factory for spawning a rice bowl with suitable physical properties."""

from __future__ import annotations

from typing import Dict, Tuple

import pybullet as p

from ... import config


def create_rice_bowl(
    client_id: int,
    pose: config.Pose6D,
    radius: float = 0.07,
    inner_height: float = 0.08,
    wall_thickness: float = 0.008,
    base_thickness: float = 0.006,
    mass: float = 0.3,
    friction: float = 1.2, 
) -> Tuple[int, Dict[str, float]]:
    """Spawn a compound rice bowl that can contain particles."""
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z + base_thickness / 2.0)
    inner_radius = radius - wall_thickness
    wall_height = inner_height

    shape_types = [
        p.GEOM_BOX,
        p.GEOM_BOX,
        p.GEOM_BOX,
        p.GEOM_BOX,
        p.GEOM_BOX,
    ]
    radii = [0.0, 0.0, 0.0, 0.0, 0.0]
    lengths = [0.0, 0.0, 0.0, 0.0, 0.0]
    half_extents = [
        [radius, radius, base_thickness / 2.0],
        [wall_thickness / 2.0, inner_radius, wall_height / 2.0],
        [wall_thickness / 2.0, inner_radius, wall_height / 2.0],
        [inner_radius, wall_thickness / 2.0, wall_height / 2.0],
        [inner_radius, wall_thickness / 2.0, wall_height / 2.0],
    ]
    
    collision_positions = [
        [0.0, 0.0, 0.0],
        [inner_radius + wall_thickness / 2.0, 0.0, wall_height / 2.0],
        [-(inner_radius + wall_thickness / 2.0), 0.0, wall_height / 2.0],
        [0.0, inner_radius + wall_thickness / 2.0, wall_height / 2.0],
        [0.0, -(inner_radius + wall_thickness / 2.0), wall_height / 2.0],
    ]
    collision_orientations = [[0.0, 0.0, 0.0, 1.0]] * len(shape_types)

    collision = p.createCollisionShapeArray(
        shapeTypes=shape_types,
        radii=radii,
        halfExtents=half_extents,
        lengths=lengths,
        collisionFramePositions=collision_positions,
        collisionFrameOrientations=collision_orientations,
        physicsClientId=client_id,
    )

    rgba_base = [0.95, 0.95, 0.95, 1.0]
    visual = p.createVisualShapeArray(
        shapeTypes=shape_types,
        radii=radii,
        halfExtents=half_extents,
        lengths=lengths,
        rgbaColors=[rgba_base] * len(shape_types),
        visualFramePositions=collision_positions,
        visualFrameOrientations=collision_orientations,
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
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        restitution=0.0,
        physicsClientId=client_id,
    )
    properties = {
        "radius": radius,
        "inner_radius": inner_radius,
        "inner_height": inner_height,
        "base_thickness": base_thickness,
        "spawn_height": inner_height * 0.7,
        "mass": mass,
    }
    return body_id, properties