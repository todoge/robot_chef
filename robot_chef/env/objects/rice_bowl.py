"""Factory for spawning a rice bowl with suitable physical properties."""

from __future__ import annotations

from typing import Dict, Tuple

from ... import config

import pybullet as p
import numpy as np
import math
import os
from typing import Tuple, Dict
import config  # assuming this defines Pose6D

def create_bowl(client_id: int, pose) -> Tuple[int, Dict[str, float]]:
    """
    Load a round bowl from URDF file.
    
    Args:
        client_id: PyBullet client ID
        pose: Pose6D object with x, y, z, roll, pitch, yaw
    
    Returns:
        body_id: PyBullet body ID
        properties: Dict with bowl dimensions
    """
    
    # Load the URDF
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    
    urdf_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bowl.urdf")
    print(f"Loading URDF from: {urdf_path}", flush=True)

    body_id = p.loadURDF(
        urdf_path,  # Make sure this file is in the same folder
        basePosition=[pose.x, pose.y, pose.z],
        baseOrientation=orientation,
        useFixedBase=False,
        physicsClientId=client_id,
    )
    
    # Set physics properties
    p.changeDynamics(
        body_id,
        -1,
        mass=0.4,
        lateralFriction=2.0,
        spinningFriction=2.0,
        rollingFriction=1.0,
        restitution=0.0,
        linearDamping=0.5,
        angularDamping=0.5,
        contactStiffness=50000,
        contactDamping=2000,
        collisionMargin=0.002,
        physicsClientId=client_id,
    )
    
    # Bowl properties (match the URDF dimensions)
    properties = {
        "outer_radius": 0.07,
        "inner_radius": 0.062,  # 0.07 - 0.008
        "inner_height": 0.05,
        "base_thickness": 0.006,
        "spawn_height": 0.035,  # 0.05 * 0.7
        "mass": 0.4,
    }
    
    return body_id, properties

def create_round_bowl(
    client_id: int,
    pose: config.Pose6D,
    outer_radius: float = 0.07,
    inner_height: float = 0.05,
    wall_thickness: float = 0.008,
    base_thickness: float = 0.006,
    mass: float = 0.4,
    friction: float = 2.0,
) -> Tuple[int, Dict[str, float]]:

    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z + base_thickness / 2.0)
    inner_radius = outer_radius - wall_thickness

    # Shape types
    shape_types = [p.GEOM_CYLINDER, p.GEOM_CYLINDER]

    # Radii for each cylinder
    radii = [outer_radius, outer_radius]

    # Half extents: only Z (half-height) is used for cylinders
    half_extents = [
        [0.0, 0.0, base_thickness / 2.0],      # base
        [0.0, 0.0, inner_height / 2.0],        # wall
    ]

    # Positions relative to body frame
    collision_positions = [
        [0.0, 0.0, 0.0],  # base at origin
        [0.0, 0.0, base_thickness / 2.0 + inner_height / 2.0],  # wall on top
    ]

    orientations = [[0, 0, 0, 1], [0, 0, 0, 1]]

    # Collision shape
    collision = p.createCollisionShapeArray(
        shapeTypes=shape_types,
        radii=radii,  # ✅ Critical!
        halfExtents=half_extents,
        collisionFramePositions=collision_positions,
        collisionFrameOrientations=orientations,
        physicsClientId=client_id,
    )

    # Visual shape
    rgba = [0.95, 0.95, 0.95, 1.0]
    visual = p.createVisualShapeArray(
        shapeTypes=shape_types,
        radii=radii,
        halfExtents=half_extents,
        rgbaColors=[rgba, rgba],
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
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        restitution=0.0,
        linearDamping=0.0,
        angularDamping=0.0,
        contactStiffness=50000,
        contactDamping=2000,
        collisionMargin=0.001,
        physicsClientId=client_id,
    )

    properties = {
        "outer_radius": outer_radius,
        "inner_radius": inner_radius,
        "inner_height": inner_height,
        "base_thickness": base_thickness,
        "wall_thickness": wall_thickness,
        "spawn_height": inner_height * 0.7,
        "mass": mass,
    }

    return body_id, properties

def extract_bowl_parameters_from_stl(stl_path: str, mass) -> Dict[str, float]:
    """
    Analyze STL mesh to estimate bowl parameters.
    
    Args:
        stl_path: Path to STL file
    
    Returns:
        Dictionary with estimated bowl parameters
    """
    try:
        import trimesh
    except ImportError:
        print("Installing trimesh... Run: pip install trimesh")
        import subprocess
        subprocess.check_call(['pip', 'install', 'trimesh'])
        import trimesh
    
    # Load mesh
    mesh = trimesh.load(stl_path)
    vertices = mesh.vertices
    
    # Get bounding box
    bounds = mesh.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    min_bounds = bounds[0]
    max_bounds = bounds[1]
    
    # Estimate parameters
    
    # 1. Height (Z dimension)
    base_z = min_bounds[2]
    top_z = max_bounds[2]
    total_height = top_z - base_z
    
    # 2. Outer radius (max XY distance from center)
    # Find center in XY plane
    center_x = (min_bounds[0] + max_bounds[0]) / 2
    center_y = (min_bounds[1] + max_bounds[1]) / 2
    
    # Calculate distances from center for all vertices
    xy_distances = np.sqrt((vertices[:, 0] - center_x)**2 + (vertices[:, 1] - center_y)**2)
    outer_radius = np.max(xy_distances)
    
    # 3. Base thickness - find vertices near bottom
    bottom_threshold = base_z + total_height * 0.1  # Bottom 10%
    bottom_vertices = vertices[vertices[:, 2] < bottom_threshold]
    
    if len(bottom_vertices) > 0:
        base_thickness = np.max(bottom_vertices[:, 2]) - base_z
    else:
        base_thickness = total_height * 0.1  # Estimate 10% of height
    
    # 4. Inner height (height of walls)
    inner_height = total_height - base_thickness
    
    # 5. Inner radius - find minimum XY distance at rim level
    # Look at vertices near the top (rim)
    rim_threshold = top_z - inner_height * 0.1  # Top 10%
    rim_vertices = vertices[vertices[:, 2] > rim_threshold]
    
    if len(rim_vertices) > 0:
        rim_distances = np.sqrt((rim_vertices[:, 0] - center_x)**2 + (rim_vertices[:, 1] - center_y)**2)
        # Inner radius is the minimum distance at rim (inner edge)
        inner_radius = np.min(rim_distances)
        wall_thickness = outer_radius - inner_radius
    else:
        # Estimate wall thickness as 10% of radius
        wall_thickness = outer_radius * 0.1
        inner_radius = outer_radius - wall_thickness
    
    # 6. Mass - estimate from volume and material density
    volume = mesh.volume  # in m^3 if STL is in meters
    # Assume ceramic/glass density: ~2500 kg/m^3
    estimated_mass = volume * 2500
    
    # If STL is in mm, convert
    if outer_radius > 1.0:  # Likely in mm
        print("STL appears to be in millimeters, converting to meters...")
        outer_radius /= 1000
        inner_radius /= 1000
        wall_thickness /= 1000
        base_thickness /= 1000
        inner_height /= 1000
        total_height /= 1000
        estimated_mass = (volume / 1e9) * 2500  # Convert mm^3 to m^3
    
    parameters = {
        "radius": float(outer_radius),
        "inner_radius": float(inner_radius),
        "base_thickness": float(base_thickness),
        "inner_height": float(inner_height),
        "spawn_height": float(inner_height * 0.7),  # For particle spawning
        "mass" : mass
    }
    
    return parameters

def create_square_bowl(
    client_id: int,
    pose: config.Pose6D,
    side_length: float = 0.14,       # overall width of the bowl
    inner_height: float = 0.05,      # vertical height of the walls
    wall_thickness: float = 0.008,   # wall thickness
    base_thickness: float = 0.006,   # base thickness
    mass: float = 0.4,
    friction: float = 2,
) -> Tuple[int, Dict[str, float]]:
    """
    Spawn a square (box-shaped) bowl that can contain particles.
    The bowl has a square base and four thin walls.
    """

    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z + base_thickness / 2.0)
    inner_side = side_length - 2 * wall_thickness  # inner open space

    # Each shape: base + 4 walls
    shape_types = [p.GEOM_BOX] * 5
    half_extents = [
        # base
        [side_length / 2.0, side_length / 2.0, base_thickness / 2.0],
        # right wall
        [wall_thickness / 2.0, inner_side / 2.0, inner_height / 2.0],
        # left wall
        [wall_thickness / 2.0, inner_side / 2.0, inner_height / 2.0],
        # front wall
        [inner_side / 2.0, wall_thickness / 2.0, inner_height / 2.0],
        # back wall
        [inner_side / 2.0, wall_thickness / 2.0, inner_height / 2.0],
    ]

    overlap_offset = wall_thickness / 2 * (2 ** 0.5) / 2  # ~0.707 * half thickness

    collision_positions = [
        [0.0, 0.0, 0.0],  # base
        [inner_side / 2.0 + wall_thickness / 2.0 - overlap_offset, 0.0, inner_height / 2.0],  # right wall
        [-(inner_side / 2.0 + wall_thickness / 2.0 - overlap_offset), 0.0, inner_height / 2.0],  # left wall
        [0.0, inner_side / 2.0 + wall_thickness / 2.0 - overlap_offset, inner_height / 2.0],  # front wall
        [0.0, -(inner_side / 2.0 + wall_thickness / 2.0 - overlap_offset), inner_height / 2.0],  # back wall
    ] 

    collision_orientations = [[0, 0, 0, 1]] * len(shape_types)

    # Collision and visual shape arrays
    collision = p.createCollisionShapeArray(
        shapeTypes=shape_types,
        halfExtents=half_extents,
        collisionFramePositions=collision_positions,
        collisionFrameOrientations=collision_orientations,
        physicsClientId=client_id,
    )

    rgba_base = [0.95, 0.95, 0.95, 1.0]
    visual = p.createVisualShapeArray(
        shapeTypes=shape_types,
        halfExtents=half_extents,
        rgbaColors=[rgba_base] * len(shape_types),
        visualFramePositions=collision_positions,
        visualFrameOrientations=collision_orientations,
        physicsClientId=client_id,
    )

    # Create the bowl as a multibody
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=base_position,
        baseOrientation=orientation,
        physicsClientId=client_id,
    )

    # Apply physics and contact properties
    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        restitution=0.0,
        linearDamping=0.5,
        angularDamping=0.5,
        contactStiffness=50000,
        contactDamping=2000,
        collisionMargin=0.002,
        physicsClientId=client_id,
    )

    # Bowl metadata
    properties = {
        "side_length": side_length,
        "inner_side": inner_side,
        "inner_height": inner_height,
        "base_thickness": base_thickness,
        "spawn_height": inner_height * 0.7,
        "mass": mass,
    }

    return body_id, properties

def create_rounded_bowl(client_id, pose, mass=0.4, friction=2.0):

    stl_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bowl.stl")
    vhacd_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "bowl_vhacd.obj")
    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=vhacd_file_path,
        physicsClientId=client_id,
    )
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=stl_file_path,
        physicsClientId=client_id,
    )
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=[pose.x, pose.y, pose.z],
        baseOrientation=orientation,
        baseInertialFramePosition=[0, 0, 0],  # usually center of mass, adjust if needed
        baseInertialFrameOrientation=[0, 0, 0, 1],  # usually identity quaternion
        physicsClientId=client_id,
    )
    # Set friction and damping to reduce sliding
    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        linearDamping=0.8,
        angularDamping=0.8,
        physicsClientId=client_id,
    )

    params = extract_bowl_parameters_from_stl(stl_file_path, mass)
    return body_id, params


def create_rice_bowl(
    client_id: int,
    pose: config.Pose6D,
    radius: float = 0.07,
    inner_height: float = 0.05,
    wall_thickness: float = 0.008,
    base_thickness: float = 0.006,
    mass: float = 0.4,
    friction: float = 2,
) -> Tuple[int, Dict[str, float]]:
    """Spawn a compound rice bowl that can contain particles."""
    orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    base_position = (pose.x, pose.y, pose.z + base_thickness / 2.0)
    inner_radius = radius - wall_thickness
    wall_height = inner_height

    shape_types = [
        p.GEOM_CYLINDER,  # base disk
        p.GEOM_BOX,
        p.GEOM_BOX,
        p.GEOM_BOX,
        p.GEOM_BOX,
    ]
    radii = [radius, 0.0, 0.0, 0.0, 0.0]
    lengths = [base_thickness, 0.0, 0.0, 0.0, 0.0]
    half_extents = [
        [0.0, 0.0, 0.0],
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
    '''
    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        restitution=0.0,
        physicsClientId=client_id,
    )
    '''
    p.changeDynamics(
        bodyUniqueId=body_id,
        linkIndex=-1,
        lateralFriction=friction,
        spinningFriction=friction,
        rollingFriction=friction * 0.5,
        restitution=0.0,
        linearDamping=0.5,  # Added: Damp linear motion
        angularDamping=0.5,  # Added: Damp rotation
        contactStiffness=50000,  # Added: Stiff contacts
        contactDamping=2000,  # Added: Contact damping
        collisionMargin=0.002,  # Added: Collision margin
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
