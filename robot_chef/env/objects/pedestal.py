import pybullet as p

from ...config import Pose6D


def create_pedestal(
    client_id: int,
    pose: Pose6D,
    half_extents: tuple[float, float, float],
) -> int:
    base_orientation = p.getQuaternionFromEuler(pose.orientation_rpy)
    col_id = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        physicsClientId=client_id,
    )
    vis_id = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=[0.35, 0.35, 0.38, 1.0],
        physicsClientId=client_id,
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=col_id,
        baseVisualShapeIndex=vis_id,
        basePosition=pose.position,
        baseOrientation=base_orientation,
        physicsClientId=client_id,
    )
    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=1.0,
        spinningFriction=0.6,
        physicsClientId=client_id,
    )
    return body_id
