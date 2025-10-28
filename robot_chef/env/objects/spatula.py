import pybullet as p
import os

def create_spatula(client_id, pose, mass=0.3, friction=2.0):

    spatula_obj_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "spatula.obj")

    mesh_scale = [1.5,1.5,1.5]

    collision_shape_id = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=spatula_obj_path,
        meshScale = mesh_scale,
        physicsClientId=client_id,
    )
    visual_shape_id = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=spatula_obj_path,
        meshScale=mesh_scale,
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

    return body_id