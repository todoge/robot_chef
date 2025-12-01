import math

from robot_chef.config import Pose6D


def apply_world_yaw(pose: Pose6D, world_yaw_deg: float) -> Pose6D:
    yaw = math.radians(world_yaw_deg)
    cos_y = math.cos(yaw)
    sin_y = math.sin(yaw)
    x = pose.x * cos_y - pose.y * sin_y
    y = pose.x * sin_y + pose.y * cos_y
    return Pose6D(
        x=x,
        y=y,
        z=pose.z,
        roll=pose.roll,
        pitch=pose.pitch,
        yaw=pose.yaw + yaw,
    )

def pause() -> None:
    print("▌▌ PAUSE ▌▌")
    input()
    print("🞂 RESUME 🞂")
    