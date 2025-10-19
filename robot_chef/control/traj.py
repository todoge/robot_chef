"""Simple joint-space trajectory following utilities."""

from __future__ import annotations

from typing import Dict, Sequence

import pybullet as p

from .. import config


def follow(
    sim,
    arm: Dict[str, object],
    joint_targets: Sequence[float],
    duration: float = 1.0,
    max_force: float = 200.0,
) -> int:
    """Linearly interpolate joint targets and drive the robot using Bullet motors."""
    joint_indices = arm["arm_joints"]
    if len(joint_indices) != len(joint_targets):
        raise ValueError("Joint target dimension mismatch.")

    current = [p.getJointState(arm["body"], j, physicsClientId=sim.client_id)[0] for j in joint_indices]
    steps = max(1, int(duration / config.SIMULATION_STEP))
    for step in range(1, steps + 1):
        alpha = step / steps
        targets = [(1.0 - alpha) * current[i] + alpha * joint_targets[i] for i in range(len(joint_targets))]
        p.setJointMotorControlArray(
            bodyUniqueId=arm["body"],
            jointIndices=joint_indices,
            controlMode=p.POSITION_CONTROL,
            targetPositions=targets,
            forces=[max_force] * len(joint_indices),
            positionGains=[0.6] * len(joint_indices),
            velocityGains=[1.0] * len(joint_indices),
            physicsClientId=sim.client_id,
        )
        sim.step_simulation(1)
    settle_steps = max(1, int(0.1 / config.SIMULATION_STEP))
    sim.step_simulation(settle_steps)
    return steps
