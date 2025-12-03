"""Particle spawning helpers for representing loose ingredients."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import pybullet as p


@dataclass
class ParticleSet:
    body_ids: List[int]
    radius: float

    @property
    def count(self) -> int:
        return len(self.body_ids)

    def count_in_pan(
        self,
        client_id: int,
        center: Sequence[float],
        inner_radius: float,
        base_height: float,
        lip_height: float,
    ) -> Tuple[int, int, float]:
        inside = 0
        cx, cy, _ = center
        for body_id in self.body_ids:
            position, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
            radial = math.hypot(position[0] - cx, position[1] - cy)
            if radial <= inner_radius - self.radius * 0.5 and base_height - self.radius <= position[2] <= lip_height + self.radius:
                inside += 1
        total = self.count
        ratio = inside / total if total else 0.0
        return total, inside, ratio


def spawn_spheres(
    client_id: int,
    count: int,
    radius: float,
    center: Sequence[float],
    bowl_radius: float,
    bowl_height: float,
    spawn_height: float,
    seed: int = 7,
    mass: float = 0.01,  
    friction: float = 0.2, 
) -> ParticleSet:
    """Spawn spherical particles inside a bowl volume."""
    rng = random.Random(seed)
    body_ids: List[int] = []
    collision = p.createCollisionShape(p.GEOM_SPHERE, radius=radius, physicsClientId=client_id)
    visual = p.createVisualShape(
        shapeType=p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[0.96, 0.94, 0.84, 1.0],
        physicsClientId=client_id,
    )
    min_z = center[2] + radius
    max_spawn = min(center[2] + max(spawn_height, radius * 3), center[2] + bowl_height - radius)
    for _ in range(count):
        r = bowl_radius * math.sqrt(rng.random())
        theta = 2.0 * math.pi * rng.random()
        x = center[0] + r * math.cos(theta)
        y = center[1] + r * math.sin(theta)
        z = min_z + rng.random() * max(0.0, max_spawn - min_z)
        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=[x, y, z],
            physicsClientId=client_id,
        )
        p.changeDynamics(
            bodyUniqueId=body_id,
            linkIndex=-1,
            lateralFriction=friction,
            spinningFriction=friction,
            rollingFriction=0.01,
            restitution=0.0,
            physicsClientId=client_id,
        )
        body_ids.append(body_id)
    return ParticleSet(body_ids=body_ids, radius=radius)