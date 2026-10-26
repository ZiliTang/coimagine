"""The training scenario: one large empty arena, one movable object, two robots, random starts
and a random goal centre kept away from the walls (so the object can actually reach it).

No static obstacles and no other layout: the object is a 1.0 x 0.25 m box, the robots are discs.
Both robots start outside contact, facing the object, on a random side/sector around it. The goal
centre is sampled inside a wall margin so a successful push is geometrically possible.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coimagine.continuous2d.scenes import Pose  # noqa: E402
from coimagine.random_start import layout, spec  # noqa: E402

ARENA = (4.0, 3.0)
OBJECT_HALF = (spec.OBJECT_WIDTH / 2.0, spec.OBJECT_HEIGHT / 2.0)


def _support_radius(angle: float) -> float:
    """Distance from the box centre to its boundary along ``angle``."""
    ca, sa = abs(math.cos(angle)), abs(math.sin(angle))
    if ca < 1e-9 or sa < 1e-9:
        return OBJECT_HALF[0] if sa < 1e-9 else OBJECT_HALF[1]
    return min(OBJECT_HALF[0] / ca, OBJECT_HALF[1] / sa)


def sample_scene(rng: np.random.Generator, index: int, split: str = "train") -> layout.InstanceSpec:
    width, height = ARENA
    # object centre keeps the whole box + margin inside the arena
    ox = float(rng.uniform(0.30 + OBJECT_HALF[0], width - 0.30 - OBJECT_HALF[0]))
    oy = float(rng.uniform(0.30 + OBJECT_HALF[1], height - 0.30 - OBJECT_HALF[1]))
    oang = float(rng.uniform(-math.pi, math.pi))
    object_pose = Pose(ox, oy, oang)

    robots = []
    base = float(rng.uniform(0.0, 2.0 * math.pi))
    spread = float(rng.uniform(0.9, 1.6))          # radians between the two robots
    for k in (0, 1):
        angle = base + k * spread
        standoff = float(rng.uniform(0.24, 0.55))   # edge gap 0.06 .. 0.37
        distance = _support_radius(angle - oang) + standoff
        x = ox + distance * math.cos(angle)
        y = oy + distance * math.sin(angle)
        facing = math.atan2(oy - y, ox - x)
        # keep the start strictly inside the arena
        x = min(max(x, spec.ROBOT_RADIUS + 0.02), width - spec.ROBOT_RADIUS - 0.02)
        y = min(max(y, spec.ROBOT_RADIUS + 0.02), height - spec.ROBOT_RADIUS - 0.02)
        robots.append(Pose(x, y, facing))

    margin = 0.60
    gx = float(rng.uniform(margin + spec.GOAL_RADIUS, width - margin - spec.GOAL_RADIUS))
    gy = float(rng.uniform(margin + spec.GOAL_RADIUS, height - margin - spec.GOAL_RADIUS))
    return layout.InstanceSpec(
        name=f"simple_{split}_{index:04d}", map="L0", split=split, index=index,
        seed=int(rng.integers(0, 1 << 30)), arena=ARENA, obstacles=(),
        object_pose=object_pose, robot_poses=tuple(robots), goal_centre=(gx, gy),
        goal_radius=spec.GOAL_RADIUS, env_version=spec.ENV_VERSION)


def sample_scenes(seed: int, count: int, split: str = "train"):
    rng = np.random.default_rng(seed)
    return [sample_scene(rng, index, split) for index in range(count)]
