"""Random scene and initial-state sampler (task book A2).

The geometry ranges below are frozen **before** any training result is seen. Only quantities
that the existing 57-dimensional input already expresses are randomised: the object's initial
pose, the goal placement, and -- for S1 -- the passage wall's position, opening centre and
opening width. Robot radius, object size, masses, friction, control gains, arena size and the
zero obstacle angle are all left at their frozen values.

Every accepted root must be strictly legal under the frozen geometric checker with the
uniform 0.005 m non-contact margin, and must not already be in absorbing success. Rejection
sampling is used; a body is never projected out of a penetration.
"""

from __future__ import annotations

import hashlib
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.legality import (
    DEFAULT_MARGIN,
    boundary_clearance,
    legality_report,
)
from ..continuous2d.scenes import Box, Pose, SceneSpec

Vec = Tuple[float, float]

#: Frozen ranges (task book section A2). SI metres and radians.
S0_ARENA = (3.2, 2.2)
S1_ARENA = (3.0, 2.0)
S0_GOAL_CENTER = (2.35, 1.10)
S0_GOAL_HALF = (0.65, 0.60)
S0_GOAL_JITTER = 0.15
S1_GOAL_CENTER_X = 2.25
S1_GOAL_HALF = (0.55, 0.55)
S1_GOAL_JITTER = 0.10
S1_WALL_X_RANGE = (1.40, 1.80)
S1_GAP_CENTER_RANGE = (0.80, 1.20)
S1_GAP_WIDTH_RANGE = (0.60, 0.85)
S1_WALL_HALF_THICKNESS = 0.09
OBJECT_WIDTH = 1.0
OBJECT_HEIGHT = 0.25
ROBOT_RADIUS = 0.18
WALL_CLEARANCE_RANGE = (0.01, 0.20)
GOAL_NEIGHBOURHOOD = 0.90
GOAL_NEIGHBOURHOOD_DISTANCE = 0.30
ROBOT_NEAR_OBJECT_PROBABILITY = 0.70
ROBOT_GAP_RANGE = (0.005, 0.25)
MAX_ROOT_ATTEMPTS = 2000

MODES: Tuple[str, ...] = ("broad", "wall", "goal")
FAMILIES: Tuple[str, ...] = ("S0", "S1")

#: Disjoint root-seed spaces per split, so no root can cross a split (task book B2).
SPLIT_SEED_BASE: Dict[str, int] = {
    "train": 100_000_000,
    "val": 200_000_000,
    "test": 300_000_000,
    "dev": 400_000_000,
}

SAMPLER_VERSION = "coverage-scale-sampler-v1"


def _digest(*parts: object) -> str:
    return hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()


def root_seed(split: str, family: str, mode: str, ordinal: int) -> int:
    """Deterministic, split-disjoint root seed."""

    if split not in SPLIT_SEED_BASE:
        raise KeyError(f"unknown split {split!r}")
    offset = int(_digest(SAMPLER_VERSION, split, family, mode, ordinal)[:6], 16)
    return SPLIT_SEED_BASE[split] + offset


def scene_id_for(family: str, seed: int) -> str:
    return f"{family}_r{seed:09d}"


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def object_corners(x: float, y: float, angle: float) -> Tuple[Vec, ...]:
    half_w, half_h = OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0
    cos, sin = math.cos(angle), math.sin(angle)
    return tuple(
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos)
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    )


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def polygon_segment_distance(corners: Sequence[Vec], ax: float, ay: float, bx: float, by: float) -> float:
    """Exact distance from a convex polygon to a segment (the segment's ends matter too).

    ``min`` over the polygon's corners alone is not the polygon-to-segment distance: a segment
    endpoint can be closest to the interior of a polygon edge. Both candidates are taken.
    """

    best = min(_point_segment_distance(cx, cy, ax, ay, bx, by) for cx, cy in corners)
    for px, py in ((ax, ay), (bx, by)):
        for index, corner in enumerate(corners):
            nxt = corners[(index + 1) % len(corners)]
            best = min(best, _point_segment_distance(px, py, corner[0], corner[1], nxt[0], nxt[1]))
    return best


def _half_extent(angle: float) -> Tuple[float, float]:
    """Half extent of the object's axis-aligned bounding box at ``angle``."""

    cos, sin = abs(math.cos(angle)), abs(math.sin(angle))
    return (
        OBJECT_WIDTH / 2.0 * cos + OBJECT_HEIGHT / 2.0 * sin,
        OBJECT_WIDTH / 2.0 * sin + OBJECT_HEIGHT / 2.0 * cos,
    )


def box_faces(box: Box) -> List[Tuple[Vec, Vec]]:
    """The four edges of an axis-aligned wall box, as world segments."""

    x0, x1 = box.cx - box.half_w, box.cx + box.half_w
    y0, y1 = box.cy - box.half_h, box.cy + box.half_h
    return [
        ((x0, y0), (x1, y0)),
        ((x1, y0), (x1, y1)),
        ((x1, y1), (x0, y1)),
        ((x0, y1), (x0, y0)),
    ]


def boundary_wall_distance(corners: Sequence[Vec], arena: Tuple[float, float], wall: str) -> float:
    """Nearest-surface distance from the object polygon to one arena boundary capsule."""

    width, height = arena
    values = []
    for x, y in corners:
        if wall == "left":
            values.append(x - 0.05)
        elif wall == "right":
            values.append(width - 0.05 - x)
        elif wall == "bottom":
            values.append(y - 0.05)
        else:
            values.append(height - 0.05 - y)
    return min(values)


def _signed_polygon_box(corners: Sequence[Vec], box: Box) -> float:
    """Signed clearance between the object polygon and the goal box (negative = overlap)."""

    from ..continuous2d_learning.contract import _box_corners, _signed_polygon_polygon

    return _signed_polygon_polygon(corners, _box_corners(box))


def inside_goal(x: float, y: float, angle: float, goal: Box, margin: float = 0.02) -> bool:
    for cx, cy in object_corners(x, y, angle):
        if abs(cx - goal.cx) > goal.half_w - margin:
            return False
        if abs(cy - goal.cy) > goal.half_h - margin:
            return False
    return True


#: Names of the pair clearances checked on a *proposed* initial state.
CLEARANCE_PAIRS: Tuple[str, ...] = (
    "robot0_boundary", "robot1_boundary",
    "robot0_obstacle", "robot1_obstacle",
    "robot0_object", "robot1_object",
    "robot_robot",
    "object_boundary", "object_obstacle",
)


def strict_clearances(scene: SceneSpec, margin: float = DEFAULT_MARGIN) -> Dict[str, float]:
    """Exact signed clearances of a proposed initial state, in metres.

    This is deliberately **stricter** than :func:`coimagine.continuous2d.legality.scene_clearances`,
    which tests the object against an obstacle only through the object's four corners. That
    corner-only test accepts a configuration in which a corner of an obstacle box lies inside
    the object polygon while every object corner is outside the box -- a real overlap the
    physics engine reports at the first step. Here the object/obstacle pair uses the same
    separating-axis clearance as :func:`...contract.geometric_violation`, and the robot pairs
    use the exact signed circle/polygon and circle/box clearances. Nothing is loosened: every
    value must be at least the uniform non-contact margin.
    """

    from ..continuous2d_learning.contract import (
        _boundary_clearance,
        _box_corners,
        _signed_circle_box,
        _signed_circle_polygon,
        _signed_polygon_polygon,
    )

    radius = scene.robot_radius
    corners = object_corners(scene.object_pose.x, scene.object_pose.y, scene.object_pose.angle)
    out: Dict[str, float] = {}
    for robot_id, pose in enumerate(scene.robot_poses):
        out[f"robot{robot_id}_boundary"] = _boundary_clearance((pose.x, pose.y), scene.arena, 0.05) - radius
        wall = float("inf")
        for box in scene.obstacles:
            wall = min(wall, _signed_circle_box(pose.x, pose.y, radius, box))
        out[f"robot{robot_id}_obstacle"] = wall
        out[f"robot{robot_id}_object"] = _signed_circle_polygon(pose.x, pose.y, radius, corners)
    first, second = scene.robot_poses
    out["robot_robot"] = math.hypot(first.x - second.x, first.y - second.y) - 2.0 * radius
    out["object_boundary"] = min(_boundary_clearance(corner, scene.arena, 0.05) for corner in corners)
    walls = float("inf")
    for box in scene.obstacles:
        walls = min(walls, _signed_polygon_polygon(corners, _box_corners(box)))
    out["object_obstacle"] = walls
    return out


def strict_legal(scene: SceneSpec, margin: float = DEFAULT_MARGIN) -> bool:
    return all(value >= margin for value in strict_clearances(scene, margin).values())


# ---------------------------------------------------------------------------
# scene geometry
# ---------------------------------------------------------------------------


def sample_scene_geometry(family: str, rng: random.Random) -> Tuple[Box, Tuple[Box, ...]]:
    """Sample the goal placement, and for S1 the passage wall, inside the frozen ranges."""

    if family == "S0":
        dx = rng.uniform(-S0_GOAL_JITTER, S0_GOAL_JITTER)
        dy = rng.uniform(-S0_GOAL_JITTER, S0_GOAL_JITTER)
        return Box(S0_GOAL_CENTER[0] + dx, S0_GOAL_CENTER[1] + dy, *S0_GOAL_HALF), ()

    arena_w, arena_h = S1_ARENA
    wall_x = rng.uniform(*S1_WALL_X_RANGE)
    gap_center = rng.uniform(*S1_GAP_CENTER_RANGE)
    gap_width = rng.uniform(*S1_GAP_WIDTH_RANGE)
    lower_height = gap_center - gap_width / 2.0
    upper_start = gap_center + gap_width / 2.0
    lower = Box(wall_x, lower_height / 2.0, S1_WALL_HALF_THICKNESS, lower_height / 2.0)
    upper = Box(
        wall_x, (upper_start + arena_h) / 2.0, S1_WALL_HALF_THICKNESS,
        (arena_h - upper_start) / 2.0,
    )
    goal_dx = rng.uniform(-S1_GOAL_JITTER, S1_GOAL_JITTER)
    goal_dy = rng.uniform(-S1_GOAL_JITTER, S1_GOAL_JITTER)
    goal = Box(
        S1_GOAL_CENTER_X + goal_dx, gap_center + goal_dy, *S1_GOAL_HALF
    )
    return goal, (lower, upper)


def _sample_object_pose(
    family: str, mode: str, arena: Tuple[float, float], goal: Box,
    obstacles: Sequence[Box], rng: random.Random,
) -> Optional[Pose]:
    """One proposal for the object centre and angle under the mode's rule."""

    width, height = arena
    if mode == "broad":
        x = rng.uniform(0.0, width)
        y = rng.uniform(0.0, height)
        return Pose(x, y, rng.uniform(-math.pi, math.pi))

    if mode == "wall":
        x = rng.uniform(0.0, width)
        y = rng.uniform(0.0, height)
        angle = rng.uniform(-math.pi, math.pi)
        corners = object_corners(x, y, angle)
        if family == "S0":
            wall = rng.choice(("left", "right", "bottom", "top"))
            distance = boundary_wall_distance(corners, arena, wall)
            if not (WALL_CLEARANCE_RANGE[0] <= distance <= WALL_CLEARANCE_RANGE[1]):
                return None
            return Pose(x, y, angle)
        else:
            # A2 adjustment (documented in docs/coverage_scale_v1/errata.md). The first
            # implementation drew a uniform object centre and accepted it only if a randomly
            # chosen wall edge happened to be 0.01-0.20 m away. The wall is a thin box, so
            # most draws land inside or far from it and the S1 acceptance rate was 0.46%.
            # The proposal now constructs the object at a perpendicular offset u drawn from
            # the *same* [0.01, 0.20] m range from one of the box's four faces, on the free
            # side of that face; the draw is still accepted only if the exact
            # polygon-to-segment distance is inside the range and the state is strictly legal.
            # The range, the mode mixture and the legality rule are unchanged.
            box = rng.choice(list(obstacles))
            face = rng.randrange(4)
            offset = rng.uniform(*WALL_CLEARANCE_RANGE)
            angle = rng.uniform(-math.pi, math.pi)
            half_x, half_y = _half_extent(angle)
            if face == 0:      # left face, free side x < cx - half_w
                x = box.cx - box.half_w - offset - half_x
                y = rng.uniform(0.0, height)
            elif face == 1:    # right face, free side x > cx + half_w
                x = box.cx + box.half_w + offset + half_x
                y = rng.uniform(0.0, height)
            elif face == 2:    # bottom face, free side y < cy - half_h
                x = rng.uniform(0.0, width)
                y = box.cy - box.half_h - offset - half_y
            else:              # top face, free side y > cy + half_h
                x = rng.uniform(0.0, width)
                y = box.cy + box.half_h + offset + half_y
            corners = object_corners(x, y, angle)
            if face == 0:
                distance = polygon_segment_distance(
                    corners, box.cx - box.half_w, box.cy - box.half_h, box.cx - box.half_w, box.cy + box.half_h
                )
            elif face == 1:
                distance = polygon_segment_distance(
                    corners, box.cx + box.half_w, box.cy - box.half_h, box.cx + box.half_w, box.cy + box.half_h
                )
            elif face == 2:
                distance = polygon_segment_distance(
                    corners, box.cx - box.half_w, box.cy - box.half_h, box.cx + box.half_w, box.cy - box.half_h
                )
            else:
                distance = polygon_segment_distance(
                    corners, box.cx - box.half_w, box.cy + box.half_h, box.cx + box.half_w, box.cy + box.half_h
                )
            if not (WALL_CLEARANCE_RANGE[0] - 1e-12 <= distance <= WALL_CLEARANCE_RANGE[1] + 1e-12):
                return None
            return Pose(x, y, angle)

    # mode == "goal"
    x = goal.cx + rng.uniform(-GOAL_NEIGHBOURHOOD, GOAL_NEIGHBOURHOOD)
    y = goal.cy + rng.uniform(-GOAL_NEIGHBOURHOOD, GOAL_NEIGHBOURHOOD)
    angle = rng.uniform(-math.pi, math.pi)
    corners = object_corners(x, y, angle)
    if _signed_polygon_box(corners, goal) > GOAL_NEIGHBOURHOOD_DISTANCE:
        return None
    if inside_goal(x, y, angle, goal):
        return None
    return Pose(x, y, angle)


def _sample_robot_pose(
    arena: Tuple[float, float], object_pose: Pose, rng: random.Random
) -> Pose:
    """70% near a random object face, 30% uniform; the mode is drawn independently per robot."""

    width, height = arena
    if rng.random() < ROBOT_NEAR_OBJECT_PROBABILITY:
        corners = object_corners(object_pose.x, object_pose.y, object_pose.angle)
        edge = rng.randrange(4)
        a, b = corners[edge], corners[(edge + 1) % 4]
        t = rng.random()
        px = a[0] + t * (b[0] - a[0])
        py = a[1] + t * (b[1] - a[1])
        edge_x, edge_y = b[0] - a[0], b[1] - a[1]
        length = math.hypot(edge_x, edge_y)
        # Outward normal of a counter-clockwise polygon: rotate the edge by -90 degrees.
        normal = (edge_y / length, -edge_x / length)
        gap = rng.uniform(*ROBOT_GAP_RANGE)
        return Pose(
            px + normal[0] * (ROBOT_RADIUS + gap),
            py + normal[1] * (ROBOT_RADIUS + gap),
            rng.uniform(-math.pi, math.pi),
        )
    return Pose(rng.uniform(0.0, width), rng.uniform(0.0, height), rng.uniform(-math.pi, math.pi))


def build_scene(family: str, mode: str, split: str, seed: int, rng: random.Random) -> Optional[SceneSpec]:
    """One candidate scene; ``None`` if the proposal is illegal and must be redrawn."""

    arena = S0_ARENA if family == "S0" else S1_ARENA
    goal, obstacles = sample_scene_geometry(family, rng)
    object_pose = _sample_object_pose(family, mode, arena, goal, obstacles, rng)
    if object_pose is None:
        return None
    robots = (
        _sample_robot_pose(arena, object_pose, rng),
        _sample_robot_pose(arena, object_pose, rng),
    )
    scene = SceneSpec(
        name=scene_id_for(family, seed),
        category=family,
        variant=-1,
        arena=arena,
        obstacles=tuple(obstacles),
        goal=goal,
        object_pose=object_pose,
        robot_poses=robots,
        description=f"coverage_scale_v1 {family} {mode} root seed {seed} ({split})",
        scene_version=f"coverage-scale-{SAMPLER_VERSION}",
    )
    # Strict legality, frozen margin, no projection. ``strict_legal`` additionally rejects a
    # configuration in which an obstacle corner lies inside the object (see its docstring).
    if not strict_legal(scene, DEFAULT_MARGIN):
        return None
    if not legality_report(scene, DEFAULT_MARGIN)["legal"]:
        return None
    if inside_goal(object_pose.x, object_pose.y, object_pose.angle, goal):
        return None  # absorbing success at t=0 is not a legal start
    return scene


@dataclass
class RootSpec:
    """One accepted root: split, family, proposal mode, seed and the full scene."""

    root_id: str
    scene_id: str
    split: str
    family: str
    mode: str
    ordinal: int
    seed: int
    scene: SceneSpec
    attempts: int

    def as_metadata(self) -> Dict[str, object]:
        return {
            "root_id": self.root_id,
            "scene_id": self.scene_id,
            "split": self.split,
            "family": self.family,
            "mode": self.mode,
            "ordinal": self.ordinal,
            "seed": self.seed,
            "attempts": self.attempts,
            "sampler_version": SAMPLER_VERSION,
        }


def sample_root(
    split: str, family: str, mode: str, ordinal: int, *, max_attempts: int = MAX_ROOT_ATTEMPTS
) -> Optional[RootSpec]:
    """Draw one legal root for a cell, or ``None`` after ``max_attempts`` rejections."""

    seed = root_seed(split, family, mode, ordinal)
    rng = random.Random(seed)
    for attempt in range(1, max_attempts + 1):
        scene = build_scene(family, mode, split, seed, rng)
        if scene is not None:
            return RootSpec(
                root_id=f"{split}/{family}/{mode}/{ordinal}",
                scene_id=scene.name,
                split=split,
                family=family,
                mode=mode,
                ordinal=ordinal,
                seed=seed,
                scene=scene,
                attempts=attempt,
            )
    return None


# ---------------------------------------------------------------------------
# the deterministic, nested root plan
# ---------------------------------------------------------------------------

#: Mode slot cycle giving the frozen 50 / 25 / 25 mode weights.
MODE_SLOTS: Tuple[str, ...] = ("broad", "wall", "broad", "goal")


def sample_roots(split: str, count: int, *, max_seed_tries: int = 64) -> List[RootSpec]:
    """The first ``count`` roots of the split's plan, in the plan's own order.

    Six ``family x mode`` cells are filled round-robin (both families per mode slot), so a
    prefix of the plan is always a valid smaller plan and the mode mixture is 50/25/25 in the
    limit. Roots whose seed is rejected after ``max_attempts`` are recorded as BLOCKED and
    skipped to the next deterministic seed of the same cell; the tolerance is never relaxed.
    """

    counters: Dict[Tuple[str, str], int] = {(family, mode): 0 for family in FAMILIES for mode in MODES}
    accepted: List[RootSpec] = []
    slot = 0
    while len(accepted) < count:
        mode = MODE_SLOTS[slot % len(MODE_SLOTS)]
        families = FAMILIES if slot % 2 == 0 else tuple(reversed(FAMILIES))
        for family in families:
            if len(accepted) >= count:
                break
            attempt = 0
            while True:
                ordinal = counters[(family, mode)]
                counters[(family, mode)] = ordinal + 1
                attempt += 1
                root = sample_root(split, family, mode, ordinal)
                if root is not None:
                    accepted.append(root)
                    break
                if attempt >= max_seed_tries:
                    raise RuntimeError(
                        f"sampler blocked: {split}/{family}/{mode} rejected {attempt} consecutive seeds"
                    )
        slot += 1
    return accepted[:count]
