"""Frozen constants and the shared scalar score for the forecast-baseline experiment.

Version: ``forecast_baseline_v1`` (see ``docs/forecast_baseline_v1/CONTRACT.md``).

This module is deliberately the only place that defines:

* the deterministic normalisation constants used by the learned model;
* the new scalar score ``J = 10*S - t/24 - d/L - 20*F``;
* the geometric validity test for a *predicted* physical state.

The score is shared by the exact reference and the learned evaluation, so the two cannot
drift apart. The old lexicographic key in ``continuous2d.scoring.ordering_key`` is left
alone and is not used by anything here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

from ..continuous2d.config import PhysicsConfig
from ..continuous2d.scenes import Box, SceneSpec
from ..continuous2d.scoring import distance_to_box, passage_cost_to_go

Vec = Tuple[float, float]

SCORE_VERSION = "forecast-scalar-v1"

#: Human-readable termination categories of one window, in the fixed class order.
TERMINATION_CLASSES: Tuple[str, ...] = ("window_complete", "success", "physical_failure")
TERMINATION_INDEX: Dict[str, int] = {name: index for index, name in enumerate(TERMINATION_CLASSES)}

#: Raw engine terminations that collapse into ``physical_failure``. The raw category is
#: still stored in every data record.
PHYSICAL_FAILURES: Tuple[str, ...] = ("numeric_failure", "out_of_bounds", "penetration_limit")

#: Deterministic normalisation constants (never fitted from data).
VELOCITY_SCALE = 2.0          # m/s
ANGULAR_VELOCITY_SCALE = 2.0  # rad/s
HOLD_SCALE = 0.55             # s, success_hold_time + one control period
DURATION_SCALE = 2.0          # s, window_steps * control_dt
MAX_OBSTACLE_SLOTS = 2

#: Thresholds copied from the frozen physics configuration; used only to decide whether a
#: predicted state is an ``invalid_prediction``.
OUT_OF_BOUNDS_MARGIN = 0.5
PENETRATION_LIMIT = 0.05

#: Score coefficients (contract C1).
SUCCESS_REWARD = 10.0
TIME_DIVISOR = 24.0
FAILURE_PENALTY = 20.0

#: A4 fixed tolerances for the continuous cumulative-error metric. A chain counts as "in
#: error" at a window when either body exceeds the position tolerance or the object exceeds
#: the angle tolerance; wrong termination and invalid prediction are counted separately.
CHAIN_POSITION_TOLERANCE = 0.05   # m
CHAIN_ANGLE_TOLERANCE = 0.10      # rad

#: Grouping thresholds for the dataset coverage report only (contract B3).
TRANSLATION_THRESHOLD = 0.01   # m
ROTATION_THRESHOLD = 0.02      # rad

GM = "window_complete"
SUCCESS = "success"
FAILURE = "physical_failure"


def termination_class(raw: str) -> str:
    """Collapse a raw engine termination into the three model classes."""

    if raw == SUCCESS:
        return SUCCESS
    if raw in PHYSICAL_FAILURES:
        return FAILURE
    return GM


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def arena_diagonal(arena: Sequence[float]) -> float:
    return math.hypot(float(arena[0]), float(arena[1]))


@dataclass(frozen=True)
class ScoreComponents:
    """The four additive terms of ``J`` plus the total (contract C1)."""

    success: int
    time: float
    distance: float
    failure: int
    diagonal: float

    @property
    def success_reward(self) -> float:
        return SUCCESS_REWARD * self.success

    @property
    def time_penalty(self) -> float:
        return self.time / TIME_DIVISOR

    @property
    def distance_penalty(self) -> float:
        return self.distance / self.diagonal

    @property
    def failure_penalty(self) -> float:
        return FAILURE_PENALTY * self.failure

    @property
    def j(self) -> float:
        return (
            self.success_reward
            - self.time_penalty
            - self.distance_penalty
            - self.failure_penalty
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "success": self.success,
            "time": self.time,
            "distance": self.distance,
            "failure": self.failure,
            "diagonal": self.diagonal,
            "success_reward": self.success_reward,
            "time_penalty": self.time_penalty,
            "distance_penalty": self.distance_penalty,
            "failure_penalty": self.failure_penalty,
            "J": self.j,
            "score_version": SCORE_VERSION,
        }


def score_components(
    *,
    success: bool,
    time: float,
    distance: float,
    failure: bool,
    diagonal: float,
) -> ScoreComponents:
    """The one scoring entry shared by exact and learned evaluation.

    ``distance`` must already be 0 on success; this is asserted rather than silently
    corrected because a non-zero success distance means the caller forgot the absorption
    rule.
    """

    if success and distance != 0.0:
        raise ValueError("distance must be 0 on success (absorption, contract C1)")
    return ScoreComponents(
        success=int(bool(success)),
        time=float(time),
        distance=float(distance),
        failure=int(bool(failure)),
        diagonal=float(diagonal),
    )


def geometric_distance(corners: Sequence[Vec], goal: Box) -> float:
    """Plain corner distance to the goal, reported independently of ``passage_cost_to_go``."""

    return distance_to_box(corners, goal)


def passage_distance(
    object_pose: Tuple[float, float, float],
    object_size: Tuple[float, float],
    goal: Box,
    obstacles: Sequence[Box],
    corners: Sequence[Vec],
) -> float:
    return passage_cost_to_go(object_pose, object_size, goal, obstacles, corners)


def state_corners(boundary_state, object_size: Tuple[float, float]) -> Tuple[Vec, ...]:
    """Object corners of a boundary state, for geometry computed from a predicted state."""

    return _object_corners(
        (boundary_state.object.x, boundary_state.object.y, boundary_state.object.angle),
        object_size,
    )


# ---------------------------------------------------------------------------
# Geometric validity of a predicted state
# ---------------------------------------------------------------------------


def _object_corners(pose: Tuple[float, float, float], size: Tuple[float, float]) -> Tuple[Vec, ...]:
    x, y, angle = pose
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    cos, sin = math.cos(angle), math.sin(angle)
    return tuple(
        (
            x + dx * cos - dy * sin,
            y + dx * sin + dy * cos,
        )
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    )


def _box_corners(box: Box) -> Tuple[Vec, ...]:
    cos, sin = math.cos(box.angle), math.sin(box.angle)
    return tuple(
        (
            box.cx + dx * cos - dy * sin,
            box.cy + dx * sin + dy * cos,
        )
        for dx, dy in (
            (-box.half_w, -box.half_h), (box.half_w, -box.half_h),
            (box.half_w, box.half_h), (-box.half_w, box.half_h),
        )
    )


def _point_segment_distance(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq < 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _signed_circle_box(cx: float, cy: float, radius: float, box: Box) -> float:
    """Signed clearance between a circle and a (possibly rotated) box; negative = overlap."""

    cos, sin = math.cos(-box.angle), math.sin(-box.angle)
    dx, dy = cx - box.cx, cy - box.cy
    lx = dx * cos - dy * sin
    ly = dx * sin + dy * cos
    clamped_x = max(-box.half_w, min(box.half_w, lx))
    clamped_y = max(-box.half_h, min(box.half_h, ly))
    outside = math.hypot(lx - clamped_x, ly - clamped_y)
    if outside > 1e-12:
        return outside - radius
    # Centre inside the box: penetration is the distance to the nearest edge plus the radius.
    depth = min(box.half_w - abs(lx), box.half_h - abs(ly))
    return -(depth + radius)


def _signed_circle_polygon(cx: float, cy: float, radius: float, corners: Sequence[Vec]) -> float:
    """Signed clearance between a circle and a convex polygon; negative = overlap."""

    inside = True
    sign = 0
    for index, corner in enumerate(corners):
        nxt = corners[(index + 1) % len(corners)]
        cross = (nxt[0] - corner[0]) * (cy - corner[1]) - (nxt[1] - corner[1]) * (cx - corner[0])
        if abs(cross) < 1e-12:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            inside = False
            break
    distances = [
        _point_segment_distance(cx, cy, corners[i][0], corners[i][1], corners[(i + 1) % len(corners)][0], corners[(i + 1) % len(corners)][1])
        for i in range(len(corners))
    ]
    nearest = min(distances)
    if inside:
        return -(nearest + radius)
    return nearest - radius


def _signed_polygon_polygon(first: Sequence[Vec], second: Sequence[Vec]) -> float:
    """Minimum translation depth between two convex polygons; negative = overlap.

    Separating-axis test over both polygons' edge normals. When no axis separates them the
    overlap is the smallest per-axis overlap, i.e. the penetration depth.
    """

    def axes(polygon: Sequence[Vec]):
        for index, corner in enumerate(polygon):
            nxt = polygon[(index + 1) % len(polygon)]
            edge = (nxt[0] - corner[0], nxt[1] - corner[1])
            length = math.hypot(edge[0], edge[1])
            if length > 1e-12:
                yield (-edge[1] / length, edge[0] / length)

    best_separation = 0.0
    min_overlap = float("inf")
    separated = False
    for axis in list(axes(first)) + list(axes(second)):
        first_proj = [point[0] * axis[0] + point[1] * axis[1] for point in first]
        second_proj = [point[0] * axis[0] + point[1] * axis[1] for point in second]
        overlap = min(max(first_proj), max(second_proj)) - max(min(first_proj), min(second_proj))
        if overlap < 0.0:
            separated = True
            best_separation = max(best_separation, -overlap)
        else:
            min_overlap = min(min_overlap, overlap)
    if separated:
        return best_separation
    return -min_overlap


def _boundary_clearance(point: Vec, arena: Sequence[float], capsule_radius: float) -> float:
    width, height = float(arena[0]), float(arena[1])
    x, y = point
    return min(
        x - capsule_radius,
        width - capsule_radius - x,
        y - capsule_radius,
        height - capsule_radius - y,
    )


def geometric_violation(
    object_state,
    robot_states,
    scene: SceneSpec,
    physics: PhysicsConfig,
) -> Dict[str, object]:
    """Is a *predicted* window-boundary state geometrically admissible?

    Returns a dictionary with ``finite``, ``in_bounds``, ``penetration`` (maximum overlap
    depth in metres) and ``violation`` (whether it exceeds the frozen failure thresholds).
    This only reads the predicted state and the known static geometry; it never consults the
    true successor.
    """

    capsule = 0.05
    values = [object_state.as_tuple(), *(state.as_tuple() for state in robot_states)]
    finite = all(math.isfinite(value) for row in values for value in row)
    if not finite:
        return {
            "finite": False, "in_bounds": False, "penetration": float("inf"),
            "violation": True, "reasons": ["non_finite"],
        }

    width, height = scene.arena
    margin = OUT_OF_BOUNDS_MARGIN
    # A3 fix: x is bounded by the arena width and y by the arena height. The previous version
    # compared both coordinates against the width, which silently widened the y failure
    # range and disagreed with PymunkBackend.in_bounds in the frozen environment.
    in_bounds = all(
        -margin <= state.x <= width + margin and -margin <= state.y <= height + margin
        for state in (object_state, *robot_states)
    )
    corners = _object_corners(
        (object_state.x, object_state.y, object_state.angle),
        (physics.object_width, physics.object_height),
    )
    in_bounds = in_bounds and all(
        -margin <= corner[0] <= width + margin and -margin <= corner[1] <= height + margin
        for corner in corners
    )

    # B1: penetration is kept per collision class so a total can be attributed to a pair.
    by_pair: Dict[str, float] = {
        "robot_object": 0.0, "robot_robot": 0.0, "robot_wall": 0.0, "object_wall": 0.0,
    }
    radius = physics.robot_radius
    for robot in robot_states:
        by_pair["robot_wall"] = max(
            by_pair["robot_wall"],
            radius - _boundary_clearance((robot.x, robot.y), scene.arena, capsule),
        )
        for box in scene.obstacles:
            by_pair["robot_wall"] = max(
                by_pair["robot_wall"], -_signed_circle_box(robot.x, robot.y, radius, box)
            )
        by_pair["robot_object"] = max(
            by_pair["robot_object"], -_signed_circle_polygon(robot.x, robot.y, radius, corners)
        )
    distance = math.hypot(robot_states[0].x - robot_states[1].x, robot_states[0].y - robot_states[1].y)
    by_pair["robot_robot"] = 2.0 * radius - distance
    for corner in corners:
        by_pair["object_wall"] = max(
            by_pair["object_wall"], -_boundary_clearance(corner, scene.arena, capsule)
        )
    for box in scene.obstacles:
        by_pair["object_wall"] = max(
            by_pair["object_wall"], -_signed_polygon_polygon(corners, _box_corners(box))
        )
    by_pair = {name: float(max(0.0, value)) for name, value in by_pair.items()}
    penetration = max([0.0, *by_pair.values()])

    reasons = []
    if not in_bounds:
        reasons.append("out_of_bounds")
    if penetration > PENETRATION_LIMIT:
        reasons.append("penetration_limit")
    return {
        "finite": True,
        "in_bounds": in_bounds,
        "penetration": float(penetration),
        "penetration_by_pair": by_pair,
        "violation": bool(reasons),
        "reasons": reasons,
        "violating_pairs": sorted(name for name, value in by_pair.items() if value > PENETRATION_LIMIT),
    }
