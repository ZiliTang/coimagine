"""The unified task score used by the planner, the random control and the data export.

The first version uses one explicit lexicographic ordering key, deliberately kept free of
tuned reward terms:

1. success first;
2. among successes, earlier completion time, then less control usage;
3. among failures, the object's geometric distance to the goal, then terminal orientation
   error, then control usage;
4. exact ties break on the stable ``candidate_id``.

The task does not specify a terminal orientation, so the orientation term is **removed**
rather than invented; :func:`orientation_error` exists, is tested against hand-worked
examples, and returns a term that is identically zero this round. The distance and angle
functions have hand-checked examples in ``tests/test_continuous2d.py``.

This key is an engineering baseline. A later selection-bias experiment that needs one
scalar ``J`` must freeze its own scoring contract; the lexicographic key must not simply be
subtracted.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from .scenes import Box

Vec = Tuple[float, float]


def distance_to_box(corners: Sequence[Vec], goal: Box) -> float:
    """Maximum distance from an object corner to the goal rectangle (0 if all inside).

    A corner inside the rectangle contributes 0; the reported number is the largest
    overhang, which is the quantity the planner should minimise once success fails.
    """

    worst = 0.0
    for x, y in corners:
        dx = max(0.0, abs(x - goal.cx) - goal.half_w)
        dy = max(0.0, abs(y - goal.cy) - goal.half_h)
        worst = max(worst, math.hypot(dx, dy))
    return worst


def centre_distance_to_box(centre: Vec, goal: Box) -> float:
    dx = max(0.0, abs(centre[0] - goal.cx) - goal.half_w)
    dy = max(0.0, abs(centre[1] - goal.cy) - goal.half_h)
    return math.hypot(dx, dy)


def angle_difference(a: float, b: float) -> float:
    """Absolute wrapped angular difference in radians, in ``[0, pi]``."""

    difference = (a - b + math.pi) % (2.0 * math.pi) - math.pi
    return abs(difference)


def extent_across_travel(angle: float, object_size: Tuple[float, float]) -> float:
    """The object's projected width perpendicular to the +x travel direction.

    A rectangle of half-extents ``(a, b)`` rotated by ``angle`` projects onto the y axis
    with full width ``2 * (a |sin| + b |cos|)``. It fits a gap when this is below the gap
    width, which is what makes an orientation change worth a preparation window.
    """

    a = object_size[0] / 2.0
    b = object_size[1] / 2.0
    return 2.0 * (a * abs(math.sin(angle)) + b * abs(math.cos(angle)))


def passage_cost_to_go(
    object_pose: Tuple[float, float, float],
    object_size: Tuple[float, float],
    goal: Box,
    obstacles: Sequence[Box],
    corners: Sequence[Vec],
) -> float:
    """Geometric cost-to-go through the passage, used only in the failure branch.

    This is a distance in metres, built entirely from public geometry. For every vertical
    wall strictly between the object and the goal it adds: how far the object still is from
    the wall, how far it is from the wall's opening, and -- when the object is too wide to
    fit through the opening at its **current orientation** -- the excess width. Rotating an
    over-wide object therefore reduces the number, which is the geometric statement that a
    preparation window is worth taking. Once the object is east of the wall the plain
    corner distance to the goal is used.

    It is a heuristic for the failure branch only: the success branch, and the actual
    return, still come from the real simulator reaching the goal region and holding there.
    """

    x, y, angle = object_pose
    base = distance_to_box(corners, goal)
    if not obstacles:
        return base
    # Group obstacle boxes into vertical walls by their x position.
    walls: dict = {}
    for box in obstacles:
        if box.angle != 0.0 or box.half_w > 0.3:
            continue
        walls.setdefault(round(box.cx, 3), []).append(box)

    for wall_x, boxes in sorted(walls.items()):
        if not (x < wall_x < goal.cx):
            continue
        boxes = sorted(boxes, key=lambda box: box.cy)
        opening: float | None = None
        gap_center = 0.0
        if len(boxes) >= 2:
            lower, upper = boxes[0], boxes[-1]
            low = lower.cy + lower.half_h
            high = upper.cy - upper.half_h
            if high > low:
                opening = high - low
                gap_center = 0.5 * (high + low)
        if opening is None:
            continue
        deficit = max(0.0, extent_across_travel(angle, object_size) - (opening - 0.02))
        return deficit + abs(wall_x - x) + abs(y - gap_center) + abs(goal.cx - wall_x)
    return base


def orientation_error(angle: float, target_angle: float | None) -> float:
    """Terminal orientation error; identically zero when no target orientation is given."""

    if target_angle is None:
        return 0.0
    return angle_difference(angle, target_angle)


def ordering_key(
    *,
    success: bool,
    success_time: float,
    control_usage: float,
    terminal_distance: float,
    orientation: float,
    candidate_id: int,
) -> Tuple[float, ...]:
    """Larger is better, so the planner maximises this tuple lexicographically."""

    if success:
        return (1.0, -success_time, -control_usage, 0.0, 0.0, 0.0, -float(candidate_id))
    return (
        0.0,
        0.0,
        0.0,
        -terminal_distance,
        -orientation,
        -control_usage,
        -float(candidate_id),
    )
