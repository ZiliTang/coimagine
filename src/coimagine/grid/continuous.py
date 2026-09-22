"""Continuous contact parameters: contact point, push direction and force.

The discrete action template ("push east", "co-push east") makes the feasible set a
switch: an action either exists or it does not. This module replaces it with a
**continuous** parameterisation, so the question "what can this robot do next?" becomes
"for which (direction, force) is the push effective?", whose answer is a **region**.

Model
-----
A robot in contact pushes with

* ``angle`` in degrees, 0 = +x (east), 90 = +y (north);
* ``force`` -- a scalar, nominally in ``[0, 2]``.

The nearest axis direction is selected by the alignment ``cos`` of the angle with that
axis, and the push succeeds when

1. the selected direction is geometrically legal (the destination 2x2 block is free);
2. the object is not latched (``constrained`` and unreleased);
3. the combined effective force reaches the resistance:
   ``sum_i traction_i * force_i >= resistance``.

Consequences that matter for the research question:

* the feasible set in ``(angle, force)`` is a **rectangle**: an angular window around a
  free axis, crossed with a force threshold;
* **the force threshold depends on the hidden contact physics** (resistance and the
  robots' tractions), which is exactly the quantity a robot cannot observe and must infer;
* nothing here is told to the robot. The threshold has to be estimated from the feedback
  of real pushes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .state import Direction, HiddenPhysics, ObjectState, RobotState, Scene

#: Axis directions with their angles in degrees.
AXES: Tuple[Tuple[Direction, float], ...] = (
    (Direction.EAST, 0.0),
    (Direction.NORTH, 90.0),
    (Direction.WEST, 180.0),
    (Direction.SOUTH, 270.0),
)

#: Nominal force range. A single gripping robot needs ``force >= resistance``.
FORCE_MIN = 0.0
FORCE_MAX = 2.0


def alignment(angle_deg: float, axis_deg: float) -> float:
    """Cosine alignment between a push angle and an axis, clamped at 0."""

    delta = math.radians(angle_deg - axis_deg)
    return max(0.0, math.cos(delta))


def nearest_axis(angle_deg: float) -> Tuple[Direction, float]:
    """The axis direction closest to ``angle_deg`` and its alignment."""

    best_direction = Direction.EAST
    best_alignment = -1.0
    for direction, axis_deg in AXES:
        value = alignment(angle_deg, axis_deg)
        if value > best_alignment:
            best_alignment = value
            best_direction = direction
    return best_direction, best_alignment


@dataclass(frozen=True)
class PushOutcome:
    moved: bool
    direction: Direction
    alignment: float
    capacity: float
    required: float
    reason: str


def _destination_free(scene: Scene, cells) -> bool:
    from .dynamics import shortest_path  # noqa: F401  (kept for API symmetry)

    for cell in cells:
        if not scene.static.in_bounds(cell) or scene.static.is_wall(cell):
            return False
        if any(cell == robot.pose for robot in scene.robots):
            return False
    return True


def push_outcome(
    scene: Scene,
    forces: Dict[int, float],
    angle_deg: float,
) -> PushOutcome:
    """Outcome of a joint push at ``angle_deg`` with per-robot forces.

    ``forces`` maps robot id to the force it applies; robots absent from the dict hold.
    """

    direction, align = nearest_axis(angle_deg)
    physics: HiddenPhysics = scene.physics
    delta = direction.delta

    # Only robots actually in contact with the object contribute capacity.
    object_cells = ObjectState(scene.obj.origin).cells
    capacity = 0.0
    for robot in scene.robots:
        force = forces.get(robot.robot_id, 0.0)
        if force <= 0.0:
            continue
        contact = (robot.pose[0] + delta[0], robot.pose[1] + delta[1]) in object_cells
        if not contact:
            continue
        traction = 1.0 if physics.traction[robot.robot_id] == "grip" else 0.0
        capacity += traction * force

    required = float(physics.resistance)
    moved_cells = ObjectState(scene.obj.origin).translated(delta).cells

    if physics.constraint == "constrained" and not scene.released:
        return PushOutcome(False, direction, align, capacity, required, "latched")
    if not _destination_free(scene, moved_cells):
        return PushOutcome(False, direction, align, capacity, required, "blocked")
    if align <= 0.0:
        return PushOutcome(False, direction, align, capacity, required, "no_direction")
    if capacity < required:
        return PushOutcome(False, direction, align, capacity, required, "insufficient_force")
    return PushOutcome(True, direction, align, capacity, required, "moved")


def feasible_mask(
    scene: Scene,
    forces: Dict[int, float],
    angles: Sequence[float],
) -> List[bool]:
    """Ground-truth feasibility over a grid of push angles."""

    return [push_outcome(scene, forces, angle).moved for angle in angles]


def force_threshold(scene: Scene, robot_ids: Sequence[int]) -> float:
    """Minimum per-robot force for a push by ``robot_ids`` to move the object.

    This is the number a robot has to *infer*: it depends on the resistance and on which
    of the pushing robots actually have traction.
    """

    physics: HiddenPhysics = scene.physics
    traction = sum(
        1.0 if physics.traction[i] == "grip" else 0.0 for i in robot_ids
    )
    if traction <= 0.0:
        return float("inf")
    return physics.resistance / traction


def angular_window_degrees() -> float:
    """Half-width of the angular window: pushing within +-45 degrees of a free axis.

    Outside that the nearest axis changes, so the window is exactly the Voronoi cell of
    the axis on the circle.
    """

    return 45.0
