"""Egocentric, occlusion-aware ray sensing and the per-robot information state.

This is the Stage 2 realisation of the project's sensing constraint:

* a robot knows the **prior static map** and its own pose (and, through the shared
  localisation stream, the teammate's pose);
* the **movable object** is only known where the robot's own rays hit it. Nothing
  about the object is broadcast.

The primitive mirrors the archived Gate A
``_old/gate_a/src/coimagine/envs/grid/environment.py::visible_cells_from_pose``
(Bresenham ray casting, first occupied cell terminates a ray). It is re-implemented
here rather than imported because ``_old/`` is off the package path, and because the
Stage 2 sensing channel needs an explicit noise model.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, FrozenSet, List, Mapping, Optional, Tuple

from .state import Cell, Scene, add, euclidean_sq

#: Observation symbols reported by the sensor.
KIND_FREE = "free"
KIND_WALL = "wall"
KIND_OBJECT = "object"
KIND_ROBOT = "robot"


def line_cells(start: Cell, end: Cell) -> Tuple[Cell, ...]:
    """Integer Bresenham line from ``start`` to ``end`` inclusive."""

    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    out: List[Cell] = [(x0, y0)]
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        out.append((x, y))
    return tuple(out)


@dataclass(frozen=True)
class OccupancyScan:
    """One robot's egocentric occupancy return.

    ``observed`` maps a visible cell to one of :data:`KIND_FREE`,
    :data:`KIND_WALL`, :data:`KIND_OBJECT`, :data:`KIND_ROBOT`. Cells absent from
    ``observed`` are outside the range or occluded, i.e. unknown.
    """

    robot_id: int
    origin: Cell
    radius: int
    observed: Mapping[Cell, str]

    @property
    def visible(self) -> FrozenSet[Cell]:
        return frozenset(self.observed)

    @property
    def hits(self) -> Mapping[Cell, str]:
        return {c: k for c, k in self.observed.items() if k != KIND_FREE}

    @property
    def observed_object_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for c, k in self.observed.items() if k == KIND_OBJECT)

    @property
    def observed_wall_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for c, k in self.observed.items() if k == KIND_WALL)

    def sensed_direction_labels(self) -> Mapping[Cell, str]:
        """Convenience for drawing: the blocking cell of each ray."""

        return dict(self.hits)


def _kind_of(scene: Scene, robot_id: int, cell: Cell) -> str:
    if scene.static.is_wall(cell):
        return KIND_WALL
    if cell in scene.obj.cells:
        return KIND_OBJECT
    for robot in scene.robots:
        if robot.robot_id != robot_id and robot.pose == cell:
            return KIND_ROBOT
    return KIND_FREE


def ray_cast(scene: Scene, robot_id: int, radius: int = 6) -> OccupancyScan:
    """Deterministic occlusion-aware lidar return for one robot.

    A ray stops at the first occupied cell; that cell is reported, and anything
    behind it is unknown.
    """

    if radius < 1:
        raise ValueError("radius must be positive")
    robot = scene.robot[robot_id]
    origin = robot.pose

    blockers: FrozenSet[Cell] = (
        scene.static.walls
        | scene.obj.cells
        | frozenset(r.pose for r in scene.robots if r.robot_id != robot_id)
    )

    observed: Dict[Cell, str] = {}
    radius_sq = radius * radius
    for cell in scene.static.cells():
        if euclidean_sq(origin, cell) > radius_sq:
            continue
        if cell == origin:
            observed[cell] = KIND_ROBOT
            continue
        line = line_cells(origin, cell)
        occluded = False
        for mid in line[1:-1]:
            if mid in blockers:
                occluded = True
                break
        if occluded:
            continue
        observed[cell] = _kind_of(scene, robot_id, cell)

    return OccupancyScan(robot_id, origin, radius, observed)


# --------------------------------------------------------------------------
# Noise model
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SensorNoise:
    """Independent per-cell sensing channel.

    This *is* the generative observation model for Stage 2; the belief filter uses
    the same parameters.

    * ``drop_prob`` -- a visible cell is not reported at all (missed return);
    * ``false_free_prob`` -- an occupied cell is reported as free;
    * ``false_occupied_prob`` -- a free cell is reported as a wall.
    """

    drop_prob: float = 0.0
    false_free_prob: float = 0.0
    false_occupied_prob: float = 0.0

    def __post_init__(self) -> None:
        for name in ("drop_prob", "false_free_prob", "false_occupied_prob"):
            value = getattr(self, name)
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")

    @property
    def is_noiseless(self) -> bool:
        return (
            self.drop_prob == 0.0
            and self.false_free_prob == 0.0
            and self.false_occupied_prob == 0.0
        )


NOISELESS = SensorNoise()


def apply_noise(
    scan: OccupancyScan, noise: SensorNoise, rng: random.Random
) -> OccupancyScan:
    """Sample a noisy realisation of a scan under :class:`SensorNoise`."""

    if noise.is_noiseless:
        return scan

    observed: Dict[Cell, str] = {}
    for cell, kind in scan.observed.items():
        if cell == scan.origin:
            observed[cell] = kind
            continue
        if rng.random() < noise.drop_prob:
            continue
        if kind in (KIND_WALL, KIND_OBJECT) and rng.random() < noise.false_free_prob:
            observed[cell] = KIND_FREE
            continue
        if kind == KIND_FREE and rng.random() < noise.false_occupied_prob:
            observed[cell] = KIND_WALL
            continue
        observed[cell] = kind

    return replace(scan, observed=observed)


# --------------------------------------------------------------------------
# Per-robot information state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotInformationState:
    """Everything one robot may legally use at a decision boundary.

    It contains the prior static map, both robot poses (shared localisation stream)
    and the robot's **own** merged occupancy observations. It contains no true
    object pose, no global occupancy and no teammate scan.
    """

    robot_id: int
    static: object  # StaticMap
    own_pose: Cell
    teammate_pose: Cell
    observed: Mapping[Cell, str] = field(default_factory=dict)
    step: int = 0

    @property
    def observed_object_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for c, k in self.observed.items() if k == KIND_OBJECT)

    @property
    def observed_free_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for c, k in self.observed.items() if k == KIND_FREE)

    def unknown_cells(self) -> FrozenSet[Cell]:
        """Non-wall cells the robot has never observed.

        The prior static map means walls are always known, even out of range; only
        non-wall cells can be unknown.
        """

        static = self.static
        return frozenset(
            c for c in static.cells() if not static.is_wall(c) and c not in self.observed
        )


def information_state(
    scene: Scene, scan: OccupancyScan, previous: Optional[RobotInformationState] = None
) -> RobotInformationState:
    """Merge a new scan into a robot's information state."""

    robot = scene.robot[scan.robot_id]
    teammate = next(r for r in scene.robots if r.robot_id != scan.robot_id)
    observed: Dict[Cell, str] = dict(previous.observed) if previous is not None else {}
    # A newer observation of a cell overwrites the older one.
    observed.update(scan.observed)

    return RobotInformationState(
        robot_id=scan.robot_id,
        static=scene.static,
        own_pose=robot.pose,
        teammate_pose=teammate.pose,
        observed=observed,
        step=(previous.step + 1 if previous is not None else 0),
    )
