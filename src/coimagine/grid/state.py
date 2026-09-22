"""Discrete 2D grid world used by the Stage 2 cooperative-pushing task.

Design constraints (``docs/2d_protocol_spec.md`` section 9):

* the **static map** (walls, obstacles, goal region) is a prior known to every robot;
* the **movable object** is not broadcast; a robot may only estimate its pose from
  its own egocentric sensing;
* each robot knows its own pose (own global localization). Both robot poses are
  treated as public, justified by a shared localisation stream; only the object is
  private.

Coordinates are integer cells ``(x, y)`` with ``x`` the column and ``y`` the row.
The drawing helpers in :mod:`coimagine.grid.viz` flip ``y`` for display.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Iterator, Mapping, Optional, Tuple

Cell = Tuple[int, int]


class Direction(str, Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    @property
    def delta(self) -> Cell:
        return {
            Direction.NORTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, -1),
            Direction.WEST: (-1, 0),
        }[self]

    @property
    def label(self) -> str:
        return {"north": "N", "east": "E", "south": "S", "west": "W"}[self.value]

    def rotated(self, steps: int) -> "Direction":
        order = (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST)
        return order[(order.index(self) + steps) % 4]

    def opposite(self) -> "Direction":
        return self.rotated(2)


def add(cell: Cell, delta: Cell) -> Cell:
    return (cell[0] + delta[0], cell[1] + delta[1])


def chebyshev(a: Cell, b: Cell) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def euclidean_sq(a: Cell, b: Cell) -> int:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


# --------------------------------------------------------------------------
# Static map
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StaticMap:
    """The prior map every robot knows: bounds, walls and the goal region."""

    width: int
    height: int
    walls: FrozenSet[Cell] = frozenset()
    target: FrozenSet[Cell] = frozenset()
    name: str = "map"

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("The grid must be at least 2x2.")
        for cell in itertools.chain(self.walls, self.target):
            if not self.in_bounds(cell):
                raise ValueError(f"Cell {cell} is out of bounds for {self.width}x{self.height}.")
        if self.walls & self.target:
            raise ValueError("The goal region cannot overlap a wall.")

    def in_bounds(self, cell: Cell) -> bool:
        return 0 <= cell[0] < self.width and 0 <= cell[1] < self.height

    def is_wall(self, cell: Cell) -> bool:
        return cell in self.walls

    def cells(self) -> Iterator[Cell]:
        for y in range(self.height):
            for x in range(self.width):
                yield (x, y)

    def free_cells(self) -> FrozenSet[Cell]:
        return frozenset(c for c in self.cells() if c not in self.walls)


# --------------------------------------------------------------------------
# Object and robots
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectState:
    """A rigid rectangular object at a discrete origin (translation only)."""

    origin: Cell
    width: int = 2
    height: int = 2

    @property
    def cells(self) -> FrozenSet[Cell]:
        return frozenset(
            (self.origin[0] + dx, self.origin[1] + dy)
            for dx in range(self.width)
            for dy in range(self.height)
        )

    def translated(self, delta: Cell) -> "ObjectState":
        return ObjectState(add(self.origin, delta), self.width, self.height)

    def within(self, region: FrozenSet[Cell]) -> bool:
        return self.cells <= region


@dataclass(frozen=True)
class RobotState:
    robot_id: int
    pose: Cell
    heading: Optional[Direction] = None


@dataclass(frozen=True)
class Scene:
    """A fully observed simulator world.

    This object is privileged: it holds the true object pose. Deployment code must
    receive a :class:`~coimagine.grid.sensing.RobotInformationState` instead.
    """

    static: StaticMap
    obj: ObjectState
    robots: Tuple[RobotState, ...]
    hidden: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.robots) != 2:
            raise ValueError("Stage 2 is fixed at two robots.")
        if len({r.robot_id for r in self.robots}) != 2:
            raise ValueError("Robot ids must be unique.")
        for robot in self.robots:
            if not self.static.in_bounds(robot.pose):
                raise ValueError(f"Robot {robot.robot_id} is out of bounds.")
            if self.static.is_wall(robot.pose):
                raise ValueError(f"Robot {robot.robot_id} starts inside a wall.")
            if robot.pose in self.obj.cells:
                raise ValueError(f"Robot {robot.robot_id} starts inside the object.")
        for cell in self.obj.cells:
            if not self.static.in_bounds(cell):
                raise ValueError("The object sticks out of the map.")
            if self.static.is_wall(cell):
                raise ValueError("The object overlaps a wall.")

    @property
    def robot(self) -> Mapping[int, RobotState]:
        return {r.robot_id: r for r in self.robots}

    def is_goal(self) -> bool:
        return self.obj.within(self.static.target)

    def occupied_cells(self) -> FrozenSet[Cell]:
        return self.obj.cells | frozenset(r.pose for r in self.robots)


# --------------------------------------------------------------------------
# Hidden physics
# --------------------------------------------------------------------------

#: Resistance thresholds map onto the coalition capacity model: a single robot has
#: capacity 1, two robots pushing the same direction have capacity 2. A hidden
#: resistance of 3 is therefore never satisfiable with two robots and is excluded.
RESISTANCE_VALUES: Tuple[int, ...] = (1, 2)
CONSTRAINT_VALUES: Tuple[str, ...] = ("free", "constrained")
TRACTION_VALUES: Tuple[str, ...] = ("grip", "slip")


@dataclass(frozen=True)
class HiddenPhysics:
    """Hidden, stationary contact conditions of one episode."""

    resistance: int = 1
    constraint: str = "free"
    traction: Tuple[str, str] = ("grip", "grip")

    def __post_init__(self) -> None:
        if self.resistance not in RESISTANCE_VALUES:
            raise ValueError(f"resistance must be one of {RESISTANCE_VALUES}")
        if self.constraint not in CONSTRAINT_VALUES:
            raise ValueError(f"constraint must be one of {CONSTRAINT_VALUES}")
        if len(self.traction) != 2 or any(t not in TRACTION_VALUES for t in self.traction):
            raise ValueError(f"traction must be two values from {TRACTION_VALUES}")

    def effective_capacity(self, pushers: Tuple[int, ...]) -> int:
        """Coalition capacity after removing robots with no grip."""

        gripping = [r for r in pushers if self.traction[r] == "grip"]
        return len(gripping)

    @property
    def label(self) -> str:
        return (
            f"r={self.resistance}, {self.constraint}, "
            f"traction=({self.traction[0]},{self.traction[1]})"
        )


def enumerate_hidden_physics() -> Tuple[HiddenPhysics, ...]:
    """The full hidden-type space: 2 resistance x 2 constraint x 2 x 2 traction."""

    return tuple(
        HiddenPhysics(resistance=r, constraint=c, traction=(ta, tb))
        for r in RESISTANCE_VALUES
        for c in CONSTRAINT_VALUES
        for ta in TRACTION_VALUES
        for tb in TRACTION_VALUES
    )
