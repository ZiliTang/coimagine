"""Frozen contract types for the deterministic grid control experiment.

See ``docs/deterministic_grid_v1/CONTRACT.md``. Everything here is observable and exact:
no hidden parameters, no noise, no private feedback and no learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, Iterator, List, Tuple

Cell = Tuple[int, int]

GRID = 6
CELLS: Tuple[Cell, ...] = tuple((x, y) for y in range(GRID) for x in range(GRID))
CELL_INDEX: Dict[Cell, int] = {cell: index for index, cell in enumerate(CELLS)}
ANCHORS: Tuple[Cell, ...] = tuple((x, y) for y in range(GRID - 1) for x in range(GRID - 1))
ANCHOR_INDEX: Dict[Cell, int] = {anchor: index for index, anchor in enumerate(ANCHORS)}

N_ROBOTS = 2
N_ACTIONS_PER_ROBOT = 9
N_JOINT_ACTIONS = N_ACTIONS_PER_ROBOT * N_ACTIONS_PER_ROBOT  # 81

#: Fixed local-action order. Index into this tuple is the action id.
ACTION_NAMES: Tuple[str, ...] = (
    "move_N", "move_S", "move_E", "move_W",
    "push_N", "push_S", "push_E", "push_W",
    "wait",
)
MOVE_ACTIONS = (0, 1, 2, 3)
PUSH_ACTIONS = (4, 5, 6, 7)
WAIT_ACTION = 8
DIRECTION: Dict[int, Cell] = {
    0: (0, 1), 1: (0, -1), 2: (1, 0), 3: (-1, 0),   # move N/S/E/W
    4: (0, 1), 5: (0, -1), 6: (1, 0), 7: (-1, 0),   # push N/S/E/W
}
#: delta -> action id, for building template candidates
MOVE_ACTION_BY_DELTA: Dict[Cell, int] = {DIRECTION[a]: a for a in MOVE_ACTIONS}
PUSH_ACTION_BY_DELTA: Dict[Cell, int] = {DIRECTION[a]: a for a in PUSH_ACTIONS}

DEFAULT_GOAL_ANCHOR: Cell = (4, 2)


def add(cell: Cell, delta: Cell) -> Cell:
    return (cell[0] + delta[0], cell[1] + delta[1])


def in_bounds(cell: Cell) -> bool:
    return 0 <= cell[0] < GRID and 0 <= cell[1] < GRID


def manhattan(a: Cell, b: Cell) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def object_cells(anchor: Cell) -> FrozenSet[Cell]:
    x, y = anchor
    return frozenset({(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)})


def decode_joint(joint_action: int) -> Tuple[int, int]:
    """Joint id ``a0 * 9 + a1`` (robot 0 major)."""

    if not 0 <= joint_action < N_JOINT_ACTIONS:
        raise ValueError(f"joint action out of range: {joint_action}")
    return divmod(joint_action, N_ACTIONS_PER_ROBOT)


def encode_joint(action0: int, action1: int) -> int:
    return action0 * N_ACTIONS_PER_ROBOT + action1


@dataclass(frozen=True)
class GridMap:
    name: str
    walls: FrozenSet[Cell]
    goal_anchor: Cell = DEFAULT_GOAL_ANCHOR

    @property
    def goal(self) -> FrozenSet[Cell]:
        x, y = self.goal_anchor
        return frozenset({(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)})

    @property
    def free_cells(self) -> FrozenSet[Cell]:
        return frozenset(cell for cell in CELLS if cell not in self.walls)

    def wall_vector(self) -> Tuple[float, ...]:
        return tuple(1.0 if cell in self.walls else 0.0 for cell in CELLS)

    def goal_vector(self) -> Tuple[float, ...]:
        goal = self.goal
        return tuple(1.0 if cell in goal else 0.0 for cell in CELLS)


@dataclass(frozen=True)
class GridState:
    robot0: Cell
    robot1: Cell
    object_anchor: Cell

    def robots(self) -> Tuple[Cell, Cell]:
        return (self.robot0, self.robot1)

    def key(self, map_name: str) -> str:
        return (
            f"{map_name}|r0={self.robot0[0]},{self.robot0[1]}"
            f"|r1={self.robot1[0]},{self.robot1[1]}"
            f"|o={self.object_anchor[0]},{self.object_anchor[1]}"
        )


#: Fixed initial state used only to check that each map has a deterministic witness.
WITNESS_INITIAL = GridState((0, 0), (0, 1), (1, 2))


def state_is_valid(state: GridState, grid_map: GridMap) -> bool:
    free = grid_map.free_cells
    obj = object_cells(state.object_anchor)
    if not obj <= free:
        return False
    positions = state.robots()
    if positions[0] == positions[1]:
        return False
    for position in positions:
        if position not in free or position in obj:
            return False
    return True


def is_goal(state: GridState, grid_map: GridMap) -> bool:
    return object_cells(state.object_anchor) <= grid_map.goal


def terminal_distance(state: GridState, grid_map: GridMap) -> float:
    """Max over the object's four cells of the Manhattan distance to the nearest goal cell."""

    if is_goal(state, grid_map):
        return 0.0
    goal = grid_map.goal
    return float(max(min(manhattan(cell, target) for target in goal) for cell in object_cells(state.object_anchor)))


def all_valid_states(grid_map: GridMap) -> List[GridState]:
    free = sorted(grid_map.free_cells)
    states: List[GridState] = []
    for anchor in ANCHORS:
        obj = object_cells(anchor)
        if not obj <= grid_map.free_cells:
            continue
        remaining = [cell for cell in free if cell not in obj]
        for first in remaining:
            for second in remaining:
                if first == second:
                    continue
                states.append(GridState(first, second, anchor))
    return states


def empty_map() -> GridMap:
    return GridMap("empty", frozenset(), DEFAULT_GOAL_ANCHOR)


def obstacle_map(obstacle: Cell = (3, 2)) -> GridMap:
    return GridMap("obstacle", frozenset({obstacle}), DEFAULT_GOAL_ANCHOR)


def default_maps() -> Tuple[GridMap, ...]:
    return (empty_map(), obstacle_map())
