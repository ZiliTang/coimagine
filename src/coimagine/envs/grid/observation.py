from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Mapping

from coimagine.core.types import Cell, Direction, EntityId, RobotId
from coimagine.envs.grid.state import TaskSpec


@dataclass(frozen=True)
class ObservedBlocker:
    entity_id: EntityId
    # The controlled benchmark assumes a recognized object template once any part
    # of the entity is observed. ``cells`` is that known footprint; only
    # ``visible_cells`` may ground current contact options.
    cells: FrozenSet[Cell]
    visible_cells: FrozenSet[Cell]
    required_pushers: int
    mass_class: str = "light"
    task_weight: float = 1.0
    allowed_push_directions: FrozenSet[Direction] = frozenset(Direction)
    display_name: str | None = None


@dataclass(frozen=True)
class LocalObservation:
    robot_id: RobotId
    own_position: Cell
    visible_cells: FrozenSet[Cell]
    walls: FrozenSet[Cell]
    free_cells: FrozenSet[Cell]
    blockers: Mapping[EntityId, ObservedBlocker]
    robots: Mapping[RobotId, Cell]
    task: TaskSpec
    revealed_entities: FrozenSet[EntityId]
