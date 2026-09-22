from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import FrozenSet, Mapping, Optional

from coimagine.core.types import Cell, Direction, EntityId, RobotId, TaskKind


@dataclass(frozen=True)
class RobotState:
    robot_id: RobotId
    position: Cell


@dataclass(frozen=True)
class BlockerState:
    entity_id: EntityId
    cells: FrozenSet[Cell]
    required_pushers: int = 1
    mass_class: str = "light"
    task_weight: float = 1.0
    allowed_push_directions: FrozenSet[Direction] = frozenset(Direction)
    display_name: Optional[str] = None


@dataclass(frozen=True)
class TaskSpec:
    kind: TaskKind = TaskKind.CLEARING
    corridor: FrozenSet[Cell] = frozenset()
    initial_blocked_cells: int = 0
    goal_region: FrozenSet[Cell] = frozenset()
    transport_entity_id: Optional[EntityId] = None
    payload_robot_id: Optional[RobotId] = None
    hold_cells: FrozenSet[Cell] = frozenset()
    guarded_passage: FrozenSet[Cell] = frozenset()
    observation_cells: FrozenSet[Cell] = frozenset()
    hidden_entities: FrozenSet[EntityId] = frozenset()
    resource_capacities: Mapping[str, int] = field(default_factory=dict)
    resource_regions: Mapping[str, FrozenSet[Cell]] = field(default_factory=dict)
    activation_dependencies: Mapping[EntityId, EntityId] = field(default_factory=dict)
    initial_blocked_weight: float = 0.0
    metadata: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class GridState:
    width: int
    height: int
    walls: FrozenSet[Cell]
    robots: Mapping[RobotId, RobotState]
    blockers: Mapping[EntityId, BlockerState]
    task: TaskSpec
    revealed_entities: FrozenSet[EntityId] = frozenset()
    step_index: int = 0

    def clone(self) -> "GridState":
        return GridState(
            width=self.width,
            height=self.height,
            walls=frozenset(self.walls),
            robots=dict(self.robots),
            blockers=dict(self.blockers),
            task=self.task,
            revealed_entities=frozenset(self.revealed_entities),
            step_index=self.step_index,
        )

    def with_updates(
        self,
        *,
        robots: Mapping[RobotId, RobotState] | None = None,
        blockers: Mapping[EntityId, BlockerState] | None = None,
        revealed_entities: FrozenSet[EntityId] | None = None,
    ) -> "GridState":
        return replace(
            self,
            robots=dict(robots or self.robots),
            blockers=dict(blockers or self.blockers),
            revealed_entities=(
                frozenset(revealed_entities)
                if revealed_entities is not None
                else self.revealed_entities
            ),
            step_index=self.step_index + 1,
        )
