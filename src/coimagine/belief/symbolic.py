from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Dict, Mapping

from coimagine.core.types import Cell, Direction, EntityId, JointOption, OptionKind, RobotId
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.observation import LocalObservation
from coimagine.envs.grid.state import BlockerState, GridState, RobotState, TaskSpec


class CellKnowledge(str, Enum):
    UNKNOWN = "unknown"
    FREE = "free"
    WALL = "wall"


@dataclass(frozen=True)
class BelievedBlocker:
    entity_id: EntityId
    # Recognized object footprint in the controlled 2-D benchmark.
    cells: frozenset[Cell]
    # Surface cells directly supported by the current real or imagined observation.
    visible_cells: frozenset[Cell]
    required_pushers: int
    last_seen_step: int
    mass_class: str = "light"
    task_weight: float = 1.0
    allowed_push_directions: frozenset[Direction] = frozenset(Direction)
    display_name: str | None = None
    certain: bool = True


@dataclass(frozen=True)
class SymbolicBelief:
    """Agent-owned symbolic information state for the controlled grid domain."""

    robot_id: RobotId
    width: int
    height: int
    own_position: Cell
    task: TaskSpec
    # Persistent local memory. It is never populated from a global map.
    cells: Mapping[Cell, CellKnowledge] = field(default_factory=dict)
    blockers: Mapping[EntityId, BelievedBlocker] = field(default_factory=dict)
    teammate_positions: Mapping[RobotId, Cell] = field(default_factory=dict)
    revealed_entities: frozenset[EntityId] = frozenset()
    # Current observation support used by the option generator.
    current_visible_cells: frozenset[Cell] = frozenset()
    current_free_cells: frozenset[Cell] = frozenset()
    current_walls: frozenset[Cell] = frozenset()
    step_index: int = 0
    source: str = "real_observation"

    def cell_knowledge(self, cell: Cell) -> CellKnowledge:
        return self.cells.get(cell, CellKnowledge.UNKNOWN)


def _freeze_cache_value(value: object) -> object:
    """Convert nested configuration values into a deterministic hashable form."""

    if isinstance(value, Mapping):
        return tuple(
            sorted(
                (str(key), _freeze_cache_value(item))
                for key, item in value.items()
            )
        )
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_cache_value(item) for item in value), key=repr))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_cache_value(item) for item in value)
    if isinstance(value, Enum):
        return value.value
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def task_cache_key(task: TaskSpec) -> tuple:
    """Canonical task signature used by semantics-preserving runtime caches."""

    return (
        task.kind.value,
        tuple(sorted(task.corridor)),
        task.initial_blocked_cells,
        tuple(sorted(task.goal_region)),
        task.transport_entity_id,
        task.payload_robot_id,
        tuple(sorted(task.hold_cells)),
        tuple(sorted(task.guarded_passage)),
        tuple(sorted(task.observation_cells)),
        tuple(sorted(task.hidden_entities)),
        tuple(sorted((str(key), int(value)) for key, value in task.resource_capacities.items())),
        tuple(
            sorted(
                (str(key), tuple(sorted(region)))
                for key, region in task.resource_regions.items()
            )
        ),
        tuple(sorted(task.activation_dependencies.items())),
        task.initial_blocked_weight,
        _freeze_cache_value(task.metadata),
    )


def belief_cache_key(belief: SymbolicBelief) -> tuple:
    """Canonical belief signature shared by option, belief, and value caches."""

    return (
        belief.robot_id,
        belief.width,
        belief.height,
        belief.own_position,
        task_cache_key(belief.task),
        tuple(sorted((cell, value.value) for cell, value in belief.cells.items())),
        tuple(
            sorted(
                (
                    entity_id,
                    tuple(sorted(blocker.cells)),
                    tuple(sorted(blocker.visible_cells)),
                    blocker.required_pushers,
                    blocker.last_seen_step,
                    blocker.mass_class,
                    blocker.task_weight,
                    tuple(sorted(direction.value for direction in blocker.allowed_push_directions)),
                    blocker.display_name,
                    blocker.certain,
                )
                for entity_id, blocker in belief.blockers.items()
            )
        ),
        tuple(sorted(belief.teammate_positions.items())),
        tuple(sorted(belief.revealed_entities)),
        tuple(sorted(belief.current_visible_cells)),
        tuple(sorted(belief.current_free_cells)),
        tuple(sorted(belief.current_walls)),
        belief.step_index,
        belief.source,
    )


class RuleBasedBeliefUpdater:
    """Deterministic symbolic filter over robot-owned local observations."""

    def initialize(self, observation: LocalObservation, width: int, height: int) -> SymbolicBelief:
        empty = SymbolicBelief(
            robot_id=observation.robot_id,
            width=width,
            height=height,
            own_position=observation.own_position,
            task=observation.task,
        )
        return self.update(empty, observation)

    def update(
        self,
        belief: SymbolicBelief,
        observation: LocalObservation,
        *,
        source: str = "real_observation",
    ) -> SymbolicBelief:
        cells: Dict[Cell, CellKnowledge] = dict(belief.cells)
        for cell in observation.free_cells:
            cells[cell] = CellKnowledge.FREE
        for cell in observation.walls:
            cells[cell] = CellKnowledge.WALL

        blockers: Dict[EntityId, BelievedBlocker] = dict(belief.blockers)
        visible_ids = set(observation.blockers)
        for entity_id, observed in observation.blockers.items():
            blockers[entity_id] = BelievedBlocker(
                entity_id=entity_id,
                cells=observed.cells,
                visible_cells=observed.visible_cells,
                required_pushers=observed.required_pushers,
                last_seen_step=belief.step_index + 1,
                mass_class=observed.mass_class,
                task_weight=observed.task_weight,
                allowed_push_directions=observed.allowed_push_directions,
                display_name=observed.display_name,
                certain=True,
            )

        for entity_id, known in list(blockers.items()):
            if entity_id in visible_ids:
                continue
            # The entity remains in memory but cannot ground a current option.
            blockers[entity_id] = replace(known, visible_cells=frozenset(), certain=False)

        teammates = {
            robot_id: position
            for robot_id, position in observation.robots.items()
            if robot_id != belief.robot_id
        }
        return SymbolicBelief(
            robot_id=belief.robot_id,
            width=belief.width,
            height=belief.height,
            own_position=observation.own_position,
            task=observation.task,
            cells=cells,
            blockers=blockers,
            teammate_positions=teammates,
            revealed_entities=frozenset(observation.revealed_entities),
            current_visible_cells=frozenset(observation.visible_cells),
            current_free_cells=frozenset(observation.free_cells),
            current_walls=frozenset(observation.walls),
            step_index=belief.step_index + 1,
            source=source,
        )


class SymbolicBeliefImaginator:
    """Optimistic, non-privileged successor-local-observation predictor.

    The predictor propagates known entities and option endpoints, renders the next
    robot-centered observation, and treats unobstructed unknown space as free. It
    never invents an unknown task entity. This is the analytic IET model used in
    Gate A; the privileged oracle instead calls the simulator observation function.
    """

    def __init__(self, env: GridEnvironment, updater: RuleBasedBeliefUpdater | None = None) -> None:
        self.env = env
        self.updater = updater or RuleBasedBeliefUpdater()

    @staticmethod
    def _predict_endpoint(current: Cell | None, option) -> Cell | None:
        if option is None:
            return current
        if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH} and option.target is not None:
            return option.target
        if option.kind == OptionKind.PUSH and option.contact_cell is not None:
            return option.contact_cell
        return current

    def predict(self, belief: SymbolicBelief, joint_option: JointOption) -> SymbolicBelief:
        options = joint_option.by_robot()
        own_position = self._predict_endpoint(belief.own_position, options.get(belief.robot_id))
        assert own_position is not None

        blockers = dict(belief.blockers)
        revealed = set(belief.revealed_entities)
        grouped: Dict[str, list] = {}
        for option in joint_option.options:
            if option.kind == OptionKind.PUSH and option.entity_id is not None:
                grouped.setdefault(option.entity_id, []).append(option)
            if option.kind == OptionKind.OBSERVE and option.entity_id is not None:
                revealed.add(option.entity_id)

        for entity_id, pushes in grouped.items():
            known = blockers.get(entity_id)
            if known is None:
                continue
            directions = {option.direction for option in pushes}
            if len(directions) == 1 and None not in directions and len(pushes) >= known.required_pushers:
                direction = next(iter(directions))
                assert direction is not None
                dx, dy = direction.delta
                blockers[entity_id] = replace(
                    known,
                    cells=frozenset((x + dx, y + dy) for x, y in known.cells),
                    visible_cells=frozenset(),
                    last_seen_step=belief.step_index + 1,
                )
            else:
                blockers[entity_id] = replace(known, visible_cells=frozenset(), certain=False)

        robot_positions: dict[RobotId, Cell] = {belief.robot_id: own_position}
        for rid, position in belief.teammate_positions.items():
            predicted = self._predict_endpoint(position, options.get(rid))
            if predicted is not None:
                robot_positions[rid] = predicted
        # Planning packets may introduce a teammate endpoint even when the teammate
        # was outside the current field of view; no raw map is required.
        for rid, option in options.items():
            if rid in robot_positions:
                continue
            predicted = self._predict_endpoint(None, option)
            if predicted is not None:
                robot_positions[rid] = predicted

        known_walls = frozenset(
            cell for cell, knowledge in belief.cells.items() if knowledge == CellKnowledge.WALL
        )
        predicted_state = GridState(
            width=belief.width,
            height=belief.height,
            walls=known_walls,
            robots={rid: RobotState(rid, position) for rid, position in robot_positions.items()},
            blockers={
                entity_id: BlockerState(
                    entity_id=entity_id,
                    cells=known.cells,
                    required_pushers=known.required_pushers,
                    mass_class=known.mass_class,
                    task_weight=known.task_weight,
                    allowed_push_directions=known.allowed_push_directions,
                    display_name=known.display_name,
                )
                for entity_id, known in blockers.items()
            },
            task=belief.task,
            revealed_entities=frozenset(set(blockers) | revealed),
            step_index=belief.step_index + 1,
        )
        observation = self.env.observe(predicted_state, belief.robot_id)
        predicted = self.updater.update(
            replace(belief, blockers=blockers, own_position=own_position),
            observation,
            source="optimistic_imagination",
        )
        return replace(predicted, revealed_entities=frozenset(revealed | set(observation.revealed_entities)))
