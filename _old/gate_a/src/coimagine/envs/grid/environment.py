from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Sequence

from coimagine.core.grid import add_cell, manhattan, translate
from coimagine.core.types import (
    Cell,
    Direction,
    ExecutionResult,
    JointOption,
    LocalOption,
    OptionKind,
    RobotId,
    TaskKind,
)
from coimagine.envs.grid.observation import LocalObservation, ObservedBlocker
from coimagine.envs.grid.state import BlockerState, GridState, RobotState

if TYPE_CHECKING:
    from coimagine.belief.symbolic import SymbolicBelief


@dataclass(frozen=True)
class Transition:
    state: GridState
    result: ExecutionResult


class GridEnvironment:
    """Option-boundary grid dynamics and occlusion-aware local sensing for Gate A."""

    def __init__(self, observation_radius: int = 6) -> None:
        if observation_radius < 1:
            raise ValueError("observation_radius must be positive")
        self.observation_radius = observation_radius

    @staticmethod
    def in_bounds(state: GridState, cell: Cell) -> bool:
        return 0 <= cell[0] < state.width and 0 <= cell[1] < state.height

    @staticmethod
    def blocker_cells(state: GridState, exclude: Optional[str] = None) -> set[Cell]:
        cells: set[Cell] = set()
        for entity_id, blocker in state.blockers.items():
            if entity_id != exclude:
                cells.update(blocker.cells)
        return cells

    @staticmethod
    def occupied_cells(state: GridState, exclude_robot: Optional[RobotId] = None) -> set[Cell]:
        occupied = set(state.walls) | GridEnvironment.blocker_cells(state)
        occupied.update(
            robot.position
            for robot_id, robot in state.robots.items()
            if robot_id != exclude_robot
        )
        return occupied

    @staticmethod
    def _line_cells(start: Cell, end: Cell) -> tuple[Cell, ...]:
        """Integer Bresenham ray including both endpoints."""

        x0, y0 = start
        x1, y1 = end
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        cells: list[Cell] = []
        while True:
            cells.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy
        return tuple(cells)

    def visible_cells_from_pose(
        self,
        state: GridState,
        own: Cell,
        *,
        exclude_robot: RobotId | None = None,
    ) -> frozenset[Cell]:
        """Render a deterministic lidar-style visibility mask.

        A ray is cast to every grid cell inside the sensing disk. The first occupied
        cell is visible and terminates the ray. This dense discrete construction is
        inexpensive in Gate A and avoids angular undersampling artifacts.
        """

        occupied = set(state.walls) | self.blocker_cells(state)
        occupied.update(
            robot.position
            for rid, robot in state.robots.items()
            if rid != exclude_robot
        )
        radius_sq = self.observation_radius * self.observation_radius
        targets = [
            (x, y)
            for x in range(max(0, own[0] - self.observation_radius), min(state.width, own[0] + self.observation_radius + 1))
            for y in range(max(0, own[1] - self.observation_radius), min(state.height, own[1] + self.observation_radius + 1))
            if (x - own[0]) ** 2 + (y - own[1]) ** 2 <= radius_sq
        ]
        visible: set[Cell] = {own}
        for target in targets:
            for cell in self._line_cells(own, target)[1:]:
                visible.add(cell)
                if cell in occupied:
                    break
        return frozenset(visible)

    def observe(self, state: GridState, robot_id: RobotId) -> LocalObservation:
        own = state.robots[robot_id].position
        visible = self.visible_cells_from_pose(state, own, exclude_robot=robot_id)
        walls = frozenset(cell for cell in state.walls if cell in visible)

        visible_blockers: dict[str, ObservedBlocker] = {}
        for entity_id, blocker in state.blockers.items():
            visible_surface = frozenset(cell for cell in blocker.cells if cell in visible)
            if not visible_surface:
                continue
            hidden = entity_id in state.task.hidden_entities and entity_id not in state.revealed_entities
            if hidden:
                # A hidden semantic entity remains a physical occluder but does not
                # ground an option until the task-specific reveal event occurs.
                continue
            visible_blockers[entity_id] = ObservedBlocker(
                entity_id=entity_id,
                cells=blocker.cells,
                visible_cells=visible_surface,
                required_pushers=blocker.required_pushers,
                mass_class=blocker.mass_class,
                task_weight=blocker.task_weight,
                allowed_push_directions=blocker.allowed_push_directions,
                display_name=blocker.display_name,
            )

        visible_robots = {
            rid: robot.position
            for rid, robot in state.robots.items()
            if robot.position in visible
        }
        occupied = set(state.walls) | self.blocker_cells(state)
        occupied.update(robot.position for rid, robot in state.robots.items() if rid != robot_id)
        free = frozenset(cell for cell in visible if cell not in occupied)
        return LocalObservation(
            robot_id=robot_id,
            own_position=own,
            visible_cells=visible,
            walls=walls,
            free_cells=free,
            blockers=visible_blockers,
            robots=visible_robots,
            task=state.task,
            revealed_entities=state.revealed_entities,
        )

    def shortest_path(self, state: GridState, robot_id: RobotId, target: Cell) -> Optional[list[Cell]]:
        start = state.robots[robot_id].position
        if start == target:
            return [start]
        if not self.in_bounds(state, target):
            return None
        blocked = self.occupied_cells(state, exclude_robot=robot_id)
        if target in blocked:
            return None
        queue = deque([start])
        parent: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cell = queue.popleft()
            for direction in Direction:
                nxt = add_cell(cell, direction.delta)
                if not self.in_bounds(state, nxt) or nxt in blocked or nxt in parent:
                    continue
                parent[nxt] = cell
                if nxt == target:
                    path = [nxt]
                    while path[-1] != start:
                        prev = parent[path[-1]]
                        assert prev is not None
                        path.append(prev)
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    @staticmethod
    def _centroid(cells: frozenset[Cell]) -> tuple[float, float]:
        if not cells:
            return 0.0, 0.0
        return (
            sum(cell[0] for cell in cells) / len(cells),
            sum(cell[1] for cell in cells) / len(cells),
        )

    def task_potential(self, state: GridState) -> float:
        task = state.task
        if task.kind in {TaskKind.CLEARING, TaskKind.REVEAL_ACT, TaskKind.STABLE_CONTROL}:
            if task.initial_blocked_weight > 0:
                blocked_weight = sum(
                    blocker.task_weight * len(blocker.cells & task.corridor)
                    for blocker in state.blockers.values()
                )
                denominator = task.initial_blocked_weight
                return max(0.0, min(1.0, 1.0 - blocked_weight / denominator))
            blocked = len(self.blocker_cells(state) & set(task.corridor))
            denominator = max(1, task.initial_blocked_cells)
            return max(0.0, min(1.0, 1.0 - blocked / denominator))

        if task.kind == TaskKind.TRANSPORT:
            entity_id = task.transport_entity_id
            blocker = state.blockers.get(entity_id) if entity_id is not None else None
            if blocker is None or not task.goal_region:
                return 0.0
            cx, cy = self._centroid(blocker.cells)
            distance = min(abs(cx - gx) + abs(cy - gy) for gx, gy in task.goal_region)
            initial_distance = float(task.metadata.get("initial_distance", max(1.0, distance)))
            return max(0.0, min(1.0, 1.0 - distance / max(1.0, initial_distance)))

        if task.kind == TaskKind.HOLD_PASS:
            rid = task.payload_robot_id
            if rid is None or rid not in state.robots or not task.goal_region:
                return 0.0
            position = state.robots[rid].position
            distance = min(manhattan(position, goal) for goal in task.goal_region)
            initial_distance = float(task.metadata.get("initial_distance", max(1, distance)))
            return max(0.0, min(1.0, 1.0 - distance / max(1.0, initial_distance)))

        raise ValueError(f"Unsupported task kind: {task.kind}")

    def is_terminal(self, state: GridState) -> bool:
        task = state.task
        if task.kind == TaskKind.TRANSPORT:
            entity_id = task.transport_entity_id
            blocker = state.blockers.get(entity_id) if entity_id is not None else None
            return blocker is not None and bool(blocker.cells & task.goal_region)
        if task.kind == TaskKind.HOLD_PASS:
            rid = task.payload_robot_id
            return rid is not None and state.robots[rid].position in task.goal_region
        return self.task_potential(state) >= 1.0

    def _validate_push_group(
        self,
        state: GridState,
        entity_id: str,
        options: Sequence[LocalOption],
    ) -> tuple[bool, str | None, frozenset[Cell] | None]:
        blocker = state.blockers.get(entity_id)
        if blocker is None:
            return False, "unknown_entity", None
        directions = {option.direction for option in options}
        if None in directions or len(directions) != 1:
            return False, "incompatible_push_directions", None
        direction = next(iter(directions))
        assert direction is not None
        if direction not in blocker.allowed_push_directions:
            return False, "disallowed_push_direction", None
        if len(options) < blocker.required_pushers:
            return False, "insufficient_pushers", None
        if len({option.robot_id for option in options}) != len(options):
            return False, "duplicate_robot", None
        if len({option.contact_cell for option in options}) != len(options):
            return False, "duplicate_contact", None

        dx, dy = direction.delta
        for option in options:
            if option.contact_cell not in blocker.cells:
                return False, "invalid_contact", None
            assert option.contact_cell is not None
            expected_support = (option.contact_cell[0] - dx, option.contact_cell[1] - dy)
            if state.robots[option.robot_id].position != expected_support:
                return False, "not_at_contact", None

        moved = translate(blocker.cells, direction)
        if any(not self.in_bounds(state, cell) for cell in moved):
            return False, "out_of_bounds", None
        occupied = set(state.walls) | self.blocker_cells(state, exclude=entity_id)
        if moved & occupied:
            return False, "blocked", None
        return True, None, moved

    @staticmethod
    def _option_destination(state: GridState, option: LocalOption) -> Cell | None:
        if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH}:
            return option.target
        if option.kind == OptionKind.PUSH:
            return option.contact_cell
        if option.robot_id in state.robots:
            return state.robots[option.robot_id].position
        return None

    def execute_optimistic(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, "SymbolicBelief"],
        joint_option: JointOption,
    ) -> Transition:
        """Rule-based local rollout used by COST and IET.

        Root candidates are grounded in real current observations. On imagined
        successor layers, this model trusts each robot's predicted free-space support
        and ignores unobserved obstacles. Known entity dynamics, contact geometry,
        resource conflicts, and joint destination conflicts remain deterministic.
        """

        if set(option.robot_id for option in joint_option.options) != set(state.robots):
            return Transition(state, ExecutionResult(False, 0.0, "incomplete_joint_option"))

        resource_usage: Dict[str, int] = {}
        for option in joint_option.options:
            for token in option.signature.resource_tokens:
                if token in state.task.resource_capacities:
                    resource_usage[token] = resource_usage.get(token, 0) + 1
        for token, count in resource_usage.items():
            capacity = int(state.task.resource_capacities[token])
            if count > capacity:
                return Transition(state, ExecutionResult(False, 1.0, "resource_capacity_conflict"))

        robot_destinations: dict[RobotId, Cell] = {}
        optimistic_paths: dict[RobotId, list[Cell]] = {}
        push_groups: dict[str, list[LocalOption]] = {}
        revealed = set(state.revealed_entities)
        duration = 1.0
        hold_robots: set[RobotId] = set()

        for option in joint_option.options:
            belief = beliefs[option.robot_id]
            if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH}:
                if option.target is None:
                    return Transition(state, ExecutionResult(False, 0.0, "missing_target"))
                if option.target not in belief.current_free_cells and option.target != belief.own_position:
                    return Transition(state, ExecutionResult(False, 1.0, "unsupported_future_target"))
                path = self._belief_shortest_path(belief, option.target)
                if path is None:
                    return Transition(state, ExecutionResult(False, 1.0, "unsupported_future_path"))
                optimistic_paths[option.robot_id] = path
                robot_destinations[option.robot_id] = option.target
                duration = max(duration, float(max(1, len(path) - 1)))
            elif option.kind == OptionKind.PUSH:
                if option.entity_id is None:
                    return Transition(state, ExecutionResult(False, 0.0, "missing_entity"))
                push_groups.setdefault(option.entity_id, []).append(option)
            elif option.kind == OptionKind.HOLD:
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
                hold_robots.add(option.robot_id)
            elif option.kind == OptionKind.OBSERVE:
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
                if option.entity_id is not None:
                    revealed.add(option.entity_id)
            elif option.kind == OptionKind.IDLE:
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
            else:
                return Transition(state, ExecutionResult(False, 0.0, "unsupported_option"))

        if state.task.kind == TaskKind.HOLD_PASS:
            payload = state.task.payload_robot_id
            if payload is not None:
                payload_option = joint_option.by_robot().get(payload)
                if payload_option and payload_option.kind in {OptionKind.APPROACH, OptionKind.REPOSITION}:
                    if not hold_robots and payload_option.target in state.task.goal_region:
                        return Transition(state, ExecutionResult(False, duration, "missing_hold"))

        blockers = dict(state.blockers)
        for entity_id, options in push_groups.items():
            blocker = blockers.get(entity_id)
            if blocker is None:
                return Transition(state, ExecutionResult(False, 1.0, "unknown_entity"))
            directions = {option.direction for option in options}
            if None in directions or len(directions) != 1:
                return Transition(state, ExecutionResult(False, 1.0, "incompatible_push_directions"))
            direction = next(iter(directions))
            assert direction is not None
            if len(options) < blocker.required_pushers:
                return Transition(state, ExecutionResult(False, 1.0, "insufficient_pushers"))
            if len({option.contact_cell for option in options}) != len(options):
                return Transition(state, ExecutionResult(False, 1.0, "duplicate_contact"))
            dx, dy = direction.delta
            for option in options:
                if option.contact_cell not in blocker.cells:
                    return Transition(state, ExecutionResult(False, 1.0, "invalid_contact"))
                assert option.contact_cell is not None
                support = (option.contact_cell[0] - dx, option.contact_cell[1] - dy)
                if state.robots[option.robot_id].position != support:
                    return Transition(state, ExecutionResult(False, 1.0, "not_at_contact"))
                robot_destinations[option.robot_id] = option.contact_cell
            moved = translate(blocker.cells, direction)
            if any(not self.in_bounds(state, cell) for cell in moved):
                return Transition(state, ExecutionResult(False, 1.0, "out_of_bounds"))
            # Reject only geometry already represented in at least one local belief.
            known_walls = {
                cell
                for belief in beliefs.values()
                for cell, knowledge in belief.cells.items()
                if knowledge.value == "wall"
            }
            known_other = {
                cell
                for belief in beliefs.values()
                for other_id, known in belief.blockers.items()
                if other_id != entity_id
                for cell in known.cells
            }
            if moved & (known_walls | known_other):
                return Transition(state, ExecutionResult(False, 1.0, "known_blocked"))
            blockers[entity_id] = BlockerState(
                entity_id=blocker.entity_id,
                cells=moved,
                required_pushers=blocker.required_pushers,
                mass_class=blocker.mass_class,
                task_weight=blocker.task_weight,
                allowed_push_directions=blocker.allowed_push_directions,
                display_name=blocker.display_name,
            )

        if len(robot_destinations.values()) != len(set(robot_destinations.values())):
            return Transition(state, ExecutionResult(False, duration, "destination_conflict"))

        paths = list(optimistic_paths.items())
        for index, (_, path_a) in enumerate(paths):
            for _, path_b in paths[index + 1 :]:
                if set(path_a[1:]) & set(path_b[1:]):
                    return Transition(
                        state,
                        ExecutionResult(False, duration, "known_swept_region_conflict"),
                    )

        robots = {
            rid: RobotState(rid, robot_destinations.get(rid, robot.position))
            for rid, robot in state.robots.items()
        }
        next_state = state.with_updates(
            robots=robots,
            blockers=blockers,
            revealed_entities=frozenset(revealed),
        )
        return Transition(next_state, ExecutionResult(True, duration, details={"optimistic": True}))

    @staticmethod
    def _belief_shortest_path(
        belief: "SymbolicBelief",
        target: Cell,
    ) -> Optional[list[Cell]]:
        """Shortest path supported by one robot's current real/imagined observation.

        The local model never routes through cells outside the current observation
        support. Under optimistic imagination, those cells already include the
        unobstructed empty-unknown prior rendered around the predicted robot pose.
        """

        start = belief.own_position
        if start == target:
            return [start]
        free = set(belief.current_free_cells) | {start}
        if target not in free:
            return None
        queue = deque([start])
        parent: Dict[Cell, Optional[Cell]] = {start: None}
        while queue:
            cell = queue.popleft()
            for direction in Direction:
                nxt = add_cell(cell, direction.delta)
                if nxt not in free or nxt in parent:
                    continue
                parent[nxt] = cell
                if nxt == target:
                    path = [nxt]
                    while path[-1] != start:
                        prev = parent[path[-1]]
                        assert prev is not None
                        path.append(prev)
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def execute(self, state: GridState, joint_option: JointOption) -> Transition:
        if set(option.robot_id for option in joint_option.options) != set(state.robots):
            return Transition(state, ExecutionResult(False, 0.0, "incomplete_joint_option"))

        push_groups: Dict[str, list[LocalOption]] = {}
        approach_paths: Dict[RobotId, list[Cell]] = {}
        robot_destinations: Dict[RobotId, Cell] = {}
        moved_blockers: Dict[str, frozenset[Cell]] = {}
        revealed = set(state.revealed_entities)
        duration = 1.0
        hold_robots: set[RobotId] = set()

        resource_usage: Dict[str, int] = {}
        for option in joint_option.options:
            for token in option.signature.resource_tokens:
                if token in state.task.resource_capacities:
                    resource_usage[token] = resource_usage.get(token, 0) + 1
        for token, count in resource_usage.items():
            capacity = int(state.task.resource_capacities[token])
            if count > capacity:
                return Transition(
                    state,
                    ExecutionResult(
                        False,
                        1.0,
                        "resource_capacity_conflict",
                        details={"resource_token": token, "usage": count, "capacity": capacity},
                    ),
                )

        for option in joint_option.options:
            if option.kind == OptionKind.PUSH:
                if option.entity_id is None:
                    return Transition(state, ExecutionResult(False, 0.0, "missing_entity"))
                push_groups.setdefault(option.entity_id, []).append(option)
            elif option.kind in {OptionKind.APPROACH, OptionKind.REPOSITION}:
                if option.target is None:
                    return Transition(state, ExecutionResult(False, 0.0, "missing_target"))
                path = self.shortest_path(state, option.robot_id, option.target)
                if path is None:
                    return Transition(state, ExecutionResult(False, 0.0, "unreachable"))
                approach_paths[option.robot_id] = path
                robot_destinations[option.robot_id] = option.target
                duration = max(duration, float(max(1, len(path) - 1)))
            elif option.kind == OptionKind.HOLD:
                if state.robots[option.robot_id].position not in state.task.hold_cells:
                    return Transition(state, ExecutionResult(False, 1.0, "not_at_hold_cell"))
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
                hold_robots.add(option.robot_id)
            elif option.kind == OptionKind.OBSERVE:
                if state.robots[option.robot_id].position not in state.task.observation_cells:
                    return Transition(state, ExecutionResult(False, 1.0, "not_at_observation_cell"))
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
                duration = max(duration, float(state.task.metadata.get("observe_duration", 1.0)))
                if option.entity_id is not None:
                    revealed.add(option.entity_id)
                else:
                    revealed.update(state.task.hidden_entities)
            elif option.kind == OptionKind.IDLE:
                robot_destinations[option.robot_id] = state.robots[option.robot_id].position
            else:
                return Transition(state, ExecutionResult(False, 0.0, "unsupported_option"))

        if state.task.kind == TaskKind.HOLD_PASS and state.task.payload_robot_id in approach_paths:
            path = approach_paths[state.task.payload_robot_id]
            if set(path) & set(state.task.guarded_passage):
                if not hold_robots:
                    return Transition(state, ExecutionResult(False, duration, "missing_hold"))

        for entity_id, options in push_groups.items():
            valid, reason, moved = self._validate_push_group(state, entity_id, options)
            if not valid or moved is None:
                return Transition(state, ExecutionResult(False, 1.0, reason))
            moved_blockers[entity_id] = moved
            for option in options:
                assert option.contact_cell is not None
                robot_destinations[option.robot_id] = option.contact_cell

        if len(robot_destinations.values()) != len(set(robot_destinations.values())):
            return Transition(state, ExecutionResult(False, duration, "destination_conflict"))

        paths = list(approach_paths.items())
        for index, (_, path_a) in enumerate(paths):
            for _, path_b in paths[index + 1 :]:
                if set(path_a[1:]) & set(path_b[1:]):
                    return Transition(state, ExecutionResult(False, duration, "swept_region_conflict"))

        blockers = dict(state.blockers)
        for entity_id, moved in moved_blockers.items():
            blocker = blockers[entity_id]
            blockers[entity_id] = BlockerState(
                entity_id=entity_id,
                cells=moved,
                required_pushers=blocker.required_pushers,
                mass_class=blocker.mass_class,
                task_weight=blocker.task_weight,
                allowed_push_directions=blocker.allowed_push_directions,
                display_name=blocker.display_name,
            )

        final_blocker_cells = (
            set().union(*(blocker.cells for blocker in blockers.values())) if blockers else set()
        )
        for destination in robot_destinations.values():
            if destination in state.walls or destination in final_blocker_cells:
                return Transition(state, ExecutionResult(False, duration, "final_collision"))

        robots = {
            robot_id: RobotState(robot_id, robot_destinations[robot_id])
            for robot_id in state.robots
        }
        next_state = state.with_updates(
            robots=robots,
            blockers=blockers,
            revealed_entities=frozenset(revealed),
        )
        return Transition(
            next_state,
            ExecutionResult(
                True,
                duration,
                details={
                    "potential_before": self.task_potential(state),
                    "potential_after": self.task_potential(next_state),
                    "revealed_entities": sorted(revealed - set(state.revealed_entities)),
                },
            ),
        )
