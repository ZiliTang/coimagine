from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Sequence

from coimagine.core.types import LocalOption, OptionKind
from coimagine.envs.grid.state import GridState
from coimagine.topology.factors import Factor


@dataclass(frozen=True)
class TupleLabel:
    options: tuple[LocalOption, ...]
    executable: bool
    extendable: bool


@dataclass(frozen=True)
class VerificationStats:
    verifier_calls: int = 0
    retained_tuples: int = 0
    executable_tuples: int = 0


class ExactFactorVerifier:
    """Exact candidate-conditioned factor labels for controlled grid tasks."""

    def __init__(self, state: GridState, max_arity: int = 3) -> None:
        self.state = state
        self.max_arity = max_arity

    def executable(self, factor: Factor, options: Sequence[LocalOption]) -> bool:
        if not options:
            return False
        if factor.factor_type.value == "entity":
            entity_id = factor.factor_id.split(":", 1)[1]
            blocker = self.state.blockers.get(entity_id)
            if blocker is None:
                return False
            pushes = [option for option in options if option.kind == OptionKind.PUSH]
            if len(pushes) != len(options):
                return True
            directions = {option.direction for option in pushes}
            return (
                None not in directions
                and len(directions) == 1
                and len(pushes) >= blocker.required_pushers
                and len({option.contact_cell for option in pushes}) == len(pushes)
            )
        if factor.factor_type.value == "resource":
            token = factor.factor_id.split(":", 1)[1]
            capacity = int(self.state.task.resource_capacities.get(token, 1))
            return len(options) <= capacity
        if factor.factor_type.value == "synchronization":
            pushes = [option for option in options if option.kind == OptionKind.PUSH]
            if not pushes:
                return True
            directions = {option.direction for option in pushes}
            return None not in directions and len(directions) == 1
        raise ValueError(f"Unsupported factor type: {factor.factor_type}")

    def all_assignments(self, factor: Factor) -> tuple[tuple[LocalOption, ...], ...]:
        robots = factor.incident_robots
        assignments: list[tuple[LocalOption, ...]] = []
        for order in range(1, min(self.max_arity, len(robots)) + 1):
            for subset in combinations(robots, order):
                pools = [factor.options_by_robot[robot_id] for robot_id in subset]
                assignments.extend(tuple(items) for items in product(*pools))
        return tuple(assignments)

    @staticmethod
    def _has_distinct_completion(
        pools: list[tuple[LocalOption, ...]],
        needed: int,
        used_contacts: set[object],
    ) -> bool:
        if needed <= 0:
            return True
        if len(pools) < needed:
            return False

        def search(index: int, remaining: int, contacts: set[object]) -> bool:
            if remaining == 0:
                return True
            if len(pools) - index < remaining:
                return False
            if index >= len(pools):
                return False
            if search(index + 1, remaining, contacts):
                return True
            for option in pools[index]:
                contact = option.contact_cell
                if contact in contacts:
                    continue
                contacts.add(contact)
                if search(index + 1, remaining - 1, contacts):
                    return True
                contacts.remove(contact)
            return False

        return search(0, needed, set(used_contacts))

    def extendable(self, factor: Factor, options: Sequence[LocalOption]) -> bool:
        selected = tuple(options)
        selected_robots = {option.robot_id for option in selected}
        remaining_robots = [
            robot_id for robot_id in factor.incident_robots if robot_id not in selected_robots
        ]
        if not remaining_robots or len(selected) >= self.max_arity:
            return False

        if factor.factor_type.value == "entity":
            entity_id = factor.factor_id.split(":", 1)[1]
            blocker = self.state.blockers.get(entity_id)
            if blocker is None:
                return False
            pushes = [option for option in selected if option.kind == OptionKind.PUSH]
            if len(pushes) != len(selected):
                return any(factor.options_by_robot.get(robot_id) for robot_id in remaining_robots)
            directions = {option.direction for option in pushes}
            if None in directions or len(directions) != 1:
                return False
            direction = next(iter(directions))
            used_contacts = {option.contact_cell for option in pushes}
            if len(used_contacts) != len(pushes):
                return False
            target_size = max(blocker.required_pushers, len(selected) + 1)
            if target_size > self.max_arity:
                return False
            pools = []
            for robot_id in remaining_robots:
                compatible = tuple(
                    option
                    for option in factor.options_by_robot.get(robot_id, ())
                    if option.kind == OptionKind.PUSH
                    and option.direction == direction
                    and option.contact_cell not in used_contacts
                )
                if compatible:
                    pools.append(compatible)
            return self._has_distinct_completion(
                pools, target_size - len(selected), used_contacts
            )

        if factor.factor_type.value == "resource":
            token = factor.factor_id.split(":", 1)[1]
            capacity = min(
                int(self.state.task.resource_capacities.get(token, 1)),
                self.max_arity,
            )
            return len(selected) < capacity and any(
                factor.options_by_robot.get(robot_id) for robot_id in remaining_robots
            )

        if factor.factor_type.value == "synchronization":
            pushes = [option for option in selected if option.kind == OptionKind.PUSH]
            directions = {option.direction for option in pushes}
            if pushes and (None in directions or len(directions) != 1):
                return False
            direction = next(iter(directions)) if directions else None
            for robot_id in remaining_robots:
                for option in factor.options_by_robot.get(robot_id, ()):
                    if option.kind != OptionKind.PUSH or direction is None or option.direction == direction:
                        return True
            return False

        raise ValueError(f"Unsupported factor type: {factor.factor_type}")

    def label(self, factor: Factor, selected: Sequence[LocalOption]) -> TupleLabel:
        selected_tuple = tuple(sorted(selected, key=lambda option: option.robot_id))
        return TupleLabel(
            selected_tuple,
            self.executable(factor, selected_tuple),
            self.extendable(factor, selected_tuple),
        )

    def recursive_labels(self, factor: Factor) -> tuple[TupleLabel, ...]:
        labels = [self.label(factor, assignment) for assignment in self.all_assignments(factor)]
        return tuple(
            sorted(
                labels,
                key=lambda item: (
                    len(item.options),
                    tuple(option.stable_key for option in item.options),
                ),
            )
        )

    def recursive_stats(self, factor: Factor) -> VerificationStats:
        labels = self.recursive_labels(factor)
        return VerificationStats(
            verifier_calls=len(labels),
            retained_tuples=sum(label.extendable for label in labels),
            executable_tuples=sum(label.executable for label in labels),
        )
