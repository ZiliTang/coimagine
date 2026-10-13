"""Exact synchronous transition for the deterministic grid (CONTRACT section 2).

Pure, deterministic and fully observable. The same function is the true environment and the
reference used to score learned predictions; no scene-id branching exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .contract import (
    DIRECTION,
    MOVE_ACTIONS,
    PUSH_ACTIONS,
    WAIT_ACTION,
    Cell,
    GridMap,
    GridState,
    add,
    decode_joint,
    in_bounds,
    object_cells,
)


@dataclass(frozen=True)
class TransitionInfo:
    object_moved: bool
    robots_moved: int
    rollback: bool
    invalid_actions: int
    push_direction: Optional[Cell]
    pushers: Tuple[int, ...]


def transition_exact(state: GridState, joint_action: int, grid_map: GridMap) -> Tuple[GridState, TransitionInfo]:
    """One synchronous step; see CONTRACT rules 1-10."""

    action0, action1 = decode_joint(joint_action)
    start = {0: state.robot0, 1: state.robot1}
    obj = object_cells(state.object_anchor)
    free = grid_map.free_cells

    move_target: Dict[int, Cell] = {}
    push_direction: Dict[int, Cell] = {}
    invalid_actions = 0

    for robot_id, action in ((0, action0), (1, action1)):
        position = start[robot_id]
        other_position = start[1 - robot_id]
        if action in MOVE_ACTIONS:
            target = add(position, DIRECTION[action])
            legal = (
                in_bounds(target)
                and target in free
                and target not in obj
                and target != other_position
            )
            if legal:
                move_target[robot_id] = target
            else:
                invalid_actions += 1
        elif action in PUSH_ACTIONS:
            delta = DIRECTION[action]
            qualified = position not in obj and add(position, delta) in obj
            if qualified:
                push_direction[robot_id] = delta
            else:
                invalid_actions += 1

    object_moved = False
    push_delta: Optional[Cell] = None
    pushers: Tuple[int, ...] = ()
    if push_direction:
        directions = set(push_direction.values())
        if len(directions) == 1:
            candidate_delta = next(iter(directions))
            candidate_pushers = tuple(sorted(r for r, d in push_direction.items() if d == candidate_delta))
            new_anchor = add(state.object_anchor, candidate_delta)
            if object_cells(new_anchor) <= free:
                object_moved = True
                push_delta = candidate_delta
                pushers = candidate_pushers

    final = dict(start)
    if object_moved and push_delta is not None:
        for robot_id in pushers:
            final[robot_id] = add(start[robot_id], push_delta)
    for robot_id, target in move_target.items():
        final[robot_id] = target

    new_obj = object_cells(add(state.object_anchor, push_delta)) if (object_moved and push_delta) else obj

    rollback = False
    if final[0] == final[1]:
        rollback = True
    if final[0] in new_obj or final[1] in new_obj:
        rollback = True
    if final[0] not in free or final[1] not in free:
        rollback = True
    if object_moved:
        for robot_id in pushers:
            if final[robot_id] == start[1 - robot_id]:
                rollback = True

    if rollback:
        next_state = state
        info = TransitionInfo(False, 0, True, invalid_actions, None, ())
    else:
        next_state = GridState(final[0], final[1], add(state.object_anchor, push_delta) if object_moved else state.object_anchor)
        moved_robots = sum(1 for robot_id in (0, 1) if final[robot_id] != start[robot_id])
        info = TransitionInfo(object_moved, moved_robots, False, invalid_actions, push_delta, pushers)
    return next_state, info


def joint_action_effect_counts(
    states: List[GridState], grid_map: GridMap
) -> Dict[str, float]:
    """Aggregate object-move / robot-move / rollback / invalid rates over states and actions."""

    object_moves = 0
    robot_moves = 0
    rollbacks = 0
    invalid = 0
    total = 0
    from .contract import N_JOINT_ACTIONS

    for state in states:
        for joint in range(N_JOINT_ACTIONS):
            _, info = transition_exact(state, joint, grid_map)
            total += 1
            object_moves += int(info.object_moved)
            robot_moves += int(info.robots_moved > 0)
            rollbacks += int(info.rollback)
            invalid += int(info.invalid_actions > 0)
    return {
        "transitions": float(total),
        "object_move_rate": object_moves / total if total else 0.0,
        "robot_move_rate": robot_moves / total if total else 0.0,
        "rollback_rate": rollbacks / total if total else 0.0,
        "invalid_action_rate": invalid / total if total else 0.0,
    }
