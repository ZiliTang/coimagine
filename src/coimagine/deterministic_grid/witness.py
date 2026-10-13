"""Deterministic witness search for the two fixed maps (B0 acceptance only).

Breadth-first search over joint actions returns a shortest real joint-action sequence from
a given initial state to the goal. The witness is used only to check that the environment
is solvable; it is never injected into evaluation candidates and never used to select a
model.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Tuple

from .contract import GridMap, GridState, N_JOINT_ACTIONS, is_goal
from .dynamics import transition_exact
from .scoring import rollout_exact


def find_witness(
    grid_map: GridMap, initial: GridState, *, max_states: int = 200_000
) -> Optional[List[int]]:
    if is_goal(initial, grid_map):
        return []
    queue: deque = deque([initial])
    parent: Dict[GridState, Tuple[GridState, int]] = {}
    seen = {initial}
    while queue and len(seen) < max_states:
        state = queue.popleft()
        for joint in range(N_JOINT_ACTIONS):
            next_state, _info = transition_exact(state, joint, grid_map)
            if next_state in seen:
                continue
            seen.add(next_state)
            parent[next_state] = (state, joint)
            if is_goal(next_state, grid_map):
                sequence: List[int] = []
                cursor = next_state
                while cursor != initial:
                    previous, action = parent[cursor]
                    sequence.append(action)
                    cursor = previous
                sequence.reverse()
                return sequence
            queue.append(next_state)
    return None


def verify_witness(grid_map: GridMap, initial: GridState, sequence: List[int]) -> Dict[str, object]:
    result = rollout_exact(initial, sequence, grid_map)
    return {
        "map": grid_map.name,
        "initial": {"robot0": initial.robot0, "robot1": initial.robot1, "object": initial.object_anchor},
        "witness_length": len(sequence),
        "witness": sequence,
        "success": result.success,
        "executed_steps": result.executed_steps,
        "final_object": result.states[-1].object_anchor,
    }
