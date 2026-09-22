from __future__ import annotations

from coimagine.core.types import TaskKind
from coimagine.envs.grid.state import BlockerState, GridState, RobotState, TaskSpec


def two_robot_approach_then_push() -> GridState:
    corridor = frozenset((x, 4) for x in range(2, 8))
    blocker_cells = frozenset({(4, 4), (5, 4)})
    return GridState(
        width=10,
        height=9,
        walls=frozenset(),
        robots={0: RobotState(0, (2, 2)), 1: RobotState(1, (7, 2))},
        blockers={
            "blocker-0": BlockerState(
                entity_id="blocker-0", cells=blocker_cells, required_pushers=2
            )
        },
        task=TaskSpec(
            kind=TaskKind.CLEARING,
            corridor=corridor,
            initial_blocked_cells=len(blocker_cells & corridor),
        ),
    )
