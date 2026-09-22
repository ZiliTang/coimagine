from __future__ import annotations

from coimagine.envs.grid.state import GridState


def render_ascii(state: GridState) -> str:
    blocker_lookup = {
        cell: entity_id for entity_id, blocker in state.blockers.items() for cell in blocker.cells
    }
    robot_lookup = {robot.position: robot_id for robot_id, robot in state.robots.items()}
    lines: list[str] = []
    for y in reversed(range(state.height)):
        row: list[str] = []
        for x in range(state.width):
            cell = (x, y)
            if cell in robot_lookup:
                row.append(str(robot_lookup[cell]))
            elif cell in blocker_lookup:
                row.append("B")
            elif cell in state.walls:
                row.append("#")
            elif cell in state.task.corridor:
                row.append("=")
            else:
                row.append(".")
        lines.append("".join(row))
    return "\n".join(lines)
