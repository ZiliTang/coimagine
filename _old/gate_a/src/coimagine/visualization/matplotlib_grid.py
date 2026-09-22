from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from coimagine.envs.grid.state import GridState


def draw_state(
    ax: Axes,
    state: GridState,
    *,
    title: str = "",
    show_grid: bool = True,
    show_robot_ids_inside: bool = True,
    entity_labels_above: bool = False,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0, state.width)
    ax.set_ylim(0, state.height)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

    for cell in state.task.corridor:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#e7e7e7", edgecolor="none"))
    for cell in state.task.goal_region:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#d9ead3", edgecolor="none"))
    for cell in state.task.guarded_passage:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#fff2cc", edgecolor="none"))
    for cell in state.task.hold_cells:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#d9d2e9", edgecolor="none"))
    for cell in state.task.observation_cells:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#cfe2f3", edgecolor="none"))
    for cell in state.walls:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#444444", edgecolor="#222222"))
    for entity_id, blocker in state.blockers.items():
        hidden = entity_id in state.task.hidden_entities and entity_id not in state.revealed_entities
        face = "#aaaaaa" if hidden else "#cc6666"
        alpha = 0.35 if hidden else 0.9
        for cell in blocker.cells:
            ax.add_patch(Rectangle(cell, 1, 1, facecolor=face, edgecolor="#6d1f1f", alpha=alpha))
        cx = sum(x + 0.5 for x, _ in blocker.cells) / len(blocker.cells)
        cy = sum(y + 0.5 for _, y in blocker.cells) / len(blocker.cells)
        label = entity_id.replace("blocker-", "B")
        if entity_labels_above:
            top_row = max(y for _, y in blocker.cells)
            label_y = min(state.height - 0.35, top_row + 1.5)
            ax.text(
                cx,
                label_y,
                label,
                ha="center",
                va="center",
                fontsize=6.1,
                color="#3E1717",
                fontweight="bold",
                bbox={
                    "boxstyle": "round,pad=0.08",
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.76,
                },
                zorder=8,
            )
        else:
            ax.text(cx, cy, label, ha="center", va="center", fontsize=7)
    for robot_id, robot in state.robots.items():
        x, y = robot.position
        circle = plt.Circle((x + 0.5, y + 0.5), 0.35, facecolor="#4c78a8", edgecolor="#1f3b5d")
        ax.add_patch(circle)
        if show_robot_ids_inside:
            ax.text(x + 0.5, y + 0.5, str(robot_id), color="white", ha="center", va="center", fontsize=8)

    if show_grid:
        for x in range(state.width + 1):
            ax.plot([x, x], [0, state.height], color="#dddddd", linewidth=0.35, zorder=0)
        for y in range(state.height + 1):
            ax.plot([0, state.width], [y, y], color="#dddddd", linewidth=0.35, zorder=0)
