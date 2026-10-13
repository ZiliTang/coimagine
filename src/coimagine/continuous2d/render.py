"""Frame rendering for the continuous prototype.

Frames come from the **recorded** per-control-step states of a real execution, not from a
generated animation: the trace drawn is the trace that ran. A GIF is written with the
Pillow writer so the deliverable needs no external ffmpeg.
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, Polygon, Rectangle  # noqa: E402

from .config import Continuous2DConfig  # noqa: E402
from .scenes import SceneSpec, scene_by_name  # noqa: E402

Vec = Tuple[float, float]


def _object_polygon(object_state: Sequence[float], size: Tuple[float, float]) -> List[Vec]:
    x, y, angle = object_state[0], object_state[1], object_state[2]
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    cos, sin = math.cos(angle), math.sin(angle)
    return [
        (x + dx * cos - dy * sin, y + dx * sin + dy * cos)
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    ]


def draw_scene(ax, scene: SceneSpec, config: Continuous2DConfig, state: Dict[str, object]) -> None:
    width, height = scene.arena
    ax.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="#333333", linewidth=1.2))
    for box in scene.obstacles:
        ax.add_patch(
            Rectangle(
                (box.cx - box.half_w, box.cy - box.half_h),
                2 * box.half_w,
                2 * box.half_h,
                color="#4a4a4a",
            )
        )
    goal = scene.goal
    ax.add_patch(
        Rectangle(
            (goal.cx - goal.half_w, goal.cy - goal.half_h),
            2 * goal.half_w,
            2 * goal.half_h,
            facecolor="#bfe3c0",
            edgecolor="#2e7d32",
            alpha=0.7,
        )
    )
    ax.add_patch(
        Polygon(
            _object_polygon(state["object"], (config.physics.object_width, config.physics.object_height)),
            closed=True,
            facecolor="#c98b4b",
            edgecolor="#6d4c1b",
            linewidth=1.0,
        )
    )
    for robot_id, robot in enumerate(state["robots"]):
        ax.add_patch(
            Circle(
                (robot[0], robot[1]),
                config.physics.robot_radius,
                facecolor="#4c78c9" if robot_id == 0 else "#c94c6e",
                edgecolor="#1b2b4c",
                linewidth=0.8,
            )
        )
    ax.set_xlim(-0.05, width + 0.05)
    ax.set_ylim(-0.05, height + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def render_episode(
    episode,
    scene_name: str,
    config: Continuous2DConfig,
    out_dir: str,
    *,
    stem: str,
    every: int = 2,
    fps: int = 20,
) -> List[str]:
    """PNG frames plus a GIF for one episode's recorded control steps."""

    scene = scene_by_name(scene_name)
    root = Path(out_dir)
    frames_dir = root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    states: List[Dict[str, object]] = []
    for entry in episode.trace:
        # Episode traces are TrajectoryResults; a bare SkillResult is accepted too.
        results = getattr(entry, "results", None) or (entry,)
        for result in results:
            for step in result.steps:
                states.append(
                    {
                        "robots": [list(state.as_tuple()) for state in step.robots],
                        "object": list(step.object.as_tuple()),
                        "skill": step.skill_labels,
                    }
                )
    if not states:
        return []

    written: List[str] = []
    figure, ax = plt.subplots(figsize=(4.6, 3.4))
    trail: List[Vec] = []
    for index, state in enumerate(states[::every]):
        ax.clear()
        trail.append((state["object"][0], state["object"][1]))
        draw_scene(ax, scene, config, state)
        if len(trail) > 1:
            ax.plot(
                [point[0] for point in trail],
                [point[1] for point in trail],
                color="#c98b4b", linewidth=0.8, alpha=0.5, zorder=0,
            )
        ax.set_title(
            f"{scene_name}  step {index * every}/{len(states)}  A:{state['skill'][0]}",
            fontsize=6.5,
        )
        path = frames_dir / f"{stem}_{index * every:04d}.png"
        figure.savefig(path, dpi=110, bbox_inches="tight")
        written.append(str(path))
    plt.close(figure)

    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        figure, ax = plt.subplots(figsize=(4.6, 3.4))

        def update(index: int):
            ax.clear()
            draw_scene(ax, scene, config, states[index])

        animation = FuncAnimation(figure, update, frames=len(states), interval=1000 // fps)
        gif_path = root / f"{stem}.gif"
        animation.save(str(gif_path), writer=PillowWriter(fps=fps))
        plt.close(figure)
        written.append(str(gif_path))
    except Exception as error:  # pragma: no cover - rendering is a deliverable, not a gate
        (root / f"{stem}_gif_error.txt").write_text(repr(error))
    return written
