"""Visualise what a learned world model imagines while it plans.

The point of these figures is to make the chain *inspectable*, not to decorate a result:

* :func:`render_branches` shows one decision boundary, with the situation the robot actually
  observes and, for **every** candidate action, the state the model imagines resulting from it,
  the model's own score for that branch, and which branch was chosen.
* :func:`render_imagination_vs_reality` puts the imagined next state next to the state the
  simulator actually produced, so a wrong imagination is visible as a wrong picture.
* :func:`render_episode_animation` writes a GIF of a whole episode: the true world on the left,
  the model's imagination that drove each decision on the right, and the chosen action.

The imagined panels are drawn by decoding the model's predicted observation vector back into a
6x6 grid. The model's prediction is whatever it is -- these figures do not clean it up.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import PillowWriter  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from ..grid.dynamics import execute  # noqa: E402
from ..grid.sensing import SensorNoise, apply_noise, ray_cast  # noqa: E402
from ..grid.state import Scene  # noqa: E402
from ..grid.viz import (  # noqa: E402
    COLOR_GRID,
    COLOR_OBJECT,
    COLOR_OBJECT_EDGE,
    COLOR_TARGET,
    COLOR_TARGET_EDGE,
    COLOR_UNKNOWN,
    COLOR_WALL,
    draw_object,
    draw_robots,
    draw_scan,
    draw_static_map,
    save_figure,
    use_paper_style,
)
from ..learning.data import _joint_for  # noqa: E402
from ..learning.features import (  # noqa: E402
    ACTION_IDS,
    CELL_KINDS,
    GRID,
    N_CELL_BLOCKS,
    observation_vector,
)
from .world_model_probe import _goal_cells, plan_with_model  # noqa: E402

#: Colours for the decoded imagined plane. Index 0 is "the model says nothing is here".
IMAGINED_COLORS = {
    0: "#f7f7f7",
    1: "#ffffff",
    2: "#5a5a5a",
    3: COLOR_OBJECT,
    4: "#4c72b0",
}
IMAGINED_LABELS = {0: "unknown", 1: "free", 2: "wall", 3: "object", 4: "robot"}


def decode_observation(observation: np.ndarray) -> np.ndarray:
    """Decode a (possibly imagined) observation vector into a 6x6 grid of block indices.

    The plane is one-hot over five blocks per cell, so a real observation decodes exactly and a
    predicted one decodes to the model's most likely content per cell.
    """

    plane = np.asarray(observation[: GRID * GRID * N_CELL_BLOCKS]).reshape(GRID, GRID, N_CELL_BLOCKS)
    kinds = plane.argmax(axis=2)
    # A cell whose blocks are all near zero is one the robot has not seen.
    kinds = np.where(plane.max(axis=2) < 0.25, 0, kinds)
    return kinds


def draw_imagined_panel(
    ax,
    observation: np.ndarray,
    static,
    *,
    title: str = "",
    highlight: bool = False,
    goal: Optional[frozenset] = None,
) -> None:
    """Draw one decoded observation as a grid picture."""

    kinds = decode_observation(observation)
    for y in range(GRID):
        for x in range(GRID):
            kind = int(kinds[y, x])
            if kind == 0 and goal is not None and (x, y) in goal:
                # The prior map is known to every robot, so the goal is drawn even where the
                # current scan says nothing: the planner uses it and so should the figure.
                color = COLOR_TARGET
            else:
                color = IMAGINED_COLORS[kind]
            ax.add_patch(
                mpatches.Rectangle((x, y), 1, 1, facecolor=color, edgecolor=COLOR_GRID, linewidth=0.6)
            )
            if kind == 0 and goal is not None and (x, y) in goal:
                ax.add_patch(
                    mpatches.Rectangle((x, y), 1, 1, fill=False, edgecolor=COLOR_TARGET_EDGE,
                                       linewidth=0.8, linestyle=(0, (2, 2)))
                )
    if goal is not None:
        for (x, y) in goal:
            if kinds[y, x] != 0:
                ax.add_patch(
                    mpatches.Rectangle((x, y), 1, 1, fill=False, edgecolor=COLOR_TARGET_EDGE,
                                       linewidth=0.8, linestyle=(0, (2, 2)))
                )
    ax.set_xlim(0, GRID)
    ax.set_ylim(0, GRID)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#c0392b" if highlight else "#bbbbbb")
        spine.set_linewidth(2.2 if highlight else 0.8)
    ax.set_title(title, fontsize=6.4, color="#c0392b" if highlight else "#333333")


def draw_truth_panel(ax, scene: Scene, scan=None, *, title: str = "") -> None:
    """Draw the simulator's ground truth, which the planner never sees."""

    draw_static_map(ax, scene.static)
    if scan is not None:
        draw_scan(ax, scan, skip=scene.static.walls)
    draw_object(ax, scene.obj)
    draw_robots(ax, scene.robots)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=6.4, color="#333333")


@dataclass
class DecisionSnapshot:
    """Everything one decision boundary produced, for the figures."""

    step: int
    scene_before: Scene
    scan: np.ndarray
    observation: np.ndarray
    branches: List[Dict[str, object]]
    chosen: int
    scene_after: Scene
    true_cost: float
    true_moved: bool


def run_episode_snapshots(
    scene: Scene,
    model,
    *,
    horizon: int = 8,
    radius: int = 5,
    depth: int = 2,
    seed: int = 0,
) -> Tuple[List[DecisionSnapshot], bool]:
    """Run one episode, recording every decision and what the model imagined at it.

    Execution uses the simulator; planning uses only ``model``.
    """

    rng = np.random.default_rng(seed)
    noise = SensorNoise()
    goal = _goal_cells(scene)
    history_a: List[str] = []
    public_history: List[str] = []
    snapshots: List[DecisionSnapshot] = []
    current = scene

    for step in range(horizon):
        if current.is_goal():
            break
        scan = apply_noise(ray_cast(current, 0, radius=radius), noise, rng)
        observation = np.asarray(
            observation_vector(scan, current.robot_pose(0), tuple(history_a), tuple(public_history))
        )
        chosen, branches = plan_with_model(model, observation, goal, depth=depth)
        action_id = ACTION_IDS[chosen]
        successor, result = execute(current, _joint_for(current, action_id))
        snapshots.append(
            DecisionSnapshot(
                step=step,
                scene_before=current,
                scan=scan,
                observation=observation,
                branches=branches,
                chosen=chosen,
                scene_after=successor,
                true_cost=float(result.cost),
                true_moved=result.y_public == "moved",
            )
        )
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        current = successor

    return snapshots, current.is_goal()


def render_branches(
    snapshot: DecisionSnapshot,
    *,
    out_dir: str = "outputs/grid/world_model",
    stem: Optional[str] = None,
    goal: Optional[frozenset] = None,
) -> List[str]:
    """One figure per decision: the observation, then what the model imagines per action."""

    use_paper_style()
    branches = snapshot.branches
    columns = len(branches)
    fig, axes = plt.subplots(1, columns + 1, figsize=(1.85 * (columns + 1), 2.75), squeeze=False)
    draw_truth_panel(
        axes[0][0], snapshot.scene_before, snapshot.scan,
        title=f"true world at t={snapshot.step}\n(planner never sees this)",
    )
    chosen_action = ACTION_IDS[snapshot.chosen]
    for index, branch in enumerate(branches):
        is_chosen = branch["action"] == chosen_action
        draw_imagined_panel(
            axes[0][index + 1],
            branch["imagined_observation"],
            snapshot.scene_before.static,
            title=(
                f"{branch['action']}\n"
                f"score {branch['score']:.1f}  p(move) {branch['moved_probability']:.2f}"
            ),
            highlight=is_chosen,
            goal=goal,
        )
    fig.suptitle(
        f"Planning at t={snapshot.step}: the model imagines every candidate action. "
        f"Red frame = the branch it chose ({chosen_action}); the frame is the ranking, "
        f"the pictures are what it actually predicted.",
        fontsize=7.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    name = stem or f"branches_t{snapshot.step:02d}"
    return save_figure(fig, out_dir, name, formats=("png", "pdf"))


def render_imagination_vs_reality(
    snapshot: DecisionSnapshot,
    *,
    out_dir: str = "outputs/grid/world_model",
    goal: Optional[frozenset] = None,
) -> List[str]:
    """The chosen branch's imagination beside what the simulator actually did."""

    use_paper_style()
    branch = next(b for b in snapshot.branches if b["action"] == ACTION_IDS[snapshot.chosen])
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.9), squeeze=False)
    draw_truth_panel(axes[0][0], snapshot.scene_before, snapshot.scan, title=f"t={snapshot.step}: current")
    draw_imagined_panel(
        axes[0][1],
        branch["imagined_observation"],
        snapshot.scene_before.static,
        title=f"imagined result of {branch['action']}",
        highlight=True,
        goal=goal,
    )
    draw_truth_panel(
        axes[0][2], snapshot.scene_after,
        title=f"what actually happened\n(cost {snapshot.true_cost:.2f}, "
              f"moved={snapshot.true_moved})",
    )
    fig.suptitle(
        "The model's imagination next to reality: every difference is the model's error",
        fontsize=7.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return save_figure(fig, out_dir, f"imagined_vs_real_t{snapshot.step:02d}", formats=("png", "pdf"))


def render_episode_animation(
    snapshots: Sequence[DecisionSnapshot],
    *,
    out_dir: str = "outputs/grid/world_model",
    stem: str = "episode",
    goal: Optional[frozenset] = None,
    fps: int = 1,
) -> str:
    """A GIF: the true world, the imagination that drove the decision, and the choice."""

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.3), squeeze=False)

    def draw(frame: int):
        for row in axes:
            for ax in row:
                ax.clear()
        snapshot = snapshots[frame]
        branch = next(b for b in snapshot.branches if b["action"] == ACTION_IDS[snapshot.chosen])
        draw_truth_panel(
            axes[0][0], snapshot.scene_before, snapshot.scan,
            title=f"t={snapshot.step}: what the robot sees and what is true",
        )
        draw_imagined_panel(
            axes[0][1],
            branch["imagined_observation"],
            snapshot.scene_before.static,
            title=f"the model imagined {branch['action']}\n(p(move)={branch['moved_probability']:.2f}, "
                  f"score={branch['score']:.1f})",
            highlight=True,
            goal=goal,
        )
        fig.suptitle(
            f"Planner chose {branch['action']} at t={snapshot.step}   |   "
            f"reality: moved={snapshot.true_moved}, cost={snapshot.true_cost:.2f}",
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        return []

    animation = FuncAnimation(fig, draw, frames=len(snapshots), interval=1000 // fps, blit=False)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}.gif")
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return path


def render_rollout_tree(
    model,
    observation: np.ndarray,
    goal: frozenset,
    static,
    *,
    out_dir: str = "outputs/grid/world_model",
    stem: str = "rollout_tree",
    depth: int = 2,
) -> List[str]:
    """The imagined tree the planner scores: every first action and its best continuation."""

    use_paper_style()
    branches = plan_with_model(model, observation, goal, depth=depth)[1]
    fig, axes = plt.subplots(2, len(branches), figsize=(1.7 * len(branches), 5.0), squeeze=False)
    for column, branch in enumerate(branches):
        draw_imagined_panel(
            axes[0][column], branch["imagined_observation"], static,
            title=f"{branch['action']}\nscore {branch['score']:.1f}", goal=goal,
        )
        continuation = branch["continuation"]
        if continuation is None:
            axes[1][column].axis("off")
        else:
            second = model.continue_from(
                model.predict_next(observation, branch["action_index"]), int(continuation)
            )
            draw_imagined_panel(
                axes[1][column], second["observation"][0], static,
                title=f"then {ACTION_IDS[int(continuation)]}", goal=goal,
            )
    fig.suptitle(
        "Two-step imagination: the top row is each first action, the bottom row is the best "
        "continuation the model finds from there",
        fontsize=7.6,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, out_dir, stem, formats=("png", "pdf"))
