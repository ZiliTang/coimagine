"""Replay: the learned policy and the equal-information reference on the same episode.

The paper needs to show *where* a learned model errs, not only how often. This renders two
closed-loop episodes of the identical scene -- one driven by the network's predicted value,
one by the exact belief-space reference -- step by step, and marks the boundaries where they
disagree.

Each cell draws the simulator's ground truth with the robot's own ray scan, the action that
was committed, and the public outcome. A red frame marks a boundary where the two policies
chose differently, which is exactly where the learned model's error enters the episode.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..grid.sensing import ray_cast  # noqa: E402
from ..grid.state import Scene  # noqa: E402
from ..grid.viz import (  # noqa: E402
    draw_object,
    draw_robots,
    draw_scan,
    draw_static_map,
    save_figure,
    use_paper_style,
)
from ..learning.mlp import MultiHeadMLP  # noqa: E402
from .rollout_probe import StepTrace, run_episode  # noqa: E402


@dataclass
class EpisodeComparison:
    scene: Scene
    learned: List[StepTrace]
    reference: List[StepTrace]
    disagreement_steps: List[int]
    learned_success: int
    reference_success: int
    learned_cost: float
    reference_cost: float


def compare_policies(
    scene: Scene,
    model: MultiHeadMLP,
    *,
    horizon: int = 8,
    depth: int = 3,
    radius: int = 5,
    seed: int = 0,
) -> EpisodeComparison:
    """Run both policies on the same scene with the same observation noise."""

    learned_trace: List[StepTrace] = []
    reference_trace: List[StepTrace] = []
    learned_success, learned_cost, _ = run_episode(
        scene, "learned_value", model, horizon=horizon, depth=depth, radius=radius,
        rng=np.random.default_rng(seed), trace=learned_trace,
    )
    reference_success, reference_cost, _ = run_episode(
        scene, "true_model_belief", model, horizon=horizon, depth=depth, radius=radius,
        rng=np.random.default_rng(seed), trace=reference_trace,
    )
    disagreements = [
        a.t
        for a, b in zip(learned_trace, reference_trace)
        if a.action != b.action
    ]
    return EpisodeComparison(
        scene=scene,
        learned=learned_trace,
        reference=reference_trace,
        disagreement_steps=disagreements,
        learned_success=learned_success,
        reference_success=reference_success,
        learned_cost=learned_cost,
        reference_cost=reference_cost,
    )


def _draw_cell(ax, step: StepTrace, *, disagree: bool) -> None:
    scene = step.scene
    draw_static_map(ax, scene.static)
    scan = step.scan if step.scan is not None else ray_cast(scene, 0, radius=5)
    draw_scan(ax, scan, skip=scene.static.walls)
    draw_object(ax, scene.obj)
    draw_robots(ax, scene.robots)
    # Square cells: without this the map is stretched and the ray circle reads as an
    # ellipse, which misrepresents the sensing range.
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    frame = "#c0392b" if disagree else "#b0b0b0"
    width = 2.4 if disagree else 0.8
    for spine in ax.spines.values():
        spine.set_edgecolor(frame)
        spine.set_linewidth(width)
    ax.set_title(
        f"t={step.t}  {step.action}\n{step.y_public}",
        fontsize=6 if disagree else 5.5,
        color="#c0392b" if disagree else "#333333",
    )


def render_comparison(
    comparison: EpisodeComparison,
    *,
    out_dir: str = "outputs/grid/replays",
    stem: str = "learned_vs_reference",
    max_steps: int = 5,
) -> List[str]:
    """Two rows: the learned policy's episode above, the reference's below."""

    use_paper_style()
    steps = min(max_steps, max(len(comparison.learned), len(comparison.reference)))
    if steps == 0:
        return []
    fig, axes = plt.subplots(2, steps, figsize=(2.1 * steps, 5.2), squeeze=False)

    for column in range(steps):
        for row, (label, trace) in enumerate(
            (("learned value", comparison.learned), ("reference", comparison.reference))
        ):
            ax = axes[row][column]
            if column < len(trace):
                step = trace[column]
                _draw_cell(
                    ax, step, disagree=step.t in comparison.disagreement_steps
                )
            else:
                ax.axis("off")
            if column == 0:
                ax.set_ylabel(label, fontsize=8)

    fig.suptitle(
        f"Learned value versus the equal-information reference on one episode "
        f"(red frame = the two policies chose differently) | "
        f"success {comparison.learned_success} vs {comparison.reference_success}, "
        f"cost {comparison.learned_cost:.1f} vs {comparison.reference_cost:.1f}",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    # The row titles sit above each panel, so the two rows need real vertical space or the
    # lower titles are drawn over the upper panels.
    fig.subplots_adjust(hspace=0.42, wspace=0.10)
    return save_figure(fig, out_dir, stem, formats=("png", "pdf"))


def find_and_render_disagreement(
    *,
    n_scenes: int = 60,
    epochs: int = 350,
    seed: int = 0,
    out_dir: str = "outputs/grid/replays",
) -> Tuple[Optional[EpisodeComparison], List[str]]:
    """Train a value model, then render the first episode where the two policies differ.

    An episode with no disagreement is uninformative for this figure, so the search is
    explicit about skipping them.
    """

    from .decision_probe import _features_for, situation_quantities
    from ..grid.planner import PHYSICS
    from ..learning.data import ACTION_IDS as _ACTIONS
    from ..learning.data import action_labels, generate_situations, pose_to_index
    from ..learning.mlp import train

    situations = generate_situations(n_scenes=n_scenes, lead_in=2, seed=seed)
    features, targets, values, physics_index, poses = [], [], [], [], []
    for situation in situations:
        _ctx, _true, belief_values, progress, enables = situation_quantities(situation)
        for action in _ACTIONS:
            features.append(_features_for(situation, action))
            moved = action_labels(situation.scene, action)[0]
            targets.append([moved, int(progress[action] > 0.5), int(enables[action] > 0.5)])
            values.append(belief_values[action])
            physics_index.append(PHYSICS.index(situation.physics))
            poses.append(pose_to_index(situation.scene.obj.origin))
    x = np.asarray(features, dtype=np.float64)
    split = x.shape[1] - len(_ACTIONS)
    model = MultiHeadMLP(seed=seed, obs_dim=split, action_dim=len(_ACTIONS))
    train(
        model, x[:, :split], x[:, split:],
        np.asarray(targets, dtype=np.float64),
        np.asarray(physics_index, dtype=np.int64),
        np.asarray(poses, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        epochs=epochs, seed=seed,
    )

    episodes = generate_situations(n_scenes=n_scenes, lead_in=0, seed=seed + 500_000)
    for index, situation in enumerate(episodes):
        comparison = compare_policies(situation.scene, model, seed=index)
        if comparison.disagreement_steps and comparison.reference_success == 1:
            paths = render_comparison(
                comparison,
                out_dir=out_dir,
                stem=f"learned_vs_reference_scene{situation.scene_id}",
            )
            return comparison, paths
    return None, []
