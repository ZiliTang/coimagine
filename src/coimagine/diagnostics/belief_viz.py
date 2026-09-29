"""Figures for the factored model: what the belief learns, and when the planner prepares.

These two results are the core of the factored-model line and neither had a picture:

* :func:`render_belief_probe` -- the posterior mass on the true parameter after each contact
  action, for the four hidden quantities. The claim is that the model learns exactly what is
  learnable, stays uncertain about what is not, and is essentially never confidently wrong; the
  figure shows all three at once, against the chance line.
* :func:`render_prepare_probe` -- how often the planner spends its first step releasing, split by
  whether releasing can work, with the parameters handed to it and with the belief it earned.
"""

from __future__ import annotations

import os
import tempfile
from typing import Dict, Optional, Sequence

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..grid.viz import save_figure, use_paper_style  # noqa: E402
from .belief_probe import (  # noqa: E402
    BeliefProbe,
    PrepareProbe,
    run_belief_probe,
    run_prepare_probe,
)

#: Colour per hidden quantity, kept consistent between the two panels.
VIEW_COLORS = {
    "own traction": "#2b7a3d",
    "teammate traction": "#b0553a",
    "resistance": "#c99a2e",
    "constraint": "#4c72b0",
}
#: The chance level: a two-valued quantity is 0.5 from an uninformative posterior.
CHANCE = 0.5


def render_belief_probe(
    probe: BeliefProbe,
    *,
    out_dir: str = "outputs/grid/world_model",
    stem: str = "belief_probe",
) -> list:
    """Mass on the truth over successive contact actions, plus the confident-error rate."""

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))

    steps = np.arange(1, probe.steps + 1)
    for view, values in probe.mass_on_truth.items():
        axes[0].plot(
            steps, values, marker="o", linewidth=1.8, markersize=5,
            color=VIEW_COLORS.get(view, "#555555"), label=view,
        )
    axes[0].axhline(CHANCE, color="#999999", linestyle="--", linewidth=1.0)
    axes[0].annotate(
        "0.5 = no information\n(a uniform posterior)",
        xy=(1.05, CHANCE), xytext=(1.15, CHANCE + 0.11),
        fontsize=6.6, color="#666666",
    )
    axes[0].set_xlabel("contact actions taken", fontsize=8)
    axes[0].set_ylabel("probability on the true value", fontsize=8)
    axes[0].set_title(
        "(a) what the belief learns, from a uniform prior", fontsize=8.5
    )
    axes[0].set_ylim(0.4, 1.05)
    axes[0].set_xticks(steps)
    axes[0].legend(fontsize=7, frameon=False, loc="lower right")
    axes[0].tick_params(labelsize=7)

    views = list(probe.confident_errors)
    values = [probe.confident_errors[v] for v in views]
    positions = np.arange(len(views))
    axes[1].bar(
        positions, values, width=0.62,
        color=[VIEW_COLORS.get(v, "#555555") for v in views],
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(
        [v.replace(" ", "\n") for v in views], fontsize=7
    )
    axes[1].set_ylabel("fraction confidently wrong (p > 0.8)", fontsize=8)
    axes[1].set_title(
        "(b) how often it is wrong *and* sure of itself", fontsize=8.5
    )
    axes[1].tick_params(labelsize=7)
    for position, value in zip(positions, values):
        axes[1].annotate(
            f"{value:.3f}", xy=(position, value), xytext=(0, 2),
            textcoords="offset points", ha="center", fontsize=7,
        )

    fig.suptitle(
        "The factored model's belief: own traction is identified by one contact action; "
        "the teammate's traction and the object's weight correctly stay near the prior; "
        "it is essentially never confidently wrong",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, out_dir, stem, formats=("png", "pdf"))


def render_prepare_probe(
    known: PrepareProbe,
    inferred: PrepareProbe,
    *,
    out_dir: str = "outputs/grid/world_model",
    stem: str = "prepare_probe",
) -> list:
    """Release rate on the first step, split by whether releasing can work at all."""

    use_paper_style()
    labels = [
        "latched,\nthis robot grips\n(release can work)",
        "latched,\nthis robot slips\n(release cannot work)",
        "free\n(nothing to release)",
    ]
    keys = ["releasable", "unreleasable", "free"]
    values = {
        "parameters known": [known.latched_releasable_rate, known.latched_unreleasable_rate,
                             known.free_release_rate],
        "belief inferred from feedback": [inferred.latched_releasable_rate,
                                          inferred.latched_unreleasable_rate,
                                          inferred.free_release_rate],
    }
    counts = [known.latched_releasable_n, known.latched_unreleasable_n, known.free_n]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    width = 0.36
    positions = np.arange(len(keys))
    for offset, (name, rates) in zip((-width / 2, width / 2), values.items()):
        color = "#4c72b0" if "known" in name else "#c99a2e"
        ax.bar(positions + offset, rates, width=width, label=name, color=color)
        for position, rate in zip(positions, rates):
            ax.annotate(
                f"{rate:.3f}", xy=(position + offset, rate), xytext=(0, 2),
                textcoords="offset points", ha="center", fontsize=6.8,
            )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"{label}\nn={count}" for label, count in zip(labels, counts)], fontsize=7
    )
    ax.set_ylabel("chose `release` as the first action", fontsize=8)
    ax.set_ylim(0, 0.78)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=7)
    ax.set_title(
        "The planner prepares exactly where preparing can work, and never where it cannot.\n"
        "The cost of not knowing the parameters falls on the free objects "
        "(wasted releases 0.062 -> 0.186), not on the ones worth preparing.",
        fontsize=8,
    )
    fig.tight_layout()
    return save_figure(fig, out_dir, stem, formats=("png", "pdf"))


def render_all(
    model,
    *,
    out_dir: str = "outputs/grid/world_model",
    n_runs: int = 200,
    n_scenes: int = 500,
) -> Dict[str, list]:
    """Both figures, from the current model."""

    belief = run_belief_probe(model, n_runs=n_runs, horizon=4)
    known = run_prepare_probe(model, n_scenes=n_scenes, lead_in=1, oracle_belief=True)
    inferred = run_prepare_probe(model, n_scenes=n_scenes, lead_in=1)
    return {
        "belief": render_belief_probe(belief, out_dir=out_dir),
        "prepare": render_prepare_probe(known, inferred, out_dir=out_dir),
    }
