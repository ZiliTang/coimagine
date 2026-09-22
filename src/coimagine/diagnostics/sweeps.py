"""Sweep figures required by the execution prompt.

Produces, on the CPU and without a display:

* a communication-budget curve (expected cost versus ``B``);
* a probe-cost sweep marking where the exact solver chooses to probe;
* a feedback-reliability sweep showing when private messaging has value.

Every point is an exact oracle value, not a sampled estimate.
"""

from __future__ import annotations

import dataclasses
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence

# Keep matplotlib's cache inside a writable temporary directory.
os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")  # no display required

import matplotlib.pyplot as plt  # noqa: E402

from ..exact import (  # noqa: E402
    NullCodec,
    ObservationCodec,
    PosteriorModeCodec,
    o_b_exact,
    o_central,
    o_central_root_actions,
    o_codec,
)
from ..toy import build_m1, build_m2  # noqa: E402

PROBE_ACTION = "PROBE"


def communication_budget_curve(output_dir: str, bits: Sequence[int] = (0, 1, 2, 3)) -> str:
    """Expected cost versus the message budget, at matched horizons.

    The left panel is the single boundary where ``O_B`` is exact; the right panel is
    ``H = 3`` where the exact object is ``O_codec``. Horizons are never mixed on one
    axis.
    """

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    single = {"M1": build_m1(horizon=1), "M2": build_m2(horizon=1, probe_cost=0.2)}
    multi = {"M1": build_m1(horizon=3), "M2": build_m2(horizon=3, probe_cost=0.2)}

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax = axes[0]
    for name, model in single.items():
        values = [o_b_exact(model, b).expected_cost for b in bits]
        ax.plot(list(bits), values, marker="o", label=f"{name}: exact O_B")
        ax.axhline(
            o_central(model).expected_cost,
            linestyle="--",
            alpha=0.5,
            label=f"{name}: O_central",
        )
    ax.set_title("Single boundary H=1 (exact O_B)")
    ax.set_xlabel("bits per robot per boundary")
    ax.set_ylabel("expected cost")
    ax.set_xticks(list(bits))
    ax.legend(fontsize=7)

    ax = axes[1]
    for name, model in multi.items():
        values = [
            o_codec(model, PosteriorModeCodec(bits=b)).expected_cost for b in bits
        ]
        ax.plot(list(bits), values, marker="s", label=f"{name}: O_codec posterior-mode")
        ax.axhline(
            o_codec(model, NullCodec()).expected_cost,
            linestyle=":",
            alpha=0.6,
            label=f"{name}: B=0 public only",
        )
    ax.set_title("H=3 (exact O_codec)")
    ax.set_xlabel("bits per robot per boundary")
    ax.set_ylabel("expected cost")
    ax.set_xticks(list(bits))
    ax.legend(fontsize=7)

    fig.tight_layout()
    path = out / "fig_communication_budget.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def probe_cost_sweep(
    output_dir: str, costs: Sequence[float] = (0.0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0)
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    values: List[float] = []
    probes: List[bool] = []
    for cost in costs:
        model = build_m2(horizon=3, probe_cost=cost, probe_strength="strong")
        values.append(o_central(model).expected_cost)
        actions = set(o_central_root_actions(model).values())
        probes.append(PROBE_ACTION in actions)

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(list(costs), values, marker="o", label="O_central expected cost")
    prober_x = [c for c, p in zip(costs, probes) if p]
    prober_y = [v for v, p in zip(values, probes) if p]
    if prober_x:
        ax.scatter(prober_x, prober_y, s=90, facecolors="none", edgecolors="tab:green",
                   label="exact policy probes", zorder=3)
    non_x = [c for c, p in zip(costs, probes) if not p]
    non_y = [v for v, p in zip(values, probes) if not p]
    if non_x:
        ax.scatter(non_x, non_y, marker="x", color="tab:red", label="exact policy does not probe", zorder=3)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("probe cost c_probe")
    ax.set_ylabel("expected cost")
    ax.set_title("M2: value of an informative probe vs its cost")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = out / "fig_probe_cost_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def feedback_strength_sweep(
    output_dir: str, strengths: Sequence[str] = ("none", "weak", "strong")
) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    central: List[float] = []
    public: List[float] = []
    message: List[float] = []
    for strength in strengths:
        model = build_m1(horizon=3, initial_strength=strength)
        c = o_central(model).expected_cost
        n = o_codec(model, NullCodec()).expected_cost
        o = o_codec(model, ObservationCodec(bits=1)).expected_cost
        central.append(c)
        public.append(n)
        message.append(n - o)

    positions = list(range(len(strengths)))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    axes[0].plot(positions, central, marker="o", label="O_central")
    axes[0].plot(positions, public, marker="s", label="O_codec null (B=0)")
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(strengths)
    axes[0].set_xlabel("initial local feedback reliability")
    axes[0].set_ylabel("expected cost")
    axes[0].set_title("M1: private evidence strength")
    axes[0].legend(fontsize=8)

    axes[1].bar(positions, message, color="tab:blue")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(strengths)
    axes[1].set_xlabel("initial local feedback reliability")
    axes[1].set_ylabel("expected-cost saving")
    axes[1].set_title("M1: value of a 1-bit message")
    fig.tight_layout()
    path = out / "fig_feedback_strength_sweep.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)


def run_all(output_dir: str) -> Dict[str, str]:
    return {
        "communication_budget": communication_budget_curve(output_dir),
        "probe_cost": probe_cost_sweep(output_dir),
        "feedback_strength": feedback_strength_sweep(output_dir),
    }
