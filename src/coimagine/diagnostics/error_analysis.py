"""Where does the learned value get the decision wrong?

The closed loop says the learned value reaches about 83% of the equal-information
reference's success rate. This module answers the follow-up question the paper needs:
**which situations produce the remaining errors, and what does the model do instead?**

For every held-out decision point it compares the learned argmin against the
belief-weighted reference argmin and records

* the contact mechanism (latched / heavy / teammate slipping / plain);
* the distance from the object to the goal, so errors near the goal are visible
  separately from errors far away;
* the confusion between the reference's action and the learned action, including how
  often the model falls back to the free "wait".

It also renders the breakdown as a figure, which is the artefact that shows where the
model fails rather than merely how often.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import __version__
from ..learning.features import ACTION_IDS
from .decision_probe import DecisionRecord, evaluate_decisions, run_decision_probe


@dataclass
class ErrorAnalysis:
    overall_disagreement: float
    by_mechanism: Dict[str, float]
    by_mechanism_n: Dict[str, int]
    by_distance: Dict[int, float]
    by_distance_n: Dict[int, int]
    learned_action_when_wrong: Dict[str, int]
    wait_fallbacks: int
    wait_used_by_reference: int
    #: Of the reference's ``wait`` choices, how many are also the oracle's best action.
    #: A high fraction means the reference is not idle-by-default: it waits where acting
    #: genuinely does not help, which is what a rational planner should do when the
    #: surviving hypotheses are unsolvable.
    reference_wait_agrees_with_oracle: int
    confusion: Dict[str, int]
    n: int
    meta: Dict[str, object] = field(default_factory=dict)


def analyse_errors(
    records: Sequence[DecisionRecord],
    *,
    policy: str = "learned_value",
    reference: str = "value_optimal",
) -> ErrorAnalysis:
    total = len(records)
    if total == 0:
        raise ValueError("no records")

    wrong = [r for r in records if r.chosen[policy] != r.chosen[reference]]
    mechanisms: Counter = Counter()
    mechanism_wrong: Counter = Counter()
    distances: Counter = Counter()
    distance_wrong: Counter = Counter()
    wrong_actions: Counter = Counter()
    confusion: Counter = Counter()
    wait_fallbacks = 0
    reference_wait = 0
    reference_wait_correct = 0

    for record in records:
        mechanism = record.mechanism
        mechanisms[mechanism] += 1
        distances[record.distance_to_goal] += 1
        if record.chosen[reference] == "wait":
            reference_wait += 1
            if record.oracle_action == "wait":
                reference_wait_correct += 1
        if record.chosen[policy] != record.chosen[reference]:
            mechanism_wrong[mechanism] += 1
            distance_wrong[record.distance_to_goal] += 1
            wrong_actions[record.chosen[policy]] += 1
            confusion[f"{record.chosen[reference]} -> {record.chosen[policy]}"] += 1
            if record.chosen[policy] == "wait" and record.chosen[reference] != "wait":
                wait_fallbacks += 1

    return ErrorAnalysis(
        overall_disagreement=len(wrong) / total,
        by_mechanism={
            m: mechanism_wrong[m] / mechanisms[m] for m in sorted(mechanisms)
        },
        by_mechanism_n={m: mechanisms[m] for m in sorted(mechanisms)},
        by_distance={
            d: distance_wrong[d] / distances[d] for d in sorted(distances)
        },
        by_distance_n={d: distances[d] for d in sorted(distances)},
        learned_action_when_wrong=dict(wrong_actions.most_common()),
        wait_fallbacks=wait_fallbacks,
        wait_used_by_reference=reference_wait,
        reference_wait_agrees_with_oracle=reference_wait_correct,
        confusion=dict(confusion.most_common()),
        n=total,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "policy": policy,
            "reference": reference,
        },
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


#: Compact mechanism codes so the axis stays readable.
MECHANISM_CODES = {"latched": "L", "heavy": "H", "teammate_slips": "S"}


def _short_mechanism(mechanism: str) -> str:
    if mechanism == "plain":
        return "plain"
    return "+".join(MECHANISM_CODES.get(part, part) for part in mechanism.split("+"))


def render_error_figure(analysis: ErrorAnalysis, path_stem: str) -> List[str]:
    """Bar chart of disagreement by mechanism and by distance to the goal."""

    from ..grid.viz import save_figure, use_paper_style

    use_paper_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.2))

    mechanisms = list(analysis.by_mechanism)
    rates = [analysis.by_mechanism[m] for m in mechanisms]
    counts = [analysis.by_mechanism_n[m] for m in mechanisms]
    axes[0].bar(range(len(mechanisms)), rates, color="#4c72b0")
    axes[0].set_xticks(range(len(mechanisms)))
    axes[0].set_xticklabels([_short_mechanism(m) for m in mechanisms], fontsize=7)
    axes[0].set_xlabel("contact mechanism (L=latched, H=heavy, S=teammate slips)", fontsize=7)
    axes[0].set_ylabel("disagreement with the reference")
    axes[0].set_ylim(0.0, 1.0)
    for i, (rate, count) in enumerate(zip(rates, counts)):
        axes[0].text(i, rate + 0.02, f"n={count}", ha="center", fontsize=6)
    axes[0].set_title("(a) by contact mechanism", fontsize=9)
    axes[0].margins(x=0.02)

    distances = list(analysis.by_distance)
    drates = [analysis.by_distance[d] for d in distances]
    dcounts = [analysis.by_distance_n[d] for d in distances]
    axes[1].bar(range(len(distances)), drates, color="#dd8452")
    axes[1].set_xticks(range(len(distances)))
    axes[1].set_xticklabels([str(d) for d in distances])
    axes[1].set_xlabel("object-to-goal distance (cells)")
    axes[1].set_ylim(0.0, 1.0)
    for i, (rate, count) in enumerate(zip(drates, dcounts)):
        axes[1].text(i, rate + 0.02, f"n={count}", ha="center", fontsize=6)
    axes[1].set_title("(b) by distance to the goal", fontsize=9)

    fig.suptitle(
        f"Learned value versus the equal-information reference "
        f"(disagreement {analysis.overall_disagreement:.1%} of {analysis.n})",
        fontsize=9,
    )
    fig.tight_layout()
    return save_figure(fig, str(Path(path_stem).parent), Path(path_stem).name, formats=("png", "pdf"))


def format_error_markdown(analysis: ErrorAnalysis) -> str:
    lines = [
        f"Held-out decision points: {analysis.n}. "
        f"Disagreement with the reference: **{analysis.overall_disagreement:.1%}**.",
        "",
        "### By contact mechanism",
        "",
        "| mechanism | n | disagreement |",
        "|---|---|---|",
    ]
    for mechanism, rate in analysis.by_mechanism.items():
        lines.append(f"| {mechanism} | {analysis.by_mechanism_n[mechanism]} | {rate:.3f} |")
    lines += [
        "",
        "### By object-to-goal distance",
        "",
        "| distance | n | disagreement |",
        "|---|---|---|",
    ]
    for distance, rate in analysis.by_distance.items():
        lines.append(f"| {distance} | {analysis.by_distance_n[distance]} | {rate:.3f} |")
    lines += [
        "",
        "### What the model picks when it is wrong",
        "",
        "| action chosen by the model | count |",
        "|---|---|",
    ]
    for action, count in analysis.learned_action_when_wrong.items():
        lines.append(f"| {action} | {count} |")
    lines += [
        "",
        "### Reference action -> model action",
        "",
        "| confusion | count |",
        "|---|---|",
    ]
    for pair, count in analysis.confusion.items():
        lines.append(f"| {pair} | {count} |")
    lines += [
        "",
        f"The model falls back to the free `wait` while the reference acts in "
        f"**{analysis.wait_fallbacks}** cases. The reference chooses `wait` in "
        f"{analysis.wait_used_by_reference} cases, and in "
        f"**{analysis.reference_wait_agrees_with_oracle}** of those the true-state oracle "
        f"also waits -- that is, acting genuinely does not help, so those are not "
        f"idling.",
    ]
    return "\n".join(lines)


def run_error_analysis(
    *,
    n_train_situations: int = 150,
    n_test_situations: int = 80,
    epochs: int = 400,
    seed: int = 0,
    output_dir: str = "outputs/grid/error_analysis",
) -> Tuple[ErrorAnalysis, Dict[str, str]]:
    started = time.perf_counter()
    probe = run_decision_probe(
        n_train_situations=n_train_situations,
        n_test_situations=n_test_situations,
        epochs=epochs,
        seed=seed,
        depth=2,
    )
    analysis = analyse_errors(probe.records)
    analysis.meta["wall_clock_seconds"] = time.perf_counter() - started

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "error_analysis.json"
    json_path.write_text(json.dumps(asdict(analysis), indent=2))
    paths["json"] = str(json_path)

    figure_paths = render_error_figure(analysis, str(out / "error_by_situation"))
    paths["figure_png"] = figure_paths[0]
    if len(figure_paths) > 1:
        paths["figure_pdf"] = figure_paths[1]

    md_path = out / "error_analysis.md"
    md_path.write_text(
        "# Where the learned value gets the decision wrong\n\n"
        f"- git revision: `{analysis.meta['git_revision']}`\n"
        f"- {n_train_situations} training / {n_test_situations} held-out situations, "
        f"lookahead depth 2\n\n"
        + format_error_markdown(analysis)
        + "\n"
    )
    paths["md"] = str(md_path)
    return analysis, paths
