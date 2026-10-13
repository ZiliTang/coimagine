"""Three figures for fit_repair_v1 (no mixed denominators).

    python experiments/fit_repair/make_figures.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def _load_evals(run_dir: Path):
    series = {}
    for cell in sorted((run_dir / "runs").iterdir()):
        rows = [json.loads(line) for line in (cell / "eval_metrics.jsonl").open()]
        series[cell.name] = {int(r["update"]): r for r in rows}
    return series


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    series = _load_evals(run_dir)
    cells = [("Old+Random", 256), ("Old+Random", 512), ("Old+Branch", 256), ("Old+Branch", 512)]
    updates = sorted({u for s in series.values() for u in s})

    def mean_curve(group, width, dataset, field):
        keys = [f"{group.replace('+', '_')}_{width}_seed{seed}" for seed in (0, 1, 2)]
        out = []
        for update in updates:
            values = []
            for key in keys:
                row = series[key].get(update)
                if row is None:
                    continue
                node = row["metrics"][dataset]
                for part in field.split("."):
                    node = node[part]
                values.append(node)
            out.append(float(np.mean(values)) if values else float("nan"))
        return out

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for group, width in cells:
        label = f"{group} w{width}"
        axes[0].plot(updates, mean_curve(group, width, "train", "position.object_mean"), label=f"train {label}")
        axes[0].plot(updates, mean_curve(group, width, "original_val", "position.object_mean"), "--", label=f"orig val {label}")
        axes[1].plot(updates, mean_curve(group, width, "train", "legality.invalid_rate"), label=f"train {label}")
        axes[1].plot(updates, mean_curve(group, width, "original_val", "legality.invalid_rate"), "--", label=f"orig val {label}")
    axes[0].set_title("object position error (m)")
    axes[1].set_title("raw invalid-prediction rate")
    for ax in axes:
        ax.set_xlabel("optimizer.step")
        ax.legend(fontsize=6)
    fig.suptitle("fit_repair_v1: training vs original-val over updates (3-seed mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_progress.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for group, width in cells:
        label = f"{group} w{width}"
        axes[0].plot(updates, mean_curve(group, width, "train", "loss.state"), label=f"train {label}")
        axes[0].plot(updates, mean_curve(group, width, "original_val", "loss.state"), "--", label=f"orig val {label}")
        axes[1].plot(updates, mean_curve(group, width, "train", "loss.termination"), label=f"train {label}")
        axes[1].plot(updates, mean_curve(group, width, "original_val", "loss.termination"), "--", label=f"orig val {label}")
    axes[0].set_title("state loss")
    axes[1].set_title("termination CE")
    for ax in axes:
        ax.set_xlabel("optimizer.step")
        ax.legend(fontsize=6)
    fig.suptitle("fit_repair_v1: state loss and termination CE over updates (3-seed mean)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_losses.png", dpi=150)
    plt.close(fig)

    table = json.load(open(run_dir / "candidates" / "aggregate.json"))
    grouped = defaultdict(list)
    for row in table:
        grouped[row["label"].split("|")[-1]].append(row)
    selections = ("best_mixed_val", "best_original_val", "fixed_6000")
    chosen = [float(np.mean([r["feasible_chosen_true_return"] for r in grouped[s] if r["feasible_chosen_true_return"] is not None])) for s in selections]
    rejection = [float(np.mean([r["rejection_rate"] for r in grouped[s]])) for s in selections]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(selections, chosen)
    axes[0].axhline(-0.3608, color="grey", linestyle=":", label="uniform candidate -0.3608")
    axes[0].set_title("feasible-chosen true return (12 runs)")
    axes[0].legend(fontsize=7)
    axes[1].bar(selections, rejection)
    axes[1].set_title("candidate rejection rate (12 runs)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=15)
    fig.suptitle("fit_repair_v1: checkpoint selection vs candidate outcome (denominators per row in CSV)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_selection.png", dpi=150)
    plt.close(fig)
    print("FIGURES_OK", sorted(p.name for p in out_dir.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
