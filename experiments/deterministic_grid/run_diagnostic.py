"""B3: run the fixed-candidate diagnostic, write raw records, metric table and two figures.

    python experiments/deterministic_grid/run_diagnostic.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coimagine.deterministic_grid.contract import default_maps  # noqa: E402
from coimagine.deterministic_grid.data import DataConfig, generate_dataset  # noqa: E402
from coimagine.deterministic_grid.diagnostic import (  # noqa: E402
    EXACT_SEED,
    DiagnosticConfig,
    run_diagnostic,
)
from coimagine.deterministic_grid.model import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = (
    "selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret",
    "chosen_true_return", "uniform_candidate_return", "candidate_best_true_return",
    "chosen_true_success", "candidate_invalid_rate", "candidate_success_rate",
)


def _aggregate(per_run: List[Dict[str, object]]) -> List[Dict[str, object]]:
    grouped: Dict[tuple, List[Dict[str, object]]] = defaultdict(list)
    for row in per_run:
        grouped[(int(row["H"]), int(row["K"]), int(row["model_seed"]))].append(row)
    rows: List[Dict[str, object]] = []
    for (depth, size, model_seed), group in sorted(grouped.items()):
        row: Dict[str, object] = {
            "H": depth, "K": size, "model_seed": model_seed,
            "n_starts": len({(r["map"], r["state_key"]) for r in group}),
            "n_runs": len(group),
        }
        for key in METRIC_KEYS:
            row[key] = float(np.mean([float(r[key]) for r in group]))
        rows.append(row)
    # mean and range across model seeds
    across: List[Dict[str, object]] = []
    for (depth, size) in sorted({(int(r["H"]), int(r["K"])) for r in per_run}):
        subset = [row for row in rows if row["H"] == depth and row["K"] == size and row["model_seed"] != EXACT_SEED]
        if not subset:
            continue
        row = {"H": depth, "K": size, "model_seed": "mean", "n_starts": subset[0]["n_starts"], "n_runs": subset[0]["n_runs"]}
        for key in METRIC_KEYS:
            values = [float(r[key]) for r in subset]
            row[key] = float(np.mean(values))
            row[key + "_min"] = float(np.min(values))
            row[key + "_max"] = float(np.max(values))
        across.append(row)
    return rows + across


def _markdown(rows: List[Dict[str, object]]) -> str:
    lines = [
        "# 固定候选机制表（每个模型种子单列 + 均值/范围）",
        "",
        "| H | K | model | starts | selected_bias | selection_excess | candidate_regret | chosen_true_return | uniform_candidate_return | candidate_best_true_return | chosen_true_success | candidate_invalid_rate |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model = "exact" if row["model_seed"] == EXACT_SEED else str(row["model_seed"])
        lines.append(
            f"| {row['H']} | {row['K']} | {model} | {row['n_starts']} | {row['selected_bias']:+.3f} | "
            f"{row['selection_excess']:+.3f} | {row['candidate_regret']:.3f} | {row['chosen_true_return']:+.3f} | "
            f"{row['uniform_candidate_return']:+.3f} | {row['candidate_best_true_return']:+.3f} | "
            f"{row['chosen_true_success']:.3f} | {row['candidate_invalid_rate']:.3f} |"
        )
    lines += ["", "K 超过该 H 的不同序列数时使用实际 K；同一全集只评估一次。", ""]
    return "\n".join(lines)


def _figures(rows: List[Dict[str, object]], out_dir: Path) -> List[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    depths = sorted({int(row["H"]) for row in rows})
    sizes = sorted({int(row["K"]) for row in rows})
    models = sorted({int(row["model_seed"]) for row in rows if row["model_seed"] != "mean"})
    written = []

    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.4), squeeze=False)
    for column, depth in enumerate(depths):
        ax = axes[0][column]
        for model_seed in models:
            values = [
                next((float(row["selected_bias"]) for row in rows
                      if row["H"] == depth and row["K"] == size and row["model_seed"] == model_seed), float("nan"))
                for size in sizes
            ]
            label = "exact" if model_seed == EXACT_SEED else f"seed {model_seed}"
            ax.plot(sizes, values, "o-" if model_seed == EXACT_SEED else "s--", label=label)
        ax.axhline(0.0, color="#888888", linewidth=0.8)
        ax.set_title(f"H = {depth}")
        ax.set_xlabel("K")
        ax.set_xticks(sizes)
        if column == 0:
            ax.set_ylabel("selected_bias = J_hat(chosen) - J_ref(chosen)")
            ax.legend(fontsize=7)
    fig.suptitle("Selection bias vs candidate-set size (fixed H, exact control)")
    fig.tight_layout()
    path = out_dir / "fig_k_selection_bias.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, axes = plt.subplots(1, len(depths), figsize=(4.2 * len(depths), 3.4), squeeze=False)
    for column, depth in enumerate(depths):
        ax = axes[0][column]
        chosen = [next(float(row["chosen_true_return"]) for row in rows
                       if row["H"] == depth and row["K"] == size and row["model_seed"] == "mean")
                  for size in sizes]
        best = [next(float(row["candidate_best_true_return"]) for row in rows
                     if row["H"] == depth and row["K"] == size and row["model_seed"] == "mean")
                for size in sizes]
        uniform = [next(float(row["uniform_candidate_return"]) for row in rows
                        if row["H"] == depth and row["K"] == size and row["model_seed"] == "mean")
                   for size in sizes]
        ax.plot(sizes, chosen, "o-", label="chosen_true_return")
        ax.plot(sizes, best, "s--", label="candidate_best")
        ax.plot(sizes, uniform, "^:", label="uniform_candidate_return")
        ax.set_title(f"H = {depth}")
        ax.set_xlabel("K")
        ax.set_xticks(sizes)
        if column == 0:
            ax.set_ylabel("J_ref")
            ax.legend(fontsize=7)
    fig.suptitle("Realised chosen return and candidate best vs candidate-set size (model mean)")
    fig.tight_layout()
    path = out_dir / "fig_k_true_return.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="smoke")
    parser.add_argument("--out-root", default="outputs/deterministic_grid_v1")
    parser.add_argument("--budget", type=int, default=30_000)
    parser.add_argument("--data-seed", type=int, default=202_610_12)
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--depths", default="1,2,4")
    parser.add_argument("--sizes", default="16,64,256")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    diag_dir = run_dir / "diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    maps = {grid_map.name: grid_map for grid_map in default_maps()}
    dataset = generate_dataset(default_maps(), DataConfig(budget=args.budget, seed=args.data_seed))
    model_seeds = [int(value) for value in args.model_seeds.split(",") if value]
    models = {seed: load_model(run_dir / "models" / f"seed{seed}.pt") for seed in model_seeds}
    hashes = {}
    for seed in model_seeds:
        import hashlib

        digest = hashlib.sha256((run_dir / "models" / f"seed{seed}.pt").read_bytes()).hexdigest()
        hashes[seed] = digest
    config = DiagnosticConfig(
        depths=tuple(int(v) for v in args.depths.split(",") if v),
        sizes=tuple(int(v) for v in args.sizes.split(",") if v),
        model_seeds=tuple(model_seeds),
    )
    result = run_diagnostic(dataset, maps, models, config=config, checkpoint_hashes=hashes)

    (diag_dir / "config.json").write_text(json.dumps(config.as_dict(), indent=2))
    with (diag_dir / "per_run.jsonl").open("w") as handle:
        for row in result["per_run"]:
            handle.write(json.dumps(row) + "\n")
    with (diag_dir / "candidate_records.jsonl").open("w") as handle:
        for row in result["candidate_records"]:
            handle.write(json.dumps(row) + "\n")
    with (diag_dir / "chosen_traces.jsonl").open("w") as handle:
        for row in result["chosen_traces"]:
            handle.write(json.dumps(row) + "\n")
    (diag_dir / "teacher_forced_vs_recursive.json").write_text(
        json.dumps(result["teacher_forced_vs_recursive"], indent=2)
    )
    (diag_dir / "eval_states.json").write_text(json.dumps(result["eval_states"], indent=2, ensure_ascii=False))
    rows = _aggregate(result["per_run"])
    (diag_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    (diag_dir / "metrics_table.md").write_text(_markdown(rows))
    written = _figures(rows, run_dir / "figures")

    # confirm no evaluation start appears in the training state groups
    train_keys = {key for key, split in dataset.state_split.items() if split == "train"}
    overlap = [entry for entry in result["eval_states"] if entry["state_key"] in train_keys]
    summary = {
        "config": config.as_dict(),
        "eval_state_count": len(result["eval_states"]),
        "eval_states_in_train": len(overlap),
        "model_checkpoint_sha256": {str(k): v for k, v in hashes.items()},
        "figures": written,
        "wall_clock": result["wall_clock"],
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(_markdown(rows))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
