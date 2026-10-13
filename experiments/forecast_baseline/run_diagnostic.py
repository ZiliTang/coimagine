"""Phase C: fixed-candidate ranking diagnostic on the continuous test scenes.

    python experiments/forecast_baseline/run_diagnostic.py --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import torch

# Single-sample inference: extra BLAS threads cost far more than they save here.
torch.set_num_threads(1)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.diagnostic import DiagnosticConfigV1, run_diagnostic
from coimagine.continuous2d_learning.model import load_model
from coimagine.continuous2d_learning.records import CollectionConfig
from coimagine.continuous2d_learning.starts import select_eval_starts

ROOT = Path(__file__).resolve().parents[2]
TABLE_KEYS = (
    "selected_bias", "candidate_mean_bias", "selection_excess", "candidate_regret",
    "chosen_true_return", "uniform_candidate_true_return", "candidate_best_true_return",
    "chosen_true_success", "candidate_success_rate", "candidate_invalid_rate",
    "candidate_true_failure_rate",
)


def _markdown(rows) -> str:
    lines = [
        "# 连续平台固定候选排序表（按起点先平均，再跨起点汇总）",
        "",
        "| H | K | model | starts | selected_bias | candidate_mean_bias | selection_excess | regret | chosen_true | uniform_true | candidate_best | chosen_succ | cand_succ | cand_invalid | cand_true_fail |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        model = "exact" if row["model_seed"] == -1 else str(row["model_seed"])
        lines.append(
            f"| {row['H']} | {row['K']} | {model} | {row['n_starts']} | "
            f"{row['selected_bias']:+.4f} | {row['candidate_mean_bias']:+.4f} | "
            f"{row['selection_excess']:+.4f} | {row['candidate_regret']:.4f} | "
            f"{row['chosen_true_return']:+.4f} | {row['uniform_candidate_true_return']:+.4f} | "
            f"{row['candidate_best_true_return']:+.4f} | {row['chosen_true_success']:.3f} | "
            f"{row['candidate_success_rate']:.3f} | {row['candidate_invalid_rate']:.3f} | "
            f"{row['candidate_true_failure_rate']:.3f} |"
        )
    return "\n".join(lines) + "\n"


def _markdown_feasible(rows) -> str:
    lines = [
        "# 连续平台固定候选：A3 仅可行候选（window_complete/success）筛选后的结果",
        "",
        "全候选按 J 排序的表见 metrics_table.md；两表不得混用。`NA` 表示该 (H,K,model) 下",
        "存在没有任何可行候选的起点，conditional_regret 记 NA，不填 0、不删起点。",
        "",
        "| H | K | model | starts | feasible | rejected | rejection_rate | feasible_chosen_true | feasible_chosen_succ | conditional_regret (n) | NA_starts |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def _fmt(value, digits=4):
        return "NA" if value is None else f"{float(value):+.{digits}f}"

    for row in rows:
        model = "exact" if row["model_seed"] == -1 else str(row["model_seed"])
        na_starts = int(row["n_starts"]) - int(row.get("conditional_regret_n_starts", 0))
        lines.append(
            f"| {row['H']} | {row['K']} | {model} | {row['n_starts']} | "
            f"{_fmt(row['feasible_count'], 2)} | {_fmt(row['rejected_count'], 2)} | "
            f"{_fmt(row['rejection_rate'])} | {_fmt(row['feasible_chosen_true_return'])} | "
            f"{_fmt(row['feasible_chosen_true_success'], 3)} | "
            f"{_fmt(row['conditional_regret'])} ({row.get('conditional_regret_n_starts', 0)}) | {na_starts} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--out-root", default="outputs/forecast_baseline_v1")
    parser.add_argument("--depths", default="1,2,3")
    parser.add_argument("--sizes", default="16,64,128")
    parser.add_argument("--candidate-seeds", default="0,1,2")
    parser.add_argument("--model-seeds", default="0,1,2")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    diag_dir = run_dir / "diagnostic"
    diag_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    model_seeds = [int(value) for value in args.model_seeds.split(",") if value]
    models = {seed: load_model(run_dir / "models" / f"seed{seed}.pt") for seed in model_seeds}
    hashes = {}
    for seed in model_seeds:
        digest = hashlib.sha256((run_dir / "models" / f"seed{seed}.pt").read_bytes()).hexdigest()
        hashes[seed] = digest

    starts = select_eval_starts(
        run_dir / "dataset" / "windows.jsonl", config=config, collection=CollectionConfig()
    )
    (diag_dir / "eval_starts.json").write_text(
        json.dumps(
            [
                {
                    **start.recoverable(),
                    "config_hash": config.config_hash(),
                    "collection": CollectionConfig().as_dict(),
                }
                for start in starts
            ],
            indent=2,
            sort_keys=True,
        )
    )
    diagnostic = DiagnosticConfigV1(
        depths=tuple(int(v) for v in args.depths.split(",") if v),
        sizes=tuple(int(v) for v in args.sizes.split(",") if v),
        candidate_seeds=tuple(int(v) for v in args.candidate_seeds.split(",") if v),
        model_seeds=tuple(model_seeds),
    )
    result = run_diagnostic(
        starts, models, config=config, diagnostic=diagnostic, checkpoint_hashes=hashes
    )

    (diag_dir / "config.json").write_text(json.dumps(diagnostic.as_dict(), indent=2, sort_keys=True))
    with (diag_dir / "per_run.jsonl").open("w") as handle:
        for row in result["per_run"]:
            handle.write(json.dumps(row) + "\n")
    with (diag_dir / "candidate_records.jsonl").open("w") as handle:
        for row in result["candidate_records"]:
            handle.write(json.dumps(row) + "\n")
    (diag_dir / "per_start.json").write_text(json.dumps(result["per_start"], indent=2, sort_keys=True))
    (diag_dir / "aggregate.json").write_text(json.dumps(result["aggregate"], indent=2, sort_keys=True))
    (diag_dir / "per_start_feasible.json").write_text(
        json.dumps(result["per_start_feasible"], indent=2, sort_keys=True)
    )
    (diag_dir / "aggregate_feasible.json").write_text(
        json.dumps(result["aggregate_feasible"], indent=2, sort_keys=True)
    )
    (diag_dir / "metrics_table_feasible.md").write_text(_markdown_feasible(result["aggregate_feasible"]))
    (diag_dir / "coverage.json").write_text(
        json.dumps(
            {"rows": result["coverage_rows"], "summary": result["coverage_summary"]},
            indent=2, sort_keys=True,
        )
    )
    (diag_dir / "recursive.json").write_text(json.dumps(result["recursive"], indent=2, sort_keys=True))
    (diag_dir / "traced_cases.json").write_text(json.dumps(result["traced_cases"], indent=2, sort_keys=True))
    (diag_dir / "metrics_table.md").write_text(_markdown(result["aggregate"]))
    summary = {
        "config": diagnostic.as_dict(),
        "n_starts": len(starts),
        "starts": [start.recoverable() for start in starts],
        "coverage_summary": result["coverage_summary"],
        "model_checkpoint_sha256": {str(key): value for key, value in hashes.items()},
        "wall_clock": result["wall_clock"],
    }
    (diag_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(_markdown(result["aggregate"]))
    print(_markdown_feasible(result["aggregate_feasible"]))
    print(json.dumps(result["coverage_summary"], indent=2))
    print(json.dumps(summary["wall_clock"]))
    print("DIAGNOSTIC_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
