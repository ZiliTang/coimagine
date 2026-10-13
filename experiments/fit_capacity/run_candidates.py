"""Phase D2: fixed-candidate review for the 12 best_mixed_val models.

    python experiments/fit_capacity/run_candidates.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.fixed_candidates import load_reference, verify_and_rebuild
from coimagine.fit_capacity.candidates import METRIC_KEYS, aggregate, evaluate_model

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    source = ROOT / args.source_run
    reference = load_reference(source, config)
    starts, verification = verify_and_rebuild(source, config, reference)
    (out_dir / "reference_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True))
    print(json.dumps(verification, indent=2), flush=True)

    rows = []
    labels = []
    for cell in sorted((run_dir / "runs").iterdir()):
        summary = json.loads((cell / "run_summary.json").read_text())
        checkpoint = summary["checkpoints"].get("best_mixed_val")
        if checkpoint is None:
            continue
        label = f"{summary['group']}|w{summary['width']}|seed{summary['seed']}"
        model = load_model(Path(checkpoint["path"]))
        rows.extend(evaluate_model(model, starts, reference, config, label=label))
        labels.append({"label": label, "updates": checkpoint["updates"], "sha256": checkpoint["sha256"]})
        print(f"evaluated {label} at updates={checkpoint['updates']}", flush=True)

    with (out_dir / "rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = aggregate(rows)
    (out_dir / "aggregate.json").write_text(json.dumps(table, indent=2, sort_keys=True))
    (out_dir / "labels.json").write_text(json.dumps(labels, indent=2, sort_keys=True))

    # 256 vs 512 paired comparison on starts where both have a feasible candidate
    by_label = {row["label"]: row for row in table}
    pairs = []
    for group in ("Old+Random", "Old+Branch"):
        for seed in (0, 1, 2):
            a = by_label.get(f"{group}|w256|seed{seed}")
            b = by_label.get(f"{group}|w512|seed{seed}")
            if not a or not b:
                continue
            entry = {"group": group, "seed": seed, "metric": METRIC_KEYS}
            for key in METRIC_KEYS:
                av, bv = a.get(key), b.get(key)
                entry[key] = {"w256": av, "w512": bv, "delta": (None if av is None or bv is None else bv - av)}
            pairs.append(entry)
    (out_dir / "paired_256_512.json").write_text(json.dumps(pairs, indent=2, sort_keys=True))

    lines = ["# D2 固定候选（best_mixed_val，H=3, K=128，按起点先平均候选种子再跨起点）", "",
             "| 模型 | 未筛选 chosen_true | candidate_best | uniform | selected_bias | regret | 拒绝率 | 无可行起点 | 仅可行 chosen_true (n) | 误拒损失 | 集合内排序损失 | 执行 regret |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, digits=4):
        return "null" if value is None else f"{float(value):+.{digits}f}"

    for row in table:
        lines.append(
            f"| {row['label']} | {fmt(row['chosen_true_return'])} | {fmt(row['candidate_best_true_return'])} | "
            f"{fmt(row['uniform_candidate_true_return'])} | {fmt(row['selected_bias'])} | {fmt(row['unfiltered_regret'])} | "
            f"{fmt(row['rejection_rate'], 3)} | {row['starts_without_feasible']} | "
            f"{fmt(row['feasible_chosen_true_return'])} ({row['feasible_chosen_true_return_n_starts']}) | "
            f"{fmt(row['misrejection_loss'])} | {fmt(row['ranking_loss'])} | {fmt(row['execution_regret'])} |"
        )
    (out_dir / "metrics_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("CANDIDATES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
