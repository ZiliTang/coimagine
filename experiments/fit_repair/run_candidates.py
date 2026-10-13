"""D2: fixed-candidate review of the 36 logical checkpoints with common-subset pairing.

    python experiments/fit_repair/run_candidates.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.fixed_candidates import load_reference, verify_and_rebuild
from coimagine.fit_capacity.candidates import aggregate, evaluate_model
from coimagine.fit_capacity.checkpoints import iter_checkpoints
from coimagine.fit_capacity.paired import compare

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
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
    cache = {}
    for checkpoint in iter_checkpoints(run_dir):
        key = checkpoint["checkpoint_hash"]
        if key not in cache:
            model = load_model(Path(checkpoint["path"]))
            cache[key] = evaluate_model(model, starts, reference, config, label=checkpoint["label"])
            for row in cache[key]:
                row.update({k: checkpoint[k] for k in ("group", "width", "model_seed", "selection", "checkpoint_hash")})
            print(f"evaluated {checkpoint['label']} updates={checkpoint['updates']}", flush=True)
        rows.extend(cache[key])

    with (out_dir / "rows.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    table = aggregate(rows)
    (out_dir / "aggregate.json").write_text(json.dumps(table, indent=2, sort_keys=True))

    by_label = {row["label"]: [r for r in rows if r["label"] == row["label"]] for row in table}
    comparisons = []
    for group in ("Old+Random", "Old+Branch"):
        for width in (256, 512):
            for seed in (0, 1, 2):
                mixed = f"{group}|w{width}|seed{seed}|best_mixed_val"
                original = f"{group}|w{width}|seed{seed}|best_original_val"
                final = f"{group}|w{width}|seed{seed}|fixed_6000"
                if mixed in by_label and original in by_label:
                    comparisons.append({"kind": "mixed->original", "group": group, "width": width, "seed": seed,
                                        **compare(by_label[mixed], by_label[original], label_a=mixed, label_b=original)})
                if mixed in by_label and final in by_label:
                    comparisons.append({"kind": "mixed->final", "group": group, "width": width, "seed": seed,
                                        **compare(by_label[mixed], by_label[final], label_a=mixed, label_b=final)})
    for group in ("Old+Random", "Old+Branch"):
        for seed in (0, 1, 2):
            for selection in ("best_mixed_val", "best_original_val", "fixed_6000"):
                first = f"{group}|w256|seed{seed}|{selection}"
                second = f"{group}|w512|seed{seed}|{selection}"
                if first in by_label and second in by_label:
                    comparisons.append({"kind": "256->512", "group": group, "selection": selection, "seed": seed,
                                        **compare(by_label[first], by_label[second], label_a=first, label_b=second)})
    (out_dir / "paired.json").write_text(json.dumps(comparisons, indent=2, sort_keys=True))

    # identity checks
    worst = 0.0
    for row in rows:
        if row["no_feasible_candidate"]:
            continue
        residual = abs((row["misrejection_loss"] + row["ranking_loss"]) - row["execution_regret"])
        worst = max(worst, residual)
    (out_dir / "identity_check.json").write_text(json.dumps({
        "rows": len(rows), "checkpoints": len(table), "max_decomposition_residual": worst,
        "exact_control_note": "the J_hat = J_ref control is the exact row already verified in data_upgrade_v1 and reused here",
    }, indent=2))

    lines = ["# D2 固定候选（36 个逻辑 checkpoint，H=3，K=128；按起点先平均候选种子再跨起点）", "",
             "| 标签 | 未筛选 chosen_true | candidate_best | uniform | selected_bias | regret | 拒绝率 | 无解 pool 行 | 无解起点 | 仅可行 chosen_true (n) | 误拒损失 | 排序损失 | 执行 regret |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, digits=4):
        return "null" if value is None else f"{float(value):+.{digits}f}"

    for row in table:
        lines.append(
            f"| {row['label']} | {fmt(row['chosen_true_return'])} | {fmt(row['candidate_best_true_return'])} | "
            f"{fmt(row['uniform_candidate_true_return'])} | {fmt(row['selected_bias'])} | {fmt(row['unfiltered_regret'])} | "
            f"{fmt(row['rejection_rate'], 3)} | {row['starts_without_feasible']} | {row['starts_without_feasible']} | "
            f"{fmt(row['feasible_chosen_true_return'])} ({row['feasible_chosen_true_return_n_starts']}) | "
            f"{fmt(row['misrejection_loss'])} | {fmt(row['ranking_loss'])} | {fmt(row['execution_regret'])} |"
        )
    (out_dir / "metrics_table.md").write_text("\n".join(lines) + "\n")
    summary = {
        "checkpoints": len(table),
        "rows": len(rows),
        "comparisons": len(comparisons),
        "improved_selection_changed": sum(
            1 for c in comparisons if c["kind"] == "mixed->original" and (c["delta_common"]["feasible_chosen_true_return"] or 0) > 0
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    print("CANDIDATES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
