"""Phase D3: fixed-candidate evaluation of the trained groups on the frozen pools.

    python experiments/data_upgrade/run_fixed_candidates.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.fixed_candidates import (
    aggregate,
    evaluate_group,
    load_reference,
    verify_and_rebuild,
)
from coimagine.data_upgrade.training import GROUP_BRANCH, GROUP_OLD, GROUP_RANDOM

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--include-old", action="store_true", help="also evaluate the three original checkpoints")
    parser.add_argument("--model-subdir", default="models")
    parser.add_argument("--tag", default="combined")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "fixed_candidates"
    out_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    source = ROOT / args.source_run
    reference = load_reference(source, config)
    starts, verification = verify_and_rebuild(source, config, reference)
    (out_dir / "reference_verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True))

    rows = []
    job_list = [(f"{group}@{args.tag}", run_dir / args.model_subdir / f"{group.replace('+', '_')}_seed{seed}.pt")
                for group in (GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH)
                for seed in (int(v) for v in args.model_seeds.split(",") if v)]
    if args.include_old:
        job_list += [("Old_v1", source / "models" / f"seed{seed}.pt")
                     for seed in (int(v) for v in args.model_seeds.split(",") if v)]
    for group, path in job_list:
        seed = int(path.stem.split("seed")[-1])
        model = load_model(path)
        result = evaluate_group(model, starts, reference, config, group=group, seed=seed)
        rows.extend(result["rows"])
        print(f"{group} seed {seed}: {len(result['rows'])} rows", flush=True)
    with (out_dir / f"rows_{args.tag}.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    summary = aggregate(rows)
    (out_dir / f"aggregate_{args.tag}.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    lines = ["# 固定候选：三组 × 三种子（全候选按 J 排序；可行性筛选单列在 aggregate_feasible）", "",
             "| 组 | H | K | model | starts | selected_bias | mean_bias | excess | regret | chosen_true | uniform_true | best_true | cand_invalid | rejection_rate | feasible_chosen_true | conditional_regret (n) |",
             "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

    def fmt(value, digits=4):
        return "NA" if value is None else f"{float(value):+.{digits}f}"

    for row in summary:
        lines.append(
            f"| {row['group']} | {row['H']} | {row['K']} | {row['model_seed']} | {row['n_starts']} | "
            f"{fmt(row['selected_bias'])} | {fmt(row['candidate_mean_bias'])} | {fmt(row['selection_excess'])} | "
            f"{fmt(row['candidate_regret'])} | {fmt(row['chosen_true_return'])} | {fmt(row['uniform_candidate_true_return'])} | "
            f"{fmt(row['candidate_best_true_return'])} | {fmt(row['candidate_invalid_rate'],3)} | {fmt(row['rejection_rate'],3)} | "
            f"{fmt(row['feasible_chosen_true_return'])} | {fmt(row['conditional_regret'])} ({row.get('conditional_regret_n_starts', 0)}) |"
        )
    (out_dir / f"metrics_table_{args.tag}.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(verification, indent=2))
    print("FIXED_CANDIDATES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
