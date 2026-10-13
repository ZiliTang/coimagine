"""Phase D2: two separate test tables for the nine group checkpoints.

    python experiments/data_upgrade/run_group_eval.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.group_eval import evaluate_branch_set
from coimagine.data_upgrade.metrics import chain_tolerance_metrics, evaluate_grouped, paired_persistence
from coimagine.data_upgrade.training import GROUP_BRANCH, GROUP_OLD, GROUP_RANDOM

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--model-subdir", default="models")
    parser.add_argument("--tag", default="combined")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "group_eval"
    model_dir = run_dir / args.model_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source_run
    original = _load(source / "dataset" / "windows.jsonl")
    original_test = [r for r in original if r["split"] == "test"]
    branch_all = _load(run_dir / "branch_data" / "branch_windows.jsonl")
    branch_test = [r for r in branch_all if r["split"] == "test"]

    report = {"checkpoint_rule": args.tag, "model_subdir": args.model_subdir,
              "original_test_windows": len(original_test), "branch_test_windows": len(branch_test), "models": {}}
    for group in (GROUP_OLD, GROUP_RANDOM, GROUP_BRANCH):
        report["models"][group] = {}
        for seed in (int(v) for v in args.model_seeds.split(",") if v):
            model = load_model(model_dir / f"{group.replace('+', '_')}_seed{seed}.pt")
            entry = {
                "original_test": {
                    "grouped": evaluate_grouped(model, original_test),
                    "persistence": paired_persistence(model, original_test),
                    "chains": chain_tolerance_metrics(model, original, splits=["test"]),
                },
                "branch_test": evaluate_branch_set(model, branch_test),
            }
            report["models"][group][str(seed)] = entry
            grouped = entry["original_test"]["grouped"]["groups"]["all"]
            branch = entry["branch_test"]
            print(
                f"{group} seed {seed}: orig obj_pos={grouped['object_position_error']:.4f} "
                f"invalid={grouped['invalid_rate']:.4f} | branch obj_pos={branch['object_position_error']:.4f} "
                f"invalid={branch['invalid_rate']:.4f} recall={branch['success_recall']} precision={branch['success_precision']}",
                flush=True,
            )
    (out_dir / f"group_eval_{args.tag}.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("GROUP_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
