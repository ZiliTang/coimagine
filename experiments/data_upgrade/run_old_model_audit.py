"""Phase B3: grouped metrics, paired persistence and tolerance-chain metrics for the three
original checkpoints.

    python experiments/data_upgrade/run_old_model_audit.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.metrics import (
    chain_tolerance_metrics,
    evaluate_grouped,
    paired_persistence,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--model-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--model-seeds", default="0,1,2")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "old_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.model_run
    windows = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    by_split = {split: [r for r in windows if r["split"] == split] for split in ("train", "val", "test")}

    report = {"models": {}}
    for seed in (int(v) for v in args.model_seeds.split(",") if v):
        model = load_model(source / "models" / f"seed{seed}.pt")
        entry = {}
        for split, records in by_split.items():
            entry[split] = {
                "grouped": evaluate_grouped(model, records),
                "persistence": paired_persistence(model, records),
            }
        entry["chains"] = {
            split: chain_tolerance_metrics(model, windows, splits=[split]) for split in ("train", "val", "test")
        }
        report["models"][str(seed)] = entry
        test_all = entry["test"]["grouped"]["groups"]["all"]
        print(
            f"seed {seed} test/all: obj_pos={test_all['object_position_error']:.4f} "
            f"robot_pos=({test_all['robot0_position_error']:.4f},{test_all['robot1_position_error']:.4f}) "
            f"angle={test_all['object_angle_error']:.4f} invalid={test_all['invalid_rate']:.4f} n={test_all['n']}"
        )
        print(f"   test persistence object_position mean diff = {entry['test']['persistence']['object_position']['mean_difference']:+.4f}")
        chains = entry["chains"]["test"]["all_chains"]
        print(f"   test chains={entry['chains']['test']['chains']} recursion cumulative termination={[round(v,3) for v in chains['cumulative_termination_rate'][:4]]}")
    (out_dir / "old_model_metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    print("OLD_MODEL_AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
