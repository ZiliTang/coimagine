"""Phase B4: test-split metrics and the state-persistence comparison.

    python experiments/forecast_baseline/model_eval.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import torch

# Single-sample inference: extra BLAS threads cost far more than they save here.
torch.set_num_threads(1)

from coimagine.continuous2d_learning.evaluation import evaluate_records, persistence_report
from coimagine.continuous2d_learning.model import load_model, load_window_records

ROOT = Path(__file__).resolve().parents[2]
METRIC_KEYS = (
    "object_position_error", "robot0_position_error", "robot1_position_error",
    "object_velocity_error", "robot0_velocity_error", "robot1_velocity_error",
    "object_angle_error", "hold_error", "duration_error", "invalid_rate",
)


def _aggregate(reports):
    out = {}
    for key in METRIC_KEYS:
        values = [float(report["groups"]["all"][key]) for report in reports]
        out[key] = {
            "mean": float(np.mean(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "per_seed": values,
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--data-root", default="outputs/forecast_baseline_v1")
    parser.add_argument("--model-seeds", default="0,1,2")
    args = parser.parse_args()

    run_dir = ROOT / args.data_root / args.run_id
    records = load_window_records(run_dir / "dataset" / "windows.jsonl", "test")
    seeds = [int(value) for value in args.model_seeds.split(",") if value]
    result = {
        "n_test_windows": len(records),
        "scenes": sorted({str(record["scene_id"]) for record in records}),
        "persistence": persistence_report(records),
        "models": {},
    }
    reports = []
    for seed in seeds:
        model = load_model(run_dir / "models" / f"seed{seed}.pt")
        report = evaluate_records(model, records)
        result["models"][str(seed)] = report
        reports.append(report)
        print(
            f"seed {seed}: object_pos={report['groups']['all']['object_position_error']:.4f} "
            f"robot_pos=({report['groups']['all']['robot0_position_error']:.4f},"
            f"{report['groups']['all']['robot1_position_error']:.4f}) "
            f"obj_angle={report['groups']['all']['object_angle_error']:.4f} "
            f"duration={report['groups']['all']['duration_error']:.4f} "
            f"invalid={report['raw_invalid_rate']:.4f} "
            f"false_success={report['false_success']}/{report['predicted_success']}"
        )
    result["aggregate"] = _aggregate(reports)
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print("persistence:", json.dumps(result["persistence"], indent=2))
    print("MODEL_EVAL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
