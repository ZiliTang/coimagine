"""Phase B1: geometry-check acceptance on the real dataset and on old model predictions.

    python experiments/data_upgrade/run_geometry_audit.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d.config import default_config
from coimagine.continuous2d_learning.model import load_model
from coimagine.data_upgrade.geometry_audit import (
    audit_engine_agreement,
    audit_predicted_states,
    audit_true_states,
    convention_probe,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_windows(path: Path):
    records = []
    with path.open() as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--model-seeds", default="0,1,2")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source_run
    windows = _load_windows(source / "dataset" / "windows.jsonl")
    config = default_config()

    true_report = audit_true_states(windows, config)
    (out_dir / "true_state_check.json").write_text(
        json.dumps({k: v for k, v in true_report.items() if k != "misclassified"}, indent=2, sort_keys=True, default=str)
    )
    with (out_dir / "misclassified_true_states.jsonl").open("w") as handle:
        for row in true_report["misclassified"]:
            handle.write(json.dumps(row) + "\n")

    probe = convention_probe(config)
    (out_dir / "convention_probe.json").write_text(json.dumps(probe, indent=2, sort_keys=True, default=str))

    engine_report = audit_engine_agreement(windows, config)
    (out_dir / "engine_agreement.json").write_text(json.dumps(engine_report, indent=2, sort_keys=True, default=str))

    predicted = {}
    for seed in (int(v) for v in args.model_seeds.split(",") if v):
        model = load_model(source / "models" / f"seed{seed}.pt")
        predicted[str(seed)] = audit_predicted_states(windows, model, config)
    (out_dir / "predicted_state_check.json").write_text(
        json.dumps(predicted, indent=2, sort_keys=True, default=str)
    )

    summary = {
        "n_windows": true_report["n_windows"],
        "true_state_violations": {
            group: row["violations"] for group, row in true_report["groups"].items()
        },
        "true_state_false_illegal_rate": {
            group: row["false_illegal_rate"] for group, row in true_report["groups"].items()
        },
        "true_state_violating_pairs": true_report["violating_pairs"],
        "engine_systematic_disagreement": engine_report["systematic_disagreement"],
        "engine_max_abs_difference": engine_report["max_abs_difference"],
        "engine_mean_abs_difference": engine_report["mean_abs_difference"],
        "engine_steps_compared": engine_report["steps"],
        "engine_convention": probe["conclusion"],
    }
    verdict = "GEOMETRY_BLOCKED" if engine_report["systematic_disagreement"] else "GEOMETRY_OK"
    summary["verdict"] = verdict
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=2))
    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
