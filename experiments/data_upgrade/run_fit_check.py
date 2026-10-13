"""Phase B2: encoding round-trip, wrapped-angle loss check and the 32-sample fit check.

    python experiments/data_upgrade/run_fit_check.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.data_upgrade.fit_check import (
    angle_residual_check,
    evaluate_fit,
    round_trip_check,
    select_fit_samples,
    train_fit_model,
)

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "fit_check"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source_run
    windows = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]

    result = {
        "round_trip": round_trip_check(windows),
        "angle_residual": angle_residual_check(),
    }
    records, selection = select_fit_samples(windows)
    result["selection"] = {key: value for key, value in selection.items() if key != "selected"}
    model, training = train_fit_model(records)
    result["training"] = training
    result["evaluation"] = evaluate_fit(model, records)
    result["verdict"] = "FIT_CHECK_OK" if result["evaluation"]["passed"] else "FIT_CHECK_FAILED"
    (out_dir / "fit_check.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str))
    torch.save({"state_dict": model.state_dict(), "seed": 0}, out_dir / "fit_check_seed0.pt")
    print(json.dumps({k: result[k] for k in ("round_trip", "angle_residual", "evaluation", "verdict")}, indent=2, default=str))
    print(f"selection: {json.dumps({k: v for k, v in result['selection'].items() if k != 'sample_ids'})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
