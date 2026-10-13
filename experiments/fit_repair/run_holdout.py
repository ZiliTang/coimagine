"""D1: holdout evaluation of the 36 logical checkpoints (original + branch test).

    python experiments/fit_repair/run_holdout.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d_learning.model import load_model
from coimagine.fit_capacity.checkpoints import iter_checkpoints
from coimagine.fit_capacity.existing import FLAT_METRICS, flatten, write_csv
from coimagine.fit_capacity.metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    original_test = [r for r in original if r["split"] == "test"]
    branch_test = [r for r in (json.loads(line) for line in (upgrade / "branch_data" / "branch_windows.jsonl").open()) if r["split"] == "test"]
    print(f"original test {len(original_test)}; branch test {len(branch_test)}", flush=True)

    cache = {}
    rows = []
    for checkpoint in iter_checkpoints(run_dir):
        key = checkpoint["checkpoint_hash"]
        if key not in cache:
            model = load_model(Path(checkpoint["path"]))
            cache[key] = {
                "original_test": evaluate(model, original_test),
                "branch_test": evaluate(model, branch_test),
            }
            print(f"evaluated {checkpoint['label']} (updates={checkpoint['updates']})", flush=True)
        for target, metrics in cache[key].items():
            rows.append({**{k: checkpoint[k] for k in
                            ("group", "width", "model_seed", "selection", "updates", "checkpoint_hash", "file_sha256", "label")},
                         "target": target, **flatten(metrics)})
    write_csv(run_dir / "holdout_metrics.csv", rows,
              fieldnames=["group", "width", "model_seed", "selection", "updates", "checkpoint_hash", "file_sha256", "label", "target"] + list(FLAT_METRICS))
    (run_dir / "holdout_details.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(f"rows {len(rows)}; unique parameter hashes {len(cache)}")
    print("HOLDOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
