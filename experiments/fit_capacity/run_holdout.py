"""Phase D1: fixed holdout evaluation of the selected checkpoints (original + branch test).

    python experiments/fit_capacity/run_holdout.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d_learning.model import load_model
from coimagine.fit_capacity.existing import FLAT_METRICS, flatten, write_csv
from coimagine.fit_capacity.metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]
SELECTIONS = ("best_mixed_val", "best_original_val", "fixed_6000")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    original_test = [r for r in original if r["split"] == "test"]
    branch_all = [json.loads(line) for line in (upgrade / "branch_data" / "branch_windows.jsonl").open()]
    branch_test = [r for r in branch_all if r["split"] == "test"]
    print(f"original test {len(original_test)} windows, branch test {len(branch_test)} windows", flush=True)

    cache = {}
    rows = []
    for cell in sorted((run_dir / "runs").iterdir()):
        summary = json.loads((cell / "run_summary.json").read_text())
        for selection in SELECTIONS:
            checkpoint = summary["checkpoints"].get(selection)
            if checkpoint is None or not Path(checkpoint["path"]).exists():
                rows.append({"cell": cell.name, "selection": selection, "status": "missing"})
                continue
            digest = checkpoint["sha256"]
            if digest not in cache:
                model = load_model(Path(checkpoint["path"]))
                cache[digest] = {
                    "original_test": evaluate(model, original_test),
                    "branch_test": evaluate(model, branch_test),
                }
                print(f"evaluated {cell.name}/{selection} (updates={checkpoint['updates']})", flush=True)
            for target, metrics in cache[digest].items():
                rows.append(
                    {
                        "cell": cell.name, "group": summary["group"], "width": summary["width"],
                        "seed": summary["seed"], "selection": selection,
                        "checkpoint_updates": checkpoint["updates"], "sha256": digest,
                        "target": target, "status": "ok", **flatten(metrics),
                    }
                )
    names = ["cell", "group", "width", "seed", "selection", "checkpoint_updates", "sha256", "target", "status"] + list(FLAT_METRICS)
    write_csv(run_dir / "holdout_metrics.csv", rows, fieldnames=names)
    details = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        details[f"{row['cell']}|{row['selection']}|{row['target']}"] = row
    (run_dir / "holdout_details.json").write_text(json.dumps(details, indent=2, sort_keys=True))
    print("HOLDOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
