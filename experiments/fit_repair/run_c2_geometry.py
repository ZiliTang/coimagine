"""C2: geometric-violation attribution with offline pose swaps (two fixed runs).

    python experiments/fit_repair/run_c2_geometry.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d_learning.model import load_model
from coimagine.fit_capacity.checkpoints import iter_checkpoints
from coimagine.fit_capacity.geometry_attribution import geometry_attribution

ROOT = Path(__file__).resolve().parents[2]
FIXED_CELLS = ("Old_Random_256_seed0", "Old_Branch_256_seed0")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "c2_geometry"
    out_dir.mkdir(parents=True, exist_ok=True)
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    base_train = [r for r in original if r["split"] == "train"]
    random_matched = [json.loads(line) for line in (upgrade / "branch_data" / "random_train_matched.jsonl").open()]
    branch_matched = [json.loads(line) for line in (upgrade / "branch_data" / "branch_train_matched.jsonl").open()]
    train_by_group = {"Old+Random": base_train + random_matched, "Old+Branch": base_train + branch_matched}
    branch_val = [r for r in (json.loads(line) for line in (upgrade / "branch_data" / "branch_windows.jsonl").open()) if r["split"] == "val"]

    report = {}
    for checkpoint in iter_checkpoints(run_dir):
        if checkpoint["cell"] not in FIXED_CELLS:
            continue
        model = load_model(Path(checkpoint["path"]))
        entry = {}
        for name, records in (
            ("train", train_by_group[str(checkpoint["group"])]),
            ("branch_val", branch_val),
        ):
            entry[name] = geometry_attribution(model, records)
        report[checkpoint["label"]] = entry
        print(f"{checkpoint['label']} done", flush=True)
    (out_dir / "geometry_attribution.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print("C2_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
