"""C1: termination-CE attribution for the 36 logical checkpoints.

    python experiments/fit_repair/run_c1_termination.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d_learning.model import load_model
from coimagine.fit_capacity.checkpoints import iter_checkpoints
from coimagine.fit_capacity.termination import termination_attribution

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-repair-20261013")
    parser.add_argument("--out-root", default="outputs/fit_repair_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "c1_termination"
    out_dir.mkdir(parents=True, exist_ok=True)
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    datasets = {
        "train": [r for r in (
            [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
            + [json.loads(line) for line in (upgrade / "branch_data" / "random_train_matched.jsonl").open()]
        ) if r["split"] in ("train",)],
        "original_val": [r for r in original if r["split"] == "val"],
        "branch_val": [r for r in (json.loads(line) for line in (upgrade / "branch_data" / "branch_windows.jsonl").open()) if r["split"] == "val"],
    }
    # the two augmented training sets differ; evaluate each checkpoint on its own training set
    random_matched = [json.loads(line) for line in (upgrade / "branch_data" / "random_train_matched.jsonl").open()]
    branch_matched = [json.loads(line) for line in (upgrade / "branch_data" / "branch_train_matched.jsonl").open()]
    base_train = [r for r in original if r["split"] == "train"]
    train_by_group = {"Old+Random": base_train + random_matched, "Old+Branch": base_train + branch_matched}

    rows = []
    for checkpoint in iter_checkpoints(run_dir):
        model = load_model(Path(checkpoint["path"]))
        entry = {"label": checkpoint["label"], **{k: checkpoint[k] for k in
                 ("group", "width", "model_seed", "selection", "updates", "checkpoint_hash")}}
        for name, records in (
            ("train", train_by_group[str(checkpoint["group"])]),
            ("original_val", datasets["original_val"]),
            ("branch_val", datasets["branch_val"]),
        ):
            entry[name] = termination_attribution(model, records)
        rows.append(entry)
        print(f"{checkpoint['label']} updates={checkpoint['updates']} done", flush=True)
    (out_dir / "termination_attribution.json").write_text(json.dumps(rows, indent=2, sort_keys=True))
    print("C1_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
