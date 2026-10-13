"""Phase D1: train the three data groups, three seeds each.

    python experiments/data_upgrade/run_training_groups.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(1)

from coimagine.continuous2d_learning.model import ModelConfig
from coimagine.data_upgrade.training import (
    GROUP_BRANCH,
    GROUP_OLD,
    GROUP_RANDOM,
    train_group,
)

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--model-seeds", default="0,1,2")
    parser.add_argument("--val-branch", choices=("on", "off"), default="on",
                        help="off = the secondary uniform rule (original val loss only)")
    parser.add_argument("--model-subdir", default="models")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    branch_dir = run_dir / "branch_data"
    source = ROOT / args.source_run
    model_dir = run_dir / args.model_subdir
    model_dir.mkdir(parents=True, exist_ok=True)

    original = _load(source / "dataset" / "windows.jsonl")
    original_train = [r for r in original if r["split"] == "train"]
    original_val = [r for r in original if r["split"] == "val"]
    branch_all = _load(branch_dir / "branch_windows.jsonl")
    branch_val = [r for r in branch_all if r["split"] == "val"]
    random_matched = _load(branch_dir / "random_train_matched.jsonl")
    branch_matched = _load(branch_dir / "branch_train_matched.jsonl")

    groups = {
        GROUP_OLD: original_train,
        GROUP_RANDOM: original_train + random_matched,
        GROUP_BRANCH: original_train + branch_matched,
    }
    summary = {
        "checkpoint_rule_selected": args.val_branch,
        "model_subdir": args.model_subdir,
        "group_sizes": {name: len(records) for name, records in groups.items()},
        "added_random": len(random_matched),
        "added_branch": len(branch_matched),
        "val_original": len(original_val),
        "val_branch": len(branch_val),
        "checkpoint_rule": "0.5 * mean original val loss + 0.5 * mean new-branch val loss",
        "models": {},
    }
    started = time.time()
    for group, records in groups.items():
        summary["models"][group] = {}
        for seed in (int(v) for v in args.model_seeds.split(",") if v):
            _model, training = train_group(
                group, records, original_val, branch_val if args.val_branch == "on" else [], seed=seed,
                config=ModelConfig(),
                checkpoint_path=model_dir / f"{group.replace('+', '_')}_seed{seed}.pt",
            )
            summary["models"][group][str(seed)] = training.as_dict()
            print(
                f"{group} seed {seed}: train={training.train_samples} best_epoch={training.best_epoch} "
                f"val_score={training.best_val_score:.5f} updates={training.updates} "
                f"wall={training.wall_clock:.1f}s",
                flush=True,
            )
    summary["wall_clock"] = time.time() - started
    name = "training_groups.json" if args.val_branch == "on" else "training_groups_originalval.json"
    (run_dir / name).write_text(json.dumps(summary, indent=2, sort_keys=True))
    print("TRAINING_GROUPS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
