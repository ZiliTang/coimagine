"""Phase D2: the single final holdout evaluation.

    python experiments/coverage_scale/run_holdout.py --run-id coverage-scale-20261013 \
        --tiers 8000,16000 --narrow-tier 16000

After the run list is frozen, every completed tier's main selection point and fixed-budget end
point, plus the equal-size narrow-coverage control, is evaluated **once** on the new test set.
The old test set is evaluated only for the first tier and the final tier, and only at the main
selection point, as a historical regression attachment. Test scores never feed back into
training, scale selection or resampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig  # noqa: E402
from coimagine.coverage_scale.evaluation import plain_summary, weighted_summary  # noqa: E402
from coimagine.coverage_scale.plan import load_registries  # noqa: E402
from coimagine.coverage_scale.shards import tier_records  # noqa: E402

OLD_TEST = "outputs/forecast_baseline_v1/forecast-20261012/dataset/windows.jsonl"


def load_model(path: Path) -> JointWindowMLP:
    blob = torch.load(path, map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--tiers", default="8000")
    parser.add_argument("--widths", default="256,512")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--narrow-tier", default="")
    parser.add_argument("--first-tier", default="8000")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    config = default_config()
    registry = load_registries(run_dir / "roots", ("train", "val", "test", "narrow"))
    registry.activate()

    new_test, test_info = tier_records(run_dir, "test", 10**9)
    old_path = ROOT / OLD_TEST
    old_test = []
    if old_path.exists():
        old_test = [json.loads(line) for line in old_path.open()]
        old_test = [record for record in old_test if record["split"] == "test"]

    widths = [int(value) for value in args.widths.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    tiers = [value for value in args.tiers.split(",") if value]

    jobs = []
    for tier in tiers:
        for width in widths:
            for seed in seeds:
                for kind in ("best_new_val_total", "final"):
                    path = run_dir / "runs" / f"{tier}_{width}_seed{seed}" / f"{kind}.pt"
                    if path.exists():
                        jobs.append((f"{tier}/{width}/{seed}/{kind}", path, tier))
    if args.narrow_tier:
        for width in widths:
            for seed in seeds:
                for kind in ("best_new_val_total", "final"):
                    path = run_dir / "narrow" / f"{args.narrow_tier}_{width}_seed{seed}" / f"{kind}.pt"
                    if path.exists():
                        jobs.append((f"narrow_{args.narrow_tier}/{width}/{seed}/{kind}", path, args.narrow_tier))

    rows = []
    started = time.time()
    for label, path, tier in jobs:
        model = load_model(path)
        new_summary = weighted_summary(model, new_test)
        entry = {
            "label": label, "tier": tier, "checkpoint": str(path.relative_to(ROOT)),
            "new_test": {
                "n": new_summary["n"],
                "weighted": new_summary["weighted"],
                "unweighted": new_summary["unweighted"],
                "legality": new_summary["legality"],
                "success": new_summary["success"],
                "cells": new_summary["cells"],
                "persistence_weighted_object_position": new_summary["persistence_weighted_object_position"],
            },
            "old_test": None,
        }
        if old_test and label.endswith("best_new_val_total") and tier in (args.first_tier, tiers[-1]):
            old_summary = plain_summary(model, old_test)
            entry["old_test"] = {
                "n": old_summary["n"], "unweighted": old_summary["unweighted"],
                "legality": old_summary["legality"], "success": old_summary["success"],
            }
        rows.append(entry)
        print(f"{label}: done ({time.time() - started:.1f}s)", flush=True)

    payload = {
        "new_test_windows": test_info,
        "old_test_rows": len(old_test),
        "rows": rows,
        "never_used_for_training": True,
    }
    (run_dir / "holdout.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({
        "labels": [row["label"] for row in rows],
        "new_test_P": {row["label"]: row["new_test"]["weighted"]["position_mean"] for row in rows},
    }, indent=1))
    print("HOLDOUT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
