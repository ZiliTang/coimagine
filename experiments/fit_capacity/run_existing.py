"""Phase A: errata inputs + full-dataset re-evaluation of the existing 18 checkpoints.

    python experiments/fit_capacity/run_existing.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.continuous2d_learning.model import load_model
from coimagine.fit_capacity.existing import (
    FLAT_METRICS,
    flatten,
    inventory,
    load_training_data,
    write_csv,
)
from coimagine.fit_capacity.metrics import evaluate

ROOT = Path(__file__).resolve().parents[2]


def _pair_analysis(run_dir: Path) -> dict:
    """A1: closed-loop pairing key must include the model seed."""

    rows = [json.loads(line) for line in (run_dir / "closed_loop_groups_origval" / "episodes.jsonl").open()]

    def key(row):
        return (row["scene_id"], row["episode_seed"], row["H"], row["model_seed"])

    report = {"pairing_key": "(scene_id, episode_seed, H, model_seed)"}
    for group in sorted({str(r["data_group"]) for r in rows}):
        keys = [key(r) for r in rows if r["data_group"] == group]
        report[group] = {"episodes": len(keys), "unique_keys": len(set(keys)), "unique": len(keys) == len(set(keys))}
    baseline = {key(r): r for r in rows if r["data_group"] == "Old@origval"}
    comparison = [r for r in rows if r["data_group"] == "Old+Branch@origval"]
    pairs = [(baseline[key(r)], r) for r in comparison if key(r) in baseline]
    deltas = [float(b["J"]) - float(a["J"]) for a, b in pairs]
    report["pairs"] = len(pairs)
    report["improved"] = sum(1 for d in deltas if d > 0)
    report["equal"] = sum(1 for d in deltas if d == 0)
    report["degraded"] = sum(1 for d in deltas if d < 0)
    report["mean_delta_J"] = sum(deltas) / len(deltas) if deltas else None
    report["best"] = max(
        (
            {"scene_id": b["scene_id"], "episode_seed": b["episode_seed"], "model_seed": b["model_seed"],
             "old_J": a["J"], "branch_J": b["J"], "delta_J": float(b["J"]) - float(a["J"])}
            for a, b in pairs
        ),
        key=lambda row: row["delta_J"],
        default=None,
    )
    report["worst"] = min(
        (
            {"scene_id": b["scene_id"], "episode_seed": b["episode_seed"], "model_seed": b["model_seed"],
             "old_J": a["J"], "branch_J": b["J"], "delta_J": float(b["J"]) - float(a["J"])}
            for a, b in pairs
        ),
        key=lambda row: row["delta_J"],
        default=None,
    )
    return report


def _count_analysis(run_dir: Path) -> dict:
    """A1: anchors vs success anchors vs windows, per split."""

    branch = [json.loads(line) for line in (run_dir / "branch_data" / "branch_windows.jsonl").open()]
    matched = [json.loads(line) for line in (run_dir / "branch_data" / "branch_train_matched.jsonl").open()]
    out = {}
    for label, rows in (("all_splits", branch), ("matched_train", matched)):
        entry = {}
        for split in sorted({str(r["split"]) for r in rows}):
            subset = [r for r in rows if r["split"] == split]
            anchors = {str((r.get("source") or {}).get("anchor_id")) for r in subset if (r.get("source") or {}).get("anchor_id")}
            success_anchors = {
                str((r.get("source") or {}).get("anchor_id"))
                for r in subset
                if (r.get("source") or {}).get("anchor_id") and int(r["success"]) == 1
            }
            entry[split] = {
                "windows": len(subset), "anchors": len(anchors),
                "success_anchors": len(success_anchors),
                "success_windows": sum(int(r["success"]) for r in subset),
            }
        out[label] = entry
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run

    audit = {
        "pairing": _pair_analysis(upgrade),
        "counts": _count_analysis(upgrade),
    }
    (run_dir / "a1_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True))
    print(json.dumps(audit["pairing"], indent=2))
    print(json.dumps(audit["counts"], indent=2))

    rows = inventory(upgrade, source)
    write_csv(run_dir / "existing_checkpoints.csv", rows)
    data = load_training_data(upgrade, source)
    metrics_rows = []
    details = {}
    for entry in rows:
        if not entry["exists"]:
            metrics_rows.append({**{k: entry[k] for k in ("rule", "group", "seed", "path", "sha256")}, "status": "missing"})
            continue
        model = load_model(ROOT / entry["path"])
        per_target = {}
        for target, records in (
            ("own_train", data["train"][entry["group"]]),
            ("original_val", data["original_val"]),
            ("branch_val", data["branch_val"]),
        ):
            result = evaluate(model, records)
            per_target[target] = result
            metrics_rows.append(
                {
                    "rule": entry["rule"], "group": entry["group"], "seed": entry["seed"],
                    "width": entry["width"], "best_epoch": entry["best_epoch"], "updates": entry["updates"],
                    "target": target, "status": "ok", **flatten(result),
                }
            )
        details[f"{entry['rule']}|{entry['group']}|{entry['seed']}"] = per_target
        print(
            f"{entry['rule']:12s} {entry['group']:11s} seed{entry['seed']} width{entry['width']} "
            f"train n={len(data['train'][entry['group']])} obj={per_target['own_train']['position']['object_mean']:.4f} "
            f"invalid={per_target['own_train']['legality']['invalid_rate']:.4f}",
            flush=True,
        )
    names = ["rule", "group", "seed", "width", "best_epoch", "updates", "target", "status"] + list(FLAT_METRICS)
    write_csv(run_dir / "existing_fit_metrics.csv", metrics_rows, fieldnames=names)
    (run_dir / "existing_fit_details.json").write_text(json.dumps(details, indent=2, sort_keys=True))
    print("EXISTING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
