"""Phase C: progress, capacity and selection tables from the training logs.

    python experiments/fit_capacity/run_tables.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PROGRESS_UPDATES = (1000, 3000, 6000)


def _load_evals(cell_dir: Path):
    rows = []
    with (cell_dir / "eval_metrics.jsonl").open() as handle:
        for line in handle:
            rows.append(json.loads(line))
    return {int(row["update"]): row for row in rows}


def _cell(run_dir: Path, group: str, width: int, seed: int):
    return run_dir / "runs" / f"{group.replace('+', '_')}_{width}_seed{seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    out = run_dir / "tables"
    out.mkdir(parents=True, exist_ok=True)
    groups = ("Old+Random", "Old+Branch")
    widths = (256, 512)
    seeds = (0, 1, 2)

    evals = {}
    summaries = {}
    for group in groups:
        for width in widths:
            for seed in seeds:
                cell = _cell(run_dir, group, width, seed)
                evals[(group, width, seed)] = _load_evals(cell)
                summaries[(group, width, seed)] = json.loads((cell / "run_summary.json").read_text())

    def field(row, dataset, path):
        node = row["metrics"][dataset]
        for key in path.split("."):
            node = node[key]
        return node

    # ---- 1. progress table -------------------------------------------------------------
    progress = []
    for group in groups:
        for width in widths:
            for update in PROGRESS_UPDATES:
                entry = {"group": group, "width": width, "updates": update}
                for seed in seeds:
                    row = evals[(group, width, seed)].get(update)
                    if row is None:
                        entry[f"seed{seed}"] = None
                        continue
                    entry[f"seed{seed}"] = {
                        "train_object_position": field(row, "train", "position.object_mean"),
                        "train_invalid_rate": field(row, "train", "legality.invalid_rate"),
                        "train_total_loss": row["train_total_loss"],
                        "original_val_object_position": field(row, "original_val", "position.object_mean"),
                        "original_val_invalid_rate": field(row, "original_val", "legality.invalid_rate"),
                        "original_val_total_loss": row["original_val_total_loss"],
                        "branch_val_object_position": field(row, "branch_val", "position.object_mean"),
                        "branch_val_invalid_rate": field(row, "branch_val", "legality.invalid_rate"),
                        "branch_val_total_loss": row["branch_val_total_loss"],
                    }
                for key in ("train_object_position", "train_invalid_rate", "train_total_loss",
                            "original_val_object_position", "original_val_invalid_rate", "original_val_total_loss",
                            "branch_val_object_position", "branch_val_invalid_rate", "branch_val_total_loss"):
                    values = [entry[f"seed{s}"][key] for s in seeds if entry[f"seed{s}"]]
                    entry[key + "_mean"] = float(np.mean(values)) if values else None
                    entry[key + "_min"] = float(np.min(values)) if values else None
                    entry[key + "_max"] = float(np.max(values)) if values else None
                progress.append(entry)

    # ---- 2. capacity table -------------------------------------------------------------
    capacity = []
    for group in groups:
        for seed in seeds:
            entry = {"group": group, "seed": seed}
            for key in ("train_object_position", "train_invalid_rate", "train_total_loss",
                        "original_val_object_position", "original_val_invalid_rate", "original_val_total_loss",
                        "branch_val_object_position", "branch_val_invalid_rate", "branch_val_total_loss"):
                entry[f"d_{key}"] = None
            for width in widths:
                row = evals[(group, width, seed)].get(6000)
                entry[f"w{width}_wall_clock"] = summaries[(group, width, seed)]["wall_clock"]
                if row is None:
                    continue
                entry[f"w{width}"] = {
                    "train_object_position": field(row, "train", "position.object_mean"),
                    "train_invalid_rate": field(row, "train", "legality.invalid_rate"),
                    "train_total_loss": row["train_total_loss"],
                    "original_val_object_position": field(row, "original_val", "position.object_mean"),
                    "original_val_invalid_rate": field(row, "original_val", "legality.invalid_rate"),
                    "original_val_total_loss": row["original_val_total_loss"],
                    "branch_val_object_position": field(row, "branch_val", "position.object_mean"),
                    "branch_val_invalid_rate": field(row, "branch_val", "legality.invalid_rate"),
                    "branch_val_total_loss": row["branch_val_total_loss"],
                }
            if "w256" in entry and "w512" in entry:
                for key in ("train_object_position", "train_invalid_rate", "train_total_loss",
                            "original_val_object_position", "original_val_invalid_rate", "original_val_total_loss",
                            "branch_val_object_position", "branch_val_invalid_rate", "branch_val_total_loss"):
                    entry[f"d_{key}"] = entry["w512"][key] - entry["w256"][key]
            capacity.append(entry)
    capacity_mean = {}
    for key in ("train_object_position", "train_invalid_rate", "train_total_loss",
                "original_val_object_position", "original_val_invalid_rate", "original_val_total_loss",
                "branch_val_object_position", "branch_val_invalid_rate", "branch_val_total_loss"):
        values = [row[f"d_{key}"] for row in capacity if row[f"d_{key}"] is not None]
        capacity_mean[key] = {"mean": float(np.mean(values)) if values else None,
                              "min": float(np.min(values)) if values else None,
                              "max": float(np.max(values)) if values else None}
    for width in widths:
        values = [row[f"w{width}_wall_clock"] for row in capacity]
        capacity_mean[f"w{width}_wall_clock"] = {"mean": float(np.mean(values)) if values else None}

    # ---- 3. selection table ------------------------------------------------------------
    selection = []
    for group in groups:
        for width in widths:
            for seed in seeds:
                summary = summaries[(group, width, seed)]
                rows = evals[(group, width, seed)]
                entry = {"group": group, "width": width, "seed": seed, "stopped_reason": summary["stopped_reason"],
                         "wall_clock": summary["wall_clock"]}
                for kind, checkpoint in summary["checkpoints"].items():
                    update = int(checkpoint["updates"])
                    row = rows.get(update, {})
                    entry[kind] = {
                        "updates": update,
                        "mixed_val_score": checkpoint.get("mixed_val_score"),
                        "four_val_loss": {
                            "original_val": row.get("metrics", {}).get("original_val", {}).get("loss"),
                            "branch_val": row.get("metrics", {}).get("branch_val", {}).get("loss"),
                        },
                        "train_object_position": field(row, "train", "position.object_mean") if row else None,
                        "train_invalid_rate": field(row, "train", "legality.invalid_rate") if row else None,
                        "original_val_object_position": field(row, "original_val", "position.object_mean") if row else None,
                        "original_val_invalid_rate": field(row, "original_val", "legality.invalid_rate") if row else None,
                        "branch_val_object_position": field(row, "branch_val", "position.object_mean") if row else None,
                        "branch_val_invalid_rate": field(row, "branch_val", "legality.invalid_rate") if row else None,
                    }
                # does "better total loss" disagree with "better physical error"?
                candidates = {kind: value for kind, value in entry.items() if isinstance(value, dict) and "updates" in value}
                if len(candidates) >= 2:
                    by_loss = min(candidates, key=lambda k: float("inf") if candidates[k]["mixed_val_score"] is None else candidates[k]["mixed_val_score"])
                    by_physics = min(
                        candidates,
                        key=lambda k: (
                            (candidates[k]["original_val_object_position"] or 0.0)
                            + (candidates[k]["branch_val_object_position"] or 0.0)
                        ),
                    )
                    entry["loss_vs_physics_disagree"] = by_loss != by_physics
                    entry["best_by_mixed_val"] = by_loss
                    entry["best_by_physics"] = by_physics
                selection.append(entry)

    (out / "progress_table.json").write_text(json.dumps(progress, indent=2, sort_keys=True))
    (out / "capacity_table.json").write_text(json.dumps({"rows": capacity, "mean": capacity_mean}, indent=2, sort_keys=True))
    (out / "selection_table.json").write_text(json.dumps(selection, indent=2, sort_keys=True))

    lines = ["# C1 训练进度表（原/分支 val 的物体位置误差、非法率、总 loss）", "",
             "| 组 | 宽度 | updates | train obj (m) | train 非法 | train loss | 原val obj | 原val 非法 | 原val loss | 分支val obj | 分支val 非法 | 分支val loss |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in progress:
        lines.append(
            f"| {row['group']} | {row['width']} | {row['updates']} | {row['train_object_position_mean']:.4f} | "
            f"{row['train_invalid_rate_mean']:.4f} | {row['train_total_loss_mean']:.4f} | "
            f"{row['original_val_object_position_mean']:.4f} | {row['original_val_invalid_rate_mean']:.4f} | "
            f"{row['original_val_total_loss_mean']:.4f} | {row['branch_val_object_position_mean']:.4f} | "
            f"{row['branch_val_invalid_rate_mean']:.4f} | {row['branch_val_total_loss_mean']:.4f} |"
        )
    lines += ["", "# C2 容量表（512 − 256，同数据同种子，6000 updates）", "",
              "| 组 | seed | Δtrain obj | Δtrain 非法 | Δtrain loss | Δ原val obj | Δ原val 非法 | Δ原val loss | Δ分支val obj | Δ分支val 非法 | Δ分支val loss | 512 墙钟 | 256 墙钟 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in capacity:
        def f(key):
            value = row[f"d_{key}"]
            return "NA" if value is None else f"{value:+.4f}"
        lines.append(
            f"| {row['group']} | {row['seed']} | {f('train_object_position')} | {f('train_invalid_rate')} | "
            f"{f('train_total_loss')} | {f('original_val_object_position')} | {f('original_val_invalid_rate')} | "
            f"{f('original_val_total_loss')} | {f('branch_val_object_position')} | {f('branch_val_invalid_rate')} | "
            f"{f('branch_val_total_loss')} | {row['w512_wall_clock']:.1f}s | {row['w256_wall_clock']:.1f}s |"
        )
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("TABLES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
