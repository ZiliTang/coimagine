"""Round-4 driver: 32-root pipeline check, three-seed training, baselines and evaluation.

    python experiments/observation_terminal_v1/train_terminal.py --check
    python experiments/observation_terminal_v1/train_terminal.py --train

Frozen by `docs/observation_terminal_v1/CONTRACT.md`: 2x256 ReLU MLP, Adam 1e-3, batch 128, at most
200 epochs, validation patience 20, seeds 0/1/2, six target groups all weighted 1.0, checkpoint on
the validation composite only, group-wise (not per-column) train standardisation, and the
privileged input slots pinned to zero so the network keeps its 86-dim input and 97,573 parameters.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.observation_terminal_v1 import baselines as BL  # noqa: E402
from experiments.observation_terminal_v1 import dataset_terminal as DT  # noqa: E402
from experiments.observation_terminal_v1.evaluate_terminal import evaluate  # noqa: E402
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    _standardise,
    predict,
    train_one_seed,
    unstandardise_predictions,
)

TIER2 = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                "phase_b_tier2")
OBS_DIR = ROOT / "outputs/observation_terminal_v1"
SEEDS = (0, 1, 2)


def load_all():
    observations: Dict[str, dict] = {}
    for split in ("train", "val", "test"):
        for line in (OBS_DIR / f"terminal_observations_{split}.jsonl").open():
            record = json.loads(line)
            observations[record["record_id"]] = record
    records, split_of = [], {}
    for split in ("train", "val", "test"):
        for line in (TIER2 / f"executions_{split}.jsonl").open():
            record = json.loads(line)
            if record["record_id"] not in observations:
                continue
            records.append(record)
            split_of[record["record_id"]] = split
    ordered = sorted(records, key=lambda row: row["record_id"])
    observation_list = [observations[row["record_id"]] for row in ordered]
    data, meta = DT.build_dataset(ordered, observation_list, split_of)
    meta["dropped_without_terminal_observation"] = len(records) - len(ordered)
    return data, ordered, observation_list, meta


def start_patch_valid(records: Sequence[dict]) -> List[List[bool]]:
    """Start-of-option patch visibility for each robot, used for the transition grouping."""

    return [[bool(record["input"][f"robot{robot_id}"].get("mask_patch_valid"))
             for robot_id in (0, 1)] for record in records]


def subset_train_roots(data, count: int):
    """Restrict the train split to the first ``count`` hash-sorted roots (check mode only)."""

    roots = sorted({root for root, split in zip(data.roots, data.split) if split == "train"})
    keep = set(roots[:count])
    mask = np.array([(split != "train") or (root in keep)
                     for root, split in zip(data.roots, data.split)])
    data.subset_mask = mask  # type: ignore[attr-defined]
    data.x = data.x[mask]
    data.x_priv = data.x_priv[mask]
    data.split = data.split[mask]
    data.roots = [r for r, k in zip(data.roots, mask) if k]
    data.commands = [c for c, k in zip(data.commands, mask) if k]
    data.command_signature = [c for c, k in zip(data.command_signature, mask) if k]
    for group in data.targets:
        for key in ("y", "mask"):
            data.targets[group][key] = data.targets[group][key][mask]
    for key, value in list(data.persistence.items()):
        if isinstance(value, np.ndarray) and value.shape[0] == len(mask):
            data.persistence[key] = value[mask]
    return data


def apply_subset(sequence, mask):
    return [item for item, keep in zip(sequence, mask) if keep]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--check-roots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--out", default=str(ROOT / "outputs/observation_terminal_v1"))
    args = parser.parse_args()
    if not (args.check or args.train):
        parser.error("pass --check or --train")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, records, observation_list, meta = load_all()
    print(f"records {len(records)} | x {data.x.shape} | meta {meta}")
    starts = start_patch_valid(records)
    if args.check:
        mask = subset_train_roots(data, args.check_roots).subset_mask
        records = apply_subset(records, mask)
        observation_list = apply_subset(observation_list, mask)
        starts = apply_subset(starts, mask)
    _standardise(data)

    payload: Dict[str, object] = {
        "stage": "check" if args.check else "train",
        "data": {"records": len(records), **meta,
                 "train": int((data.split == "train").sum()),
                 "val": int((data.split == "val").sum()),
                 "test": int((data.split == "test").sum()),
                 "train_roots": len({r for r, s in zip(data.roots, data.split) if s == "train"})},
        "config": {"hidden": 256, "layers": 2, "optimizer": "adam", "lr": 1e-3,
                   "batch_size": 128, "max_epochs": args.epochs, "patience": 20,
                   "seeds": [int(s) for s in args.seeds.split(",")],
                   "group_weights": {spec.name: spec.weight for spec in data.specs},
                   "normalisation": "group-wise, flattened over masked train entries; "
                                    "mean/std shape (1,) broadcast to the group's columns",
                   "normalisation_shapes": {name: list(np.atleast_1d(
                       data.targets[name]["mean"]).shape)
                       for name in data.targets if "mean" in data.targets[name]}},
        "baselines": {}, "runs": {},
    }

    for split in ("val", "test"):
        for name, function in (("P0_persistence", BL.persistence_predictions),
                               ("P1_command_grouped", BL.command_grouped_predictions)):
            predictions, info = function(data, split)
            entry = evaluate(data, observation_list, predictions, split, name,
                             permutation_seed=0, start_patch_valid=starts)
            entry["baseline_info"] = {k: v for k, v in info.items()
                                      if k != "common_subset_index"}
            payload["baselines"][f"{name}|{split}"] = entry
            print(f"baseline {name} {split}: mid2d p50 "
                  f"{entry['terminal_observation_geometry']['r0_mid_2d_m']['p50']:.5f} m  "
                  f"angle p50 {entry['terminal_observation_geometry']['r0_normal_angle_deg']['p50']:.3f} deg",
                  flush=True)

    for seed in (int(s) for s in args.seeds.split(",")):
        started = time.perf_counter()
        trained = train_one_seed(data, use_privileged=False, seed=seed, epochs=args.epochs,
                                 patience=20)
        model_input = trained["x"]
        predictions = unstandardise_predictions(data, predict(trained["model"], model_input,
                                                              data.specs))
        key = f"deployable_seed{seed}"
        checkpoint = out_dir / "checkpoints"
        checkpoint.mkdir(parents=True, exist_ok=True)
        path = checkpoint / f"{key}.pt"
        torch.save({
            "state_dict": trained["model"].state_dict(),
            "specs": [spec.__dict__ for spec in data.specs],
            "input_dim": trained["input_dim"], "parameters": trained["parameters"],
            "best_val_loss": trained["best_val"], "best_epoch": trained["best_epoch"],
            "seed": seed, "use_privileged": False,
            "target_mean": {name: np.atleast_1d(data.targets[name]["mean"]).tolist()
                            for name in data.targets if "mean" in data.targets[name]},
            "target_std": {name: np.atleast_1d(data.targets[name]["std"]).tolist()
                           for name in data.targets if "std" in data.targets[name]},
            "normalisation_shapes": {name: list(np.atleast_1d(
                data.targets[name]["mean"]).shape)
                for name in data.targets if "mean" in data.targets[name]},
            "exit_classes": list(DT.EXIT_CLASSES),
            "geometry_columns": DT.GEOMETRY_COLUMNS,
            "pose_columns": DT.POSE_COLUMNS,
            "velocity_columns": DT.VELOCITY_COLUMNS,
            "validity_columns": DT.VALIDITY_COLUMNS,
        }, path)
        payload["runs"][key] = {
            "parameters": trained["parameters"], "input_dim": trained["input_dim"],
            "best_val_loss": trained["best_val"], "best_epoch": trained["best_epoch"],
            "train_seconds": time.perf_counter() - started,
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "checkpoint": str(path),
            "curves": trained["curves"],
            "val": evaluate(data, observation_list, predictions, "val", key, permutation_seed=0,
                            start_patch_valid=starts),
            "test": evaluate(data, observation_list, predictions, "test", key, permutation_seed=0,
                             start_patch_valid=starts),
        }
        print(f"{key}: best_val {trained['best_val']:.4f} epoch {trained['best_epoch']} "
              f"params {trained['parameters']} ({payload['runs'][key]['train_seconds']:.1f}s)",
              flush=True)

    tag = "check" if args.check else "train"
    (out_dir / f"terminal_report_{tag}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    (out_dir / f"terminal_report_{tag}.md").write_text(render(payload))
    print(render(payload)[:4000])
    print("TRAIN_TERMINAL_OK")
    return 0


def _f(value, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "-"
    return f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = ["# observation_terminal_v1 — 评价", "",
             f"stage `{payload['stage']}`", f"data {payload['data']}", ""]
    rows = []
    for name, entry in payload["baselines"].items():
        rows.append((name, entry))
    for name, run in payload["runs"].items():
        rows.append((f"{name}|val", run["val"]))
        rows.append((f"{name}|test", run["test"]))
    lines += ["| model|split | r0 mid2d p50 (m) | r0 mid2d p90 | r0 angle p50 (deg) | "
              "r0 angle p90 | pose dx p50 | dur p50 (s) | exit acc |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, entry in rows:
        geometry = entry["terminal_observation_geometry"]
        lines.append(
            f"| {name} | {_f(geometry['r0_mid_2d_m']['p50'], 5)} | "
            f"{_f(geometry['r0_mid_2d_m']['p90'], 5)} | "
            f"{_f(geometry['r0_normal_angle_deg']['p50'], 3)} | "
            f"{_f(geometry['r0_normal_angle_deg']['p90'], 3)} | "
            f"{_f(entry['robot_pose_delta']['r0_dx_m']['p50'], 4)} | "
            f"{_f(entry['duration']['duration_s']['p50'], 4)} | "
            f"{_f(entry['exit_class']['accuracy'], 3)} |")
    lines += ["", "## 动作差值 / 秩相关（r0 中点 x）", "",
              "| model|split | 动作差值 p50 (m) | 秩相关 ρ | 置换 ρ | 可定义 root |", "|---|---:|---:|---:|---:|"]
    for name, entry in rows:
        sensitivity = entry["action_sensitivity"]
        permuted = entry.get("action_sensitivity_permuted")
        lines.append(
            f"| {name} | {_f(sensitivity['action_difference_error']['r0_mid_x']['p50'], 5)} | "
            f"{_f(sensitivity['rank_correlation']['r0_mid_x']['mean'], 3)} | "
            f"{_f(permuted['rank_correlation']['r0_mid_x']['mean'], 3) if permuted else '-'} | "
            f"{sensitivity['rank_correlation']['r0_mid_x']['roots_defined']}/"
            f"{sensitivity['rank_correlation']['r0_mid_x']['roots_attempted']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
