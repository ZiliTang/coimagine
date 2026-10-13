"""Phase B driver: 32-root pipeline check, tier-1 training, evaluation, tier-2 decision.

    python experiments/option_prediction_mechanism_v1/phase_b/train_b.py \
        --data outputs/.../phase_b/records_tier1.jsonl --out outputs/.../phase_b --check
    python experiments/option_prediction_mechanism_v1/phase_b/train_b.py \
        --data outputs/.../phase_b/records_tier1.jsonl --out outputs/.../phase_b --tier 1

The freeze is the contract: 2x256 ReLU MLP, Adam 1e-3, batch 128, <=200 epochs, validation
patience 20, seeds 0/1/2, six target groups all weighted 1.0. Checkpoints are chosen on the
validation composite. The test split is scored only for the final report.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import (  # noqa: E402
    EXIT_CLASSES,
    build_supervised,
    load_records,
)
from experiments.option_prediction_mechanism_v1.phase_b.evaluate_b import (  # noqa: E402
    angle_from_sincos,
    per_root_error,
    binary_report,
    classification_report,
    continuous_errors,
    grouped_errors,
    same_root_spread,
)
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    SupervisedData,
    _standardise,
    predict,
    persistence_predictions,
    train_one_seed,
    unstandardise_predictions,
)

SEEDS = (0, 1, 2)
GEOMETRY_COLUMNS = ["r0_signed_distance_m", "r0_tangential_offset_m", "r0_normal_sin",
                    "r0_normal_cos", "r1_signed_distance_m", "r1_tangential_offset_m",
                    "r1_normal_sin", "r1_normal_cos"]
POSE_COLUMNS = ["r0_dx_m", "r0_dy_m", "r0_dtheta_rad", "r1_dx_m", "r1_dy_m", "r1_dtheta_rad"]
VELOCITY_COLUMNS = ["r0_vx_mps", "r0_vy_mps", "r0_omega_radps",
                    "r1_vx_mps", "r1_vy_mps", "r1_omega_radps"]
VALIDITY_COLUMNS = ["r0_patch_valid", "r0_contacting", "r1_patch_valid", "r1_contacting"]


def _angle_error(prediction: np.ndarray, truth: np.ndarray, mask: np.ndarray, row: int,
                 robot_id: int) -> float:
    sin_index, cos_index = 4 * robot_id + 2, 4 * robot_id + 3
    if mask[row, sin_index] <= 0 or mask[row, cos_index] <= 0:
        return float("nan")
    predicted = math.atan2(prediction[row, sin_index], prediction[row, cos_index])
    actual = math.atan2(truth[row, sin_index], truth[row, cos_index])
    return abs(math.degrees((predicted - actual + math.pi) % (2 * math.pi) - math.pi))


#: The evaluation interface is in **original physical units** for every group. Network outputs are
#: standardised and must be converted with ``unstandardise_predictions`` exactly once before they
#: reach :func:`evaluate`; the persistence baseline already produces original units and must not be
#: converted again (that double conversion was the round-3 unit bug).


def _subset(data: SupervisedData, records, train_roots: int):
    train_root_ids = sorted({r for r, s in zip(data.roots, data.split) if s == "train"})
    keep = set(train_root_ids[:train_roots])
    mask = np.array([(s != "train") or (r in keep) for r, s in zip(data.roots, data.split)])
    data.x = data.x[mask]
    data.x_priv = data.x_priv[mask]
    data.split = data.split[mask]
    data.commands = [c for c, k in zip(data.commands, mask) if k]
    data.roots = [r for r, k in zip(data.roots, mask) if k]
    for group in data.targets:
        for key in ("y", "mask"):
            data.targets[group][key] = data.targets[group][key][mask]
    for group in data.persistence:
        data.persistence[group] = data.persistence[group][mask]
    return data, [record for record, keep in zip(records, mask) if keep]


def _meta(records: Sequence[Dict[str, object]]):
    classes = np.array([r["interaction_class"] for r in records])
    exit_class = np.array([r["label"]["exit_class"] for r in records])
    motion = np.array(["static" if int(r["truth"]["micro_motion"]) else "motion"
                       for r in records])
    end = np.array(["normal" if r["label"]["settled_observed"] else
                    "after_abort" if r["label"]["settled_after_abort"] else
                    "abort" if r["label"]["primary_exit_reason"] else "unsettled"
                    for r in records])
    return {"interaction_class": classes, "exit_reason": exit_class, "motion": motion, "end": end}


def evaluate(data: SupervisedData, records, predictions: Dict[str, np.ndarray], split: str,
             label: str) -> Dict[str, object]:
    """Score predictions that are already in the targets' **original physical units**.

    Convention (pinned by ``tests/test_option_prediction_mechanism_b.py``):
    * a network's output is standardised and MUST be passed through
      :func:`unstandardise_predictions` exactly once before reaching here;
    * :func:`persistence_predictions` already returns original units and MUST NOT be converted
      again -- doing so was the round-3 bug, which scaled the baseline by ``std`` and shifted it by
      ``mean`` a second time and made every persistence error look far worse (or better) than it is;
    * discrete groups (class indices, binary logits) are passed through unchanged.
    """

    index = np.where(data.split == split)[0]
    meta = _meta(records)
    out: Dict[str, object] = {"label": label, "split": split, "n": int(len(index)),
                              "n_roots": len({data.roots[i] for i in index})}
    geometry_raw = predictions["terminal_geometry"]
    truth_geometry = data.targets["terminal_geometry"]["y_raw"]
    geometry_mask = data.targets["terminal_geometry"]["mask"]
    out["terminal_geometry"] = continuous_errors(
        geometry_raw[index], truth_geometry[index], geometry_mask[index],
        data.specs[0], GEOMETRY_COLUMNS)
    for robot_id in (0, 1):
        out["terminal_geometry"][f"r{robot_id}_normal_angle_deg"] = angle_from_sincos(
            geometry_raw[index], truth_geometry[index], geometry_mask[index],
            4 * robot_id + 2, 4 * robot_id + 3)
    for group, columns in (("robot_pose_delta", POSE_COLUMNS),
                           ("robot_residual_velocity", VELOCITY_COLUMNS)):
        raw = predictions[group]
        out[group] = continuous_errors(
            raw[index], data.targets[group]["y_raw"][index],
            data.targets[group]["mask"][index], None, columns)
    duration_raw = predictions["duration"]
    out["duration"] = continuous_errors(
        duration_raw[index], data.targets["duration"]["y_raw"][index],
        data.targets["duration"]["mask"][index], None, ["duration_s"])
    out["observation_validity"] = binary_report(
        predictions["observation_validity"][index],
        data.targets["observation_validity"]["y"][index],
        data.targets["observation_validity"]["mask"][index], VALIDITY_COLUMNS)
    out["exit_class"] = classification_report(
        predictions["exit_class"][index], data.targets["exit_class"]["y"][index],
        data.targets["exit_class"]["mask"][index], len(EXIT_CLASSES), EXIT_CLASSES)
    # grouped views: signed-distance error by configuration / motion / end bucket
    distance_error = np.abs(geometry_raw[index, 0] - truth_geometry[index, 0])
    valid = geometry_mask[index, 0] > 0
    for key in ("interaction_class", "motion", "end"):
        out[f"signed_distance_by_{key}"] = grouped_errors(
            key, distance_error[valid], meta[key][index][valid])
    # duration error by end bucket
    duration_error = np.abs(duration_raw[index, 0] - data.targets["duration"]["y_raw"][index, 0])
    out["duration_by_end"] = grouped_errors("end", duration_error, meta["end"][index])
    # root-level aggregation: a root's eight branches count as one experiment
    root_ids = [data.roots[i] for i in index]
    valid_sd = geometry_mask[index, 0] > 0
    out["per_root"] = {
        "signed_distance_m": per_root_error(distance_error[valid_sd],
                                            np.asarray(root_ids)[valid_sd]),
        "normal_angle_deg": per_root_error(
            np.asarray([_angle_error(geometry_raw, truth_geometry, geometry_mask, i, 0)
                        for i in index]),
            np.asarray(root_ids)),
    }
    # does the action change the terminal state, and does the model capture it?
    out["same_root_action_spread"] = same_root_spread(
        [data.roots[i] for i in index], [data.commands[i] for i in index],
        truth_geometry[index], geometry_mask[index], geometry_raw[index])
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tier", default="check")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--check-roots", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    records = load_records(Path(args.data))
    data = build_supervised(records)
    if args.check:
        data, records = _subset(data, records, args.check_roots)
    _standardise(data)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "tier": args.tier, "check": bool(args.check), "seeds": list(seeds),
        "config": {"hidden": 256, "layers": 2, "optimizer": "adam", "lr": 1e-3,
                   "batch_size": 128, "max_epochs": args.epochs, "patience": 20,
                   "group_weights": {spec.name: spec.weight for spec in data.specs}},
        "data": {"records": len(records), "roots": len(set(data.roots)),
                 "train_records": int((data.split == "train").sum()),
                 "val_records": int((data.split == "val").sum()),
                 "test_records": int((data.split == "test").sum()),
                 "train_roots": len({r for r, s in zip(data.roots, data.split) if s == "train"})},
        "runs": {},
    }

    # persistence baseline (no network) is reported for every split; it is also what the
    # tier-2 trigger is measured against
    for split in ("val", "test"):
        baseline = persistence_predictions(data, split)
        payload.setdefault("baseline", {})[split] = evaluate(data, records, baseline, split,
                                                             "persistence")

    for arm, privileged in (("deployable", False), ("privileged", True)):
        for seed in seeds:
            started = time.perf_counter()
            trained = train_one_seed(data, use_privileged=privileged, seed=seed,
                                     epochs=args.epochs, patience=20)
            x = trained["x"]
            index_test = np.where(data.split == "test")[0]
            index_val = np.where(data.split == "val")[0]
            predictions = unstandardise_predictions(
                data, predict(trained["model"], x, data.specs))
            key = f"{arm}_seed{seed}"
            checkpoint_dir = out_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"{key}.pt"
            torch.save({
                "state_dict": trained["model"].state_dict(),
                "specs": [spec.__dict__ for spec in data.specs],
                "input_dim": trained["input_dim"],
                "best_val_loss": trained["best_val"],
                "best_epoch": trained["best_epoch"],
                "seed": seed, "arm": arm, "use_privileged": privileged,
                "target_mean": {name: data.targets[name]["mean"].tolist()
                                for name in data.targets if "mean" in data.targets[name]},
                "target_std": {name: data.targets[name]["std"].tolist()
                               for name in data.targets if "std" in data.targets[name]},
                "exit_classes": list(EXIT_CLASSES),
            }, checkpoint_path)
            payload["runs"][key] = {
                "arm": arm, "seed": seed, "parameters": trained["parameters"],
                "input_dim": trained["input_dim"], "best_val_loss": trained["best_val"],
                "best_epoch": trained["best_epoch"], "train_seconds": time.perf_counter() - started,
                "curves": trained["curves"],
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": __import__("hashlib").sha256(
                    checkpoint_path.read_bytes()).hexdigest(),
                "val": evaluate(data, records, predictions, "val", key),
                "test": evaluate(data, records, predictions, "test", key),
            }
            print(f"{key}: best_val={trained['best_val']:.4f} "
                  f"epoch={trained['best_epoch']} params={trained['parameters']} "
                  f"({time.time() - started:.1f}s)", flush=True)

    # tier-2 trigger: at least one task-relevant target beats persistence on *validation*
    payload["tier2_decision"] = _tier2_decision(payload)
    (out_dir / f"train_report_{args.tier}.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    (out_dir / f"train_report_{args.tier}.md").write_text(render(payload))
    print(render(payload))
    print("TRAIN_B_OK")
    return 0


def _ratio(model_value, baseline_value) -> float:
    if model_value is None or baseline_value in (None, 0.0):
        return float("nan")
    return float(model_value) / float(baseline_value)


def _tier2_decision(payload: dict) -> dict:
    """Frozen trigger: tier 1 trained without fault AND beat persistence on validation."""

    baseline = payload.get("baseline", {}).get("val", {})
    evidence = {}
    triggered = False
    for key, run in payload["runs"].items():
        if run["arm"] != "deployable":
            continue
        val = run["val"]
        candidates = {
            "geometry_signed_distance_mae_p50": (
                val["terminal_geometry"]["r0_signed_distance_m"]["p50"],
                baseline["terminal_geometry"]["r0_signed_distance_m"]["p50"]),
            "geometry_tangential_offset_mae_p50": (
                val["terminal_geometry"]["r0_tangential_offset_m"]["p50"],
                baseline["terminal_geometry"]["r0_tangential_offset_m"]["p50"]),
            "normal_angle_mae_p50_deg": (
                val["terminal_geometry"]["r0_normal_angle_deg"]["p50"],
                baseline["terminal_geometry"]["r0_normal_angle_deg"]["p50"]),
            "duration_mae_p50_s": (val["duration"]["duration_s"]["p50"],
                                   baseline["duration"]["duration_s"]["p50"]),
        }
        wins = {name: {"model": model, "persistence": base, "ratio": _ratio(model, base)}
                for name, (model, base) in candidates.items()}
        better = [name for name, row in wins.items()
                  if row["ratio"] == row["ratio"] and row["ratio"] < 0.8]
        evidence[key] = {"targets": wins, "beats_persistence_on": better}
        if better:
            triggered = True
    return {"triggered": triggered, "evidence": evidence,
            "rule": "tier 2 is added only if tier 1 trained without data/label faults and at "
                    "least one task-relevant validation target is at least 20% better than the "
                    "persistence baseline (ratio < 0.8)"}


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and value != value:
        return "-"
    return f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = ["# Phase B training report", "",
             f"tier `{payload['tier']}`, seeds {payload['seeds']}, config {payload['config']}",
             f"data: {payload['data']}", ""]
    base = payload.get("baseline", {})
    if base:
        lines += ["## Persistence baseline", "",
                  "| split | r0 sd p50 (m) | r0 tang p50 (m) | r0 angle p50 (deg) | r1 sd p50 (m) | "
                  "r1 tang p50 (m) | duration p50 (s) | exit acc |", 
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for split, block in base.items():
            geometry = block["terminal_geometry"]
            lines.append(
                f"| {split} | {_fmt(geometry['r0_signed_distance_m']['p50'])} | "
                f"{_fmt(geometry['r0_tangential_offset_m']['p50'])} | "
                f"{_fmt(geometry['r0_normal_angle_deg']['p50'])} | "
                f"{_fmt(geometry['r1_signed_distance_m']['p50'])} | "
                f"{_fmt(geometry['r1_tangential_offset_m']['p50'])} | "
                f"{_fmt(block['duration']['duration_s']['p50'])} | "
                f"{_fmt(block['exit_class']['accuracy'], 3)} |")
        lines.append("")
    for key, run in payload["runs"].items():
        lines += [f"## {key}", "",
                  f"params {run['parameters']}, input dim {run['input_dim']}, "
                  f"best val {run['best_val_loss']:.4f} @ epoch {run['best_epoch']}, "
                  f"{run['train_seconds']:.1f}s", ""]
        for split in ("val", "test"):
            block = run[split]
            geometry = block["terminal_geometry"]
            lines += [f"### {split} (n={block['n']}, roots={block['n_roots']})", "",
                      "| target | p50 | p90 |", "|---|---:|---:|"]
            for name in ("r0_signed_distance_m", "r0_tangential_offset_m",
                         "r0_normal_angle_deg", "r1_signed_distance_m",
                         "r1_tangential_offset_m", "r1_normal_angle_deg"):
                row = geometry[name]
                unit = "deg" if "angle" in name else "m"
                lines.append(f"| {name} ({unit}) | {_fmt(row['p50'])} | {_fmt(row['p90'])} |")
            for group in ("robot_pose_delta", "robot_residual_velocity", "duration"):
                for name, row in block[group].items():
                    lines.append(f"| {name} | {_fmt(row['p50'])} | {_fmt(row['p90'])} |")
            lines += ["", f"exit class accuracy {_fmt(block['exit_class']['accuracy'], 3)}; "
                          f"same-root true spread p50 "
                          f"{_fmt(block['same_root_action_spread']['true_spread']['p50'])} "
                          f"vs predicted "
                          f"{_fmt(block['same_root_action_spread']['predicted_spread']['p50'])}",
                      ""]
    decision = payload["tier2_decision"]
    lines += ["## Tier-2 decision", "", f"triggered: **{decision['triggered']}**",
              f"rule: {decision['rule']}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
