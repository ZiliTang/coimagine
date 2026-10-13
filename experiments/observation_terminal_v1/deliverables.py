"""Round-4 deliverables: 12 fixed spot checks, one test root's eight actions, normalisation export.

    python experiments/observation_terminal_v1/deliverables.py

Spot checks are selected by a fixed hash of ``record_id`` inside each category that actually exists
(exit class x terminal visibility of robot 0), so the selection is reproducible and no category is
invented when it is empty. The test-root dump shows all eight joint commands of one hash-chosen test
root with their inputs, labels and the seed-0 predictions, which is the audit trail the task book
asks for.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.observation_terminal_v1 import dataset_terminal as DT  # noqa: E402
from experiments.observation_terminal_v1.train_terminal import load_all, start_patch_valid  # noqa: E402
from experiments.option_prediction_mechanism_v1.phase_b.model import (  # noqa: E402
    MultiHeadMLP,
    _standardise,
    predict,
    unstandardise_predictions,
)

OUT = ROOT / "outputs/observation_terminal_v1"


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def select_spot_checks(records: List[dict], observations: List[dict], count: int = 12) -> List[int]:
    buckets: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        observation = observations[index]
        visible = bool(observation["terminal_observation"]["0"].get("patch_valid"))
        category = f"{observation['exit_class']}|{'visible' if visible else 'invisible'}"
        buckets[category].append(index)
    chosen: List[int] = []
    for category in sorted(buckets):
        ordered = sorted(buckets[category], key=lambda i: _digest(records[i]["record_id"]))
        if ordered:
            chosen.append(ordered[0])
    for category in sorted(buckets):
        ordered = sorted(buckets[category], key=lambda i: _digest(records[i]["record_id"]))
        for index in ordered[1:]:
            if len(chosen) >= count:
                break
            chosen.append(index)
        if len(chosen) >= count:
            break
    return chosen[:count]


def describe(index: int, record: dict, observation: dict) -> dict:
    geometry = {}
    for robot_id in (0, 1):
        start = record["input"][f"robot{robot_id}"]["start_pose_world"]
        terminal = observation["terminal_robots"][robot_id]
        summary = observation["terminal_observation"][str(robot_id)]
        terminal_geometry = DT.terminal_geometry(summary, start, terminal)
        delta, velocity = DT.robot_motion(summary, start, terminal)
        geometry[f"robot{robot_id}"] = {
            "start_pose_world": [float(v) for v in start],
            "terminal_pose_world": [float(v) for v in terminal],
            "start_segment_own_frame": record["input"][f"robot{robot_id}"].get(
                "patch_segment_own_frame"),
            "start_normal_sin_cos": record["input"][f"robot{robot_id}"].get(
                "patch_normal_sin_cos"),
            "terminal_segment_robot_frame": summary.get("patch_segment_robot"),
            "terminal_midpoint_and_normal_in_RI0": (None if terminal_geometry is None
                                                    else [float(v) for v in terminal_geometry]),
            "terminal_patch_valid": bool(summary.get("patch_valid")),
            "terminal_contacting": bool(summary.get("contacting")),
            "pose_delta_in_RI0": [float(v) for v in delta],
            "residual_velocity_in_RI0": [float(v) for v in velocity],
        }
    return {
        "record_id": record["record_id"], "root_id": record["root_id"],
        "split": record["split"], "command_index": record["command_index"],
        "command": record["input"]["command"],
        "exit_class": observation["exit_class"],
        "primary_exit_reason": observation["primary_exit_reason"],
        "duration_seconds": observation["duration_seconds"],
        "category": f"{observation['exit_class']}|"
                    f"{'visible' if geometry['robot0']['terminal_patch_valid'] else 'invisible'}",
        "robots": geometry,
        "spot_check_hash": _digest(record["record_id"])[:16],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data, records, observations, meta = load_all()
    starts = start_patch_valid(records)

    chosen = select_spot_checks(records, observations)
    spot = {
        "selection_rule": "sha256(record_id) ascending within each existing "
                          "(exit_class x robot-0 terminal visibility) category",
        "categories_available": sorted({f"{observations[i]['exit_class']}|"
                                        f"{'visible' if observations[i]['terminal_observation']['0'].get('patch_valid') else 'invisible'}"
                                        for i in range(len(records))}),
        "checks": [describe(index, records[index], observations[index]) for index in chosen],
    }
    (OUT / "spot_checks.json").write_text(json.dumps(spot, indent=1, sort_keys=True, default=float))

    # one hash-chosen test root, all eight commands, with seed-0 predictions
    _standardise(data)
    checkpoint_path = OUT / "checkpoints" / f"deployable_seed{args.seed}.pt"
    predictions = None
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = MultiHeadMLP(payload["input_dim"], data.specs)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        # the deployable arm keeps the privileged slots pinned to zero, exactly as in training
        model_input = np.hstack([data.x, np.zeros_like(data.x_priv)])
        predictions = unstandardise_predictions(data, predict(model, model_input, data.specs))
    test_roots = sorted({root for root, split in zip(data.roots, data.split) if split == "test"})
    root = sorted(test_roots, key=_digest)[0]
    indices = [i for i, r in enumerate(data.roots) if r == root]
    dump = {"root_id": root, "selection_rule": "sha256(root_id) ascending among test roots",
            "checkpoint": str(checkpoint_path) if predictions is not None else None,
            "actions": []}
    for index in sorted(indices, key=lambda i: data.commands[i]):
        entry = {
            "record_id": records[index]["record_id"],
            "command_index": data.commands[index],
            "command": records[index]["input"]["command"],
            "input_x": [float(v) for v in data.x[index]],
            "input_x_privileged_all_zero": [float(v) for v in data.x_priv[index]],
            "exit_class": observations[index]["exit_class"],
            "labels": {spec.name: np.atleast_1d(
                data.targets[spec.name].get("y_raw",
                                            data.targets[spec.name]["y"])[index]).tolist()
                for spec in data.specs},
            "masks": {spec.name: (np.atleast_1d(
                data.targets[spec.name]["mask"][index]).tolist()) for spec in data.specs},
        }
        if predictions is not None:
            entry["predictions"] = {name: np.atleast_1d(values[index]).tolist()
                                    for name, values in predictions.items()}
        dump["actions"].append(entry)
    (OUT / "test_root_dump.json").write_text(json.dumps(dump, indent=1, sort_keys=True, default=float))

    normalisation = {
        "scheme": "group-wise: masked train entries flattened, one mean/std per group, shape (1,)",
        "groups": {name: {"mean": np.atleast_1d(data.targets[name]["mean"]).tolist(),
                          "std": np.atleast_1d(data.targets[name]["std"]).tolist(),
                          "shape": list(np.atleast_1d(data.targets[name]["mean"]).shape)}
                   for name in data.targets if "mean" in data.targets[name]},
        "fitted_on": "train split only",
        "per_column_train_stats_for_reference": _per_column(data),
    }
    (OUT / "normalisation.json").write_text(
        json.dumps(normalisation, indent=1, sort_keys=True, default=float))
    print(f"spot checks: {len(spot['checks'])} categories={spot['categories_available']}")
    print(f"test root dump: {root} actions={len(dump['actions'])} predictions={predictions is not None}")
    print(json.dumps(normalisation["groups"], indent=1))
    print("DELIVERABLES_OK")
    return 0


def _per_column(data) -> Dict[str, list]:
    out: Dict[str, list] = {}
    train = data.split == "train"
    for spec in data.specs:
        if spec.kind != "continuous":
            continue
        block = data.targets[spec.name]
        values = np.atleast_2d(block.get("y_raw", block["y"])[train])
        mask = np.atleast_2d(block["mask"][train])
        stats = []
        for column in range(values.shape[1]):
            valid = mask[:, column] > 0
            if valid.any():
                stats.append({"mean": float(values[valid, column].mean()),
                              "std": float(values[valid, column].std())})
            else:
                stats.append({"mean": None, "std": None})
        out[spec.name] = stats
    return out


if __name__ == "__main__":
    raise SystemExit(main())
