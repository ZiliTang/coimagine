"""Phase A2: inventory and full-dataset re-evaluation of the existing checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from ..continuous2d_learning.model import JointWindowMLP, ModelConfig
from ..data_upgrade.training import GROUP_BRANCH, GROUP_OLD, GROUP_RANDOM
from .metrics import evaluate

RULE_DIRS = {"mixed_val": "models", "original_val": "models_originalval"}
GROUP_FILES = {GROUP_OLD: "Old", GROUP_RANDOM: "Old_Random", GROUP_BRANCH: "Old_Branch"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inventory(run_dir: Path, source_run: Path) -> List[Dict[str, object]]:
    """Every expected (rule, group, seed) checkpoint, with hash and provenance."""

    training = {
        "mixed_val": json.loads((run_dir / "training_groups.json").read_text()),
        "original_val": json.loads((run_dir / "training_groups_originalval.json").read_text()),
    }
    rows: List[Dict[str, object]] = []
    for rule, directory in RULE_DIRS.items():
        for group, stem in GROUP_FILES.items():
            for seed in (0, 1, 2):
                path = run_dir / directory / f"{stem}_seed{seed}.pt"
                entry: Dict[str, object] = {
                    "rule": rule, "group": group, "seed": seed,
                    "path": str(path.relative_to(Path(__file__).resolve().parents[3])),
                    "exists": path.exists(),
                    "sha256": _sha256(path) if path.exists() else None,
                    "width": None, "best_epoch": None, "updates": None, "train_samples": None,
                }
                record = training[rule]["models"].get(group, {}).get(str(seed))
                if record:
                    entry["best_epoch"] = record.get("best_epoch")
                    entry["updates"] = record.get("updates")
                    entry["train_samples"] = record.get("train_samples")
                if path.exists():
                    blob = torch.load(path, map_location="cpu")
                    entry["width"] = int(blob["config"]["hidden"])
                    entry["layers"] = int(blob["config"]["layers"])
                    entry["checkpoint_seed"] = int(blob.get("seed", -1))
                rows.append(entry)
    return rows


def load_training_data(run_dir: Path, source_run: Path) -> Dict[str, List[Dict[str, object]]]:
    original = [json.loads(line) for line in (source_run / "dataset" / "windows.jsonl").open()]
    branch_dir = run_dir / "branch_data"
    branch_all = [json.loads(line) for line in (branch_dir / "branch_windows.jsonl").open()]
    matched_random = [json.loads(line) for line in (branch_dir / "random_train_matched.jsonl").open()]
    matched_branch = [json.loads(line) for line in (branch_dir / "branch_train_matched.jsonl").open()]
    return {
        "original_train": [r for r in original if r["split"] == "train"],
        "original_val": [r for r in original if r["split"] == "val"],
        "branch_val": [r for r in branch_all if r["split"] == "val"],
        "train": {
            GROUP_OLD: [r for r in original if r["split"] == "train"],
            GROUP_RANDOM: [r for r in original if r["split"] == "train"] + matched_random,
            GROUP_BRANCH: [r for r in original if r["split"] == "train"] + matched_branch,
        },
    }


FLAT_METRICS = (
    "loss.state", "loss.hold", "loss.duration", "loss.termination", "loss.total",
    "position.object_mean", "position.object_p90", "position.robot0_mean", "position.robot1_mean",
    "angle.object_mean", "velocity.object_speed_mean",
    "legality.invalid_rate", "legality.non_finite_count",
    "legality.penetration_p50", "legality.penetration_p90", "legality.penetration_p99",
    "termination.true_success", "termination.predicted_success",
    "termination.success_recall", "termination.success_precision",
    "termination.predicted_success_and_illegal",
    "persistence.object_position_error", "persistence.object_angle_error",
    "persistence.paired_object_position_difference", "persistence.paired_object_angle_difference",
)


def flatten(metrics: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {"n": metrics.get("n")}
    for dotted in FLAT_METRICS:
        section, key = dotted.split(".", 1)
        value = metrics.get(section, {})
        out[dotted] = value.get(key) if isinstance(value, dict) else None
    return out


def write_csv(path: Path, rows: Sequence[Dict[str, object]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    names = list(fieldnames) if fieldnames else list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in names})
