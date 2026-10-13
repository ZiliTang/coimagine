"""Item 4: one unified evaluation after the model list is frozen.

    python experiments/coverage_scale/run_eval_256k.py

Order is fixed and enforced by this script:

1. write the frozen model manifest (paths + SHA256 + parameter hashes) and never add to it;
2. evaluate the new test set once for every frozen checkpoint;
3. score the fixed candidate set (reusing the cached exact reference, so the truth is computed
   exactly once across both rounds) for the main selection points of the 128k and 256k tiers.

The test set is never read by the training scripts, and no test number feeds back into the
model list or into the decision to keep scaling. H=1 and H=3 are tabulated separately; pairing
is on ``(start_id, candidate_seed)`` and only rows feasible on both sides enter the paired
means.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(4)

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig  # noqa: E402
from coimagine.coverage_scale import candidate_eval as ce  # noqa: E402
from coimagine.coverage_scale.evaluation import weighted_summary  # noqa: E402
from coimagine.coverage_scale.plan import load_registries  # noqa: E402
from coimagine.coverage_scale.shards import tier_records  # noqa: E402

DATA_RUN = "outputs/coverage_scale_v1/coverage-scale-20261013"
OUT_RUN = "outputs/coverage_scale_v1/coverage-scale-256k"
PAIRED_KEYS = (
    "chosen_true_return", "feasible_chosen_true_return", "candidate_best_true_return",
    "uniform_candidate_true_return", "misrejection_loss", "ranking_loss", "execution_regret",
    "rejection_rate", "no_feasible_candidate", "selected_bias", "selection_excess",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_model(path: Path) -> JointWindowMLP:
    blob = torch.load(path, map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def paired(rows, label_a: str, label_b: str, horizon: int):
    def index(label):
        return {(row["start_id"], row["candidate_seed"]): row
                for row in rows if str(row["label"]) == label and int(row["H"]) == horizon}

    a, b = index(label_a), index(label_b)
    common = sorted(set(a) & set(b))
    entry = {"pair": f"{label_a}->{label_b}", "H": horizon, "common_rows": len(common),
             "feasible_both": 0, "solved_to_unsolved": 0, "unsolved_to_solved": 0}
    diffs = defaultdict(list)
    for key in common:
        left, right = a[key], b[key]
        if left["feasible_chosen_true_return"] is None or right["feasible_chosen_true_return"] is None:
            if left["feasible_chosen_true_return"] is None and right["feasible_chosen_true_return"] is not None:
                entry["unsolved_to_solved"] += 1
            elif left["feasible_chosen_true_return"] is not None and right["feasible_chosen_true_return"] is None:
                entry["solved_to_unsolved"] += 1
            continue
        entry["feasible_both"] += 1
        for name in PAIRED_KEYS:
            left_value, right_value = left.get(name), right.get(name)
            if left_value is None or right_value is None:
                continue
            diffs[name].append(float(right_value) - float(left_value))
    for name in PAIRED_KEYS:
        values = diffs.get(name, [])
        entry[f"delta_{name}"] = (sum(values) / len(values)) if values else None
        entry[f"delta_{name}_n"] = len(values)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default="128000,256000")
    parser.add_argument("--kinds", default="best_new_val_total,final,update60000")
    parser.add_argument("--candidate-kinds", default="best_new_val_total")
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    data_dir = ROOT / DATA_RUN
    out_dir = ROOT / OUT_RUN
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registries(data_dir / "roots", ("train", "val", "test"))
    registry.activate()

    tiers = [value for value in args.tiers.split(",") if value]
    kinds = [value for value in args.kinds.split(",") if value]
    candidate_kinds = [value for value in args.candidate_kinds.split(",") if value]

    # ---- 1. freeze the model list ----------------------------------------
    frozen = []
    for tier in tiers:
        base = data_dir / "runs" if tier == "128000" else out_dir / "runs"
        for width in (256, 512):
            for seed in (0, 1, 2):
                for kind in kinds:
                    path = base / f"{tier}_{width}_seed{seed}" / f"{kind}.pt"
                    if not path.exists():
                        continue
                    blob = torch.load(path, map_location="cpu")
                    frozen.append({
                        "label": f"{tier}/{width}/{seed}/{kind}",
                        "tier": tier, "width": width, "seed": seed, "kind": kind,
                        "path": str(path.relative_to(ROOT)),
                        "file_sha256": sha256(path),
                        "parameter_hash": blob.get("parameter_hash"),
                        "updates": blob.get("updates"),
                        "new_val_total_loss": blob.get("new_val_total_loss"),
                    })
    manifest_path = out_dir / "model_list.json"
    frozen_hash = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous["model_list_sha256"] != frozen_hash:
            raise SystemExit("the frozen model list changed; refusing to evaluate a different list")
    manifest_path.write_text(json.dumps({
        "model_list_sha256": frozen_hash,
        "frozen_before_any_test_evaluation": True,
        "models": frozen,
        "candidate_evaluation_kinds": candidate_kinds,
    }, indent=1, sort_keys=True))
    print(f"frozen model list: {len(frozen)} checkpoints, sha256 {frozen_hash[:16]}", flush=True)

    # ---- 2. the single new-test pass --------------------------------------
    new_test, test_info = tier_records(data_dir, "test", 10**9)
    rows = []
    started = time.time()
    for entry in frozen:
        model = load_model(ROOT / entry["path"])
        summary = weighted_summary(model, new_test)
        rows.append({
            "label": entry["label"], "tier": entry["tier"], "width": entry["width"],
            "seed": entry["seed"], "kind": entry["kind"],
            "new_test": {
                "n": summary["n"],
                "weighted": summary["weighted"],
                "unweighted": summary["unweighted"],
                "legality": summary["legality"],
                "success": summary["success"],
                "cells": summary["cells"],
                "persistence_weighted_object_position": summary["persistence_weighted_object_position"],
            },
        })
        print(f"test {entry['label']}: done ({time.time()-started:.1f}s)", flush=True)
    (out_dir / "holdout_256k.json").write_text(json.dumps({
        "test": test_info, "rows": rows, "used_for_selection": False,
    }, indent=1, sort_keys=True))

    # ---- 3. fixed candidates, reference reused ----------------------------
    config = default_config()
    starts = [ce.make_start(root, config) for root in ce.select_test_starts(data_dir)]
    reference_path = data_dir / "candidates" / "reference.json"
    reference = json.loads(reference_path.read_text())["reference"]
    print(f"candidate reference reused: {len(reference)} pools", flush=True)

    candidate_rows = []
    for tier in tiers:
        base = data_dir / "runs" if tier == "128000" else out_dir / "runs"
        for width in (256, 512):
            for seed in (0, 1, 2):
                for kind in candidate_kinds:
                    path = base / f"{tier}_{width}_seed{seed}" / f"{kind}.pt"
                    if not path.exists():
                        continue
                    label = f"{tier}/{width}/{seed}/{kind}"
                    model = load_model(path)
                    candidate_rows.extend(
                        ce.evaluate_model_on_starts(model, starts, reference, config, label=label)
                    )
                    print(f"candidates {label}: done", flush=True)
    cand_dir = out_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    (cand_dir / "rows.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in candidate_rows))
    labels = sorted({str(row["label"]) for row in candidate_rows})
    cross_tier_pairs = [
        (a, b) for a in labels for b in labels
        if a.split("/")[0] == "128000" and b.split("/")[0] == "256000"
        and a.split("/")[1:] == b.split("/")[1:]
    ]
    per_horizon = {}
    for horizon in (1, 3):
        per_horizon[str(horizon)] = {
            "aggregate": ce.aggregate_by_horizon(candidate_rows, horizon),
            "paired": [paired(candidate_rows, a, b, horizon) for a, b in cross_tier_pairs],
        }
    (cand_dir / "aggregate_by_horizon.json").write_text(json.dumps({
        "horizons": [1, 3], "labels": labels, "by_horizon": per_horizon,
        "cross_tier_pairs": len(cross_tier_pairs),
        "reference_reused": True, "K": ce.POOL_SIZE, "candidate_seeds": list(ce.CANDIDATE_SEEDS),
    }, indent=1, sort_keys=True))
    print(json.dumps({horizon: [row for row in per_horizon[horizon]["aggregate"]
                                if row["label"].endswith("best_new_val_total")][:4]
                      for horizon in ("1", "3")}, indent=1))
    print("EVAL_256K_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
