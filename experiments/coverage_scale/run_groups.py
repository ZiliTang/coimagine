"""Item 3: grouped metrics on the same new validation set for the 128k and 256k models.

    python experiments/coverage_scale/run_groups.py

Windows are grouped by the frozen thresholds into object translation / rotation / stationary
and into contact / no contact. The two partitions are independent and overlap; the overlap
counts are reported. For every group the object, robot 0 and robot 1 position errors are given
separately, each next to its own persistence baseline, plus the invalid rate and success
windows. Nothing here is averaged into a single "P" before the per-body numbers are shown.
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

from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig  # noqa: E402
from coimagine.coverage_scale.evaluation import grouped_summary  # noqa: E402
from coimagine.coverage_scale.plan import load_registries  # noqa: E402
from coimagine.coverage_scale.shards import tier_records  # noqa: E402

DATA_RUN = "outputs/coverage_scale_v1/coverage-scale-20261013"
OUT_RUN = "outputs/coverage_scale_v1/coverage-scale-256k"
BODY_KEYS = ("object_position", "robot0_position", "robot1_position")
PERSISTENCE_KEYS = ("persistence_object", "persistence_robot0", "persistence_robot1")


def load_model(path: Path) -> JointWindowMLP:
    blob = torch.load(path, map_location="cpu")
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kinds", default="best_new_val_total")
    args = parser.parse_args()

    data_dir = ROOT / DATA_RUN
    out_dir = ROOT / OUT_RUN
    out_dir.mkdir(parents=True, exist_ok=True)
    registry = load_registries(data_dir / "roots", ("train", "val", "test"))
    registry.activate()
    val, val_info = tier_records(data_dir, "val", 10**9)

    kinds = [value for value in args.kinds.split(",") if value]
    models = []
    for tier, base in (("128000", data_dir / "runs"), ("256000", out_dir / "runs")):
        for width in (256, 512):
            for seed in (0, 1, 2):
                for kind in kinds:
                    path = base / f"{tier}_{width}_seed{seed}" / f"{kind}.pt"
                    if path.exists():
                        models.append((tier, width, seed, kind, path))
    print(f"models: {len(models)}; val windows {len(val)} / roots {val_info['independent_roots']}", flush=True)

    rows = []
    started = time.time()
    for tier, width, seed, kind, path in models:
        model = load_model(path)
        summary = grouped_summary(model, val)
        rows.append({"tier": tier, "width": width, "seed": seed, "kind": kind,
                     "checkpoint": str(path.relative_to(ROOT)), "summary": summary})
        print(f"{tier}/{width}/{seed}/{kind}: done ({time.time()-started:.1f}s)", flush=True)

    payload = {"val": val_info, "groups": rows}
    (out_dir / "groups.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    # -- aggregate over seeds, per (tier, width, kind) ----------------------
    aggregated = {}
    for tier, width, seed, kind, _ in models:
        key = f"{tier}/{width}/{kind}"
        aggregated.setdefault(key, []).append((seed, next(
            row["summary"] for row in rows
            if (row["tier"], row["width"], row["seed"], row["kind"]) == (tier, width, seed, kind)
        )))
    table = {}
    for key, entries in sorted(aggregated.items()):
        groups = entries[0][1]["groups"]
        entry = {"seeds": len(entries), "groups": {}}
        for name in groups:
            cell = {"n_windows": groups[name]["n_windows"], "n_roots": groups[name]["n_roots"],
                    "success_windows": groups[name]["success_windows"]}
            weighted_available = all(
                item[1]["groups"][name]["weighted"].get("object_position") is not None for item in entries
            )
            for body, persistence in zip(BODY_KEYS, PERSISTENCE_KEYS):
                cell[body] = sum(item[1]["groups"][name][body] for item in entries) / len(entries)
                cell[persistence] = sum(item[1]["groups"][name][persistence] for item in entries) / len(entries)
                cell["weighted_" + body] = (
                    sum(item[1]["groups"][name]["weighted"][body] for item in entries) / len(entries)
                    if weighted_available else None
                )
            cell["invalid_rate"] = sum(item[1]["groups"][name]["invalid"] for item in entries) / len(entries)
            cell["weighted_invalid_rate"] = (
                sum(item[1]["groups"][name]["weighted"]["invalid"] for item in entries) / len(entries)
                if weighted_available else None
            )
            cell["position_mean"] = sum(item[1]["groups"][name]["position_mean"] for item in entries) / len(entries)
            cell["weighted_position_mean"] = (
                sum(item[1]["groups"][name]["weighted"]["position_mean"] for item in entries) / len(entries)
                if weighted_available else None
            )
            entry["groups"][name] = cell
        entry["overlap"] = entries[0][1]["overlap"]
        entry["partitions"] = entries[0][1]["partitions"]
        table[key] = entry
    (out_dir / "groups_table.json").write_text(json.dumps(table, indent=1, sort_keys=True))
    print(json.dumps({key: {"all": {k: round(v, 4) for k, v in value["groups"]["all"].items()
                                   if isinstance(v, float)}} for key, value in table.items()}, indent=1))
    print("GROUPS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
