"""Coverage table and manifest for the collected data (task book B and section 8).

    python experiments/coverage_scale/make_coverage.py --run-id coverage-scale-20261013

Reads the per-shard root metadata written during collection (not the windows themselves) and
reports, per ``family x mode`` cell and per split: roots, windows, contact windows, success
windows, physical-failure windows, and the sampler acceptance rate of each cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.coverage_scale.recipe import BRANCHES_PER_ANCHOR, RECIPE_VERSION, MAX_REAL_WINDOWS  # noqa: E402
from coimagine.coverage_scale.sampler import MODES  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_stats(run_dir: Path):
    per_split = defaultdict(lambda: {
        "roots": 0, "windows": 0, "contacts": 0, "successes": 0, "failures": 0,
        "guide_only_success": 0, "no_feasible": 0, "cells": defaultdict(lambda: {
            "roots": 0, "windows": 0, "contacts": 0, "successes": 0, "failures": 0,
            "attempts": [], "roots_with_contact": 0, "roots_with_success": 0,
        }),
    })
    for split_dir in sorted((run_dir / "shards").glob("*")):
        split = split_dir.name
        for meta_path in sorted(split_dir.glob("shard_*.meta.json")):
            meta = json.loads(meta_path.read_text())
            bucket = per_split[split]
            for stat in meta["stats"]:
                cell_key = f"{stat['family']}/{stat['mode']}"
                cell = bucket["cells"][cell_key]
                bucket["roots"] += 1
                bucket["windows"] += stat["windows_total"]
                bucket["contacts"] += stat["contact_windows"]
                bucket["successes"] += int(bool(stat["success"]))
                bucket["failures"] += stat["failure_windows"]
                bucket["guide_only_success"] += int(bool(stat["guide_success"]) and not stat["success"])
                bucket["no_feasible"] += int(bool(stat["no_feasible_plan"]))
                cell["roots"] += 1
                cell["windows"] += stat["windows_total"]
                cell["contacts"] += stat["contact_windows"]
                cell["successes"] += int(bool(stat["success"]))
                cell["failures"] += stat["failure_windows"]
                cell["attempts"].append(stat["attempts"])
                cell["roots_with_contact"] += int(stat["contact_windows"] > 0)
                cell["roots_with_success"] += int(bool(stat["success"]))
    out = {}
    for split, bucket in per_split.items():
        cells = {}
        for key, cell in bucket["cells"].items():
            attempts = cell.pop("attempts")
            cell["mean_attempts"] = sum(attempts) / len(attempts)
            cell["acceptance_per_attempt"] = 1.0 / cell["mean_attempts"]
            cell["mean_windows_per_root"] = cell["windows"] / max(1, cell["roots"])
            cells[key] = cell
        payload = dict(bucket)
        payload["cells"] = cells
        payload["mean_windows_per_root"] = payload["windows"] / max(1, payload["roots"])
        out[split] = payload
    return out


def motion_counts(run_dir: Path) -> dict:
    """Success / contact / motion counts over the exact tiers that are actually trained on."""

    from coimagine.coverage_scale.shards import tier_records

    out = {}
    for split, target in (("train", 128000), ("val", 10**9), ("test", 10**9), ("narrow", 16017)):
        if not (run_dir / "shards" / split).exists():
            continue
        records, info = tier_records(run_dir, split, target)
        windows = len(records)
        keys = ("translation", "rotation", "stationary", "contact", "success", "failure")
        counts = {key: 0 for key in keys}
        per_cell: dict = {}
        success_roots = set()
        contact_roots = set()
        for record in records:
            before, after = record["state"], record["next_state"]
            dx = float(after["object"][0]) - float(before["object"][0])
            dy = float(after["object"][1]) - float(before["object"][1])
            translation = (dx * dx + dy * dy) ** 0.5 > 0.01
            rotation = abs(float(after["object"][2]) - float(before["object"][2])) > 0.02
            contact = bool(record["contact"]["had_robot_object_contact"])
            success = bool(int(record["success"]))
            failure = str(record["termination_class"]) == "physical_failure"
            flags = {
                "translation": int(translation), "rotation": int(rotation),
                "stationary": int(not translation and not rotation), "contact": int(contact),
                "success": int(success), "failure": int(failure),
            }
            for key, value in flags.items():
                counts[key] += value
            root_id = str(record["root_id"])
            if success:
                success_roots.add(root_id)
            if contact:
                contact_roots.add(root_id)
            cell = f"{record.get('family', '?')}/{record.get('mode', '?')}"
            row = per_cell.setdefault(cell, {key: 0 for key in keys})
            for key, value in flags.items():
                row[key] += value
        out[split] = {
            "windows": windows, "independent_roots": info["independent_roots"],
            "counts": counts, "per_cell": per_cell,
            "roots_with_contact": len(contact_roots), "roots_with_success": len(success_roots),
            "thresholds": {"translation_m": 0.01, "rotation_rad": 0.02},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    stats = collect_stats(run_dir)
    motion = motion_counts(run_dir)
    registry_hashes = {}
    for split in ("train", "val", "test", "narrow"):
        path = run_dir / "roots" / f"scenes_{split}.json"
        if path.exists():
            registry_hashes[split] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    payload = {
        "run_id": args.run_id,
        "recipe_version": RECIPE_VERSION,
        "max_real_windows_per_root": MAX_REAL_WINDOWS,
        "branches_per_anchor": BRANCHES_PER_ANCHOR,
        "split_stats": stats,
        "motion_counts": motion,
        "scene_registry": registry_hashes,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    (run_dir / "coverage_table.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    lines = ["# coverage_scale_v1 - data coverage", "",
             f"recipe `{RECIPE_VERSION}`, at most {MAX_REAL_WINDOWS} real windows per root.", ""]
    for split in ("train", "val", "test", "narrow"):
        if split not in stats:
            continue
        bucket = stats[split]
        lines.append(f"## {split}: {bucket['roots']} collection roots, {bucket['windows']} windows, "
                     f"{bucket['mean_windows_per_root']:.1f} windows/root")
        lines.append("")
        lines.append("| family/mode | roots | windows | contact windows | root success | failure windows | mean attempts | acceptance/attempt |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key in sorted(bucket["cells"]):
            cell = bucket["cells"][key]
            lines.append(
                f"| {key} | {cell['roots']} | {cell['windows']} | {cell['contacts']} | "
                f"{cell['roots_with_success']} | {cell['failures']} | {cell['mean_attempts']:.1f} | "
                f"{100*cell['acceptance_per_attempt']:.2f}% |"
            )
        lines.append("")
    lines.append("## success / contact / motion counts over the trained tiers")
    lines.append("")
    lines.append("| split | windows | independent roots | roots with contact | roots with success | translation windows | rotation windows | stationary | contact | success | physical failure |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split in ("train", "val", "test", "narrow"):
        if split not in motion:
            continue
        entry = motion[split]
        counts = entry["counts"]
        lines.append(
            f"| {split} | {entry['windows']} | {entry['independent_roots']} | "
            f"{entry['roots_with_contact']} | {entry['roots_with_success']} | "
            f"{counts['translation']} | {counts['rotation']} | {counts['stationary']} | "
            f"{counts['contact']} | {counts['success']} | {counts['failure']} |"
        )
    lines.append("")
    (run_dir / "coverage_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("COVERAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
