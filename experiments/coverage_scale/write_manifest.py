"""Write ``docs/coverage_scale_v1/manifest.json`` (task book section 8).

    python experiments/coverage_scale/write_manifest.py --run-id coverage-scale-20261013
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

TIERS = ("8000", "16000", "32000", "64000", "128000")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # pragma: no cover
        return ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    from coimagine.continuous2d.config import default_config
    from coimagine.coverage_scale.sampler import (
        MODE_SLOTS, S0_GOAL_CENTER, S0_GOAL_HALF, S0_GOAL_JITTER, S1_GAP_CENTER_RANGE,
        S1_GAP_WIDTH_RANGE, S1_GOAL_CENTER_X, S1_GOAL_HALF, S1_GOAL_JITTER, S1_WALL_X_RANGE,
        SPLIT_SEED_BASE, WALL_CLEARANCE_RANGE,
    )
    from coimagine.coverage_scale.shards import contiguous_done, tier_records
    from coimagine.coverage_scale.trainer import eval_schedule, update_budget

    data = {}
    for split, target in (("train", 128000), ("val", 10**9), ("test", 10**9), ("narrow", 16017)):
        if not (run_dir / "shards" / split).exists():
            continue
        _, info = tier_records(run_dir, split, target)
        data[split] = info
        data[split]["contiguous_shards"] = len(contiguous_done(run_dir, split))

    runs = []
    for tier in TIERS:
        for cell in sorted((run_dir / "runs").glob(f"{tier}_*_seed*")):
            summary = cell / "run_summary.json"
            if summary.exists():
                payload = json.loads(summary.read_text())
                runs.append({
                    "tier": tier, "width": payload["width"], "seed": payload["seed"],
                    "updates": payload["updates"], "windows": payload["windows"],
                    "sample_order_hash": payload["sample_order_hash"],
                    "parameter_hash": payload["checkpoints"]["best_new_val_total"]["parameter_hash"],
                    "file_sha256": payload["checkpoints"]["best_new_val_total"]["file_sha256"],
                    "wall_clock": payload["wall_clock"],
                })
    narrow = []
    for cell in sorted((run_dir / "narrow").glob("*_seed*")):
        summary = cell / "run_summary.json"
        if summary.exists():
            payload = json.loads(summary.read_text())
            narrow.append({
                "tier": payload["tier"], "width": payload["width"], "seed": payload["seed"],
                "updates": payload["updates"], "windows": payload["windows"],
                "parameter_hash": payload["checkpoints"]["best_new_val_total"]["parameter_hash"],
            })

    registries = {}
    for split in ("train", "val", "test", "narrow"):
        path = run_dir / "roots" / f"scenes_{split}.json"
        if path.exists():
            registries[split] = {"sha256": sha256(path), "bytes": path.stat().st_size}

    manifest = {
        "run_id": args.run_id,
        "task_book": "docs/tasks/20261013_0225_CoImagine_过夜数据覆盖与模型规模联合扩展任务书.md",
        "contract": "docs/coverage_scale_v1/CONTRACT.md",
        "errata": "docs/coverage_scale_v1/errata.md",
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": git("status", "--porcelain").splitlines(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": default_config().as_dict(),
        "sampler_ranges": {
            "S0_goal_center": S0_GOAL_CENTER, "S0_goal_half": S0_GOAL_HALF,
            "S0_goal_jitter": S0_GOAL_JITTER,
            "S1_wall_x": S1_WALL_X_RANGE, "S1_gap_center": S1_GAP_CENTER_RANGE,
            "S1_gap_width": S1_GAP_WIDTH_RANGE,
            "S1_goal_center_x": S1_GOAL_CENTER_X, "S1_goal_half": S1_GOAL_HALF,
            "S1_goal_jitter": S1_GOAL_JITTER,
            "wall_clearance": WALL_CLEARANCE_RANGE,
            "mode_slots": MODE_SLOTS,
            "split_seed_base": SPLIT_SEED_BASE,
        },
        "datasets": data,
        "update_budget": {tier: update_budget(int(tier)) for tier in TIERS},
        "eval_schedule_128000": list(eval_schedule(update_budget(128000))),
        "runs": runs,
        "narrow_runs": narrow,
        "scene_registry": registries,
    }
    (ROOT / "docs" / "coverage_scale_v1").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "coverage_scale_v1" / "manifest.json").write_text(
        json.dumps(manifest, indent=1, sort_keys=True)
    )
    print(json.dumps({"datasets": data, "runs": len(runs), "narrow_runs": len(narrow)}, indent=1))
    print("MANIFEST_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
