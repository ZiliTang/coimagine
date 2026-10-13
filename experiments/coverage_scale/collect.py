"""Collect the coverage-scale dataset: val/test first, then nested train tiers.

    python experiments/coverage_scale/collect.py --run-id coverage-scale-20261013 \
        --workers 4 --train-shards 300

The master materialises the deterministic root plan for every split (so no two workers race
on the plan file), then runs ``workers`` independent worker processes over a global task
queue. Each worker owns whole shards, writes them atomically, and appends one progress line
per finished shard. A shard only counts once its ``_done`` marker exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.coverage_scale.plan import RootPlan  # noqa: E402
from coimagine.coverage_scale.shards import (  # noqa: E402
    ROOTS_PER_SHARD,
    collect_shard,
    is_done,
    scan_done,
    shard_dir,
)
from coimagine.coverage_scale.sampler import FAMILIES, MODES  # noqa: E402
from coimagine.continuous2d.config import default_config  # noqa: E402

VAL_TEST_ROOTS = 128
FIRST_TIER_WINDOWS = 8000
APPROX_WINDOWS_PER_ROOT = 28


def train_shards_for(windows: int) -> int:
    roots = int(windows / APPROX_WINDOWS_PER_ROOT) + 1
    return (roots + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD + 1


def build_tasks(train_shards: int, first_tier_shards: int) -> list:
    """val and test are frozen first, then the 8k train tier, then the rest of train."""

    val_shards = (VAL_TEST_ROOTS + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD
    tasks = []
    tasks += [("train", shard) for shard in range(first_tier_shards)]
    tasks += [("val", shard) for shard in range(val_shards)]
    tasks += [("test", shard) for shard in range(val_shards)]
    tasks += [("train", shard) for shard in range(first_tier_shards, train_shards)]
    return tasks


def progress_path(run_dir: Path) -> Path:
    return run_dir / "collection_progress.jsonl"


def worker_main(args) -> int:
    run_dir = ROOT / args.out_root / args.run_id
    config = default_config()
    plan = RootPlan(run_dir / "roots")
    tasks = build_tasks(args.train_shards, args.first_tier_shards)
    log = progress_path(run_dir).open("a")
    done_count = 0
    for index, (split, shard) in enumerate(tasks):
        if index % args.workers != args.worker_id:
            continue
        if is_done(run_dir, split, shard):
            continue
        if args.deadline and time.time() > args.deadline:
            break
        started = time.time()
        meta = collect_shard(run_dir, split, shard, config, plan)
        log.write(json.dumps({
            "worker": args.worker_id, "split": split, "shard": shard,
            "windows": meta["windows"], "root_count": meta["root_count"],
            "wall_clock": time.time() - started, "at": time.time(),
        }) + "\n")
        log.flush()
        done_count += 1
        if done_count % 5 == 0:
            print(f"worker {args.worker_id}: {done_count} shards done", flush=True)
    log.close()
    return 0


def master_main(args) -> int:
    run_dir = ROOT / args.out_root / args.run_id
    (run_dir / "roots").mkdir(parents=True, exist_ok=True)
    plan = RootPlan(run_dir / "roots")

    val_shards = (VAL_TEST_ROOTS + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD
    started = time.time()
    print(f"materialising root plans: train {args.train_shards * ROOTS_PER_SHARD} roots, "
          f"val/test {VAL_TEST_ROOTS} roots", flush=True)
    for split, count in (("train", args.train_shards * ROOTS_PER_SHARD),
                         ("val", VAL_TEST_ROOTS), ("test", VAL_TEST_ROOTS)):
        roots = plan.ensure(split, count)
        counts = {}
        for root in roots:
            counts[f"{root.family}/{root.mode}"] = counts.get(f"{root.family}/{root.mode}", 0) + 1
        print(f"  {split}: {len(roots)} roots {counts}", flush=True)
    (run_dir / "root_plan_summary.json").write_text(json.dumps({
        "train_shards": args.train_shards,
        "val_test_roots": VAL_TEST_ROOTS,
        "mode_weights": {mode: sum(1 for r in plan.peek("train") if r.mode == mode) / max(1, len(plan.peek("train")))
                         for mode in MODES},
        "families": list(FAMILIES),
        "materialise_wall_clock": time.time() - started,
    }, indent=2, sort_keys=True))

    tasks = build_tasks(args.train_shards, args.first_tier_shards)
    deadline = args.deadline_seconds and (time.time() + args.deadline_seconds)
    processes = []
    for worker_id in range(args.workers):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--worker-id", str(worker_id),
            "--workers", str(args.workers), "--run-id", args.run_id, "--out-root", args.out_root,
            "--train-shards", str(args.train_shards), "--first-tier-shards", str(args.first_tier_shards),
        ]
        if deadline:
            command += ["--deadline", str(deadline)]
        processes.append(subprocess.Popen(command))
    for process in processes:
        process.wait()
    summary = {
        "workers": args.workers, "tasks": len(tasks),
        "done": {split: len(scan_done(run_dir, split)) for split in ("train", "val", "test")},
        "wall_clock": time.time() - started,
    }
    (run_dir / "collection_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=1), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--train-shards", type=int, default=300)
    parser.add_argument("--first-tier-shards", type=int, default=0)
    parser.add_argument("--deadline", type=float, default=0.0)
    parser.add_argument("--deadline-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if args.first_tier_shards == 0:
        args.first_tier_shards = train_shards_for(FIRST_TIER_WINDOWS)
    if args.worker_id >= 0:
        return worker_main(args)
    return master_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
