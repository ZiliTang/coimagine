"""Phase D1: the equal-size narrow-coverage control.

    python experiments/coverage_scale/collect_narrow.py --run-id coverage-scale-20261013 \
        --target-windows 16000 --workers 4

Starts from the six original training initial states and uses the *same* B1 recipe and the same
dedup semantics as the wide-coverage group. Only extra collection rounds are added, so the
number of independent initial states is still six; that is reported, never hidden.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.coverage_scale.plan import NARROW_SCENES, NarrowPlan  # noqa: E402
from coimagine.coverage_scale.shards import (  # noqa: E402
    ROOTS_PER_SHARD,
    collect_shard,
    is_done,
    scan_done,
)
from coimagine.continuous2d.config import default_config  # noqa: E402

ATTEMPT_MULTIPLIER = 10


def rounds_for(target_windows: int, windows_per_round: float = 29.6) -> int:
    return int(target_windows * ATTEMPT_MULTIPLIER / windows_per_round) + len(NARROW_SCENES)


def worker_main(args) -> int:
    run_dir = ROOT / args.out_root / args.run_id
    config = default_config()
    plan = NarrowPlan(run_dir / "roots")
    shards = args.shards
    progress = (run_dir / "narrow_progress.jsonl").open("a")
    for shard in range(shards):
        if shard % args.workers != args.worker_id:
            continue
        if is_done(run_dir, "narrow", shard):
            continue
        meta = collect_shard(run_dir, "narrow", shard, config, plan)
        progress.write(json.dumps({
            "worker": args.worker_id, "split": "narrow", "shard": shard,
            "windows": meta["windows"], "root_count": meta["root_count"],
        }) + "\n")
        progress.flush()
    progress.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--target-windows", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-id", type=int, default=-1)
    parser.add_argument("--deadline-seconds", type=float, default=1800.0)
    parser.add_argument("--shards", type=int, default=0,
                        help="explicit shard count; 0 uses the 10x attempt rule")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    rounds = rounds_for(args.target_windows)
    shards = args.shards or (rounds + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD
    args.shards = shards
    if args.worker_id >= 0:
        return worker_main(args)

    print(f"narrow control: target {args.target_windows} windows, {rounds} rounds, {shards} shards, "
          f"independent initial states {len(NARROW_SCENES)}", flush=True)
    processes = []
    for worker_id in range(args.workers):
        command = [
            sys.executable, str(Path(__file__).resolve()), "--run-id", args.run_id,
            "--out-root", args.out_root, "--target-windows", str(args.target_windows),
            "--workers", str(args.workers), "--worker-id", str(worker_id),
            "--shards", str(shards),
        ]
        processes.append(subprocess.Popen(command))
    for process in processes:
        process.wait()
    summary = {
        "target_windows": args.target_windows, "rounds": rounds, "shards": shards,
        "independent_initial_states": list(NARROW_SCENES),
        "done_shards": len(scan_done(run_dir, "narrow")),
    }
    (run_dir / "narrow_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps(summary, indent=1))
    print("NARROW_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
