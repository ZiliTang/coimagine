"""Phase C: one data tier x two widths x three seeds, six runs per tier.

    python experiments/coverage_scale/run_matrix.py --tier 8000 --run-id coverage-scale-20261013 --parallel 3

A cell is ``(tier, width, seed)``. The master spawns independent per-cell processes (each
pinned to 4 torch threads) and keeps at most ``--parallel`` alive; a finished cell is skipped
on a rerun, and every cell keeps a resumable ``resume_state.pt``.
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

WIDTHS = (256, 512)
SEEDS = (0, 1, 2)


def load_tier(run_dir: Path, split: str, target: int):
    from coimagine.coverage_scale.plan import load_registries
    from coimagine.coverage_scale.shards import select_monitor, tier_records

    registry = load_registries(run_dir / "roots", ("train", "val", "test"))
    registry.activate()
    records, info = tier_records(run_dir, split, target)
    return records, info, registry


def run_single(run_dir: Path, tier: str, width: int, seed: int, *, split: str = "train",
               wait_seconds: float = 3600.0) -> int:
    import torch

    torch.set_num_threads(4)
    from coimagine.coverage_scale.shards import select_monitor, tier_records
    from coimagine.coverage_scale.trainer import run_cell, update_budget

    target = int(tier)
    deadline = time.time() + wait_seconds
    while True:
        train, info, _ = load_tier(run_dir, split, target)
        if info["reached_target"]:
            break
        if time.time() > deadline:
            raise SystemExit(f"{split} tier {tier} still incomplete after waiting: {info}")
        print(f"waiting for {split} tier {tier}: {info['actual_windows']} / {target} windows", flush=True)
        time.sleep(20)
    monitor = select_monitor(train, 4096)
    val, val_info, _ = load_tier(run_dir, "val", 10**9)
    cell_dir = (run_dir / "runs" if split == "train" else run_dir / split) / f"{tier}_{width}_seed{seed}"
    result = run_cell(
        tier, width, seed, train, monitor, val, cell_dir,
        max_updates=update_budget(len(train)),
    )
    meta = {
        "tier": tier, "width": width, "seed": seed, "split": split,
        "train": info, "val": val_info, "monitor_windows": len(monitor),
        "max_updates": result.max_updates, "updates": result.updates,
        "wall_clock": result.wall_clock, "sample_order_hash": result.sample_order_hash,
        "stopped_reason": result.stopped_reason,
    }
    (cell_dir / "cell_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--tier", default="8000")
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--split", default="train")
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--deadline-seconds", type=float, default=0.0)
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    if args.single:
        return run_single(run_dir, args.tier, args.width, args.seed, split=args.split)

    widths = [int(value) for value in args.widths.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    cells = [(width, seed) for width in widths for seed in seeds]
    deadline = time.time() + args.deadline_seconds if args.deadline_seconds else None

    running: list = []
    started = time.time()
    index = 0
    results = []
    while index < len(cells) or running:
        while len(running) < args.parallel and index < len(cells):
            width, seed = cells[index]
            index += 1
            cell_dir = (run_dir / "runs" if args.split == "train" else run_dir / args.split) / f"{args.tier}_{width}_seed{seed}"
            if (cell_dir / "run_summary.json").exists():
                results.append({"tier": args.tier, "width": width, "seed": seed, "status": "skipped"})
                continue
            if deadline and time.time() > deadline:
                results.append({"tier": args.tier, "width": width, "seed": seed, "status": "budget_exhausted"})
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--run-id", args.run_id, "--out-root", args.out_root, "--tier", str(args.tier),
                "--single", "--width", str(width), "--seed", str(seed),
                "--split", args.split,
            ]
            cell_dir.mkdir(parents=True, exist_ok=True)
            log = (cell_dir / "stdout.log").open("w")
            running.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), width, seed))
        if not running:
            continue
        time.sleep(5)
        for entry in list(running):
            process, width, seed = entry
            if process.poll() is not None:
                running.remove(entry)
                cell_dir = (run_dir / "runs" if args.split == "train" else run_dir / args.split) / f"{args.tier}_{width}_seed{seed}"
                summary_path = cell_dir / "run_summary.json"
                status = "ok" if summary_path.exists() else f"failed(rc={process.returncode})"
                record = {"tier": args.tier, "width": width, "seed": seed, "status": status}
                if summary_path.exists():
                    summary = json.loads(summary_path.read_text())
                    record.update({
                        "updates": summary["updates"], "wall_clock": summary["wall_clock"],
                        "seen_samples": summary["seen_samples"],
                        "sample_order_hash": summary["sample_order_hash"],
                        "checkpoints": summary["checkpoints"],
                    })
                results.append(record)
                print(f"cell {args.tier} w{width} seed{seed}: {status}", flush=True)

    summary_path = run_dir / (f"tier_{args.tier}_summary.json" if args.split == "train"
                              else f"{args.split}_{args.tier}_summary.json")
    summary_path.write_text(json.dumps({
        "tier": args.tier, "cells": results, "wall_clock": time.time() - started,
        "parallel": args.parallel, "deadline_seconds": args.deadline_seconds,
    }, indent=2, sort_keys=True))
    print(json.dumps({"tier": args.tier, "cells": results}, indent=1))
    print("MATRIX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
