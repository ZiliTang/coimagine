"""The 256k continuation tier: 256/512 x seeds 0/1/2, from scratch, uncapped budget.

    python experiments/coverage_scale/run_matrix_256k.py --parallel 2

Data is *appended* to the frozen coverage-scale run (same generator, recipe, split and root
plan), and only the training budget / evaluation interval change, so the 128k tier and every
frozen validation, test and candidate artifact stay untouched.
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

DATA_RUN = "outputs/coverage_scale_v1/coverage-scale-20261013"
OUT_RUN = "outputs/coverage_scale_v1/coverage-scale-256k"
WIDTHS = (256, 512)
SEEDS = (0, 1, 2)


def load(run_dir: Path, out_dir: Path, target: int):
    from coimagine.coverage_scale.plan import load_registries
    from coimagine.coverage_scale.shards import tier_records

    registry = load_registries(run_dir / "roots", ("train", "val", "test"))
    registry.activate()
    train, info = tier_records(run_dir, "train", target)
    val, val_info = tier_records(run_dir, "val", 10**9)
    return train, info, val, val_info


def run_single(out_dir: Path, data_dir: Path, target: int, width: int, seed: int,
               *, wait_seconds: float = 3600.0) -> int:
    import torch

    torch.set_num_threads(4)
    from coimagine.coverage_scale.shards import select_monitor
    from coimagine.coverage_scale.trainer_ext import run_cell_ext, update_budget_uncapped

    deadline = time.time() + wait_seconds
    while True:
        train, info, val, val_info = load(data_dir, out_dir, target)
        if info["reached_target"]:
            break
        if time.time() > deadline:
            raise SystemExit(f"tier {target} still incomplete after waiting: {info}")
        print(f"waiting for {target}: {info['actual_windows']} windows", flush=True)
        time.sleep(30)
    monitor = select_monitor(train, 4096)
    cell_dir = out_dir / "runs" / f"{target}_{width}_seed{seed}"
    result = run_cell_ext(target, width, seed, train, monitor, val, cell_dir,
                          max_updates=update_budget_uncapped(len(train)))
    meta = {
        "tier": target, "width": width, "seed": seed, "train": info, "val": val_info,
        "monitor_windows": len(monitor), "max_updates": result.max_updates,
        "updates": result.updates, "epochs_seen": result.epochs_seen,
        "wall_clock": result.wall_clock, "sample_order_hash": result.sample_order_hash,
        "initial_parameter_hash": result.initial_parameter_hash,
        "stopped_reason": result.stopped_reason,
    }
    (cell_dir / "cell_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(json.dumps(meta), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-windows", type=int, default=256000)
    parser.add_argument("--parallel", type=int, default=2)
    parser.add_argument("--widths", default=",".join(str(w) for w in WIDTHS))
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    parser.add_argument("--single", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=0, help=argparse.SUPPRESS)
    args = parser.parse_args()

    out_dir = ROOT / OUT_RUN
    data_dir = ROOT / DATA_RUN
    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    if args.single:
        return run_single(out_dir, data_dir, args.target_windows, args.width, args.seed)

    widths = [int(value) for value in args.widths.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    cells = [(width, seed) for width in widths for seed in seeds]
    running: list = []
    index = 0
    started = time.time()
    results = []
    while index < len(cells) or running:
        while len(running) < args.parallel and index < len(cells):
            width, seed = cells[index]
            index += 1
            cell_dir = out_dir / "runs" / f"{args.target_windows}_{width}_seed{seed}"
            if (cell_dir / "run_summary.json").exists():
                results.append({"width": width, "seed": seed, "status": "skipped"})
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()),
                "--target-windows", str(args.target_windows), "--single",
                "--width", str(width), "--seed", str(seed),
            ]
            cell_dir.mkdir(parents=True, exist_ok=True)
            log = (cell_dir / "stdout.log").open("w")
            running.append((subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT), width, seed))
        if not running:
            continue
        time.sleep(10)
        for entry in list(running):
            process, width, seed = entry
            if process.poll() is not None:
                running.remove(entry)
                cell_dir = out_dir / "runs" / f"{args.target_windows}_{width}_seed{seed}"
                summary = cell_dir / "run_summary.json"
                status = "ok" if summary.exists() else f"failed(rc={process.returncode})"
                record = {"width": width, "seed": seed, "status": status}
                if summary.exists():
                    payload = json.loads(summary.read_text())
                    record.update({"updates": payload["updates"], "wall_clock": payload["wall_clock"],
                                   "epochs_seen": payload["epochs_seen"]})
                results.append(record)
                print(f"cell 256k w{width} seed{seed}: {status}", flush=True)
    (out_dir / "tier_256000_summary.json").write_text(json.dumps({
        "target_windows": args.target_windows, "cells": results,
        "wall_clock": time.time() - started, "parallel": args.parallel,
    }, indent=2, sort_keys=True))
    print(json.dumps(results, indent=1))
    print("MATRIX_256K_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
