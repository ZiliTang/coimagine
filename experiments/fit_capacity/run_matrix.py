"""Phase B: the 12-cell training matrix (2 data groups x 2 widths x 3 seeds).

    python experiments/fit_capacity/run_matrix.py --run-id <id> [--widths 256,512] [--seeds 0,1,2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

torch.set_num_threads(4)

from coimagine.data_upgrade.training import GROUP_BRANCH, GROUP_RANDOM
from coimagine.fit_capacity.trainer import TOTAL_UPDATES, run

ROOT = Path(__file__).resolve().parents[2]
WALL_BUDGET_SECONDS = 60 * 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="fit-capacity-20261012")
    parser.add_argument("--out-root", default="outputs/fit_capacity_v1")
    parser.add_argument("--upgrade-run", default="outputs/data_upgrade_v1/data-upgrade-20261012")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    parser.add_argument("--widths", default="256,512")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--total-updates", type=int, default=TOTAL_UPDATES)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    upgrade = ROOT / args.upgrade_run
    source = ROOT / args.source_run
    original = [json.loads(line) for line in (source / "dataset" / "windows.jsonl").open()]
    original_train = [r for r in original if r["split"] == "train"]
    original_val = [r for r in original if r["split"] == "val"]
    branch_all = [json.loads(line) for line in (upgrade / "branch_data" / "branch_windows.jsonl").open()]
    branch_val = [r for r in branch_all if r["split"] == "val"]
    matched = {
        GROUP_RANDOM: [json.loads(line) for line in (upgrade / "branch_data" / "random_train_matched.jsonl").open()],
        GROUP_BRANCH: [json.loads(line) for line in (upgrade / "branch_data" / "branch_train_matched.jsonl").open()],
    }
    datasets = {group: original_train + extra for group, extra in matched.items()}
    for group, records in datasets.items():
        print(f"{group}: {len(records)} training windows", flush=True)

    widths = [int(v) for v in args.widths.split(",") if v]
    seeds = [int(v) for v in args.seeds.split(",") if v]
    cells = [(group, width, seed) for group in (GROUP_RANDOM, GROUP_BRANCH) for width in widths for seed in seeds]
    summary = {"run_id": args.run_id, "total_updates": args.total_updates, "cells": [], "wall_budget_seconds": WALL_BUDGET_SECONDS}
    started = time.time()
    first = True
    for group, width, seed in cells:
        cell_dir = run_dir / "runs" / f"{group.replace('+', '_')}_{width}_seed{seed}"
        if args.skip_existing and (cell_dir / "run_summary.json").exists():
            record = json.loads((cell_dir / "run_summary.json").read_text())
            summary["cells"].append(record)
            print(f"skip {group} {width} seed{seed} (already complete)", flush=True)
            continue
        result = run(
            group, width, seed, datasets[group], original_val, branch_val, cell_dir,
            total_updates=args.total_updates,
        )
        summary["cells"].append(result.as_dict())
        elapsed = time.time() - started
        print(
            f"{group} width{width} seed{seed}: updates={result.updates} seen={result.seen_samples} "
            f"wall={result.wall_clock:.1f}s elapsed={elapsed:.1f}s reason={result.stopped_reason}",
            flush=True,
        )
        if first:
            first = False
            projected = result.wall_clock / max(1, min(500, result.updates)) * args.total_updates * len(cells)
            summary["projection"] = {
                "first_run_wall_clock": result.wall_clock,
                "projected_total_seconds": projected,
                "within_budget": projected < WALL_BUDGET_SECONDS,
            }
            print(f"projection: one run {result.wall_clock:.1f}s -> all {len(cells)} runs ~{projected/60:.1f} min "
                  f"(budget {WALL_BUDGET_SECONDS/60:.0f} min)", flush=True)
        if elapsed > WALL_BUDGET_SECONDS:
            summary["stopped_early"] = True
            summary["stopped_reason"] = "wall-clock budget reached; resumable state saved per run"
            print("wall-clock budget reached; stopping with resumable state", flush=True)
            break
    summary["wall_clock"] = time.time() - started
    (run_dir / "training_matrix.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"cells recorded: {len(summary['cells'])}; total wall {summary['wall_clock']:.1f}s")
    print("MATRIX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
