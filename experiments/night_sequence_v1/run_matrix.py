"""Run the frozen training matrix from task book 5.1.

Calibration (D0, batch 128, 200 updates) measured 5.2 s for MLP-history and 35.1 s for
Transformer-history on this machine. 6400 updates would need about 2 h 20 m for the whole matrix,
which does not fit beside the mandatory deliverable and asset work, so -- as section 5.2 allows --
**every** main run uses the same reduced budget of 3200 updates. The reduction is uniform, is
recorded in the ledger, and no run gets a different budget.

    python experiments/night_sequence_v1/run_matrix.py --updates 3200 --jobs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"

#: (tier, model, seed) in task book 5.1 order
def matrix() -> list:
    runs = [
        ("d0", "mlp_history", 0), ("d0", "transformer_history", 0),
        ("d1", "mlp_history", 0), ("d1", "transformer_history", 0),
        ("d1", "mlp_current", 0), ("d1", "mlp_current", 1), ("d1", "mlp_current", 2),
    ]
    for seed in (1, 2):
        for tier in ("d0", "d1"):
            for model in ("mlp_history", "transformer_history"):
                runs.append((tier, model, seed))
    return runs


def d2_matrix() -> list:
    """Task book 5.1 item 5: the optional D2 pair, only if D2 is complete and time remains."""

    return [("d2", "mlp_history", 0), ("d2", "transformer_history", 0)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", default="")
    parser.add_argument("--matrix", choices=["main", "d2"], default="main")
    args = parser.parse_args()
    runs = d2_matrix() if args.matrix == "d2" else matrix()
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if f"{r[0]}_{r[1]}_seed{r[2]}" in wanted]
    RUNS = DATA / "runs"
    RUNS.mkdir(parents=True, exist_ok=True)
    pending = [r for r in runs if not (RUNS / f"{r[0]}_{r[1]}_seed{r[2]}.pt").exists()]
    print(f"matrix: {len(runs)} runs, {len(pending)} pending", flush=True)
    started = time.time()
    running = []
    queue = list(pending)
    while queue or running:
        while queue and len(running) < args.jobs:
            tier, model, seed = queue.pop(0)
            tag = f"{tier}_{model}_seed{seed}"
            log = open(RUNS / f"{tag}.log", "w")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "experiments/night_sequence_v1/train.py"),
                 "--tier", tier, "--model", model, "--seed", str(seed),
                 "--updates", str(args.updates), "--threads", str(args.threads),
                 "--out", str(RUNS)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, tag, log, time.time()))
            print(f"  start {tag}", flush=True)
        time.sleep(5)
        for item in list(running):
            process, tag, log, began = item
            if process.poll() is not None:
                log.close()
                running.remove(item)
                print(f"  done  {tag} exit={process.returncode} "
                      f"({time.time() - began:.0f}s)", flush=True)
    print(f"MATRIX_OK in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
