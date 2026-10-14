"""Phase 3 matrix: nine residual runs and three corrected-direct runs, all on D1.

The three direct runs are not a re-use of an old number: they are trained under this round's protocol
with the round-7 engineering fixes in place, so the residual comparison is between two
parameterisations of the same pipeline. Twelve runs, 3200 updates each, three architectures, three
seeds; the model size, the batch size and the loss weights are unchanged from round 6.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/residual_prediction_v1"


def matrix() -> list:
    runs = []
    for seed in (0, 1, 2):
        for kind in ("mlp_current", "mlp_history", "transformer_history"):
            runs.append((kind, "residual", "per_group", seed))
    # corrected direct control, one architecture, all seeds
    for seed in (0, 1, 2):
        runs.append(("mlp_history", "direct", "per_group", seed))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    runs = matrix()
    if args.only:
        wanted = set(args.only.split(","))
        runs = [r for r in runs if f"{r[0]}_{r[1]}_{r[2]}_seed{r[3]}" in wanted]
    checkpoints = OUT / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    pending = [r for r in runs
               if not (checkpoints / f"{r[0]}_{r[1]}_{r[2]}_seed{r[3]}.pt").exists()]
    print(f"matrix: {len(runs)} runs, {len(pending)} pending", flush=True)
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    started = time.time()
    running = []
    queue = list(pending)
    while queue or running:
        while queue and len(running) < args.jobs:
            kind, parameterisation, normalisation, seed = queue.pop(0)
            tag = f"{kind}_{parameterisation}_{normalisation}_seed{seed}"
            log = open(logs / f"{tag}.log", "w")
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "experiments/residual_prediction_v1/multitask.py"),
                 "--tier", "d1", "--kind", kind, "--seed", str(seed),
                 "--parameterisation", parameterisation, "--normalisation", normalisation,
                 "--updates", str(args.updates), "--threads", str(args.threads),
                 "--out", str(OUT)],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            running.append((process, tag, log, time.time()))
            print(f"  start {tag}", flush=True)
        time.sleep(5)
        for item in list(running):
            process, tag, log, began = item
            if process.poll() is not None:
                log.close()
                running.remove(item)
                print(f"  done  {tag} exit={process.returncode} ({time.time() - began:.0f}s)",
                      flush=True)
    print(f"MATRIX_OK in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
