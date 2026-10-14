"""The round's run matrix, ordered exactly as the task book section 8 prescribes.

Stage A  the ~0.9M and ~1.6M direct-prediction Transformers under the inherited protocol.
Stage B  the ~1.6M parameter-matched pair (direct multi-step vs latent autoregressive), seed 0.
Stage C  seeds 1 and 2 of that pair.
Stage D  the ~4.0M pair, only if the earlier stages finish with budget left.

Every run is a separate process so a single crash cannot take the matrix down, and a finished
checkpoint is skipped so the matrix is restartable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/transformer_scale_latent_v1"
RESIDUAL = ROOT / "outputs/residual_prediction_v1"


def direct_runs() -> list:
    """(tag, kwargs) for the capacity curve of the direct-prediction Transformer."""

    # sizes chosen by counting the round-6 architecture, not by guessing: the closest
    # configurations to 0.9M / 1.6M / 4.0M are 895,937 / 1,635,569 / 4,149,569 parameters.
    # (176,5,704)=2,009,569 and (256,6,1024)=4,939,329 were tried first and are kept as extra
    # points under their own tags rather than passed off as the requested sizes.
    return [
        ("capacity_0p9M_seed0", {"d_model": 128, "heads": 4, "layers": 4, "ffn": 512}),
        ("capacity_1p6M_seed0", {"d_model": 176, "heads": 4, "layers": 4, "ffn": 704}),
        ("capacity_2p0M_seed0", {"d_model": 176, "heads": 4, "layers": 5, "ffn": 704}),
        ("capacity_4p0M_seed0", {"d_model": 256, "heads": 4, "layers": 5, "ffn": 1024}),
    ]


def pair_runs(seeds, decoder, latent_decoder) -> list:
    size = f"160,4,4,640,{decoder}"
    runs = []
    for seed in seeds:
        runs.append(("direct_multistep", size, latent_decoder, seed))
    for seed in seeds:
        runs.append(("latent_autoregressive", size, latent_decoder, seed))
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="a")
    parser.add_argument("--updates", type=int, default=3200)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--skip", default="", help="comma-separated tags to leave out")
    parser.add_argument("--decoder", type=int, default=320)
    parser.add_argument("--latent-decoder", type=int, default=192)
    args = parser.parse_args()
    logs = OUT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (OUT / "checkpoints").mkdir(parents=True, exist_ok=True)
    commands = []
    skipped = {name for name in args.skip.split(",") if name}
    if args.stage in ("a", "all"):
        for tag, kwargs in direct_runs():
            if tag in skipped or (OUT / "checkpoints" / f"{tag}.pt").exists():
                continue
            commands.append((tag, [sys.executable,
                                   str(ROOT / "experiments/transformer_scale_latent_v1/capacity.py"),
                                   "--tag", tag, "--updates", str(args.updates),
                                   "--threads", str(args.threads),
                                   "--kwargs", json.dumps(kwargs)]))
    if args.stage in ("b", "c", "all"):
        seeds = [0] if args.stage == "b" else args.seeds
        for kind, size, latent_decoder, seed in pair_runs(seeds, args.decoder,
                                                          args.latent_decoder):
            tag = f"{kind}_h8_lr0.0003_seed{seed}"
            if tag in skipped or (OUT / "checkpoints" / f"{tag}.pt").exists():
                continue
            commands.append((tag, [sys.executable,
                                   str(ROOT / "experiments/transformer_scale_latent_v1/train.py"),
                                   "--kind", kind, "--size", size, "--seed", str(seed),
                                   "--horizon", "8", "--updates", str(args.updates),
                                   "--lr", "3e-4", "--threads", str(args.threads),
                                   "--latent-decoder", str(latent_decoder), "--tag", tag]))
    print(f"matrix: {len(commands)} pending", flush=True)
    running = []
    queue = list(commands)
    started = time.time()
    while queue or running:
        while queue and len(running) < args.jobs:
            tag, command = queue.pop(0)
            handle = open(logs / f"{tag}.log", "w")
            process = subprocess.Popen(command, cwd=ROOT, stdout=handle,
                                       stderr=subprocess.STDOUT)
            running.append((process, tag, handle, time.time()))
            print(f"  start {tag}", flush=True)
        time.sleep(5)
        for item in list(running):
            process, tag, handle, began = item
            if process.poll() is not None:
                handle.close()
                running.remove(item)
                print(f"  done  {tag} exit={process.returncode} ({time.time() - began:.0f}s)",
                      flush=True)
    print(f"MATRIX_OK in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
