"""Interrupt-and-resume control: a resumed run must be bit-identical to an uninterrupted one.

Three runs of the same configuration and seed are compared:

    A: 4 updates, uninterrupted
    B: 2 updates, then a *fresh process* that resumes from the checkpoint and runs to 4

The final model parameters must match exactly. The checkpoints are produced by the normal
training path (`--save-every`), not by a special branch-only saver.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "experiments/random_start_reposition_v1/rl.py"


def run(arguments, out: Path) -> None:
    command = [sys.executable, str(SCRIPT), "--out", str(out), "--scan", "8",
               "--dev-instances", "4", "--episodes-per-update", "2", "--eval-every", "100"] + arguments
    subprocess.run(command, check=True, cwd=ROOT, stdout=subprocess.DEVNULL,
                   stderr=subprocess.STDOUT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/random_start_reposition_v1"))
    args = parser.parse_args()
    out = Path(args.out)
    work = out / "results" / "resume_check"
    work.mkdir(parents=True, exist_ok=True)

    uninterrupted = work / "uninterrupted"
    split = work / "split"
    run(["--updates", "4", "--prefix", "a", "--save-every", "2"], uninterrupted)
    run(["--updates", "2", "--prefix", "a", "--save-every", "2"], split)
    run(["--updates", "4", "--prefix", "a", "--save-every", "2", "--resume"], split)

    first = torch.load(uninterrupted / "checkpoints" / "a_final.pt", map_location="cpu")
    second = torch.load(split / "checkpoints" / "a_final.pt", map_location="cpu")
    identical = all(torch.equal(first[key], second[key]) for key in first)
    report = {"ok": bool(identical), "tensors_compared": len(first),
              "note": "the resumed process loads model, optimizer, torch and numpy RNG state and "
                      "the real update index; the update sequence is seeded by the update index, "
                      "so the two runs must agree exactly"}
    (work / "resume_check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print("RESUME_OK" if identical else "RESUME_FAILED")
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
