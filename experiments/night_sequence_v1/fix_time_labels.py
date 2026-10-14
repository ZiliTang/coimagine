"""Correct the stored time labels without re-simulating.

The executor's exit instant is ``physics_steps * dt``. The stage machine can break **before** the
force call, so the exiting period does not advance the world and ``physics_steps == steps - 1``;
the original writer used ``steps * dt`` (the number of *sensed* periods, which includes O0) and so
reported a duration one control period too long.

Re-running 137k executions is unnecessary because the claim is verifiable on the stored data: the
separately sensed terminal observation must be bit-identical to the last stored period's own state
if and only if the exiting period did not step. This script checks that invariant for every record
and only then rewrites the scalar fields.

    python experiments/night_sequence_v1/fix_time_labels.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    args = parser.parse_args()
    paths = sorted((DATA / "sequences").glob("*.npz"))
    total = 0
    offenders = []
    for path in paths:
        block = np.load(path, allow_pickle=True)
        for row in range(len(block["steps"])):
            steps = int(block["steps"][row])
            total += 1
            difference = float(np.abs(block["own_world"][row, steps - 1]
                                      - block["terminal_own_world"][row]).max())
            if difference > args.tolerance:
                offenders.append({"path": path.name, "row": row, "difference": difference})
    if offenders:
        print(json.dumps({"invariant_violations": len(offenders),
                          "first": offenders[:5]}, indent=1))
        print("REFUSING to patch: the exiting period may have stepped for these records")
        return 1
    print(f"invariant holds for all {total} sequences in {len(paths)} shards")

    if not args.apply:
        print("dry run; pass --apply to rewrite the stored scalars")
        return 0

    rewritten = 0
    for path in paths:
        block = dict(np.load(path, allow_pickle=True))
        steps = block["steps"].astype(np.float64)
        corrected = block["T_last_sensed"].astype(np.float32)
        block["T_exit"] = corrected
        block["duration_seconds"] = corrected
        block["truth_steps"] = (steps - 1).astype(np.int32)
        block["physics_steps_match_rows"] = np.ones(len(steps), dtype=np.int8)
        temporary = path.with_suffix(".npz.tmp")
        with open(temporary, "wb") as handle:
            np.savez_compressed(handle, **block)
        temporary.replace(path)
        rewritten += 1
        if rewritten % 500 == 0:
            print(f"  rewrote {rewritten}/{len(paths)}", flush=True)
    print(f"rewrote {rewritten} shards")
    print("FIX_TIME_LABELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
