"""Diagnose where the new local control loses pushing speed and where it stalls.

    python experiments/local_contact_option/diagnose_pushing.py

Reads only the per-control-period logs of an existing run, so it never re-simulates and can
never show a different trajectory from the measured one. For every arm-N execution it splits the
PUSH phase into three questions:

1. **reference ramp** - how long does the rate-limited reference take to reach the commanded
   speed, and what fraction of the effective push is spent below it;
2. **tracking** - how much of the commanded speed the robot actually achieves, and whether the
   force is limited (a limited force means the controller is asking for more than it can give)
   or simply too small (a proportional law with a low gain stalls against a load);
3. **geometry** - how far the sensed surface is at contact, and whether periods with contact but
   no motion ("stuck") cluster at a particular stand-off distance.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

PRIMARY = ("solo_central", "dual_same_side")
STALL_SPEED = 0.02        # m/s below which an actively pushing robot counts as stalled
MOVE_EPS = 1e-5           # m per control period below which the object has not moved


def _norm(vector) -> float:
    return math.hypot(vector[0], vector[1])


def analyse_root(path: Path, *, root_id: str, variant: str,
                 commanded_speed: Optional[float] = None) -> Optional[dict]:
    # one log file holds every arm of the same (root, variant); analyse arm N only
    rows = [json.loads(line) for line in path.open()]
    rows = [row for row in rows if row.get("arm") == "N"]
    if not rows:
        return None
    periods = []
    for row in rows:
        observed = row.get("observed") or {}
        entry = observed.get("0")
        if entry is None:
            continue
        periods.append((row, entry))
    if not periods:
        return None
    commanded = commanded_speed
    push = [(row, entry) for row, entry in periods
            if (row.get("phase") or row.get("stage")) == "push"]
    if not push:
        return None

    now = [row["time"] for row, _ in push]
    speed = [_norm(row["true"]["robots"][0][3:5]) for row, _ in push]
    reference = [entry.get("reference") or (0.0, 0.0) for _, entry in push]
    reference_norm = [_norm(value) for value in reference]
    force = [_norm(row["force"][0]) for row, _ in push]
    limit = 15.0
    contact = [bool(entry.get("contact")) for _, entry in push]
    gap = []
    for row, entry in push:
        distance = entry.get("patch_distance")
        gap.append(None if distance is None else max(0.0, distance - 0.18))
    normal_speed = []
    for row, entry in push:
        normal = entry.get("patch_normal")
        velocity = row["true"]["robots"][0][3:5]
        if normal is None:
            normal_speed.append(None)
        else:
            normal_speed.append(velocity[0] * normal[0] + velocity[1] * normal[1])
    object_positions = [tuple(row["true"]["object"][:2]) for row, _ in push]
    moved = [0.0] + [
        math.dist(object_positions[index], object_positions[index - 1])
        for index in range(1, len(object_positions))
    ]

    target = commanded if commanded else max(reference_norm)
    ramp_periods = sum(1 for value in reference_norm if value < 0.95 * target)
    stalled = sum(1 for value, is_contact in zip(speed, contact)
                  if is_contact and value < STALL_SPEED)
    stuck = sum(1 for value, delta, is_contact in zip(speed, moved, contact)
                if is_contact and delta < MOVE_EPS and value < STALL_SPEED)
    saturated = sum(1 for value in force if value >= limit - 1e-6)
    forward_gap = [value for value in gap if value is not None]
    return {
        "root_id": root_id, "variant": variant,
        "commanded_speed": target,
        "push_periods": len(push),
        "push_seconds": len(push) * 0.05 - (len(push) - 1) * 0.0 if push else 0.0,
        "reference_p90": statistics.quantiles(reference_norm, n=10)[8] if len(reference_norm) > 9 else max(reference_norm),
        "actual_mean": statistics.fmean(speed),
        "actual_p50": statistics.median(speed),
        "ratio_actual_to_commanded": statistics.fmean(speed) / target if target else None,
        "force_mean": statistics.fmean(force),
        "force_p90": statistics.quantiles(force, n=10)[8] if len(force) > 9 else max(force),
        "saturated_periods": saturated,
        "stalled_periods": stalled,
        "stuck_periods": stuck,
        "contact_loss_periods": sum(1 for value in contact if not value),
        "ramp_periods_below_95pct": ramp_periods,
        "gap_p50": statistics.median(forward_gap) if forward_gap else None,
        "gap_min": min(forward_gap) if forward_gap else None,
        "normal_speed_mean": statistics.fmean([v for v in normal_speed if v is not None])
        if any(v is not None for v in normal_speed) else None,
        "object_travel": sum(moved),
        "object_travel_per_push_second": sum(moved) / max(1e-9, len(push) * 0.05),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-repair-v2")
    parser.add_argument("--out-root", default="outputs/local_contact_option_repair_v2")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    roots_meta = {row["root_id"]: row
                  for row in json.loads((run_dir / f"roots_{args.split}.json").read_text())["roots"]}
    records = defaultdict(list)
    for path in sorted((run_dir / "logs" / args.split).glob("*__baseline.jsonl")):
        key = path.name.split("__")[0]
        match = [name for name in roots_meta if name.replace("/", "_") == key]
        if not match:
            continue
        meta = roots_meta[match[0]]
        entry = analyse_root(path, root_id=match[0], variant="baseline",
                             commanded_speed=float(meta["speed"][0]))
        if entry is None:
            continue
        entry["interaction_class"] = meta["interaction_class"]
        records[entry["interaction_class"]].append(entry)

    summary = {}
    for interaction_class, rows in sorted(records.items()):
        summary[interaction_class] = {
            "n": len(rows),
            "commanded_speed": statistics.median([r["commanded_speed"] for r in rows]),
            "ratio_actual_to_commanded": statistics.median(
                [r["ratio_actual_to_commanded"] for r in rows if r["ratio_actual_to_commanded"]]),
            "force_mean": statistics.median([r["force_mean"] for r in rows]),
            "force_p90": statistics.median([r["force_p90"] for r in rows]),
            "saturated_period_share": statistics.median(
                [r["saturated_periods"] / max(1, r["push_periods"]) for r in rows]),
            "stalled_period_share": statistics.median(
                [r["stalled_periods"] / max(1, r["push_periods"]) for r in rows]),
            "stuck_period_share": statistics.median(
                [r["stuck_periods"] / max(1, r["push_periods"]) for r in rows]),
            "ramp_share_below_95pct": statistics.median(
                [r["ramp_periods_below_95pct"] / max(1, r["push_periods"]) for r in rows]),
            "object_travel_per_push_second": statistics.median(
                [r["object_travel_per_push_second"] for r in rows]),
            "gap_p50_at_push": statistics.median(
                [r["gap_p50"] for r in rows if r["gap_p50"] is not None] or [0.0]),
            "contact_loss_periods": statistics.median([r["contact_loss_periods"] for r in rows]),
        }
    payload = {"split": args.split, "run_id": args.run_id, "summary": summary,
               "per_root": {key: value for key, value in records.items()}}
    out = Path(args.out) if args.out else run_dir / "push_diagnosis.json"
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1, sort_keys=True))
    print("DIAGNOSE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
