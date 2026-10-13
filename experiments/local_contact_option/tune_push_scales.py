"""Aggressive-first tuning of the reference scale parameters (dev split only).

    python experiments/local_contact_option/tune_push_scales.py --stage aggressive
    python experiments/local_contact_option/tune_push_scales.py --stage backoff

The diagnosis showed three conservative scale parameters, not a bad geometry: during PUSH the
robot reached only ~50% of the commanded speed with a mean force of ~1.5 N against a 15 N limit
and zero saturation (a proportional law fighting a ~20 N s/m load at Kv = 20), the reference
ramp spent 13-23% of the push below 95% of the command (a_ref_max = 0.5 m/s^2), and APPROACH
burned 1.7 s of a 6.3 s execution because its ceiling was 0.08 m/s.

This script changes only those numbers, on the **dev** roots, and never touches the test split.
It reports the pushing effect (travel per push second, achieved/commanded speed, approach time)
next to the safety signals that decide when to stop being aggressive (contact retention,
penetration, overshoot, false stable, settle coverage).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import build_roots  # noqa: E402
from coimagine.local_contact_option.runner import run_split  # noqa: E402

PRIMARY = ("solo_central", "dual_same_side")
PUSHABLE = ("solo_central", "solo_offset", "dual_same_side", "dual_rotate")

#: name -> the configuration overrides applied on top of the frozen base
STAGES: Dict[str, Dict[str, Dict[str, float]]] = {
    "aggressive": {
        "current": {},
        # every scale that can make the push slow, moved together and far
        "aggressive": {
            "kv": 120.0, "a_ref_max": 3.0,
            "approach_max_speed": 0.30, "approach_contact_speed": 0.06,
            "approach_gain": 4.0, "ready_precontact_speed": 0.02,
        },
        "kv_only": {"kv": 120.0},
        "ramp_only": {"a_ref_max": 3.0},
        "approach_only": {
            "approach_max_speed": 0.30, "approach_contact_speed": 0.06,
            "approach_gain": 4.0, "ready_precontact_speed": 0.02,
        },
    },
    "select": {
        # the pre-change configuration on the enlarged dev split, for a fair reference
        "old_baseline": {"kv": 20.0},
        # enlarged dev split: the gain is the lever, so sweep it finely with a fixed fast approach
        "kv20": {"kv": 20.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                 "approach_contact_speed": 0.06, "approach_gain": 4.0,
                 "ready_precontact_speed": 0.02},
        "kv25": {"kv": 25.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                 "approach_contact_speed": 0.06, "approach_gain": 4.0,
                 "ready_precontact_speed": 0.02},
        "kv30": {"kv": 30.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                 "approach_contact_speed": 0.06, "approach_gain": 4.0,
                 "ready_precontact_speed": 0.02},
        "kv35": {"kv": 35.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                 "approach_contact_speed": 0.06, "approach_gain": 4.0,
                 "ready_precontact_speed": 0.02},
        "kv40": {"kv": 40.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                 "approach_contact_speed": 0.06, "approach_gain": 4.0,
                 "ready_precontact_speed": 0.02},
    },
    "backoff": {
        # a_ref_max turned out to contribute ~5%, so the grid is spent on the gain (the
        # dominant lever) and on the approach ceiling (which buys wall clock, not distance)
        "kv40_fastap": {"kv": 40.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                        "approach_contact_speed": 0.06, "approach_gain": 4.0,
                        "ready_precontact_speed": 0.02},
        "kv60_fastap": {"kv": 60.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                        "approach_contact_speed": 0.06, "approach_gain": 4.0,
                        "ready_precontact_speed": 0.02},
        "kv80_fastap": {"kv": 80.0, "a_ref_max": 3.0, "approach_max_speed": 0.30,
                        "approach_contact_speed": 0.06, "approach_gain": 4.0,
                        "ready_precontact_speed": 0.02},
        "kv60_midap": {"kv": 60.0, "a_ref_max": 3.0, "approach_max_speed": 0.16,
                       "approach_contact_speed": 0.04, "approach_gain": 3.0,
                       "ready_precontact_speed": 0.015},
        "kv60_slowap": {"kv": 60.0, "a_ref_max": 3.0, "approach_max_speed": 0.10,
                        "approach_contact_speed": 0.03, "approach_gain": 2.5,
                        "ready_precontact_speed": 0.012},
        "kv80_midap": {"kv": 80.0, "a_ref_max": 3.0, "approach_max_speed": 0.16,
                       "approach_contact_speed": 0.04, "approach_gain": 3.0,
                       "ready_precontact_speed": 0.015},
    },
}


def _median(values: Sequence[float]) -> float | None:
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def summarise(records: List[dict], logs: Path) -> Dict[str, object]:
    primary = [row for row in records
               if row["interaction_class"] in PRIMARY and row["variant"] == "baseline"]
    pushable = [row for row in records
                if row["interaction_class"] in PUSHABLE and row["variant"] == "baseline"]
    settled = [row for row in records if row["metrics"]["settled_observed"]]
    false_stable = [row for row in settled if row["metrics"]["false_stable"]]
    perturbed = [row for row in primary if row["variant"] == "pos_10mm"]
    push_seconds, approach_seconds, ready_seconds = [], [], []
    for row in primary:
        path = logs / f"{row['root_id'].replace('/', '_')}__{row['variant']}.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.open()]
        rows = [item for item in rows if item.get("arm") == "N"]
        counts: Dict[str, int] = {}
        for item in rows:
            phase = item.get("phase") or item.get("stage")
            counts[phase] = counts.get(phase, 0) + 1
        push_seconds.append(counts.get("push", 0) * 0.05)
        approach_seconds.append(counts.get("approach", 0) * 0.05)
        ready_seconds.append(counts.get("ready", 0) * 0.05)
    travel = [row["metrics"]["displacement"] / max(1e-9, row["metrics"].get("active_contact_time") or 1e-9)
              for row in primary]
    return {
        "n_primary": len(primary),
        "effective_push_ratio": _ratio(pushable, lambda row: row["metrics"]["reached_effective_push"]),
        "contact_lost": sum(1 for row in records if row["metrics"]["exit_reason"] == "contact_lost"),
        "face_switch": sum(1 for row in records if row["metrics"]["exit_reason"] == "face_switch"),
        "approach_timeout": sum(1 for row in records if row["metrics"]["exit_reason"] == "approach_timeout"),
        "settled_ratio": _ratio(records, lambda row: row["metrics"]["settled_observed"]),
        "false_stable_ratio": _ratio(settled, lambda row: row["metrics"]["false_stable"]),
        "false_stable_n": len(false_stable),
        "displacement_median": _median([row["metrics"]["displacement"] for row in primary]),
        "travel_per_push_second": _median(travel),
        "final_speed_median": _median([row["metrics"]["final_speed"] for row in primary]),
        "wall_clock_median": _median([row["metrics"]["wall_clock_seconds"] for row in primary]),
        "approach_seconds": _median(approach_seconds),
        "push_seconds": _median(push_seconds),
        "ready_seconds": _median(ready_seconds),
        "max_penetration": max(row["metrics"].get("max_penetration") or 0.0 for row in records),
        "force_saturation_max": max(row["metrics"]["force_saturation_ratio"] for row in records),
        "overshoot_ratio_p90": _median([row["metrics"].get("speed_overshoot_ratio") or 0.0 for row in records]),
        "pos_10mm_p90": _paired_p90(records, "pos_10mm"),
    }


def _ratio(rows, predicate) -> float | None:
    rows = list(rows)
    return (sum(1 for row in rows if predicate(row)) / len(rows)) if rows else None


def _paired_p90(records, variant: str) -> float | None:
    base = {row["root_id"]: row["metrics"]["relative_object"] for row in records
            if row["variant"] == "baseline" and row["arm"] == "N"}
    values = []
    for row in records:
        if row["arm"] != "N" or row["variant"] != variant:
            continue
        origin = base.get(row["root_id"])
        if origin is None:
            continue
        now = row["metrics"]["relative_object"]
        values.append(math.hypot(now[0] - origin[0], now[1] - origin[1]))
    if not values:
        return None
    values.sort()
    return values[min(len(values) - 1, int(0.9 * len(values)))]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="aggressive", choices=sorted(STAGES))
    parser.add_argument("--base-kv", type=float, default=20.0,
                        help="the frozen v1 selection; overridden by every setting")
    parser.add_argument("--variants", default="baseline,pos_10mm")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--out-dir", default="outputs/local_contact_option_tuning")
    parser.add_argument("--only", default=None, help="run a single named setting")
    parser.add_argument("--per-class", type=int, default=4,
                        help="dev roots per class; the first sweep used 4 (8 primary roots) and "
                             "was too small: kv=40 looked clean there but penetrated 5x more on "
                             "the test split")
    args = parser.parse_args()

    roots = build_roots("dev", args.per_class)
    out_root = ROOT / args.out_dir
    results = {}
    settings = STAGES[args.stage]
    if args.only:
        settings = {args.only: settings[args.only]}
    for name, overrides in settings.items():
        config = OptionConfig(kv=args.base_kv)
        config = replace(config, **overrides)
        run_dir = out_root / f"{args.stage}__{name}"
        payload = run_split(
            roots, out_dir=run_dir, option_config=config,
            variants=args.variants.split(","), arms=["N"], workers=args.workers,
            tag="dev",
        )
        records = [json.loads(line) for line in Path(payload["executions_path"]).open()]
        summary = summarise(records, run_dir / "logs" / "dev")
        results[name] = {"config": {"kv": config.kv, "a_ref_max": config.a_ref_max,
                                    "approach_max_speed": config.approach_max_speed,
                                    "approach_contact_speed": config.approach_contact_speed,
                                    "approach_gain": config.approach_gain,
                                    "ready_precontact_speed": config.ready_precontact_speed},
                         **summary}
        print(f"{name:14s} " + json.dumps({key: (round(value, 4) if isinstance(value, float) else value)
                                           for key, value in summary.items()}), flush=True)
    (out_root / f"{args.stage}_summary.json").write_text(json.dumps(results, indent=1, sort_keys=True))

    print("\n| setting | travel/push s | displacement med | eff push | settled | false stable | approach s | push s | wall s | max pen | sat | 10mm P90 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, row in results.items():
        print("| {name} | {t} | {d} | {e} | {s} | {f} | {a} | {p} | {w} | {pen} | {sat} | {p90} |".format(
            name=name,
            t=_f(row["travel_per_push_second"]), d=_f(row["displacement_median"]),
            e=_f(row["effective_push_ratio"]), s=_f(row["settled_ratio"]),
            f=f'{_f(row["false_stable_ratio"])} ({row["false_stable_n"]})',
            a=_f(row["approach_seconds"]), p=_f(row["push_seconds"]),
            w=_f(row["wall_clock_median"]), pen=_f(row["max_penetration"], 5),
            sat=_f(row["force_saturation_max"], 4), p90=_f(row["pos_10mm_p90"], 4)))
    print("TUNING_OK")
    return 0


def _f(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
