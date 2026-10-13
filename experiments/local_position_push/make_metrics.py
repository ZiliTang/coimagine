"""C: the six metric groups the task book requires for the V/X comparison.

    python experiments/local_position_push/make_metrics.py

Groups:
 1. push completion, contact ratios, abort rates, approach and PUSH duration;
 2. displacement along the commanded direction, rotation, PUSH-only motion, real linear/angular
    speed, with "no motion" separated from "normal advance";
 3. force request vs applied (peak, P95), saturation share, impulse, penetration, normal and
    tangential force components, gap_signed, reference-position error;
 4. truth terminal residual speed, observation-settle coverage, false-stable rate, timeouts;
 5. same-root perturbation effect on the terminal translation and angle, by perturbation kind;
 6. robot contact-state combinations and tangential slip along the surface (evaluator side).

Plus the paired root bootstrap (position and angle separately), the "both moved" subset with the
2 cm / 3 deg thresholds stated up front, and the motion bins as an auxiliary view.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

ARMS = ("V", "X")
PRIMARY = ("solo_central", "dual_same_side")
PUSHABLE = ("solo_central", "solo_offset", "dual_same_side", "dual_rotate")
#: stated before looking at the results, and never adjusted afterwards
MOTION_LINEAR = 0.02   # m
MOTION_ANGLE = math.radians(3.0)
BOOTSTRAP_SAMPLES = 2000
BOOTSTRAP_SEED = 20261013


def _median(values):
    values = [value for value in values if value is not None]
    return statistics.median(values) if values else None


def _mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def _quantile(values, q):
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    position = q * (len(values) - 1)
    low = int(math.floor(position))
    high = min(len(values) - 1, low + 1)
    weight = position - low
    return values[low] * (1 - weight) + values[high] * weight


def _ratio(rows, predicate):
    rows = list(rows)
    return (sum(1 for row in rows if predicate(row)) / len(rows)) if rows else None


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def load(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.open()]


def _log_rows(run_dir: Path, split: str, row: dict, arm: str) -> List[dict]:
    path = run_dir / "logs" / split / f"{row['root_id'].replace('/', '_')}__{row['variant']}.jsonl"
    if not path.exists():
        return []
    return [item for item in (json.loads(line) for line in path.open())
            if item.get("arm") == arm]


def force_statistics(run_dir: Path, split: str, records: Sequence[dict]) -> Dict[str, dict]:
    """Group 3, straight from the per-control-period logs of the PUSH phase."""

    out: Dict[str, dict] = {}
    for arm in ARMS:
        applied, requested, saturation, impulse, penetration = [], [], [], [], []
        normal_force, tangential_force, gaps, errors = [], [], [], []
        for row in records:
            if row["arm"] != arm or row["variant"] != "baseline":
                continue
            periods = _log_rows(run_dir, split, row, arm)
            push = [item for item in periods if (item.get("phase") or item.get("stage")) == "push"]
            if not push:
                continue
            for item in push:
                entry = (item.get("observed") or {}).get("0") or {}
                force = item["force"][0]
                magnitude = math.hypot(*force)
                applied.append(magnitude)
                raw = entry.get("force_raw")
                if raw is not None:
                    requested.append(math.hypot(*raw))
                    saturation.append(1.0 if math.hypot(*raw) > 15.0 + 1e-9 else 0.0)
                normal = entry.get("patch_normal")
                if normal:
                    normal_force.append(force[0] * normal[0] + force[1] * normal[1])
                    tangential_force.append(force[0] * (-normal[1]) + force[1] * normal[0])
                if entry.get("gap_signed") is not None:
                    gaps.append(entry["gap_signed"])
                if entry.get("error"):
                    errors.append(math.hypot(*entry["error"]))
                impulse.append(magnitude * 0.05)
            penetration.append(row["metrics"].get("max_penetration") or 0.0)
        out[arm] = {
            "applied_peak": max(applied) if applied else None,
            "applied_p95": _quantile(applied, 0.95),
            "applied_median": _median(applied),
            "requested_peak": max(requested) if requested else None,
            "requested_p95": _quantile(requested, 0.95),
            "saturation_share": _mean(saturation),
            "normal_force_p95": _quantile([abs(v) for v in normal_force], 0.95),
            "tangential_force_p95": _quantile([abs(v) for v in tangential_force], 0.95),
            "gap_p05": _quantile(gaps, 0.05),
            "gap_median": _median(gaps),
            "gap_p95": _quantile(gaps, 0.95),
            "error_p95": _quantile(errors, 0.95),
            "impulse_total_median": _median(impulse),
            "max_penetration": max(penetration) if penetration else None,
        }
    return out


def slip_statistics(run_dir: Path, split: str, records: Sequence[dict]) -> Dict[str, dict]:
    """Group 6: tangential slip of the contact along the surface, evaluated on the truth side.

    The matrix maps the observed contact coordinate of the *initial* period onto the current one
    through the object's own pose (an evaluator-only computation), so the number is the material
    slip along the face. It is never fed to a controller.
    """

    out: Dict[str, dict] = {}
    for arm in ARMS:
        slips, combinations = [], Counter()
        for row in records:
            if row["arm"] != arm or row["variant"] != "baseline":
                continue
            periods = _log_rows(run_dir, split, row, arm)
            push = [item for item in periods if (item.get("phase") or item.get("stage")) == "push"]
            if len(push) < 2:
                continue
            for item in push:
                observed = item.get("observed") or {}
                key = "".join("1" if (observed.get(side) or {}).get("contact") else "0"
                              for side in ("0", "1"))
                combinations[key] += 1
            first, last = push[0]["true"]["object"], push[-1]["true"]["object"]
            start_contact = (push[0].get("observed") or {}).get("0", {}).get("contact_point_robot")
            if start_contact is None:
                continue
            # material coordinate of the initial contact, followed through the object pose
            cos, sin = math.cos(first[2]), math.sin(first[2])
            world_x = first[0] + start_contact[0] * cos - start_contact[1] * sin
            world_y = first[1] + start_contact[0] * sin + start_contact[1] * cos
            cos2, sin2 = math.cos(-last[2]), math.sin(-last[2])
            dx, dy = world_x - last[0], world_y - last[1]
            material = (dx * cos2 - dy * sin2, dx * sin2 + dy * cos2)
            end_contact = (push[-1].get("observed") or {}).get("0", {}).get("contact_point_robot")
            if end_contact is None:
                continue
            slips.append(math.hypot(material[0] - end_contact[0], material[1] - end_contact[1]))
        out[arm] = {"slip_median": _median(slips), "slip_p90": _quantile(slips, 0.9),
                    "contact_combinations": dict(combinations)}
    return out


def group1(records) -> Dict[str, dict]:
    out = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        primary = [row for row in rows if row["interaction_class"] in PRIMARY
                   and row["variant"] == "baseline"]
        exits = Counter(row["metrics"]["exit_reason"] for row in rows)
        contacts = [row["metrics"].get("push_contact_ratio") for row in rows
                    if row["metrics"].get("push_contact_ratio") is not None]
        out[arm] = {
            "n": len(rows),
            "effective_push_ratio": _ratio(rows, lambda row: row["metrics"]["reached_effective_push"]),
            "effective_push_ratio_primary": _ratio(primary,
                                                   lambda row: row["metrics"]["reached_effective_push"]),
            "contact_ratio_median": _median(contacts),
            "contact_lost_ratio": _ratio(rows, lambda row: row["metrics"]["exit_reason"] == "contact_lost"),
            "patch_lost_ratio": _ratio(rows, lambda row: row["metrics"]["exit_reason"] == "patch_lost"),
            "patch_support_lost_ratio": _ratio(
                rows, lambda row: row["metrics"]["exit_reason"] == "patch_support_lost"),
            "face_switch_ratio": _ratio(rows, lambda row: row["metrics"]["exit_reason"] == "face_switch"),
            "approach_seconds": _median([row["metrics"].get("approach_seconds") for row in rows]),
            "push_seconds": _median([row["metrics"].get("push_wall_time") for row in rows]),
            "wall_clock_seconds": _median([row["metrics"]["wall_clock_seconds"] for row in rows]),
            "exits": dict(exits),
        }
    return out


def group2(records) -> Dict[str, dict]:
    out = {}
    for arm in ARMS:
        primary = [row for row in records if row["arm"] == arm
                   and row["interaction_class"] in PRIMARY and row["variant"] == "baseline"]
        moved = [row for row in primary if row["metrics"]["displacement"] >= MOTION_LINEAR]
        turned = [row for row in primary if abs(row["metrics"]["rotation"]) >= MOTION_ANGLE]
        push_only = [row["metrics"].get("stage_motion", {}).get("push", {}).get("linear")
                     for row in primary]
        out[arm] = {
            "displacement_median": _median([row["metrics"]["displacement"] for row in primary]),
            "displacement_p90": _quantile([row["metrics"]["displacement"] for row in primary], 0.9),
            "rotation_median": _median([row["metrics"]["rotation"] for row in primary]),
            "rotation_p90": _quantile([row["metrics"]["rotation"] for row in primary], 0.9),
            "push_only_median": _median(push_only),
            "linear_speed_median": _median([row["metrics"]["final_speed"] for row in primary]),
            "moved_2cm_ratio": _ratio(primary, lambda row: row["metrics"]["displacement"] >= MOTION_LINEAR),
            "turned_3deg_ratio": _ratio(primary, lambda row: abs(row["metrics"]["rotation"]) >= MOTION_ANGLE),
            "no_motion_ratio": _ratio(primary, lambda row: row["metrics"]["displacement"] < 0.005),
            "both_moved_roots": len(moved), "n_primary": len(primary),
            "both_moved_displacement_median": _median([row["metrics"]["displacement"] for row in moved]),
            "both_turned_rotation_median": _median([row["metrics"]["rotation"] for row in turned]),
        }
    return out


def group4(records) -> Dict[str, dict]:
    out = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        settled = [row for row in rows if row["metrics"]["settled_observed"]]
        out[arm] = {
            "settled_ratio": _ratio(rows, lambda row: row["metrics"]["settled_observed"]),
            "false_stable_ratio": _ratio(settled, lambda row: row["metrics"]["false_stable"]),
            "false_stable_n": sum(1 for row in settled if row["metrics"]["false_stable"]),
            "settled_after_abort": sum(1 for row in rows if row["metrics"].get("settled_after_abort")),
            "settle_timeout_ratio": _ratio(rows, lambda row: row["metrics"]["exit_reason"] == "settle_timeout"),
            "settle_unobserved_ratio": _ratio(
                rows, lambda row: row["metrics"]["exit_reason"] == "settle_unobserved"),
            "terminal_object_speed_median": _median([row["metrics"]["final_speed"] for row in rows]),
            "terminal_object_omega_median": _median([row["metrics"]["final_omega"] for row in rows]),
            "terminal_robot_speed_max_median": _median(
                [max((row["metrics"].get("terminal_robot_speeds") or [0.0])) for row in rows]),
            "terminal_robot_omega_max_median": _median(
                [max((row["metrics"].get("terminal_robot_omegas") or [0.0])) for row in rows]),
        }
    return out


def group5(records) -> Dict[str, dict]:
    """Same-root perturbation effect on the terminal translation and angle, per kind."""

    out: Dict[str, dict] = {}
    kinds = ("pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay", "param")
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm]
        base = {row["root_id"]: row["metrics"]["relative_object"] for row in rows
                if row["variant"] == "baseline"}
        entries = {}
        for variant in kinds:
            positions, angles = [], []
            for row in rows:
                if row["variant"] != variant:
                    continue
                origin = base.get(row["root_id"])
                if origin is None:
                    continue
                now = row["metrics"]["relative_object"]
                positions.append(math.hypot(now[0] - origin[0], now[1] - origin[1]))
                angles.append(abs(_wrap(now[2] - origin[2])))
            entries[variant] = {
                "n": len(positions),
                "position_median": _median(positions), "position_p90": _quantile(positions, 0.9),
                "position_p95": _quantile(positions, 0.95),
                "angle_median": _median(angles), "angle_p90": _quantile(angles, 0.9),
                "angle_p95": _quantile(angles, 0.95),
            }
        out[arm] = entries
    return out


def paired_bootstrap(records, variant: str, samples: int = BOOTSTRAP_SAMPLES,
                     seed: int = BOOTSTRAP_SEED) -> Dict[str, object]:
    """Root-level paired bootstrap of the X - V difference, position and angle separately."""

    by_key = {(row["root_id"], row["arm"], row["variant"]): row for row in records}
    roots = sorted({row["root_id"] for row in records})
    per_root: Dict[str, Dict[str, List[float]]] = {}
    for root in roots:
        v = by_key.get((root, "V", variant))
        x = by_key.get((root, "X", variant))
        v_base = by_key.get((root, "V", "baseline"))
        x_base = by_key.get((root, "X", "baseline"))
        if not all((v, x, v_base, x_base)):
            continue
        vv, vv0 = v["metrics"]["relative_object"], v_base["metrics"]["relative_object"]
        xx, xx0 = x["metrics"]["relative_object"], x_base["metrics"]["relative_object"]
        per_root[root] = {
            "position": [math.hypot(xx[0] - xx0[0], xx[1] - xx0[1])
                         - math.hypot(vv[0] - vv0[0], vv[1] - vv0[1])],
            "angle": [abs(_wrap(xx[2] - xx0[2])) - abs(_wrap(vv[2] - vv0[2]))],
        }
    usable = sorted(per_root)
    out: Dict[str, object] = {"variant": variant, "roots": len(usable)}
    for key in ("position", "angle"):
        pooled = [value for root in usable for value in per_root[root][key]]
        rng = random.Random(seed)
        draws = []
        for _ in range(samples):
            chosen = [usable[rng.randrange(len(usable))] for _ in range(len(usable))]
            values = [value for root in chosen for value in per_root[root][key]]
            draws.append(_median(values))
        draws.sort()
        out[key] = {
            "n": len(pooled), "median": _median(pooled),
            "low": draws[int(0.025 * len(draws))],
            "high": draws[min(len(draws) - 1, int(0.975 * len(draws)))],
        }
    return out


def motion_bins(records) -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    for arm in ARMS:
        rows = [row for row in records if row["arm"] == arm and row["variant"] == "baseline"]
        buckets: Dict[str, List[float]] = defaultdict(list)
        for row in rows:
            displacement = row["metrics"]["displacement"]
            key = "0-2cm" if displacement < 0.02 else ("2-10cm" if displacement < 0.10 else "10cm+")
            buckets[key].append(abs(row["metrics"]["rotation"]))
        out[arm] = {key: {"n": len(values), "rotation_median": _median(values)}
                    for key, values in sorted(buckets.items())}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    records = load(run_dir / f"executions_{args.split}.jsonl")
    payload = {
        "split": args.split, "n": len(records),
        "group1_completion": group1(records),
        "group2_motion": group2(records),
        "group3_force": force_statistics(run_dir, args.split, records),
        "group4_terminal": group4(records),
        "group5_perturbation": group5(records),
        "group6_slip": slip_statistics(run_dir, args.split, records),
        "paired_bootstrap": {variant: paired_bootstrap(records, variant)
                             for variant in ("pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05",
                                             "noise", "delay", "param")},
        "motion_bins": motion_bins(records),
        "thresholds": {"motion_linear_m": MOTION_LINEAR, "motion_angle_deg": 3.0,
                       "note": "stated before the results and never adjusted"},
        "paired_delta": {},
    }
    # the primary paired comparison: median X - V on the primary denominator, baseline variant
    by_key = {(row["root_id"], row["arm"], row["variant"]): row for row in records}
    deltas = {"displacement": [], "rotation": [], "wall_clock": [], "final_speed": []}
    for row in records:
        if row["arm"] != "X" or row["variant"] != "baseline":
            continue
        if row["interaction_class"] not in PRIMARY:
            continue
        other = by_key.get((row["root_id"], "V", "baseline"))
        if other is None:
            continue
        deltas["displacement"].append(row["metrics"]["displacement"] - other["metrics"]["displacement"])
        deltas["rotation"].append(row["metrics"]["rotation"] - other["metrics"]["rotation"])
        deltas["wall_clock"].append(row["metrics"]["wall_clock_seconds"] - other["metrics"]["wall_clock_seconds"])
        deltas["final_speed"].append(row["metrics"]["final_speed"] - other["metrics"]["final_speed"])
    payload["paired_delta"] = {key: {"n": len(values), "median": _median(values),
                                     "low": _quantile(values, 0.025), "high": _quantile(values, 0.975)}
                               for key, values in deltas.items()}
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    lines = [f"# local_position_push_v1 — {args.split} (n={len(records)})", "",
             "## 1. completion and contact", "",
             "| arm | eff push | eff push (primary) | contact ratio | contact_lost | patch_lost | support_lost | approach s | push s | wall s |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = payload["group1_completion"][arm]
        lines.append(f"| {arm} | {_f(row['effective_push_ratio'])} | {_f(row['effective_push_ratio_primary'])} | "
                     f"{_f(row['contact_ratio_median'])} | {_f(row['contact_lost_ratio'])} | "
                     f"{_f(row['patch_lost_ratio'])} | {_f(row['patch_support_lost_ratio'])} | "
                     f"{_f(row['approach_seconds'],2)} | {_f(row['push_seconds'],2)} | "
                     f"{_f(row['wall_clock_seconds'],2)} |")
    lines += ["", "## 2. motion (primary denominator, baseline)", "",
              "| arm | displacement med | p90 | rotation med | push-only med | moved>=2cm | turned>=3deg | no motion |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = payload["group2_motion"][arm]
        lines.append(f"| {arm} | {_f(row['displacement_median'])} | {_f(row['displacement_p90'])} | "
                     f"{_f(row['rotation_median'])} | {_f(row['push_only_median'])} | "
                     f"{_f(row['moved_2cm_ratio'])} | {_f(row['turned_3deg_ratio'])} | "
                     f"{_f(row['no_motion_ratio'])} |")
    lines += ["", "## 3. force (PUSH periods, baseline)", "",
              "| arm | applied peak | applied P95 | requested P95 | saturation | normal P95 | tangential P95 | gap P05 | gap med | error P95 | pen max |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = payload["group3_force"][arm]
        lines.append(f"| {arm} | {_f(row['applied_peak'],2)} | {_f(row['applied_p95'],2)} | "
                     f"{_f(row['requested_p95'],2)} | {_f(row['saturation_share'],3)} | "
                     f"{_f(row['normal_force_p95'],2)} | {_f(row['tangential_force_p95'],3)} | "
                     f"{_f(row['gap_p05'],4)} | {_f(row['gap_median'],4)} | {_f(row['error_p95'],4)} | "
                     f"{_f(row['max_penetration'],5)} |")
    lines += ["", "## 4. terminal state and settle", "",
              "| arm | settled | false stable | after abort | settle timeout | object v med | object w med | robot v max med |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        row = payload["group4_terminal"][arm]
        lines.append(f"| {arm} | {_f(row['settled_ratio'])} | {_f(row['false_stable_ratio'])} ({row['false_stable_n']}) | "
                     f"{row['settled_after_abort']} | {_f(row['settle_timeout_ratio'])} | "
                     f"{_f(row['terminal_object_speed_median'],4)} | {_f(row['terminal_object_omega_median'],4)} | "
                     f"{_f(row['terminal_robot_speed_max_median'],4)} |")
    lines += ["", "## 5. same-root perturbation sensitivity (X - V median, root bootstrap)", "",
              "| variant | position delta | 95% CI | angle delta (rad) | 95% CI | X pos P90 | V pos P90 | X ang P90 | V ang P90 |",
              "|---|---:|---|---:|---|---:|---:|---:|---:|"]
    for variant in ("pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay", "param"):
        boot = payload["paired_bootstrap"][variant]
        x = payload["group5_perturbation"]["X"][variant]
        v = payload["group5_perturbation"]["V"][variant]
        lines.append(f"| {variant} | {_f(boot['position']['median'],4)} | "
                     f"[{_f(boot['position']['low'],4)}, {_f(boot['position']['high'],4)}] | "
                     f"{_f(boot['angle']['median'],4)} | "
                     f"[{_f(boot['angle']['low'],4)}, {_f(boot['angle']['high'],4)}] | "
                     f"{_f(x['position_p90'],4)} | {_f(v['position_p90'],4)} | "
                     f"{_f(x['angle_p90'],4)} | {_f(v['angle_p90'],4)} |")
    lines += ["", "## 6. contact combinations and tangential slip (baseline)", "",
              "| arm | slip median | slip P90 | contact states |", "|---|---:|---:|---|"]
    for arm in ARMS:
        row = payload["group6_slip"][arm]
        lines.append(f"| {arm} | {_f(row['slip_median'],4)} | {_f(row['slip_p90'],4)} | "
                     f"{row['contact_combinations']} |")
    (run_dir / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("METRICS_OK")
    return 0


def _f(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
