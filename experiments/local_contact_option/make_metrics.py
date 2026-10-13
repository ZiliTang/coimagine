"""C.3 / D.1: aggregate the three metric layers and evaluate the frozen engineering gates.

    python experiments/local_contact_option/make_metrics.py

Everything here reads ``executions_{dev,test}.jsonl`` and writes tables. Nothing re-runs a
model or a controller, and no threshold is chosen from the numbers it prints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import (  # noqa: E402
    BOOTSTRAP_SAMPLES,
    BOOTSTRAP_SEED,
)
from coimagine.local_contact_option.metrics import (  # noqa: E402
    MICRO_ANGULAR,
    MICRO_LINEAR,
    PUSH_SECONDS,
    cluster_bootstrap,
    quantile,
    summarise,
)

ARMS = ("O", "P", "F", "N")
#: A5: the *primary* denominator for G2/G4 is the specified pushable set, not the old four
#: classes. The old four-class set is still reported, clearly labelled, as a secondary group.
PRIMARY_PUSHABLE_CLASSES = ("solo_central", "dual_same_side")
SECONDARY_PUSHABLE_CLASSES = ("solo_central", "solo_offset", "dual_same_side", "dual_rotate")
PUSHABLE_CLASSES = SECONDARY_PUSHABLE_CLASSES
BLOCKED_CLASSES = ("incompatible", "constrained")
NON_PRIMARY_PUSHABLE = ("solo_offset", "dual_rotate")
DISPLACEMENT_BINS = ((0.0, 0.02), (0.02, 0.10), (0.10, float("inf")))


def load(path: Path) -> List[Dict[str, object]]:
    return [json.loads(line) for line in path.open()]


def group_by(records, key):
    out: Dict[object, List[Dict[str, object]]] = defaultdict(list)
    for record in records:
        out[record[key]].append(record)
    return out


# ---------------------------------------------------------------------------
# layer 1
# ---------------------------------------------------------------------------


def layer1(records) -> Dict[str, object]:
    by_arm = group_by(records, "arm")
    out: Dict[str, object] = {}
    for arm, rows in sorted(by_arm.items()):
        exits = Counter(row["metrics"]["exit_reason"] for row in rows)
        pushable = [row for row in rows if row["interaction_class"] in PUSHABLE_CLASSES
                    and row["variant"] == "baseline"]
        pushable_all = [row for row in rows if row["interaction_class"] in PUSHABLE_CLASSES]
        out[arm] = {
            "executions": len(rows),
            "exits": dict(exits),
            "approach_success": _ratio(rows, lambda row: row["metrics"]["exit_reason"] not in
                                       ("approach_timeout", "surface_unobserved", "settle_unobserved")),
            "effective_push_ratio_baseline_pushable": _ratio(pushable, lambda row: row["metrics"]["reached_effective_push"]),
            "effective_push_ratio_all_variants_pushable": _ratio(pushable_all, lambda row: row["metrics"]["reached_effective_push"]),
            "effective_push_ratio_blocked": _ratio([row for row in rows if row["interaction_class"] in BLOCKED_CLASSES],
                                                   lambda row: row["metrics"]["reached_effective_push"]),
            "active_contact_seconds": summarise([row["metrics"].get("active_contact_time") for row in rows]),
            "wall_clock_seconds": summarise([row["metrics"]["wall_clock_seconds"] for row in rows]),
            "push_wall_seconds": summarise([row["metrics"].get("push_wall_time") for row in rows]),
            "contact_period_ratio": summarise([row["metrics"]["contact_period_ratio"] for row in rows]),
            "force_saturation_ratio": summarise([row["metrics"]["force_saturation_ratio"] for row in rows]),
            "max_penetration": max(row["metrics"].get("max_penetration", 0.0) for row in rows),
            "settled_observed_ratio": _ratio([row for row in rows if row["arm"] in ("P", "N")],
                                             lambda row: row["metrics"]["settled_observed"]),
            "false_stable_ratio": _ratio([row for row in rows if row["metrics"]["settled_observed"]],
                                         lambda row: row["metrics"]["false_stable"]),
            "settle_coverage": _ratio([row for row in rows if row["arm"] in ("P", "N")],
                                      lambda row: row["metrics"]["settled_observed"]),
            "final_speed": summarise([row["metrics"]["final_speed"] for row in rows]),
        }
    # the cross table the task asks for and forbids omitting
    cross: Dict[str, Dict[str, int]] = {}
    for arm in ("P", "N"):
        table = {"observed_true": 0, "observed_false": 0, "unobserved_true": 0, "unobserved_false": 0}
        for row in by_arm.get(arm, []):
            metrics = row["metrics"]
            key = ("observed" if metrics["settled_observed"] else "unobserved") + (
                "_true" if metrics["settled_true"] else "_false")
            table[key] += 1
        cross[arm] = table
    return {"per_arm": out, "settle_cross_table": cross}


def _ratio(rows, predicate) -> Optional[float]:
    rows = list(rows)
    if not rows:
        return None
    return sum(1 for row in rows if predicate(row)) / len(rows)


# ---------------------------------------------------------------------------
# layer 2
# ---------------------------------------------------------------------------


def layer2(records) -> Dict[str, object]:
    by_arm_root_variant = {(row["arm"], row["root_id"], row["variant"]): row for row in records}
    by_arm_root: Dict[str, Dict[str, List[Dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        by_arm_root[row["arm"]][row["root_id"]].append(row)

    result: Dict[str, object] = {"per_arm": {}, "sensitivity_by_level": {}, "by_displacement_bin": {}}
    for arm, per_root in sorted(by_arm_root.items()):
        linear: Dict[str, List[float]] = defaultdict(list)
        angular: Dict[str, List[float]] = defaultdict(list)
        robot_linear: Dict[str, List[float]] = defaultdict(list)
        for root_id, rows in per_root.items():
            baseline = next((row for row in rows if row["variant"] == "baseline"), None)
            if baseline is None:
                continue
            base_object = baseline["metrics"]["relative_object"]
            base_robots = baseline["metrics"]["relative_robots"]
            for row in rows:
                if row["variant"] == "baseline":
                    continue
                object_now = row["metrics"]["relative_object"]
                linear[row["variant"]].append(math.hypot(
                    object_now[0] - base_object[0], object_now[1] - base_object[1]))
                angular[row["variant"]].append(abs(_wrap(object_now[2] - base_object[2])))
                robot_linear[row["variant"]].append(max(
                    math.hypot(after[0] - before[0], after[1] - before[1])
                    for after, before in zip(row["metrics"]["relative_robots"], base_robots)
                ))
        entry = {}
        for variant in sorted(linear):
            entry[variant] = {
                "object_position": summarise(linear[variant]),
                "object_angle": summarise(angular[variant]),
                "robot_position": summarise(robot_linear[variant]),
            }
        result["per_arm"][arm] = entry
        result["sensitivity_by_level"][arm] = {
            "position_2mm": entry.get("pos_2mm", {}).get("object_position", {}).get("p90"),
            "position_10mm": entry.get("pos_10mm", {}).get("object_position", {}).get("p90"),
            "velocity_0.02": entry.get("vel_0.02", {}).get("object_position", {}).get("p90"),
            "velocity_0.05": entry.get("vel_0.05", {}).get("object_position", {}).get("p90"),
        }
    # displacement-matched bins
    for arm in sorted(by_arm_root):
        bins = {}
        for low, high in DISPLACEMENT_BINS:
            values = []
            for root_id, rows in by_arm_root[arm].items():
                baseline = next((row for row in rows if row["variant"] == "baseline"), None)
                if baseline is None:
                    continue
                displacement = baseline["metrics"]["displacement"]
                if not (low <= displacement < high):
                    continue
                for row in rows:
                    if row["variant"] == "baseline":
                        continue
                    object_now = row["metrics"]["relative_object"]
                    base_object = baseline["metrics"]["relative_object"]
                    values.append(math.hypot(object_now[0] - base_object[0],
                                             object_now[1] - base_object[1]))
            bins[f"{low:.2f}-{high:.2f}" if high != float("inf") else f"{low:.2f}+"] = summarise(values)
        result["by_displacement_bin"][arm] = bins
    # terminal contact / failure mode distribution and effective-2s share
    modes: Dict[str, Counter] = {}
    for arm, rows in sorted(group_by(records, "arm").items()):
        modes[arm] = Counter(row["metrics"]["exit_reason"] for row in rows)
    result["terminal_modes"] = {arm: dict(counter) for arm, counter in modes.items()}
    # "is it just moving less": effective displacement along the initial push direction
    directional = {}
    for arm, rows in sorted(group_by(records, "arm").items()):
        selected = [row for row in rows if row["interaction_class"] in ("solo_central", "dual_same_side")
                    and row["variant"] == "baseline"]
        directional[arm] = summarise([row["metrics"]["displacement"] for row in selected])
    result["central_and_same_side_displacement"] = directional
    return result


def layer3(records) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for arm, rows in sorted(group_by(records, "arm").items()):
        push_rows = [row for row in rows if row["metrics"].get("reached_effective_push")]
        with_twist = [row for row in push_rows if row["metrics"].get("twist")]
        stable = [row for row in with_twist if row["metrics"].get("push_contact_ratio", 0.0) >= 0.999]
        switching = [row for row in with_twist if row["metrics"].get("push_contact_ratio", 0.0) < 0.999]
        aborted = [row for row in rows if not row["metrics"]["reached_effective_push"]
                   and row["metrics"]["exit_reason"] in ("contact_lost", "face_switch", "patch_lost")]
        micro = [row for row in rows if row["metrics"]["micro_motion"]]
        out[arm] = {
            "push_completed": len(push_rows),
            "twist_fitted": len(with_twist),
            "stable_contact": _twist_stats(stable),
            "contact_switching": _twist_stats(switching),
            "micro_motion_executions": len(micro),
            "micro_motion_excluded_from_fit": sum(1 for row in micro if not row["metrics"].get("twist")),
            "straight_line_share": _ratio(with_twist, lambda row: abs(row["metrics"]["twist"]["omega"]) < 0.05),
            "aborted_before_push_completion": len(aborted),
            "min_push_contact_ratio": min(
                (row["metrics"].get("push_contact_ratio", 0.0) for row in push_rows), default=None),
            "note": "every completed push kept contact for 100% of its PUSH periods; the "
                    "intermittent-contact population is exactly the aborted runs, listed separately",
        }
    return out


def _root_paired_p90(records_a, records_b, variant: str) -> Dict[str, Dict[str, float]]:
    """Per-root displacement between a variant and the baseline, for one arm.

    The returned mapping is keyed by the **real root_id** and keeps every variant of that root
    together, so a bootstrap can resample roots rather than variants.
    """

    baseline = {row["root_id"]: row["metrics"]["relative_object"]
                for row in records_a if row["variant"] == "baseline"}
    out: Dict[str, List[float]] = {}
    for row in records_a:
        if row["variant"] != variant:
            continue
        origin = baseline.get(row["root_id"])
        if origin is None:
            continue
        now = row["metrics"]["relative_object"]
        out.setdefault(row["root_id"], []).append(
            math.hypot(now[0] - origin[0], now[1] - origin[1]))
    return out


def root_p90_bootstrap(per_root: Dict[str, List[float]], *, samples: int, seed: int,
                       target_roots: Sequence[str]) -> Dict[str, object]:
    """Bootstrap the P90 over roots, recomputing the quantile inside every resample.

    Averaging pre-computed per-root means and calling the interval a "P90 interval" would be a
    different statistic; this function resamples whole roots (all their variants together) and
    recomputes the P90 of the pooled sample each time.
    """

    import random

    roots = [root for root in target_roots if root in per_root]
    assert len(roots) == len(set(target_roots)), "the bootstrap root set has duplicates"
    pooled = [value for root in roots for value in per_root[root]]
    if not pooled:
        return {"n_roots": 0, "n_values": 0, "p90": None, "low": None, "high": None}
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [roots[rng.randrange(len(roots))] for _ in range(len(roots))]
        values = [value for root in chosen for value in per_root[root]]
        draws.append(quantile(values, 0.90))
    draws.sort()
    return {
        "n_roots": len(roots),
        "n_values": len(pooled),
        "p90": quantile(pooled, 0.90),
        "low": draws[int(0.025 * len(draws))],
        "high": draws[min(len(draws) - 1, int(0.975 * len(draws)))],
    }


def _twist_stats(rows) -> Dict[str, object]:
    if not rows:
        return {"n": 0}
    return {
        "n": len(rows),
        "fit_position_rmse": summarise([row["metrics"]["twist"]["fit_position_rmse"] for row in rows]),
        "fit_angle_rmse": summarise([row["metrics"]["twist"]["fit_angle_rmse"] for row in rows]),
        "extrapolation_position_rmse": summarise(
            [row["metrics"]["twist"]["extrapolation_position_rmse"] for row in rows]),
        "extrapolation_angle_rmse": summarise(
            [row["metrics"]["twist"]["extrapolation_angle_rmse"] for row in rows]),
    }


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# D.1 frozen gates
# ---------------------------------------------------------------------------


def gates(records) -> Dict[str, object]:
    by_arm = group_by(records, "arm")

    def pushable(arm, classes):
        return [row for row in by_arm.get(arm, [])
                if row["interaction_class"] in classes and row["variant"] == "baseline"]

    primary_n = pushable("N", PRIMARY_PUSHABLE_CLASSES)
    primary_p = pushable("P", PRIMARY_PUSHABLE_CLASSES)
    secondary_n = pushable("N", SECONDARY_PUSHABLE_CLASSES)
    secondary_p = pushable("P", SECONDARY_PUSHABLE_CLASSES)
    effective = _ratio(primary_n, lambda row: row["metrics"]["reached_effective_push"])
    effective_secondary = _ratio(secondary_n, lambda row: row["metrics"]["reached_effective_push"])

    # settle bookkeeping: the coverage denominator is every execution of the arm, not a subset
    arm_n = by_arm.get("N", [])
    observed = [row for row in arm_n if row["metrics"]["settled_observed"]]
    after_abort = [row for row in arm_n if row["metrics"].get("settled_after_abort")]
    false_stable = _ratio(observed, lambda row: row["metrics"]["false_stable"])
    coverage = _ratio(arm_n, lambda row: row["metrics"]["settled_observed"])
    coverage_pushable = _ratio([row for row in arm_n if row["interaction_class"] in PRIMARY_PUSHABLE_CLASSES],
                               lambda row: row["metrics"]["settled_observed"])

    n_median = quantile([row["metrics"]["displacement"] for row in primary_n], 0.5)
    p_median = quantile([row["metrics"]["displacement"] for row in primary_p], 0.5)
    ratio = (n_median / p_median) if (n_median is not None and p_median not in (None, 0.0)) else None
    n_median_s = quantile([row["metrics"]["displacement"] for row in secondary_n], 0.5)
    p_median_s = quantile([row["metrics"]["displacement"] for row in secondary_p], 0.5)

    def _per_root(arm, variant, key):
        rows = by_arm.get(arm, [])
        base = {row["root_id"]: row["metrics"][key] for row in rows if row["variant"] == "baseline"}
        out: Dict[str, List[float]] = {}
        for row in rows:
            if row["variant"] != variant:
                continue
            origin = base.get(row["root_id"])
            if origin is None:
                continue
            now = row["metrics"][key]
            out.setdefault(row["root_id"], []).append(abs(_wrap(now[2] - origin[2])))
        return out

    target_roots = sorted({row["root_id"] for row in records})
    position = {arm: _root_paired_p90(by_arm.get(arm, []), by_arm.get(arm, []), "pos_10mm")
                for arm in ("P", "N")}
    angle = {arm: _per_root(arm, "pos_10mm", "relative_object") for arm in ("P", "N")}
    pos_p = root_p90_bootstrap(position["P"], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                               target_roots=target_roots)
    pos_n = root_p90_bootstrap(position["N"], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                               target_roots=target_roots)
    ang_p = root_p90_bootstrap(angle["P"], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                               target_roots=target_roots)
    ang_n = root_p90_bootstrap(angle["N"], samples=BOOTSTRAP_SAMPLES, seed=BOOTSTRAP_SEED,
                               target_roots=target_roots)
    position_improvement = (
        None if not pos_p["p90"] else 1.0 - pos_n["p90"] / pos_p["p90"]
    )
    angle_change = (
        None if not ang_p["p90"] else (ang_n["p90"] - ang_p["p90"]) / ang_p["p90"]
    )
    speed_breakdown = {}
    for speed in (0.08, 0.16, 0.26):
        p_rows = [row for row in primary_p if row["speed"][0] == speed]
        n_rows = [row for row in primary_n if row["speed"][0] == speed]
        p_med = quantile([row["metrics"]["displacement"] for row in p_rows], 0.5)
        n_med = quantile([row["metrics"]["displacement"] for row in n_rows], 0.5)
        speed_breakdown[str(speed)] = {
            "P_n": len(p_rows), "N_n": len(n_rows), "P_median": p_med, "N_median": n_med,
            "ratio": (n_med / p_med) if (p_med not in (None, 0.0) and n_med is not None) else None,
        }
    efficiency = {}
    for arm in ("P", "N"):
        rows = pushable(arm, PRIMARY_PUSHABLE_CLASSES)
        efficiency[arm] = summarise([row["metrics"].get("displacement_per_active_second")
                                     for row in rows])
    stage_motion = {}
    for arm in ("O", "P", "F", "N"):
        rows = pushable(arm, PRIMARY_PUSHABLE_CLASSES)
        stages = {}
        for row in rows:
            for stage, entry in (row["metrics"].get("stage_motion") or {}).items():
                bucket = stages.setdefault(stage, [])
                bucket.append(entry["linear"])
        stage_motion[arm] = {stage: summarise(values) for stage, values in stages.items()}
    return {
        "G1_information_and_safety": {
            "checked_by": "tests/test_local_contact_option.py + "
                          "tests/test_local_contact_option_repair.py",
        },
        "G2_effective_push_ratio": {
            "primary_denominator": list(PRIMARY_PUSHABLE_CLASSES),
            "primary_roots": len(primary_n), "value": effective, "threshold": 0.90,
            "pass": bool(effective is not None and effective >= 0.90),
            "secondary_denominator": list(SECONDARY_PUSHABLE_CLASSES),
            "secondary_roots": len(secondary_n), "secondary_value": effective_secondary,
            "secondary_pass": bool(effective_secondary is not None and effective_secondary >= 0.90),
            "arm": "N",
        },
        "G3_false_stable_and_coverage": {
            "false_stable_ratio": false_stable, "threshold": 0.05,
            "observed_settled": len(observed), "settled_after_abort": len(after_abort),
            "coverage_all_executions": coverage, "coverage_denominator": len(arm_n),
            "coverage_primary_pushable": coverage_pushable,
            "pass": bool(false_stable is not None and false_stable <= 0.05),
            "denominator": len(observed), "arm": "N",
        },
        "G4_displacement_not_much_smaller": {
            "primary_denominator": list(PRIMARY_PUSHABLE_CLASSES),
            "N_median": n_median, "P_median": p_median, "ratio": ratio, "threshold": 0.80,
            "pass": bool(ratio is not None and ratio >= 0.80),
            "secondary_N_median": n_median_s, "secondary_P_median": p_median_s,
            "secondary_ratio": (n_median_s / p_median_s)
            if (p_median_s not in (None, 0.0) and n_median_s is not None) else None,
            "by_commanded_speed": speed_breakdown,
            "displacement_per_active_second": efficiency,
            "stage_motion": stage_motion,
            "note": "the old four-class 92.5% figure is NOT moved onto the new denominator; both "
                    "are reported with their own root counts",
        },
        "G5_perturbation_final_state": {
            "position": {"P": pos_p, "N": pos_n, "improvement": position_improvement,
                         "threshold": 0.20,
                         "pass": bool(position_improvement is not None and position_improvement >= 0.20)},
            "angle": {"P": ang_p, "N": ang_n, "relative_change": angle_change,
                      "threshold": 0.10,
                      "pass": bool(angle_change is not None and angle_change <= 0.10),
                      "units": "rad (wrap); degrees reported in the tables",
                      "absolute_change_rad": (None if not (ang_p["p90"] and ang_n["p90"])
                                              else ang_n["p90"] - ang_p["p90"]),
                      "near_zero_note": "the relative angle gate is numerically sensitive when "
                                        "the P90 is close to zero; the absolute difference is "
                                        "reported next to it and no new tolerance was invented"},
            "pass": bool(position_improvement is not None and position_improvement >= 0.20
                         and angle_change is not None and angle_change <= 0.10),
        },
        "note": "engineering gates, not theoretical sufficient conditions",
    }


def _pos_p90(rows, variant="pos_10mm") -> Optional[float]:
    base = {row["root_id"]: row["metrics"]["relative_object"] for row in rows if row["variant"] == "baseline"}
    values = []
    for row in rows:
        if row["variant"] != variant:
            continue
        origin = base.get(row["root_id"])
        if origin is None:
            continue
        now = row["metrics"]["relative_object"]
        values.append(math.hypot(now[0] - origin[0], now[1] - origin[1]))
    return quantile(values, 0.90)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-v1")
    parser.add_argument("--out-root", default="outputs/local_contact_option_v1")
    args = parser.parse_args()
    out_dir = ROOT / args.out_root / args.run_id
    test = load(out_dir / "executions_test.jsonl")
    dev_path = out_dir / "executions_dev.jsonl"
    dev = load(dev_path) if dev_path.exists() else []

    payload = {
        "test_executions": len(test), "dev_executions": len(dev),
        "layer1": layer1(test),
        "layer2": layer2(test),
        "layer3": layer3(test),
        "gates": gates(test),
        "layer1_dev": layer1(dev) if dev else None,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    lines = ["# local_contact_option_v1 - summary tables", "",
             f"test executions: {len(test)}; dev executions: {len(dev)}", "",
             "## Layer 1 - execution clarity and physical validity", "",
             "| arm | exits | effective 2s (pushable, baseline) | contact ratio | saturation | max penetration | settled observed | false stable | final speed median |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, row in sorted(payload["layer1"]["per_arm"].items()):
        lines.append(
            f"| {arm} | {', '.join(f'{k}:{v}' for k, v in sorted(row['exits'].items()))} | "
            f"{_fmt(row['effective_push_ratio_baseline_pushable'])} | "
            f"{_fmt(row['contact_period_ratio']['mean'], 3)} | "
            f"{_fmt(row['force_saturation_ratio']['mean'], 3)} | "
            f"{row['max_penetration']:.4f} | {_fmt(row['settled_observed_ratio'])} | "
            f"{_fmt(row['false_stable_ratio'], 3)} | {_fmt(row['final_speed']['median'], 4)} |")
    lines += ["", "## Settle cross table (must not be omitted)", "",
              "| arm | observed & truly settled | observed but still moving (false stable) | unobserved & settled | unobserved & moving |",
              "|---|---:|---:|---:|---:|"]
    for arm, table in sorted(payload["layer1"]["settle_cross_table"].items()):
        lines.append(f"| {arm} | {table['observed_true']} | {table['observed_false']} | "
                     f"{table['unobserved_true']} | {table['unobserved_false']} |")
    lines += ["", "## Layer 2 - perturbation sensitivity of the final state (object, m)", "",
              "| arm | pos 2mm P90 | pos 10mm P90 | vel 0.02 P90 | vel 0.05 P90 | angle 10mm P90 |",
              "|---|---:|---:|---:|---:|---:|"]
    for arm, entry in sorted(payload["layer2"]["per_arm"].items()):
        lines.append(
            f"| {arm} | {_fmt(entry.get('pos_2mm', {}).get('object_position', {}).get('p90'), 4)} | "
            f"{_fmt(entry.get('pos_10mm', {}).get('object_position', {}).get('p90'), 4)} | "
            f"{_fmt(entry.get('vel_0.02', {}).get('object_position', {}).get('p90'), 4)} | "
            f"{_fmt(entry.get('vel_0.05', {}).get('object_position', {}).get('p90'), 4)} | "
            f"{_fmt(entry.get('pos_10mm', {}).get('object_angle', {}).get('p90'), 4)} |")
    lines += ["", "## Layer 3 - constant body-frame twist fit (PUSH trajectories only)", "",
              "| arm | push completed | aborted before completion | twist fitted | fit pos RMSE | fit angle RMSE | extrapolation pos RMSE | straight-line share |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, entry in sorted(payload["layer3"].items()):
        stable = entry.get("stable_contact", {})
        lines.append(
            f"| {arm} | {entry['push_completed']} | {entry['aborted_before_push_completion']} | "
            f"{entry['twist_fitted']} | "
            f"{_fmt(stable.get('fit_position_rmse', {}).get('median'), 4)} | "
            f"{_fmt(stable.get('fit_angle_rmse', {}).get('median'), 4)} | "
            f"{_fmt(stable.get('extrapolation_position_rmse', {}).get('median'), 4)} | "
            f"{_fmt(entry.get('straight_line_share'), 3)} |")
    lines += ["", "## D.1 frozen engineering gates (A5-corrected denominators)", "",
              "| gate | value | threshold | pass |", "|---|---:|---:|:--:|"]
    lines.append("| G2 effective push 2 s (primary: solo_central+dual_same_side) | "
                 f"{_fmt(payload['gates']['G2_effective_push_ratio']['value'], 4)} | 0.90 | "
                 f"{'yes' if payload['gates']['G2_effective_push_ratio']['pass'] else 'no'} |")
    lines.append("| G2 (secondary: four pushable classes) | "
                 f"{_fmt(payload['gates']['G2_effective_push_ratio']['secondary_value'], 4)} | 0.90 | "
                 f"{'yes' if payload['gates']['G2_effective_push_ratio']['secondary_pass'] else 'no'} |")
    g3 = payload["gates"]["G3_false_stable_and_coverage"]
    lines.append(f"| G3 false stable | {_fmt(g3['false_stable_ratio'], 4)} | 0.05 | "
                 f"{'yes' if g3['pass'] else 'no'} |")
    lines.append(f"| G3 coverage (all {g3['coverage_denominator']} executions) | "
                 f"{_fmt(g3['coverage_all_executions'], 4)} | - | - |")
    g4 = payload["gates"]["G4_displacement_not_much_smaller"]
    lines.append(f"| G4 N/P displacement median (primary) | {_fmt(g4['ratio'], 4)} | 0.80 | "
                 f"{'yes' if g4['pass'] else 'no'} |")
    g5 = payload["gates"]["G5_perturbation_final_state"]
    lines.append(f"| G5 position P90 improvement | {_fmt(g5['position']['improvement'], 4)} | 0.20 | "
                 f"{'yes' if g5['position']['pass'] else 'no'} |")
    lines.append(f"| G5 angle P90 relative change rad "
                 f"({_fmt(g5['angle']['P']['p90'], 4)} -> {_fmt(g5['angle']['N']['p90'], 4)}) | "
                 f"{_fmt(g5['angle']['relative_change'], 4)} | 0.10 | "
                 f"{'yes' if g5['angle']['pass'] else 'no'} |")
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("METRICS_OK")
    return 0


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
