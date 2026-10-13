"""Decide between two hypotheses: is the controller bad, or is the settle test too strict?

    python experiments/local_position_push/analyse_settle.py

Two independent measurements, both from the recorded logs only:

A. **Truth audit of every non-settled exit.** At the exit moment, were the object and both
   robots actually at rest? A non-settled exit whose truth state *was* at rest is a false
   negative of the settle test.

B. **Exact counterfactual threshold sweep.** During BRAKE/SETTLE the applied force is
   ``clip_norm(Kv * (v_ref - v), F_max)`` with ``v_ref`` ramping to zero; the settle test only
   decides *when to stop*, never what force is applied. So re-evaluating the test at another
   surface-speed threshold on the logged SETTLE periods is an exact counterfactual, not a
   re-simulation -- provided the threshold is raised (a lowered threshold could have made a run
   continue past the point where the log stops).

C. **Estimator noise floor in situ.** For SETTLE periods in which the truth says nothing is
   moving, what does the surface-speed estimator report? That is the false-alarm floor the
   0.02 m/s threshold is being compared against.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import (  # noqa: E402
    BRAKE_SETTLE_LIMIT,
    SETTLE_ANGULAR,
    SETTLE_HOLD,
    SETTLE_LINEAR,
    SETTLE_SURFACE,
)

CONTROL_DT = 0.05
STILL_SPEED = 0.005      # m/s; "nothing is really moving" for the noise-floor measurement
STILL_OMEGA = 0.01       # rad/s


def _percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}
    def q(fraction):
        position = fraction * (len(values) - 1)
        low = int(math.floor(position))
        high = min(len(values) - 1, low + 1)
        weight = position - low
        return values[low] * (1 - weight) + values[high] * weight
    return {"n": len(values), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": values[-1]}


def load_execution(path: Path, arm: str) -> Optional[dict]:
    settle: List[dict] = []
    first: Optional[dict] = None
    brake_seconds = 0.0
    for line in path.open():
        row = json.loads(line)
        if row.get("arm") != arm:
            continue
        if first is None:
            first = row
        stage = row.get("stage")
        if stage == "brake":
            brake_seconds += CONTROL_DT
        elif stage == "settle":
            settle.append(row)
    if first is None:
        return None
    return {"settle": settle, "first": first, "brake_seconds": brake_seconds}


def would_settle(settle_rows: Sequence[dict], brake_seconds: float, threshold: float,
                 active_robots: Sequence[int], hold: float = SETTLE_HOLD) -> Optional[float]:
    """Time (s) at which the run would have settled under ``threshold``, or None.

    Every *active* robot has to hold simultaneously, exactly like the protocol requires.
    """

    needed = max(1, int(round(hold / CONTROL_DT)))
    run = 0
    for index, row in enumerate(settle_rows):
        elapsed = brake_seconds + (index + 1) * CONTROL_DT
        if elapsed > BRAKE_SETTLE_LIMIT:
            return None
        observed = row.get("observed") or {}
        ok = True
        for robot_id in active_robots:
            entry = observed.get(str(robot_id)) or {}
            own = entry.get("own")
            if own is None:
                ok = False
                break
            speed = math.hypot(own[3], own[4])
            omega = abs(own[5])
            surface = entry.get("surface_speed")
            if not (speed < SETTLE_LINEAR and omega < SETTLE_ANGULAR
                    and surface is not None and surface < threshold):
                ok = False
                break
        run = run + 1 if ok else 0
        if run >= needed:
            return elapsed
    return None


def hold_window_is_truly_still(settle_rows: Sequence[dict], brake_seconds: float,
                               threshold: float, active_robots: Sequence[int],
                               hold: float = SETTLE_HOLD) -> Optional[bool]:
    """For the window that would settle, was *every* body truly still the whole time?

    This is the honest false-positive check: a point sample at one instant is not enough,
    the whole 0.30 s hold has to be genuinely quiet.
    """

    needed = max(1, int(round(hold / CONTROL_DT)))
    run = 0
    for index, row in enumerate(settle_rows):
        elapsed = brake_seconds + (index + 1) * CONTROL_DT
        if elapsed > BRAKE_SETTLE_LIMIT:
            return None
        observed = row.get("observed") or {}
        ok = True
        for robot_id in active_robots:
            entry = observed.get(str(robot_id)) or {}
            own = entry.get("own")
            surface = entry.get("surface_speed")
            if own is None or surface is None:
                ok = False
                break
            if not (math.hypot(own[3], own[4]) < SETTLE_LINEAR and abs(own[5]) < SETTLE_ANGULAR
                    and surface < threshold):
                ok = False
                break
        run = run + 1 if ok else 0
        if run >= needed:
            window = settle_rows[index - needed + 1:index + 1]
            for item in window:
                truth = item["true"]
                if (math.hypot(truth["object"][3], truth["object"][4]) >= SETTLE_LINEAR
                        or abs(truth["object"][5]) >= SETTLE_ANGULAR):
                    return False
                for state in truth["robots"]:
                    if (math.hypot(state[3], state[4]) >= SETTLE_LINEAR
                            or abs(state[5]) >= SETTLE_ANGULAR):
                        return False
            return True
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    records = [json.loads(line) for line in (run_dir / f"executions_{args.split}.jsonl").open()]
    by_key = {(row["root_id"], row["variant"], row["arm"]): row for row in records}
    log_dir = run_dir / "logs" / args.split
    roots = {row["root_id"]: row for row in
             json.loads((run_dir / f"roots_{args.split}.json").read_text())["roots"]}

    audit: Dict[str, dict] = {}
    thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 0.20, 0.30]
    sweep: Dict[str, Dict[str, float]] = {}
    noise: Dict[str, List[float]] = defaultdict(list)
    noise_truth: Dict[str, List[float]] = defaultdict(list)
    stable_noise: Dict[str, List[float]] = defaultdict(list)
    unstable_noise: Dict[str, List[float]] = defaultdict(list)
    stable_hit: Dict[str, int] = defaultdict(int)
    stable_miss: Dict[str, int] = defaultdict(int)

    for arm in ("V", "X"):
        truth_rest_not_settled = 0
        truth_moving_settled = 0
        exits = Counter()
        settle_seconds_at_exit: List[float] = []
        would: Dict[str, int] = Counter()
        truth_still: Dict[str, int] = Counter()
        truth_moving: Dict[str, int] = Counter()
        total = 0
        for key, record in sorted(by_key.items()):
            if key[2] != arm:
                continue
            path = log_dir / f"{key[0].replace('/', '_')}__{key[1]}.jsonl"
            if not path.exists():
                continue
            loaded = load_execution(path, arm)
            if loaded is None:
                continue
            total += 1
            metrics = record["metrics"]
            active = roots[key[0]]["active_robots"]
            exits[metrics["exit_reason"]] += 1
            settled = bool(metrics["settled_observed"])
            # A: truth at the exit moment, object AND both robots
            object_speed = metrics["final_speed"]
            object_omega = metrics["final_omega"]
            robot_speed = max(metrics.get("terminal_robot_speeds") or [0.0])
            robot_omega = max(metrics.get("terminal_robot_omegas") or [0.0])
            truth_rest = (object_speed < SETTLE_LINEAR and object_omega < SETTLE_ANGULAR
                          and robot_speed < SETTLE_LINEAR and robot_omega < SETTLE_ANGULAR)
            if truth_rest and not settled:
                truth_rest_not_settled += 1
            if settled and not truth_rest:
                truth_moving_settled += 1
            if not settled:
                settle_seconds_at_exit.append((metrics.get("stage_seconds") or {}).get("settle", 0.0))
            # B: counterfactual sweep.
            # A run that already settled under the deployed threshold would also settle under
            # any looser one, and its log stops at the settling period (the settling period
            # itself is not logged, because the stage machine breaks before the row is written).
            # Recomputing it would undercount by exactly that missing period, so only the
            # non-settled runs are recomputed -- and for those every period up to the break
            # *is* logged, so the recomputation is exact.
            for threshold in thresholds:
                if settled:
                    would[f"{threshold:.2f}"] += 1
                else:
                    verdict = hold_window_is_truly_still(loaded["settle"], loaded["brake_seconds"],
                                                         threshold, active)
                    if verdict is not None:
                        would[f"{threshold:.2f}"] += 1
                        (truth_still if verdict else truth_moving)[f"{threshold:.2f}"] += 1
            # C: estimator noise floor on truly-static periods.
            # The estimate compares two scans ~0.05 s apart, so a still period that directly
            # follows a moving one still carries that motion. Require stillness in *two*
            # consecutive periods before calling the estimate a false alarm.
            previous_still = False
            previous_points = None
            for row in loaded["settle"]:
                entry = (row.get("observed") or {}).get("0") or {}
                own = entry.get("own")
                surface = entry.get("surface_speed")
                truth = row["true"]
                speeds = [math.hypot(truth["object"][3], truth["object"][4])]
                omegas = [abs(truth["object"][5])]
                for state in truth["robots"]:
                    speeds.append(math.hypot(state[3], state[4]))
                    omegas.append(abs(state[5]))
                still = max(speeds) < STILL_SPEED and max(omegas) < STILL_OMEGA
                if own is not None and surface is not None and still and previous_still:
                    noise[arm].append(surface)
                    points = entry.get("patch_points")
                    stable = (previous_points is not None and points == previous_points)
                    (stable_noise if stable else unstable_noise)[arm].append(surface)
                    (stable_hit if surface < SETTLE_SURFACE else stable_miss)[arm] += 1 \
                        if stable else 0
                if still:
                    noise_truth[arm].append(max(speeds))
                previous_still = still
                previous_points = entry.get("patch_points")
        audit[arm] = {
            "executions": total,
            "exits": dict(exits),
            "settled": exits.get("settled_observed", 0),
            "settle_rate": exits.get("settled_observed", 0) / max(1, total),
            "truth_at_rest_but_not_settled": truth_rest_not_settled,
            "settled_but_truth_moving": truth_moving_settled,
            "settle_seconds_median_when_not_settled": statistics.median(settle_seconds_at_exit)
            if settle_seconds_at_exit else None,
        }
        sweep[arm] = {key: value / max(1, total) for key, value in would.items()}
        audit[arm]["newly_settled_truth_still"] = dict(truth_still)
        audit[arm]["newly_settled_truth_moving"] = dict(truth_moving)

    payload = {
        "split": args.split,
        "thresholds": thresholds,
        "audit": audit,
        "counterfactual_settle_rate": sweep,
        "estimator_noise_floor_on_static_periods": {arm: _percentiles(values)
                                                    for arm, values in noise.items()},
        "truth_speed_on_those_periods": {arm: _percentiles(values)
                                         for arm, values in noise_truth.items()},
        "static_periods_split_by_patch_stability": {
            arm: {"patch_point_count_unchanged": _percentiles(stable_noise[arm]),
                  "patch_point_count_changed": _percentiles(unstable_noise[arm]),
                  "below_threshold_when_unchanged": stable_hit[arm],
                  "above_threshold_when_unchanged": stable_miss[arm]}
            for arm in ("V", "X")},
        "note": ("the counterfactual sweep is exact for thresholds >= the deployed "
                 f"{SETTLE_SURFACE} m/s, because BRAKE/SETTLE forces never depend on the test; "
                 "a lower threshold could have let a run continue past where its log stops and "
                 "is therefore not evaluated"),
    }
    (run_dir / "settle_audit.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    lines = ["# 判稳到底是不是太严？", "",
             "## A. 未判稳的退出，真值当时是否真的静止", "",
             "| 臂 | 执行数 | 判稳率 | 未判稳但真值已静止 | 判稳了但真值仍在动 | 未判稳时已进入 SETTLE 的中位秒数 |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        row = audit[arm]
        lines.append(f"| {arm} | {row['executions']} | {row['settle_rate']:.3f} | "
                     f"**{row['truth_at_rest_but_not_settled']}** | {row['settled_but_truth_moving']} | "
                     f"{_f(row['settle_seconds_median_when_not_settled'],2)} |")
    lines += ["", "## B. 精确反事实：把表面速度门槛放宽会怎样", "",
              "| 门槛 m/s | V 判稳率 | X 判稳率 | 新增判稳中真值确实静止 | 新增判稳中真值仍在动 |",
              "|---:|---:|---:|---:|---:|"]
    for threshold in thresholds:
        key = f"{threshold:.2f}"
        still = audit["V"]["newly_settled_truth_still"].get(key, 0) + \
            audit["X"]["newly_settled_truth_still"].get(key, 0)
        moving = audit["V"]["newly_settled_truth_moving"].get(key, 0) + \
            audit["X"]["newly_settled_truth_moving"].get(key, 0)
        lines.append(f"| {key} | {sweep['V'].get(key, 0.0):.3f} | {sweep['X'].get(key, 0.0):.3f} | "
                     f"{still} | **{moving}** |")
    lines += ["", f"部署门槛 = {SETTLE_SURFACE} m/s。门槛 ≥ 部署值时的反事实是精确的（BRAKE/SETTLE 的力与"
              "判据无关），小于部署值不评估。", "",
              "## C. 真值确定静止时，估计器报多少（虚警地板）", "",
              "| 臂 | n | p50 | p90 | p95 | max |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        row = payload["estimator_noise_floor_on_static_periods"][arm]
        lines.append(f"| {arm} | {row['n']} | {_f(row['p50'],4)} | {_f(row['p90'],4)} | "
                     f"{_f(row['p95'],4)} | {_f(row['max'],4)} |")
    lines += ["", "## D. 虚警来自哪里：片段构成是否变化", "",
              "| 臂 | 片段点数不变时的估计 | 片段点数变了时的估计 | 点数不变但超过门槛的比例 |",
              "|---|---:|---:|---:|"]
    for arm in ("V", "X"):
        row = payload["static_periods_split_by_patch_stability"][arm]
        stable = row["patch_point_count_unchanged"]
        unstable = row["patch_point_count_changed"]
        hit = row["below_threshold_when_unchanged"]
        miss = row["above_threshold_when_unchanged"]
        total = hit + miss
        lines.append(f"| {arm} | n={stable['n']} p50={_f(stable['p50'],4)} p90={_f(stable['p90'],4)} | "
                     f"n={unstable['n']} p50={_f(unstable['p50'],4)} p90={_f(unstable['p90'],4)} | "
                     f"{miss / total:.3f} ({miss}/{total}) |")
    (run_dir / "settle_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("SETTLE_AUDIT_OK")
    return 0


def _f(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    raise SystemExit(main())
