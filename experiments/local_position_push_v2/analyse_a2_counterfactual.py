"""A2 (log-only): threshold and budget counterfactuals, with their applicability conditions.

    python experiments/local_position_push_v2/analyse_a2_counterfactual.py --split test

**What this is.** During BRAKE/SETTLE the applied force is
``clip_norm(Kv * (v_ref - v), F_max)`` with ``v_ref`` ramping to zero, and the stage machine
never reads the settle *test* to decide what force to apply. The test only decides when to stop.
Re-evaluating a different threshold or a longer budget on the logged settle periods is therefore
an **exact** counterfactual, not a re-simulation -- *provided* the change only removes periods
from a run (threshold raised) or only looks at periods that were logged (budget extended within
the log). A threshold *below* the deployed one could have let a run continue past where its log
stops and is not evaluated.

**What this is not.** It is a decision-layer counterfactual. The robot trajectories are the
recorded ones; nothing here shows that a closed-loop continuation starting from those states
would improve, and no claim of that kind may be based on this table. Two further limits are
reported explicitly rather than papered over:

  * a longer budget helps only if the estimate *ever* came below the threshold within the log.
    If it never did, the log cannot say whether it would have eventually;
  * the truth check is reported twice -- bodies below ``SETTLE_LINEAR``/``SETTLE_ANGULAR`` (the
    definition used so far) and the rotation-aware true surface normal speed below the candidate
    threshold (the definition that matches what the estimator is trying to measure). The first is
    weaker on a rotating object.

Written to ``<run-dir>/a2_counterfactual.{json,md}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.local_contact_option.contract import (  # noqa: E402
    BRAKE_SETTLE_LIMIT,
    SETTLE_ANGULAR,
    SETTLE_HOLD,
    SETTLE_LINEAR,
    SETTLE_SURFACE,
)
from experiments.local_position_push_v2.audit_a1 import (  # noqa: E402
    true_surface_normal_speed,
)

CONTROL_DT = 0.05
NEEDED = max(1, int(round(SETTLE_HOLD / CONTROL_DT)))
#: The settle test can only choose *between* these outcomes. An abort is decided during PUSH and
#: the controller keeps the abort reason as the terminal exit even if quiet is reached later, so
#: an aborted run can never be re-classified as settled by a threshold or budget change; such
#: runs are excluded from the counterfactual and reported separately.
SETTLE_DECIDED = ("settled_observed", "settle_timeout", "settle_unobserved")
THRESHOLDS = (0.02, 0.025, 0.03, 0.04, 0.05, 0.06)
BUDGETS = (4.0, 5.0, 6.0, 8.0)


def _truth_bodies_low(row: dict, active: Sequence[int]) -> bool:
    truth = row["true"]
    if (math.hypot(truth["object"][3], truth["object"][4]) >= SETTLE_LINEAR
            or abs(truth["object"][5]) >= SETTLE_ANGULAR):
        return False
    for robot_id in active:
        state = truth["robots"][robot_id]
        if (math.hypot(state[3], state[4]) >= SETTLE_LINEAR or abs(state[5]) >= SETTLE_ANGULAR):
            return False
    return True


def _truth_surface_low(row: dict, active: Sequence[int], threshold: float) -> bool:
    """Rotation-aware: the surface point speed |v + omega x r| . n below the candidate value.

    A period whose truth value cannot be built (no local normal that period) is *not* counted as
    a violation -- the question is how many accepted windows were demonstrably still moving, and
    an unjudgeable period is not evidence either way.
    """

    for robot_id in active:
        value = true_surface_normal_speed(row, robot_id)
        if value is not None and value >= threshold:
            return False
    return True


def _criterion_holds(row: dict, active: Sequence[int], threshold: float) -> bool:
    observed = row.get("observed") or {}
    for robot_id in active:
        entry = observed.get(str(robot_id)) or {}
        own = entry.get("own")
        if own is None:
            return False
        speed = entry.get("surface_speed")
        if (math.hypot(own[3], own[4]) >= SETTLE_LINEAR or abs(own[5]) >= SETTLE_ANGULAR
                or speed is None or speed >= threshold):
            return False
    return True


def _fires(settle_rows: Sequence[dict], active: Sequence[int], threshold: float,
           budget: float, brake_seconds: float = 0.0) -> bool:
    """Would the on-board criterion fire under ``threshold`` within ``budget`` of BRAKE+SETTLE?

    ``brake_seconds`` is the time already spent in the BRAKE stage when SETTLE begins; the budget
    in the controller is on their sum, so leaving it out would overstate the available time.
    """

    run = 0
    for index, row in enumerate(settle_rows):
        if brake_seconds + (index + 1) * CONTROL_DT > budget:
            return False
        run = run + 1 if _criterion_holds(row, active, threshold) else 0
        if run >= NEEDED:
            return True
    return False


def _window_truth(settle_rows: Sequence[dict], active: Sequence[int], threshold: float,
                  budget: float, brake_seconds: float = 0.0) -> tuple:
    """The 0.30 s window that would be accepted, judged two ways."""

    run = 0
    for index, row in enumerate(settle_rows):
        if brake_seconds + (index + 1) * CONTROL_DT > budget:
            return None, None
        run = run + 1 if _criterion_holds(row, active, threshold) else 0
        if run >= NEEDED:
            window = settle_rows[index - NEEDED + 1:index + 1]
            bodies = all(_truth_bodies_low(item, active) for item in window)
            surface = all(_truth_surface_low(item, active, threshold) for item in window)
            return bodies, surface
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    records = [json.loads(line) for line in (run_dir / f"executions_{args.split}.jsonl").open()]
    roots = {row["root_id"]: row for row in
             json.loads((run_dir / f"roots_{args.split}.json").read_text())["roots"]}
    log_dir = run_dir / "logs" / args.split

    threshold_table: Dict[str, Dict[str, dict]] = {}
    abort_tables: Dict[str, Dict[str, dict]] = {}
    budget_table: Dict[str, Dict[str, dict]] = {}
    for arm in ("V", "X"):
        subset = [record for record in records if record["arm"] == arm]
        loaded = []
        for record in subset:
            active = roots[record["root_id"]]["active_robots"]
            path = log_dir / f"{record['root_id'].replace('/', '_')}__{record['variant']}.jsonl"
            if not path.exists():
                continue
            rows = [json.loads(line) for line in path.open()]
            rows = [row for row in rows if row.get("arm") == arm]
            settle = [row for row in rows if row.get("stage") == "settle"]
            brake_seconds = CONTROL_DT * sum(1 for row in rows if row.get("stage") == "brake")
            loaded.append((record, active, settle, brake_seconds))
        total = len(loaded)
        eligible = [item for item in loaded
                    if item[0]["metrics"]["exit_reason"] in SETTLE_DECIDED]
        aborts = [item for item in loaded if item[0]["metrics"]["exit_reason"] not in SETTLE_DECIDED]
        excluded_aborts = len(aborts)

        threshold_table[arm] = {}
        abort_table: Dict[str, dict] = abort_tables.setdefault(arm, {})
        for threshold in THRESHOLDS:
            settled = 0
            new_bodies_bad = new_surface_bad = newly = 0
            for record, active, settle, brake_seconds in eligible:
                already = record["metrics"]["exit_reason"] == "settled_observed"
                if already:
                    settled += 1
                    continue
                if _fires(settle, active, threshold, BRAKE_SETTLE_LIMIT, brake_seconds):
                    settled += 1
                    newly += 1
                    bodies, surface = _window_truth(settle, active, threshold,
                                                    BRAKE_SETTLE_LIMIT, brake_seconds)
                    new_bodies_bad += int(bodies is False)
                    new_surface_bad += int(surface is False)
            # rates are reported over *all* executions so they stay comparable with 0.674/0.617;
            # an abort is never settled under any threshold, so it contributes zero to the
            # numerator and stays in the denominator
            threshold_table[arm][f"{threshold:.3f}"] = {
                "settle_rate": settled / max(1, total),
                "settle_rate_settle_decided": settled / max(1, len(eligible)),
                "settled": settled, "eligible": len(eligible), "total": total,
                "excluded_aborts": excluded_aborts,
                "newly_settled": newly,
                "newly_truth_bodies_bad": new_bodies_bad,
                "newly_truth_surface_bad": new_surface_bad,
                "false_positive_rate_bodies": new_bodies_bad / max(1, newly),
                "false_positive_rate_surface": new_surface_bad / max(1, newly),
            }
            # the abort population, kept separate on purpose
            reached_quiet = 0
            for record, active, settle, brake_seconds in aborts:
                if _fires(settle, active, threshold, BRAKE_SETTLE_LIMIT, brake_seconds):
                    reached_quiet += 1
            abort_table[f"{threshold:.3f}"] = {
                "abort_executions": len(aborts), "reached_quiet_within_budget": reached_quiet,
                "reclassified_as_settled": 0,
                "note": "the controller keeps the abort reason as the terminal exit, so reaching "
                        "quiet here does not change the recorded outcome",
            }

        budget_table[arm] = {}
        for budget in BUDGETS:
            within_log = True
            settled = 0
            for record, active, settle, brake_seconds in eligible:
                already = record["metrics"]["exit_reason"] == "settled_observed"
                if already:
                    settled += 1
                    continue
                logged_seconds = brake_seconds + len(settle) * CONTROL_DT
                if budget > logged_seconds + 1e-9:
                    within_log = False
                if _fires(settle, active, SETTLE_SURFACE, budget, brake_seconds):
                    settled += 1
            budget_table[arm][f"{budget:.1f}"] = {
                "settled_eligible": settled, "eligible": len(eligible), "total": total,
                "settle_rate_settle_decided": settled / max(1, len(eligible)),
                "exact_within_log": within_log,
            }

        # how many settle_timeout runs ever saw the estimate below the deployed threshold?
        timeouts = [(record, active, settle) for record, active, settle, _ in loaded
                    if record["metrics"]["exit_reason"] == "settle_timeout"]
        ever_low = 0
        min_estimate: List[float] = []
        for record, active, settle in timeouts:
            values = []
            for row in settle:
                for robot_id in active:
                    entry = (row.get("observed") or {}).get(str(robot_id)) or {}
                    value = entry.get("surface_speed")
                    if value is not None:
                        values.append(value)
            if values:
                min_estimate.append(min(values))
                ever_low += int(min(values) < SETTLE_SURFACE)
        budget_table[arm]["timeout_diagnosis"] = {
            "timeout_executions": len(timeouts),
            "min_estimate_below_deployed_threshold_ever": ever_low,
            "min_estimate_never_below_threshold": len(timeouts) - ever_low,
            "median_min_estimate": sorted(min_estimate)[len(min_estimate) // 2]
            if min_estimate else None,
        }

        # per-variant settle rate at the deployed threshold (so the aggregate cannot hide it)
        by_variant: Dict[str, Counter] = defaultdict(Counter)
        for record, active, settle, _brake in loaded:
            by_variant[record["variant"]]["n"] += 1
            if record["metrics"]["exit_reason"] == "settled_observed":
                by_variant[record["variant"]]["settled"] += 1
        budget_table[arm]["by_variant"] = {
            variant: {"n": counts["n"], "settled": counts["settled"],
                      "settle_rate": counts["settled"] / max(1, counts["n"])}
            for variant, counts in sorted(by_variant.items())}

    payload = {
        "split": args.split, "run_id": args.run_id,
        "applicability": {
            "exact_for": "thresholds >= the deployed value, and budgets <= the logged length; "
                         "the BRAKE/SETTLE force never depends on the settle test",
            "not_exact_for": "a threshold below the deployed value (a run could have continued "
                             "past the end of its log)",
            "scope": "decision layer only. The trajectories are the recorded ones; this table "
                     "does not show that a closed-loop continuation would improve",
            "truth_checks": ["bodies below SETTLE_LINEAR/SETTLE_ANGULAR (weaker on a rotating "
                             "object)", "rotation-aware true surface normal speed below the "
                             "candidate threshold (what the estimator measures)"],
        },
        "constants": {"SETTLE_SURFACE": SETTLE_SURFACE, "SETTLE_LINEAR": SETTLE_LINEAR,
                      "SETTLE_ANGULAR": SETTLE_ANGULAR, "SETTLE_HOLD": SETTLE_HOLD,
                      "BRAKE_SETTLE_LIMIT": BRAKE_SETTLE_LIMIT},
        "threshold_counterfactual": threshold_table,
        "abort_population_counterfactual": abort_tables,
        "budget_counterfactual": budget_table,
    }
    (run_dir / "a2_counterfactual.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    (run_dir / "a2_counterfactual.md").write_text(render(payload))
    print(render(payload))
    print("A2_COUNTERFACTUAL_OK")
    return 0


def _f(value, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = ["# A2（仅日志）：门槛与预算反事实", "",
             f"split `{payload['split']}`，run `{payload['run_id']}`。", "",
             "**适用范围**：门槛 **只评估 ≥ 部署值 0.02**、预算只评估 **落在日志长度内** 的部分；"
             "因为 BRAKE/SETTLE 的力从不由判据决定，这些是精确反事实。"
             "反事实只在**由判据决定结局**的执行上做（`settled_observed` / `settle_timeout` / "
             "`settle_unobserved`）；中止类退出由 PUSH 决定，任何门槛或预算都不会把它变成判稳。"
             "**这不是闭环结论**：轨迹是记录下来的那一条，本表不能说明接续会改善。", "",
             "## 门槛反事实（部署 0.02 m/s）", "",
             "| 臂 | 门槛 m/s | 判稳率(全部) | 判稳率(判据决定集) | 新增判稳 | 新增中真值体速不达标 | 新增中真值表面法向速度不达标 | 误报率(体速) | 误报率(表面) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        for threshold, row in payload["threshold_counterfactual"][arm].items():
            lines.append(
                f"| {arm} | {threshold} | {row['settle_rate']:.3f} | "
                f"{row['settle_rate_settle_decided']:.3f} | {row['newly_settled']} | "
                f"{row['newly_truth_bodies_bad']} | {row['newly_truth_surface_bad']} | "
                f"{row['false_positive_rate_bodies']:.3f} | "
                f"{row['false_positive_rate_surface']:.3f} |")
    lines += ["", "## 中止类退出：门槛放宽后有多少达到判据（**不改变记录结局**）", "",
              "中止结局由 PUSH 决定，控制器把中止原因留作终态退出，所以即使随后达到判据也不会被"
              "记成判稳。此表只说明「有多少中止执行其实已经静下来」。", "",
              "| 臂 | 门槛 m/s | 中止执行 | 达到判据 |", "|---|---:|---:|---:|"]
    for arm in ("V", "X"):
        for threshold, row in payload["abort_population_counterfactual"][arm].items():
            lines.append(f"| {arm} | {threshold} | {row['abort_executions']} | "
                         f"{row['reached_quiet_within_budget']} |")
    lines += ["", "## 预算反事实（部署上限 4.0 s）", "",
              "| 臂 | 预算 s | 判稳数(判据决定集) | 判稳率(判据决定集) | 日志内可精确评估 |",
              "|---|---:|---:|---:|---|"]
    for arm in ("V", "X"):
        for budget, row in payload["budget_counterfactual"][arm].items():
            if not isinstance(row, dict) or "settle_rate_settle_decided" not in row:
                continue
            lines.append(f"| {arm} | {budget} | {row['settled_eligible']}/{row['eligible']} | "
                         f"{row['settle_rate_settle_decided']:.3f} | "
                         f"{'是' if row['exact_within_log'] else '否（超出日志）'} |")
    lines += ["", "## 超时执行：估计量是否曾经低于门槛", "",
              "| 臂 | 超时执行 | 曾经低于 0.02 | 从未低于 0.02 | 最小估计中位数 |", "|---|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        row = payload["budget_counterfactual"][arm]["timeout_diagnosis"]
        lines.append(f"| {arm} | {row['timeout_executions']} | "
                     f"{row['min_estimate_below_deployed_threshold_ever']} | "
                     f"{row['min_estimate_never_below_threshold']} | "
                     f"{_f(row['median_min_estimate'],4)} |")
    lines += ["", "若估计量在整个日志里从未低于门槛，那么「给更多时间」在这份日志上无法被验证；"
              "只有曾经低于门槛的执行，延长预算才有可能改变结果。", "",
              "## 部署门槛下的逐变体判稳率", "",
              "| 臂 | 变体 | n | 判稳 | 判稳率 |", "|---|---|---:|---:|---:|"]
    for arm in ("V", "X"):
        for variant, row in payload["budget_counterfactual"][arm]["by_variant"].items():
            lines.append(f"| {arm} | {variant} | {row['n']} | {row['settled']} | "
                         f"{row['settle_rate']:.3f} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
