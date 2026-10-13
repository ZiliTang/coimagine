"""A1: classification audit of how every execution ended, and of the settle decision itself.

    python experiments/local_position_push_v2/audit_a1.py --split test

Round 2 asks for the accounts to be separated instead of lumped into one "settle rate". For every
execution this script records, independently:

  push_completed          did the effective-push budget (2.0 s of simultaneous contact) complete?
  exit_reason             the recorded terminal reason of the stage machine;
  exit_truth_low_speed    at the exit instant, were object *and* every active robot below the
                          settle thresholds (SETTLE_LINEAR / SETTLE_ANGULAR)?
  window_truth_low_speed  was that true for the *whole* trailing SETTLE_HOLD window (0.30 s)?
  local_settle            did the on-board criterion accept quiet (recorded settled_observed)?
  settle_after_abort      did an aborted run nevertheless satisfy the on-board criterion later?

Two measurement questions from the round-2 brief are answered explicitly:

  Frame.  ``update_surface_estimate`` maps each scan's returns into the **world** frame with
  ``world_points(patch, robot_pose)`` and matches world points, so the robot's own motion is
  already compensated: the estimate is the world-frame surface velocity along the normal, *not* a
  robot-relative quantity. ``check_frame`` verifies that numerically.

  Rotation.  The honest truth reference for a surface point is ``v_object + omega x r`` with
  ``r`` the lever arm from the object centre to the observed contact point; projecting that onto
  the local normal gives the true normal surface speed. A reference built from ``|v_object|``
  alone understates the motion of an off-centre contact on a rotating object.

Written to ``<run-dir>/audit_a1.{json,md}``. Read-only: nothing here reruns the controller.
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
ABORT_EXITS = ("contact_lost", "patch_lost", "surface_unobserved", "patch_support_lost")
#: "the surface really is not moving": stricter than the deployed threshold, used only to build
#: the false-alarm population, so that the discipline does not depend on the deployed value
STATIC_FLOOR = 0.005


def _percentiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = sorted(value for value in values if value is not None)
    if not values:
        return {"n": 0, "p50": None, "p90": None, "p95": None, "max": None}

    def q(fraction: float) -> float:
        position = fraction * (len(values) - 1)
        low = int(math.floor(position))
        high = min(len(values) - 1, low + 1)
        weight = position - low
        return values[low] * (1 - weight) + values[high] * weight

    return {"n": len(values), "p50": q(0.5), "p90": q(0.9), "p95": q(0.95), "max": values[-1]}


def _speed(state: Sequence[float]) -> float:
    return math.hypot(state[3], state[4])


def truth_low_speed(row: dict, active: Sequence[int]) -> bool:
    """Object *and* every active robot below the settle thresholds at one logged period."""

    truth = row["true"]
    if _speed(truth["object"]) >= SETTLE_LINEAR or abs(truth["object"][5]) >= SETTLE_ANGULAR:
        return False
    for robot_id in active:
        state = truth["robots"][robot_id]
        if _speed(state) >= SETTLE_LINEAR or abs(state[5]) >= SETTLE_ANGULAR:
            return False
    return True


def true_surface_normal_speed(row: dict, robot_id: int) -> Optional[float]:
    """|v_object + omega x r| projected on the local normal, at the observed contact point.

    This is the rotation-aware truth reference. ``r`` is the lever arm from the object centre to
    the contact point; the projection uses the *world* patch normal, the same quantity the
    estimator projects onto. Returns ``None`` when the period has no usable normal or point.
    """

    observed = (row.get("observed") or {}).get(str(robot_id)) or {}
    normal = observed.get("patch_normal")
    if normal is None:
        return None
    own = observed.get("own")
    point = None
    contact = observed.get("contact_point_robot")
    if contact is not None and own is not None:
        cos, sin = math.cos(own[2]), math.sin(own[2])
        point = (own[0] + contact[0] * cos - contact[1] * sin,
                 own[1] + contact[0] * sin + contact[1] * cos)
    else:
        segment = observed.get("segment_world")
        if segment:
            point = (0.5 * (segment[0][0] + segment[1][0]),
                     0.5 * (segment[0][1] + segment[1][1]))
    if point is None:
        return None
    obj = row["true"]["object"]
    rx, ry = point[0] - obj[0], point[1] - obj[1]
    vx = obj[3] - obj[5] * ry
    vy = obj[4] + obj[5] * rx
    return abs(vx * normal[0] + vy * normal[1])


def load_settle_rows(path: Path, arm: str) -> List[dict]:
    rows = []
    for line in path.open():
        row = json.loads(line)
        if row.get("arm") == arm:
            rows.append(row)
    return rows


def criterion_fires(settle_rows: Sequence[dict], active: Sequence[int]) -> bool:
    """Did the *on-board* criterion hold for SETTLE_HOLD somewhere in the logged settle periods?

    This is the quantity ``settled_after_abort`` was meant to record. The recorded metric is
    structurally always False: ``settled`` is defined as ``exit_reason == settled_observed`` and
    the abort path forces ``exit_reason = abort_exit``, so ``settled and abort_exit`` cannot both
    hold (option.py). Recomputing it here is what makes the abort population auditable.
    """

    needed = max(1, int(round(SETTLE_HOLD / CONTROL_DT)))
    run_length = 0
    for row in settle_rows:
        observed = row.get("observed") or {}
        ok = True
        for robot_id in active:
            entry = observed.get(str(robot_id)) or {}
            own = entry.get("own")
            if own is None:
                ok = False
                break
            speed = entry.get("surface_speed")
            if (math.hypot(own[3], own[4]) >= SETTLE_LINEAR or abs(own[5]) >= SETTLE_ANGULAR
                    or speed is None or speed >= SETTLE_SURFACE):
                ok = False
                break
        run_length = run_length + 1 if ok else 0
        if run_length >= needed:
            return True
    return False


def check_frame() -> Dict[str, object]:
    """Verify numerically that the estimator is world-frame and robot-motion invariant.

    A static surface is scanned from two different robot poses. In the robot frame its returns
    move; in the world frame they do not. If the estimator were robot-relative it would report
    the ego-motion; being world-frame it must report ~0.
    """

    from coimagine.local_contact_option.control import LocalControlState, update_surface_estimate
    from coimagine.local_contact_option.sensing import LocalPatch

    def make_patch(points, distance, normal, timestamp):
        return LocalPatch(
            points_robot=tuple(points), segment_robot=(points[0], points[-1]), distance=distance,
            normal_world=normal, tangent_world=(-normal[1], normal[0]), confidence=1.0,
            timestamp=timestamp, is_target=True, arc_rays=len(points), occluder_labels=())

    normal = (0.0, 1.0)
    xs = (0.10, 0.15, 0.20, 0.25)
    # a wall at world y = 0.50; the robot approaches it by 0.01 m between the two scans, so the
    # *robot-frame* returns move down by 0.01 while the world points stay exactly put
    first = make_patch([(x, 0.30) for x in xs], 0.30, normal, 0.0)
    second = make_patch([(x, 0.29) for x in xs], 0.29, normal, 0.05)
    state = LocalControlState()
    update_surface_estimate(state, first, (0.0, 0.20, 0.0), 0.0, 0.0)
    speed, signed, _rms, matches, reason = update_surface_estimate(
        state, second, (0.0, 0.21, 0.0), 0.05, 0.05)

    # control: a surface that really translates 4 mm in 0.05 s must register 0.08 m/s
    shifted = make_patch([(x, 0.304) for x in xs], 0.304, normal, 0.05)
    state2 = LocalControlState()
    update_surface_estimate(state2, first, (0.0, 0.20, 0.0), 0.0, 0.0)
    moving, _s, _r, _m, _why = update_surface_estimate(
        state2, shifted, (0.0, 0.20, 0.0), 0.05, 0.05)
    return {
        "method": "two scans of a static surface from two robot poses 0.01 m apart; "
                  "world_points() maps returns with the robot pose before matching, so the "
                  "ego-motion cancels and the world points are identical",
        "static_surface_estimate": speed,
        "static_surface_reason": reason,
        "static_surface_matches": matches,
        "static_surface_signed": signed,
        "reference_translation_estimate": moving,
        "expected_static": 0.0,
        "expected_translation": 0.004 / 0.05,
        "frame": "world",
        "verdict": ("robot motion does not enter the estimate"
                    if speed is not None and abs(speed) < 1e-6 else
                    "UNEXPECTED: estimator responded to robot motion"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-suffix", default="")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    records = [json.loads(line) for line in (run_dir / f"executions_{args.split}.jsonl").open()]
    roots = {row["root_id"]: row for row in
             json.loads((run_dir / f"roots_{args.split}.json").read_text())["roots"]}
    log_dir = run_dir / "logs" / args.split

    per_arm: Dict[str, dict] = {}
    per_variant: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    exit_audit: Dict[str, Counter] = defaultdict(Counter)
    exit_truth: Dict[str, List[float]] = defaultdict(list)
    exit_surface_truth: Dict[str, List[float]] = defaultdict(list)
    static_estimate: Dict[str, List[float]] = defaultdict(list)
    static_truth: Dict[str, List[float]] = defaultdict(list)
    moving_estimate: Dict[str, List[float]] = defaultdict(list)
    moving_truth: Dict[str, List[float]] = defaultdict(list)
    static_by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    moving_by_variant: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    abort_fired: Dict[str, int] = defaultdict(int)
    abort_total: Dict[str, int] = defaultdict(int)
    recorded_after_abort: Dict[str, int] = defaultdict(int)
    frame_truth_low: Dict[str, Counter] = defaultdict(Counter)
    n_missing_logs = 0

    for arm in ("V", "X"):
        exits: Counter = Counter()
        total = 0
        for record in sorted(records, key=lambda row: (row["root_id"], row["variant"])):
            if record["arm"] != arm:
                continue
            active = roots[record["root_id"]]["active_robots"]
            path = log_dir / f"{record['root_id'].replace('/', '_')}__{record['variant']}.jsonl"
            if not path.exists():
                n_missing_logs += 1
                continue
            rows = load_settle_rows(path, arm)
            settle = [row for row in rows if row.get("stage") == "settle"]
            total += 1
            metrics = record["metrics"]
            reason = metrics["exit_reason"]
            exits[reason] += 1
            per_variant[arm][record["variant"]][reason] += 1
            if metrics["push_completed"]:
                per_variant[arm][record["variant"]]["push_completed"] += 1
            if metrics.get("reached_effective_push"):
                per_variant[arm][record["variant"]]["reached_effective_push"] += 1

            # exit-instant truth, independent of the on-board verdict
            instant_low = (metrics["final_speed"] < SETTLE_LINEAR
                           and metrics["final_omega"] < SETTLE_ANGULAR
                           and max(metrics.get("terminal_robot_speeds") or [0.0]) < SETTLE_LINEAR
                           and max(metrics.get("terminal_robot_omegas") or [0.0]) < SETTLE_ANGULAR)
            # full trailing hold window
            window = settle[-max(1, int(round(SETTLE_HOLD / CONTROL_DT))):]
            window_low = bool(window) and all(truth_low_speed(row, active) for row in window)
            frame_truth_low[arm][(bool(instant_low), bool(window_low))] += 1
            exit_audit[arm][reason] += 1
            if reason != "settled_observed":
                exit_audit[arm]["not_settled"] += 1
                exit_truth[arm].append(metrics["final_speed"])
                if window:
                    surface_truth = [true_surface_normal_speed(window[-1], robot_id)
                                     for robot_id in active]
                    surface_truth = [value for value in surface_truth if value is not None]
                    if surface_truth:
                        exit_surface_truth[arm].append(max(surface_truth))
                exit_audit[arm]["exit_instant_truth_low"] += int(instant_low)
                exit_audit[arm]["window_truth_low"] += int(window_low)
                last = (settle[-1].get("observed") or {})
                if any((last.get(str(robot_id)) or {}).get("surface_speed") is None
                       for robot_id in active):
                    exit_audit[arm]["exit_surface_unavailable"] += 1
            if reason in ABORT_EXITS:
                abort_total[arm] += 1
                recorded_after_abort[arm] += int(metrics.get("settled_after_abort", False))
                abort_fired[arm] += int(criterion_fires(settle, active))

            # estimator accuracy against the rotation-aware truth, split by truth motion
            previous_static = False
            for row in settle:
                surface_truth = [true_surface_normal_speed(row, robot_id) for robot_id in active]
                surface_truth = [value for value in surface_truth if value is not None]
                estimates = [(row.get("observed") or {}).get(str(robot_id)) or {}
                             for robot_id in active]
                values = [entry.get("surface_speed") for entry in estimates]
                if not surface_truth or any(value is None for value in values):
                    previous_static = False
                    continue
                estimates_max = max(values)
                truth_max = max(surface_truth)
                still = truth_max < STATIC_FLOOR and truth_low_speed(row, active)
                if still and previous_static:
                    static_estimate[arm].append(estimates_max)
                    static_truth[arm].append(truth_max)
                    static_by_variant[arm][record["variant"]].append(estimates_max)
                elif truth_max >= SETTLE_SURFACE:
                    moving_estimate[arm].append(estimates_max)
                    moving_truth[arm].append(truth_max)
                    moving_by_variant[arm][record["variant"]].append(estimates_max)
                previous_static = still

        per_arm[arm] = {
            "executions": total,
            "exits": {key: value for key, value in sorted(exits.items())},
            "settle_rate": exits.get("settled_observed", 0) / max(1, total),
        }

    # push completion and the abort accounting, read from the records once more
    push_stats: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        arm = record["arm"]
        push_stats[arm]["executions"] += 1
        push_stats[arm]["push_completed"] += int(bool(record["metrics"]["push_completed"]))
        push_stats[arm]["reached_effective_push"] += int(
            bool(record["metrics"].get("reached_effective_push")))

    payload = {
        "split": args.split,
        "run_id": args.run_id,
        "definitions": {
            "exit_instant_truth_low": "object and every active robot below SETTLE_LINEAR/"
                                      "SETTLE_ANGULAR at the recorded exit",
            "window_truth_low": f"that held for the whole trailing {SETTLE_HOLD} s of logged "
                                "settle periods",
            "local_settle": "recorded settled_observed (the on-board criterion)",
            "settle_after_abort_recomputed": "an aborted run whose logged settle periods still "
                                             "satisfy the on-board criterion for SETTLE_HOLD",
            "truth_surface_reference": "|v_object + omega x r| . normal at the observed contact "
                                       "point (rotation-aware)",
            "static_floor": STATIC_FLOOR,
        },
        "frame_check": check_frame(),
        "arms": per_arm,
        "push_stats": {arm: dict(push_stats[arm]) for arm in ("V", "X")},
        "exit_classification": {arm: dict(exit_audit[arm]) for arm in ("V", "X")},
        "settle_after_abort": {
            arm: {"abort_executions": abort_total[arm],
                  "recorded_settled_after_abort": recorded_after_abort[arm],
                  "recomputed_criterion_fired": abort_fired[arm]}
            for arm in ("V", "X")},
        "exit_truth_window_cross": {
            arm: {f"instant_low={a},window_low={b}": count
                  for (a, b), count in sorted(frame_truth_low[arm].items())}
            for arm in ("V", "X")},
        "not_settled_truth": {
            arm: {
                "exit_object_speed": _percentiles(exit_truth[arm]),
                "exit_true_surface_normal_speed_rotation_aware": _percentiles(
                    exit_surface_truth[arm]),
            } for arm in ("V", "X")},
        "estimator_on_static_periods": {
            arm: {"estimate": _percentiles(static_estimate[arm]),
                  "truth_surface_speed": _percentiles(static_truth[arm])}
            for arm in ("V", "X")},
        "estimator_when_moving": {
            arm: {"estimate": _percentiles(moving_estimate[arm]),
                  "truth_surface_speed": _percentiles(moving_truth[arm])}
            for arm in ("V", "X")},
        "estimator_on_static_periods_by_variant": {
            arm: {variant: {"estimate": _percentiles(values)}
                  for variant, values in sorted(static_by_variant[arm].items())}
            for arm in ("V", "X")},
        "estimator_when_moving_by_variant": {
            arm: {variant: {"estimate": _percentiles(values)}
                  for variant, values in sorted(moving_by_variant[arm].items())}
            for arm in ("V", "X")},
        "per_variant": {arm: {variant: dict(counts) for variant, counts in
                              sorted(per_variant[arm].items())} for arm in ("V", "X")},
        "constants": {
            "SETTLE_LINEAR": SETTLE_LINEAR, "SETTLE_ANGULAR": SETTLE_ANGULAR,
            "SETTLE_SURFACE": SETTLE_SURFACE, "SETTLE_HOLD": SETTLE_HOLD,
            "BRAKE_SETTLE_LIMIT": BRAKE_SETTLE_LIMIT, "CONTROL_DT": CONTROL_DT,
        },
        "missing_logs": n_missing_logs,
    }
    suffix = f"_{args.out_suffix}" if args.out_suffix else ""
    (run_dir / f"audit_a1{suffix}.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    (run_dir / f"audit_a1{suffix}.md").write_text(render(payload))
    print(render(payload))
    print("AUDIT_A1_OK")
    return 0


def _f(value, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = ["# A1 分类审计", "",
             f"split `{payload['split']}`，run `{payload['run_id']}`。", ""]
    frame = payload["frame_check"]
    lines += ["## 0. 估计量坐标系核对", "",
              f"- 坐标系：**{frame['frame']}**。{frame['method']}。",
              f"- 静止表面（机器人向表面靠近 0.01 m、物体不动）估计 = "
              f"{_f(frame['static_surface_estimate'])} m/s（期望 {frame['expected_static']}），"
              f"匹配点 {frame['static_surface_matches']}。",
              f"- 对照：真实平移 0.004 m / 0.05 s → 估计 "
              f"{_f(frame['reference_translation_estimate'])} m/s"
              f"（期望 {frame['expected_translation']:.4f}）。",
              f"- 判定：{frame['verdict']}。", ""]
    for arm in ("V", "X"):
        arm_row = payload["arms"][arm]
        push = payload["push_stats"][arm]
        lines += [f"## 臂 {arm}", "",
                  f"- 执行 {arm_row['executions']}；有效推动完成 `push_completed` "
                  f"{push['push_completed']}/{arm_row['executions']}；"
                  f"进入有效推动 `reached_effective_push` {push['reached_effective_push']}。",
                  f"- 局部判稳（`settled_observed`）：{arm_row['exits'].get('settled_observed', 0)}"
                  f" → 判稳率 {arm_row['settle_rate']:.3f}。",
                  f"- 退出原因：{arm_row['exits']}。", ""]
        cls = payload["exit_classification"][arm]
        not_settled = cls.get("not_settled", 0)
        lines += [f"- 未判稳 {not_settled} 例中：退出瞬间真值低速 {cls.get('exit_instant_truth_low', 0)}"
                  f"；完整 0.30 s 保持窗真值低速 {cls.get('window_truth_low', 0)}"
                  f"；退出时局部表面速度不可测 {cls.get('exit_surface_unavailable', 0)}。", ""]
    lines += ["## 退出瞬间真值 vs 完整窗口真值（交叉表）", "",
              "| 臂 | (瞬间低速, 窗口低速) | 执行数 |", "|---|---|---:|"]
    for arm in ("V", "X"):
        for key, count in payload["exit_truth_window_cross"][arm].items():
            lines.append(f"| {arm} | {key} | {count} |")
    lines += ["", "## 中止后判稳", "",
              "| 臂 | 中止执行 | 记录 `settled_after_abort` | 重算判据成立 |", "|---|---:|---:|---:|"]
    for arm in ("V", "X"):
        row = payload["settle_after_abort"][arm]
        lines.append(f"| {arm} | {row['abort_executions']} | "
                     f"{row['recorded_settled_after_abort']} | "
                     f"{row['recomputed_criterion_fired']} |")
    lines += ["", "记录的 `settled_after_abort` 结构性恒为 False（`exit_reason` 被中止原因覆盖），"
              "所以上表第 3 列才是真实的中止后判稳数。", "",
              "## 未判稳执行的退出真值", "",
              "| 臂 | 指标 | n | p50 | p90 | p95 | max |", "|---|---|---:|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        truth = payload["not_settled_truth"][arm]
        for label, key in (("退出瞬间物体线速度 m/s", "exit_object_speed"),
                           ("退出时真值表面法向速度（含旋转）m/s",
                            "exit_true_surface_normal_speed_rotation_aware")):
            row = truth[key]
            lines.append(f"| {arm} | {label} | {row['n']} | {_f(row['p50'])} | {_f(row['p90'])} | "
                         f"{_f(row['p95'])} | {_f(row['max'])} |")
    lines += ["", "## 估计量的虚警地板（真值确静止的连续两周期；含旋转的真值参考）", "",
              "| 臂 | 指标 | n | p50 | p90 | max |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        block = payload["estimator_on_static_periods"][arm]
        for label, key in (("估计量 m/s", "estimate"), ("真值表面法向速度 m/s", "truth_surface_speed")):
            row = block[key]
            lines.append(f"| {arm} | {label} | {row['n']} | {_f(row['p50'])} | {_f(row['p90'])} | "
                         f"{_f(row['max'])} |")
    lines += ["", "## 真值确实在动时，估计量是否看见", "",
              "| 臂 | 指标 | n | p50 | p90 | max |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        block = payload["estimator_when_moving"][arm]
        for label, key in (("估计量 m/s", "estimate"), ("真值表面法向速度 m/s", "truth_surface_speed")):
            row = block[key]
            lines.append(f"| {arm} | {label} | {row['n']} | {_f(row['p50'])} | {_f(row['p90'])} | "
                         f"{_f(row['max'])} |")
    lines += ["", "## 估计量虚警地板的**逐变体**分解（真值确静止的两周期）", "",
              "| 臂 | 变体 | n | p50 | p90 | max |", "|---|---|---:|---:|---:|---:|"]
    for arm in ("V", "X"):
        for variant, block in payload["estimator_on_static_periods_by_variant"][arm].items():
            row = block["estimate"]
            lines.append(f"| {arm} | {variant} | {row['n']} | {_f(row['p50'])} | {_f(row['p90'])} | "
                         f"{_f(row['max'])} |")
    lines += ["", "## 逐变体退出原因", ""]
    for arm in ("V", "X"):
        lines += [f"### {arm}", "", "| 变体 | 计数 |", "|---|---|"]
        for variant, counts in payload["per_variant"][arm].items():
            compact = ", ".join(f"{key}={value}" for key, value in sorted(counts.items()))
            lines.append(f"| {variant} | {compact} |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
