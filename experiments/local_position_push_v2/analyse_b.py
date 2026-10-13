"""B: can the *observable* terminal local geometry be predicted from observable start inputs?

    python experiments/local_position_push_v2/analyse_b.py --split test --arm V

Round 1 predicted the object's terminal pose in the robot's start frame. That target contains the
object's **global orientation**, which is unobservable through a local scan, so a low R^2 there
said nothing about the option. This script redoes the analysis with a target that a robot could
actually measure at the end of the skill:

  signed_distance   m     (segment centre - robot) . normal        -- how far the surface now is
  tangential_offset m     (segment centre - robot) . tangent       -- how far it slid sideways
  normal_angle      deg   local normal direction in the robot's start frame

plus the discrete outcomes that must not be dropped: whether a terminal patch was observable at
all, whether contact was still held, and whether the observed face changed. The estimator never
sees the object's global pose; neither does the target.

Feature sets, all scored with **leave-one-root-out** cross validation (the eight variants of a
root are never split):

  skill      only what was asked (speed, alpha, active set)
  observed   skill + the whitelist: own start velocity, own start local patch geometry, contact
             flag, teammate relative pose. This is the *deployable* set.
  truth      skill + the object's true initial pose and velocity, expressed in the robot's start
             frame. Not deployable; reported only as an upper reference.

Reported per target: median and mean absolute error **in physical units**, the effective sample
count, and R^2 as an auxiliary number. The residual of the deployable model is then stratified by
the *hidden* object orientation relative to the start frame -- the quantity the local scan cannot
observe -- to show whether the error is systematically tied to it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

TARGETS = ("signed_distance", "tangential_offset", "normal_angle_deg")


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rot(point: Sequence[float], pose: Sequence[float]) -> Tuple[float, float, float]:
    cos, sin = math.cos(-pose[2]), math.sin(-pose[2])
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    return (dx * cos - dy * sin, dx * sin + dy * cos, _wrap(point[2] - pose[2]))


def local_geometry(row: dict, robot_id: int = 0):
    """signed distance, tangential offset and world normal from one logged observation."""

    obs = (row.get("observed") or {}).get(str(robot_id)) or {}
    segment = obs.get("segment_world")
    normal = obs.get("patch_normal")
    own = obs.get("own")
    if not segment or normal is None or own is None:
        return None
    cx = 0.5 * (segment[0][0] + segment[1][0])
    cy = 0.5 * (segment[0][1] + segment[1][1])
    dx, dy = cx - own[0], cy - own[1]
    tangent = (-normal[1], normal[0])
    return (dx * normal[0] + dy * normal[1],
            dx * tangent[0] + dy * tangent[1],
            (normal[0], normal[1]))


def rows_of(path: Path, arm: str) -> List[dict]:
    out = []
    for line in path.open():
        row = json.loads(line)
        if row.get("arm") == arm:
            out.append(row)
    return out


def build(run_dir: Path, split: str, arm: str):
    records = [json.loads(line) for line in (run_dir / f"executions_{split}.jsonl").open()]
    roots = {row["root_id"]: row for row in
             json.loads((run_dir / f"roots_{split}.json").read_text())["roots"]}
    log_dir = run_dir / "logs" / split
    skill_rows, observed_rows, truth_rows, targets, groups = [], [], [], [], []
    discrete: List[dict] = []
    for record in records:
        if record["arm"] != arm:
            continue
        path = log_dir / f"{record['root_id'].replace('/', '_')}__{record['variant']}.jsonl"
        if not path.exists():
            continue
        rows = rows_of(path, arm)
        start_row = next((row for row in rows if row.get("stage") == "push"), None)
        if start_row is None or not rows:
            continue
        end_row = rows[-1]
        start_obs = (start_row.get("observed") or {}).get("0") or {}
        end_obs = (end_row.get("observed") or {}).get("0") or {}
        own_start = start_obs.get("own")
        if own_start is None:
            continue
        frame = (own_start[0], own_start[1], own_start[2])
        meta = roots[record["root_id"]]
        active = list(meta["active_robots"])
        mate = start_row["true"]["robots"][1]
        relation = _rot(mate, frame)

        skill = [float(record["speed"][0]) if record["speed"] and record["speed"][0] else 0.0,
                 math.cos(float(meta["alpha_rad"][0] or 0.0)),
                 math.sin(float(meta["alpha_rad"][0] or 0.0)),
                 float(len(active))]
        start_geom = local_geometry(start_row, 0)
        if start_geom is None:
            signed0, tang0, normal0 = 0.0, 0.0, (0.0, 0.0)
            observed_extra = [0.0] * 8 + [0.0, 0.0, 0.0, 0.0]
        else:
            signed0, tang0, normal0 = start_geom
            angle0 = _wrap(math.atan2(normal0[1], normal0[0]) - frame[2])
            segment = start_obs["segment_world"]
            span = math.dist(segment[0], segment[1])
            observed_extra = [
                math.cos(angle0), math.sin(angle0), signed0, tang0, span,
                float(start_obs.get("patch_points") or 0), float(start_obs.get("patch_confidence") or 0.0),
                float(bool(start_obs.get("contact"))),
                relation[0], relation[1],
                math.cos(math.atan2(mate[1] - frame[1], mate[0] - frame[0]) - frame[2]),
                math.sin(math.atan2(mate[1] - frame[1], mate[0] - frame[0]) - frame[2]),
            ]
        observed_rows.append(skill + [own_start[3], own_start[4], own_start[5]] + observed_extra)
        object_rel = _rot(start_row["true"]["object"], frame)
        truth_rows.append(skill + [
            object_rel[0], object_rel[1], math.cos(object_rel[2]), math.sin(object_rel[2]),
            start_row["true"]["object"][3], start_row["true"]["object"][4],
            start_row["true"]["object"][5],
        ])
        skill_rows.append(skill)

        end_geom = local_geometry(end_row, 0)
        patch_available = end_geom is not None
        contact_end = bool(end_obs.get("contact"))
        face_changed = False
        if start_geom is not None and end_geom is not None:
            dot = max(-1.0, min(1.0, start_geom[2][0] * end_geom[2][0]
                                + start_geom[2][1] * end_geom[2][1]))
            face_changed = math.degrees(math.acos(dot)) > 45.0
        discrete.append({
            "root_id": record["root_id"], "variant": record["variant"],
            "exit_reason": record["metrics"]["exit_reason"],
            "terminal_patch_available": patch_available,
            "contact_at_end": contact_end,
            "face_changed": face_changed,
            "hidden_object_angle_deg": math.degrees(object_rel[2]),
        })
        if patch_available:
            targets.append([end_geom[0], end_geom[1],
                            math.degrees(_wrap(math.atan2(end_geom[2][1], end_geom[2][0])
                                               - frame[2]))])
            groups.append(record["root_id"])
        else:
            targets.append([float("nan")] * 3)
            groups.append(record["root_id"])
    return (np.array(skill_rows), np.array(observed_rows), np.array(truth_rows),
            np.array(targets), groups, discrete)


def leave_one_root_out_folds(groups: Sequence[str]):
    """Yield ``(root, train_mask, test_mask)`` with every variant of a root on one side.

    The eight variants of a root share one frozen snapshot, so they are not independent samples;
    a fold that split them would leak. This is the single place that decides the split.
    """

    unique = sorted(set(groups))
    for root in unique:
        test = np.array([group == root for group in groups])
        train = ~test
        yield root, train, test


def ridge_cv(features: np.ndarray, targets: np.ndarray, groups: Sequence[str],
             lam: float = 1e-2):
    """Leave-one-root-out predictions for every target column (NaN rows are skipped)."""

    valid = ~np.isnan(targets).any(axis=1)
    predictions = np.full_like(targets, np.nan)
    for _root, train_mask, test_mask in leave_one_root_out_folds(groups):
        test = test_mask & valid
        train = train_mask & valid
        if test.sum() == 0 or train.sum() <= features.shape[1]:
            continue
        mean = features[train].mean(axis=0)
        std = features[train].std(axis=0)
        std[std < 1e-9] = 1.0
        x_train = (features[train] - mean) / std
        x_test = (features[test] - mean) / std
        design = np.hstack([x_train, np.ones((x_train.shape[0], 1))])
        penalty = lam * np.eye(design.shape[1])
        penalty[-1, -1] = 0.0
        weights = np.linalg.solve(design.T @ design + penalty, design.T @ targets[train])
        predictions[test] = np.hstack([x_test, np.ones((x_test.shape[0], 1))]) @ weights
    scored = valid & ~np.isnan(predictions).any(axis=1)
    residual = targets[scored] - predictions[scored]
    total = targets[scored] - targets[scored].mean(axis=0)
    per_column = 1.0 - (residual ** 2).sum(axis=0) / np.maximum((total ** 2).sum(axis=0), 1e-12)
    return predictions, residual, per_column, scored


def summarise(name: str, features: np.ndarray, targets: np.ndarray, groups: Sequence[str]):
    predictions, residual, r2, scored = ridge_cv(features, targets, groups)
    out = {"features": name, "n": int(scored.sum()), "roots": len(set(groups)),
           "per_target": {}}
    for index, target in enumerate(TARGETS):
        column = residual[:, index]
        out["per_target"][target] = {
            "mae_median": float(np.median(np.abs(column))),
            "mae_mean": float(np.mean(np.abs(column))),
            "r2": float(r2[index]),
            "units": "m" if target != "normal_angle_deg" else "deg",
        }
    return out, predictions, residual, scored


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--arm", default="V")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    skill, observed, truth, targets, groups, discrete = build(run_dir, args.split, args.arm)

    scored_sets = {}
    payload_sets = {}
    combined = np.hstack([observed, truth[:, 4:]])
    for name, features in (("skill_only", skill), ("observable", observed),
                           ("truth_reference", truth),
                           ("observable_plus_truth", combined)):
        summary, predictions, residual, scored = summarise(name, features, targets, groups)
        payload_sets[name] = summary
        scored_sets[name] = (predictions, residual, scored)

    # discrete outcomes
    total = len(discrete)
    discrete_counts = {
        "executions": total,
        "terminal_patch_available": sum(1 for row in discrete if row["terminal_patch_available"]),
        "terminal_patch_unavailable_kept": sum(
            1 for row in discrete if not row["terminal_patch_available"]),
        "contact_at_end": sum(1 for row in discrete if row["contact_at_end"]),
        "face_changed": sum(1 for row in discrete if row["face_changed"]),
        "exit_reasons": {},
    }
    for row in discrete:
        key = row["exit_reason"]
        discrete_counts["exit_reasons"][key] = discrete_counts["exit_reasons"].get(key, 0) + 1

    # is the deployable model's residual tied to the hidden object orientation?
    predictions, residual, scored = scored_sets["observable"]
    hidden = np.array([row["hidden_object_angle_deg"] for row in discrete])
    hidden = hidden[scored]
    stratification = {}
    for index, target in enumerate(TARGETS):
        column = residual[:, index]
        order = np.argsort(hidden)
        bins = np.array_split(order, 4)
        means = [float(np.mean(column[chunk])) for chunk in bins if len(chunk)]
        # Spearman correlation without scipy: Pearson on ranks
        def _rank(values):
            order = np.argsort(values)
            ranks = np.empty(len(values))
            ranks[order] = np.arange(len(values))
            return ranks
        if len(column) > 3:
            rho = float(np.corrcoef(_rank(hidden), _rank(column))[0, 1])
        else:
            rho = float("nan")
        stratification[target] = {
            "hidden_angle_quartile_residual_means": means,
            "spread_across_quartiles": float(max(means) - min(means)) if len(means) > 1 else None,
            "spearman_rho_residual_vs_hidden_angle": rho,
            "n": int(len(column)),
        }

    payload = {
        "split": args.split, "arm": args.arm, "run_id": args.run_id,
        "targets": list(TARGETS),
        "validation": "leave-one-root-out; all 8 variants of a root held out together",
        "feature_sets": payload_sets,
        "discrete_outcomes": discrete_counts,
        "residual_vs_hidden_object_orientation": stratification,
        "note": ("targets contain no object global pose component; the truth_reference set is an "
                 "upper bound that is not a deployable input"),
    }
    (run_dir / f"analyse_b_{args.arm}.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    (run_dir / f"analyse_b_{args.arm}.md").write_text(render(payload))
    print(render(payload))
    print("ANALYSE_B_OK")
    return 0


def _f(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def render(payload: dict) -> str:
    lines = ["# B. 可观测终态转移（留一 root）", "",
             f"臂 `{payload['arm']}`，split `{payload['split']}`，run `{payload['run_id']}`。",
             f"目标：{', '.join(payload['targets'])}（表达在机器人自身起始坐标系）。",
             f"验证：{payload['validation']}。", "",
             "| 特征集 | 目标 | 单位 | n | 中位绝对误差 | 平均绝对误差 | R²（辅助） |",
             "|---|---|---|---:|---:|---:|---:|"]
    for name, block in payload["feature_sets"].items():
        for target, row in block["per_target"].items():
            lines.append(f"| {name} | {target} | {row['units']} | {block['n']} | "
                         f"{_f(row['mae_median'])} | {_f(row['mae_mean'])} | {_f(row['r2'],3)} |")
    discrete = payload["discrete_outcomes"]
    lines += ["", "## 离散结局（全部执行都保留，不丢样本）", "",
              f"- 执行数 {discrete['executions']}；终态片段可观测 "
              f"{discrete['terminal_patch_available']}，不可观测 "
              f"{discrete['terminal_patch_unavailable_kept']}（保留为离散结果）。",
              f"- 终态仍在接触 {discrete['contact_at_end']}；观测面发生变化 {discrete['face_changed']}。",
              f"- 退出原因：{discrete['exit_reasons']}。", "",
              "## 可部署模型的残差 vs **隐藏**物体朝向", "",
              "| 目标 | n | 四分层残差均值 | 层间极差 | Spearman ρ |", "|---|---:|---|---:|---:|"]
    for target, row in payload["residual_vs_hidden_object_orientation"].items():
        means = ", ".join(_f(value) for value in row["hidden_angle_quartile_residual_means"])
        lines.append(f"| {target} | {row['n']} | {means} | "
                     f"{_f(row['spread_across_quartiles'])} | "
                     f"{_f(row['spearman_rho_residual_vs_hidden_angle'],3)} |")
    lines += ["", "隐藏角 = 物体真值朝向相对机器人起始朝向的夹角，本地扫描原理上不可观测；"
              "若残差随它系统变化，说明该终态目标仍依赖隐藏状态。", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
