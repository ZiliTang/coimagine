"""Can the terminal state be predicted from what the robot can *observe*?

    python experiments/local_position_push/analyse_predictability.py

The option is only useful as a learning primitive if its terminal state is a predictable
function of the information available at skill start. Three nested feature sets, all evaluated
with **leave-one-root-out** cross validation (variants of one root are not independent):

  truth    -- the object's true initial pose plus the robot poses (an upper bound that no
              deployment predictor can reach, because the object's global pose is unobservable)
  observed -- only the whitelist: the robot's own start pose, the local patch geometry and the
              skill parameters
  skill    -- the skill parameters alone (how much is explained by "what was asked" only)

Targets: the object's terminal pose expressed in the robot's own start frame, and separately the
terminal pose *change* (terminal minus initial, same frame).
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


def _wrap(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _rot(point: Sequence[float], pose: Sequence[float]) -> Tuple[float, float, float]:
    """Express ``point`` = (x, y, angle) in the frame of ``pose`` = (x, y, angle)."""

    cos, sin = math.cos(-pose[2]), math.sin(-pose[2])
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    return (dx * cos - dy * sin, dx * sin + dy * cos, _wrap(point[2] - pose[2]))


def build_dataset(run_dir: Path, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    records = [json.loads(line) for line in (run_dir / f"executions_{split}.jsonl").open()]
    log_dir = run_dir / "logs" / split
    roots = {row["root_id"]: row for row in
             json.loads((run_dir / f"roots_{split}.json").read_text())["roots"]}
    truth_rows, observed_rows, skill_rows, targets, groups = [], [], [], [], []
    for record in records:
        if record["arm"] != "V":
            continue
        path = log_dir / f"{record['root_id'].replace('/', '_')}__{record['variant']}.jsonl"
        if not path.exists():
            continue
        first_truth = first_observed = None
        for line in path.open():
            row = json.loads(line)
            if row.get("arm") != "V":
                continue
            # the first PUSH period is the skill's start state
            if (row.get("phase") or row.get("stage")) == "push":
                first_truth = row["true"]
                first_observed = (row.get("observed") or {}).get("0") or {}
                break
        if first_truth is None or first_observed is None:
            continue
        terminal = record.get("terminal_object")
        if terminal is None:
            continue
        meta = roots[record["root_id"]]
        robot0 = first_truth["robots"][0]
        start_frame = (robot0[0], robot0[1], robot0[2])
        # target: object terminal pose and terminal-minus-initial, both in the robot's start frame
        terminal_rel = _rot(terminal, start_frame)
        initial_rel = _rot(first_truth["object"], start_frame)
        target = np.array([terminal_rel[0], terminal_rel[1],
                           math.sin(_wrap(terminal_rel[2])), math.cos(_wrap(terminal_rel[2])),
                           terminal_rel[0] - initial_rel[0], terminal_rel[1] - initial_rel[1],
                           math.sin(_wrap(terminal_rel[2] - initial_rel[2])),
                           math.cos(_wrap(terminal_rel[2] - initial_rel[2]))])
        targets.append(target)
        groups.append(record["root_id"])
        skill = [float(record["speed"][0]) if record["speed"] else 0.0,
                 math.cos(float(meta["alpha_rad"][0] or 0.0)),
                 math.sin(float(meta["alpha_rad"][0] or 0.0)),
                 float(len(meta["active_robots"]))]
        skill_rows.append(skill)
        # observed: own pose is the frame, so what remains observable is the local patch and the
        # contact state; express the patch normal in the robot frame and keep its geometry
        normal = first_observed.get("patch_normal")
        angle = math.atan2(normal[1], normal[0]) - robot0[2] if normal else 0.0
        distance = first_observed.get("patch_distance") or 0.0
        points = first_observed.get("patch_points") or 0
        confidence = first_observed.get("patch_confidence") or 0.0
        segment = first_observed.get("segment_world")
        span = math.dist(segment[0], segment[1]) if segment else 0.0
        tangential = first_observed.get("tangential_offset") or 0.0
        mate = first_truth["robots"][1]
        relation = _rot(mate, start_frame)
        observed_rows.append(skill + [
            math.cos(angle), math.sin(angle), distance, float(points), confidence, span,
            tangential, relation[0], relation[1], math.cos(mate[2] - robot0[2]),
            math.sin(mate[2] - robot0[2]),
        ])
        # truth: the object's true initial pose in the robot start frame, plus robot relations
        object_rel = _rot(first_truth["object"], start_frame)
        truth_rows.append(skill + [
            object_rel[0], object_rel[1], math.cos(_wrap(object_rel[2])),
            math.sin(_wrap(object_rel[2])), relation[0], relation[1],
            math.cos(_wrap(first_truth["object"][2] - robot0[2])),
            math.sin(_wrap(first_truth["object"][2] - robot0[2])),
        ])
    return (np.array(truth_rows), np.array(observed_rows), np.array(skill_rows),
            np.array(targets), groups)


def ridge_cv(features: np.ndarray, targets: np.ndarray, groups: Sequence[str],
             lam: float = 1e-3) -> Dict[str, float]:
    """Leave-one-root-out R^2 for each target column, plus the pooled mean."""

    unique = sorted(set(groups))
    predictions = np.zeros_like(targets)
    for root in unique:
        test = np.array([group == root for group in groups])
        if test.sum() == 0 or (~test).sum() == 0:
            continue
        train = ~test
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
    residual = ((targets - predictions) ** 2).sum(axis=0)
    total = ((targets - targets.mean(axis=0)) ** 2).sum(axis=0)
    per_column = 1.0 - residual / np.maximum(total, 1e-12)
    return {
        "r2_position": float(np.mean(per_column[[0, 1]])),
        "r2_angle": float(np.mean(per_column[[2, 3]])),
        "r2_delta_position": float(np.mean(per_column[[4, 5]])),
        "r2_delta_angle": float(np.mean(per_column[[6, 7]])),
        "overall": float(1.0 - residual.sum() / max(total.sum(), 1e-12)),
        "n": int(len(groups)), "roots": len(unique),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    truth, observed, skill, targets, groups = build_dataset(run_dir, args.split)
    payload = {
        "split": args.split, "rows": len(groups), "roots": len(set(groups)),
        "truth_features": ridge_cv(truth, targets, groups),
        "observed_features": ridge_cv(observed, targets, groups),
        "skill_features_only": ridge_cv(skill, targets, groups),
        "target": "object terminal pose and terminal-minus-initial pose, in the robot's start frame",
        "validation": "leave-one-root-out; all 8 variants of a root are held out together",
    }
    (run_dir / "predictability.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    lines = ["# 终态可预测性（V 臂，留一 root 交叉验证）", "",
             "目标：物体终态位姿、以及终态−初态的位姿变化，都表达在机器人自身起始坐标系里。", "",
             "| 特征集 | 位置 R² | 角度 R² | 增量位置 R² | 增量角度 R² | 总体 R² |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, key in (("真值初始状态（上界）", "truth_features"),
                      ("仅可观测（白名单）", "observed_features"),
                      ("仅技能参数", "skill_features_only")):
        row = payload[key]
        lines.append(f"| {name} | {row['r2_position']:.3f} | {row['r2_angle']:.3f} | "
                     f"{row['r2_delta_position']:.3f} | {row['r2_delta_angle']:.3f} | "
                     f"{row['overall']:.3f} |")
    lines += ["", f"样本 {payload['rows']} 条，{payload['roots']} 个 root。", ""]
    (run_dir / "predictability.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print("PREDICTABILITY_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
