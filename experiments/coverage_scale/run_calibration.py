"""Phase A acceptance: the random-scene data link, end to end (task book A3).

    python experiments/coverage_scale/run_calibration.py --run-id coverage-scale-20261013

Runs on 24 independent ``dev`` roots (4 per family x mode cell), which never enter a split.
Checks, in order: strict initial-state legality; full SceneSpec serialisation round-trip;
same-seed reproducibility and different-seed variability; arena containment of the goal and
positive obstacle sizes; the 57-dimensional encoding; one learned prediction; the frozen
geometric check; and one exact candidate reference. Writes ``calibration.json``, a demo
figure with one root per cell, and the acceptance table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.continuous2d.scenes import scene_by_name  # noqa: E402
from coimagine.continuous2d_learning.contract import geometric_violation  # noqa: E402
from coimagine.continuous2d_learning.features import INPUT_DIM, encode_input  # noqa: E402
from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig, predict_window  # noqa: E402
from coimagine.continuous2d_learning.pools import candidate_pool  # noqa: E402
from coimagine.continuous2d_learning.rollout import evaluate_sequence_exact  # noqa: E402
from coimagine.coverage_scale.plan import RootPlan  # noqa: E402
from coimagine.coverage_scale.recipe import collect_root  # noqa: E402
from coimagine.coverage_scale.sampler import (  # noqa: E402
    FAMILIES,
    MODES,
    RootSpec,
    sample_root,
    scene_id_for,
    strict_clearances,
    strict_legal,
)
from coimagine.coverage_scale.scene_registry import (  # noqa: E402
    SceneRegistry,
    scene_from_dict,
    scene_to_dict,
)


def _encodable(scene, config) -> bool:
    env = Continuous2DEnv(config)
    env.reset(scene)
    observation = env.observation()
    from coimagine.continuous2d.skills import Hold

    features = encode_input(scene, _boundary(env), (Hold(), Hold()), config.physics)
    return len(features) == INPUT_DIM


def _boundary(env):
    from coimagine.continuous2d_learning.features import BoundaryState

    snapshot = env.export_state()
    return BoundaryState(object=snapshot.object, robots=snapshot.robots, success_counter=snapshot.success_counter)


def _draw(ax, scene) -> None:
    width, height = scene.arena
    ax.add_patch(patches.Rectangle((0, 0), width, height, fill=False, lw=1.0, ec="black"))
    for box in scene.obstacles:
        ax.add_patch(patches.Rectangle(
            (box.cx - box.half_w, box.cy - box.half_h), 2 * box.half_w, 2 * box.half_h,
            facecolor="0.6", edgecolor="black", lw=0.6,
        ))
    ax.add_patch(patches.Rectangle(
        (scene.goal.cx - scene.goal.half_w, scene.goal.cy - scene.goal.half_h),
        2 * scene.goal.half_w, 2 * scene.goal.half_h,
        facecolor="tab:green", alpha=0.18, edgecolor="tab:green", lw=0.8,
    ))
    corners = [
        (scene.object_pose.x + dx * math.cos(scene.object_pose.angle) - dy * math.sin(scene.object_pose.angle),
         scene.object_pose.y + dx * math.sin(scene.object_pose.angle) + dy * math.cos(scene.object_pose.angle))
        for dx, dy in ((-0.5, -0.125), (0.5, -0.125), (0.5, 0.125), (-0.5, 0.125))
    ]
    ax.add_patch(patches.Polygon(corners, closed=True, facecolor="tab:orange", alpha=0.65, edgecolor="black", lw=0.6))
    for pose in scene.robot_poses:
        ax.add_patch(patches.Circle((pose.x, pose.y), scene.robot_radius, facecolor="tab:blue", alpha=0.6))
    ax.set_xlim(-0.05, width + 0.05)
    ax.set_ylim(-0.05, height + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--roots-per-cell", type=int, default=4)
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config = default_config()
    report: dict = {"roots_per_cell": args.roots_per_cell, "cells": {}, "checks": {}}

    roots = []
    for family in FAMILIES:
        for mode in MODES:
            cell = []
            for ordinal in range(args.roots_per_cell):
                root = sample_root("dev", family, mode, ordinal)
                assert root is not None, f"dev/{family}/{mode}/{ordinal} blocked"
                cell.append(root)
            roots.extend(cell)
            report["cells"][f"{family}/{mode}"] = {
                "roots": [r.as_metadata() for r in cell],
                "attempts": [r.attempts for r in cell],
                "mean_attempts": sum(r.attempts for r in cell) / len(cell),
                "acceptance_per_attempt": 1.0 / (sum(r.attempts for r in cell) / len(cell)),
            }

    # -- geometry acceptance -------------------------------------------------
    for root in roots:
        clearances = strict_clearances(root.scene)
        assert strict_legal(root.scene), f"illegal root {root.root_id}: {clearances}"
        goal = root.scene.goal
        width, height = root.scene.arena
        assert goal.cx - goal.half_w >= -1e-9 and goal.cx + goal.half_w <= width + 1e-9
        assert goal.cy - goal.half_h >= -1e-9 and goal.cy + goal.half_h <= height + 1e-9
        for box in root.scene.obstacles:
            assert box.half_w > 0 and box.half_h > 0
    report["checks"]["strict_legality_all_roots"] = True
    report["checks"]["goal_inside_arena_all_roots"] = True

    # -- serialisation round trip, across a fresh registry -------------------
    registry = SceneRegistry.from_scenes(root.scene for root in roots)
    path = run_dir / "calibration_scenes.json"
    digest = registry.save(path)
    loaded = SceneRegistry.load(path)
    for root in roots:
        original = scene_to_dict(root.scene)
        restored = scene_to_dict(loaded.get(root.scene_id))
        assert original == restored, f"round trip mismatch for {root.scene_id}"
    report["checks"]["scene_serialisation_round_trip"] = True
    report["checks"]["registry_sha256"] = digest
    report["checks"]["registry_scenes"] = len(loaded)

    # -- determinism and variability ----------------------------------------
    for family in FAMILIES:
        for mode in MODES:
            first = sample_root("dev", family, mode, 0)
            again = sample_root("dev", family, mode, 0)
            assert scene_to_dict(first.scene) == scene_to_dict(again.scene)

    def _signature(scene):
        return (
            scene.goal.cx, scene.goal.cy,
            scene.object_pose.x, scene.object_pose.y, scene.object_pose.angle,
            tuple((box.cx, box.cy, box.half_w, box.half_h) for box in scene.obstacles),
            tuple((pose.x, pose.y, pose.angle) for pose in scene.robot_poses),
        )

    changed_object = changed_robot = changed_map = 0
    comparisons = 0
    for family in FAMILIES:
        for mode in MODES:
            scenes = [sample_root("dev", family, mode, index).scene for index in range(args.roots_per_cell)]
            for i in range(len(scenes)):
                for j in range(i + 1, len(scenes)):
                    comparisons += 1
                    a, b = scenes[i], scenes[j]
                    changed_object += int(
                        a.object_pose != b.object_pose
                    )
                    changed_robot += int(a.robot_poses != b.robot_poses)
                    changed_map += int(
                        a.goal != b.goal or a.obstacles != b.obstacles or a.arena != b.arena
                    )
    report["checks"]["same_seed_reproducible"] = True
    report["checks"]["different_seed_pairs"] = comparisons
    report["checks"]["pairs_with_changed_object_pose"] = changed_object
    report["checks"]["pairs_with_changed_robot_poses"] = changed_robot
    report["checks"]["pairs_with_changed_map"] = changed_map
    assert changed_object == comparisons and changed_robot == comparisons
    assert changed_map >= 0.5 * comparisons, "map geometry rarely changes between seeds"

    # -- end-to-end smoke on every cell --------------------------------------
    registry.activate()
    smoke = {}
    model = JointWindowMLP(ModelConfig(hidden=64))
    from coimagine.continuous2d.skills import Hold

    hold_pair = (Hold(), Hold())
    for family in FAMILIES:
        for mode in MODES:
            root = sample_root("dev", family, mode, 0)
            scene = scene_by_name(root.scene_id)
            assert scene is not None
            env = Continuous2DEnv(config)
            env.reset(scene)
            observation = env.observation()
            boundary = _boundary(env)
            features = encode_input(scene, boundary, hold_pair, config.physics)
            assert len(features) == INPUT_DIM
            prediction = predict_window(model, scene, boundary, hold_pair, config.physics)
            violation = geometric_violation(
                prediction.next_state.object, prediction.next_state.robots, scene, config.physics
            )
            planner = Continuous2DEnv(config)
            snapshot = env.export_state()
            pool = candidate_pool(observation, config, 1, seed=0, cap=4)
            evaluation = evaluate_sequence_exact(
                planner, snapshot, scene, pool[0], config, candidate_id=0
            )
            payload = collect_root(root, config)
            smoke[f"{family}/{mode}"] = {
                "scene_id": root.scene_id,
                "encoded_dim": len(features),
                "prediction_finite": bool(violation["finite"]),
                "prediction_penetration": float(violation["penetration"]),
                "exact_reference_J": float(evaluation.j),
                "exact_reference_exit": evaluation.exit_class,
                "collected_windows": payload["stats"]["windows_total"],
                "collected_roots": 1,
            }
    report["smoke"] = smoke
    report["checks"]["end_to_end_smoke"] = True

    # -- demo figure, one root per cell --------------------------------------
    figure, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for axis, family in zip(axes, FAMILIES):
        for subplot, mode in zip(axis, MODES):
            root = sample_root("dev", family, mode, 0)
            _draw(subplot, root.scene)
            subplot.set_title(f"{family} / {mode}\n{root.scene_id}", fontsize=9)
    figure.suptitle("coverage_scale_v1 dev calibration: one root per family x mode cell (57-d input geometry)")
    figure.tight_layout()
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "A_calibration_roots.png", dpi=130)
    plt.close(figure)

    report["acceptance_table"] = {
        cell: {"mean_attempts": row["mean_attempts"], "acceptance_per_attempt": row["acceptance_per_attempt"]}
        for cell, row in report["cells"].items()
    }
    report["status"] = "CALIBRATION_OK"
    (run_dir / "calibration.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    print(json.dumps(report["acceptance_table"], indent=1))
    print("CALIBRATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
