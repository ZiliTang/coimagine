"""S1: sample and freeze the random-start instance panels, and draw the scene atlas.

Deliverables
    configs/instances_<map>_<split>.json   frozen instance lists (with a content hash)
    results/scene_sampling.json            rejection statistics, realised distributions
    results/scene_legality.json            independent re-check of the section-2.2 contract
    videos/atlas_<map>.png                 12 fixed instances per map, with a scale bar
    videos/coverage_<map>.png              spatial coverage of robots and goal bearings
    videos/lidar_<map>.png                 one 360-degree scan drawn from the returns alone
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import geometry as G  # noqa: E402
from coimagine.random_start import layout, lidar  # noqa: E402
from coimagine.random_start.spec import (  # noqa: E402
    LIDAR_RANGE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    ROBOT_RADIUS,
    SPLIT_SIZES,
)

for _candidate in ("/home/tzl/.local/share/fonts/wqy.ttc", "/tmp/wqy.ttc"):
    if Path(_candidate).exists():
        font_manager.fontManager.addfont(_candidate)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_candidate).get_name(), "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "outputs/random_start_reposition_v1"


def _object_polygon(instance) -> np.ndarray:
    return np.asarray(G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                    instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                    OBJECT_HEIGHT / 2.0))


def _draw_scene(axis, instance, title: str, scale_bar: bool = True) -> None:
    width, height = instance.arena
    axis.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="#333333",
                             linewidth=1.2))
    for box in instance.obstacles:
        axis.add_patch(Rectangle((box.cx - box.half_w, box.cy - box.half_h),
                                 2 * box.half_w, 2 * box.half_h, color="#4a4a4a"))
    axis.add_patch(Circle(instance.goal_centre, instance.goal_radius, facecolor="#bfe3c0",
                          edgecolor="#2e7d32", alpha=0.85, linewidth=1.1))
    axis.add_patch(Polygon(_object_polygon(instance), closed=True, facecolor="#c98b4b",
                           edgecolor="#6d4c1b", linewidth=1.0))
    for robot_id, pose in enumerate(instance.robot_poses):
        color = "#4c78c9" if robot_id == 0 else "#c94c6e"
        axis.add_patch(Circle((pose.x, pose.y), ROBOT_RADIUS, facecolor=color,
                              edgecolor="#1b2b4c", linewidth=0.8))
        axis.plot([pose.x, pose.x + 0.18 * math.cos(pose.angle)],
                  [pose.y, pose.y + 0.18 * math.sin(pose.angle)], color="#1b2b4c", linewidth=0.9)
        axis.text(pose.x - 0.10, pose.y + 0.22, f"r{robot_id}", fontsize=6.5, color=color)
    if scale_bar:
        axis.plot([0.15, 0.65], [-0.03, -0.03], color="#111111", linewidth=2.0, clip_on=False)
        axis.text(0.40, -0.075, "0.5 m", ha="center", fontsize=6.0, clip_on=False)
    axis.set_xlim(-0.08, width + 0.08)
    axis.set_ylim(-0.10, height + 0.08)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(title, fontsize=7.5)


def atlas(instances: Sequence, path: Path, columns: int = 4) -> Dict[str, object]:
    rows = int(math.ceil(len(instances) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(3.0 * columns, 2.2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for axis, instance in zip(axes, instances):
        _draw_scene(axis, instance, f"{instance.name}  goal {math.dist(instance.goal_centre, (instance.object_pose.x, instance.object_pose.y)):.2f} m")
    for axis in axes[len(instances):]:
        axis.axis("off")
    figure.suptitle("fixed random-start instances (object = orange, robots = blue/red, "
                    "goal = green)", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {"path": str(path), "instances": [instance.name for instance in instances]}


def coverage_figure(instances: Sequence, stats: Dict[str, object], path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.2, 6.2))
    gaps, bearings, goal_bearings, goal_distances = [], [], [], []
    for instance in instances:
        object_poly = G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                    instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                    OBJECT_HEIGHT / 2.0)
        for pose in instance.robot_poses:
            gaps.append(G.circle_polygon_gap((pose.x, pose.y), ROBOT_RADIUS, object_poly))
            bearings.append(math.atan2(pose.y - instance.object_pose.y,
                                       pose.x - instance.object_pose.x))
        goal_bearings.append(math.atan2(instance.goal_centre[1] - instance.object_pose.y,
                                        instance.goal_centre[0] - instance.object_pose.x))
        goal_distances.append(math.dist(instance.goal_centre,
                                        (instance.object_pose.x, instance.object_pose.y)))
    axes[0][0].hist(gaps, bins=24, color="#4c78c9")
    axes[0][0].set_title("robot-object gap (m), contract [0.12, 0.50]")
    axes[0][0].axvline(0.12, color="#c62828", linestyle="--", linewidth=0.9)
    axes[0][0].axvline(0.50, color="#c62828", linestyle="--", linewidth=0.9)
    axes[0][1].hist(np.degrees(bearings), bins=24, color="#4c78c9")
    axes[0][1].set_title("robot bearing about the object (deg)")
    axes[0][2].hist(np.degrees(goal_bearings), bins=24, color="#2e7d32")
    axes[0][2].set_title("goal bearing about the object (deg)")
    axes[1][0].hist(goal_distances, bins=24, color="#2e7d32")
    axes[1][0].set_title("object-goal distance (m), contract [0.80, 1.20]")
    axes[1][0].axvline(0.80, color="#c62828", linestyle="--", linewidth=0.9)
    axes[1][0].axvline(1.20, color="#c62828", linestyle="--", linewidth=0.9)
    axes[1][1].scatter(np.degrees(bearings[: len(instances)]),
                       np.degrees(goal_bearings), s=6, color="#333333")
    axes[1][1].set_title("goal bearing vs robot 0 bearing (deg)")
    axes[1][2].plot(np.degrees(goal_bearings), np.degrees(bearings[: len(instances)]),
                    linestyle="none", marker=".", ms=3, color="#4c78c9")
    axes[1][2].set_title("robot 0 bearing vs goal bearing (deg)")
    for row in axes:
        for axis in row:
            axis.tick_params(labelsize=7)
    figure.suptitle(f"realised sampling coverage ({len(instances)} instances); "
                    "no angle is claimed to be exactly uniform", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)


def lidar_figure(instance, path: Path, robot_id: int = 0) -> Dict[str, object]:
    robots = [(pose.x, pose.y, pose.angle, 0.0, 0.0, 0.0) for pose in instance.robot_poses]
    obstacles = tuple(G.box_polygon(box.cx, box.cy, box.angle, box.half_w, box.half_h)
                      for box in instance.obstacles)
    scan = lidar.scan_lidar(instance.arena, obstacles,
                            (instance.object_pose.x, instance.object_pose.y,
                             instance.object_pose.angle), robots, robot_id, 0, 0.0)
    patches = lidar.scan_patches(scan, (robots[robot_id][0], robots[robot_id][1]))
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    colors = {lidar.CATEGORY_OBJECT: "#e08a2e", lidar.CATEGORY_STATIC: "#7a7a7a",
              lidar.CATEGORY_TEAMMATE: "#a04cc9"}
    for axis, title in ((axes[0], "world view (diagnostic truth)"),
                        (axes[1], "returned scan only (what the robot has)")):
        _draw_scene(axis, instance, title, scale_bar=(axis is axes[0]))
        if axis is axes[0]:
            axis.add_patch(Circle((robots[robot_id][0], robots[robot_id][1]), LIDAR_RANGE,
                                  fill=False, edgecolor="#9e9e9e", linestyle=":", linewidth=0.8))
        for index, distance in enumerate(scan.distances):
            angle = scan.ray_angles[index]
            if not scan.valid[index]:
                if axis is axes[1]:
                    axis.plot([scan.origin[0], scan.origin[0] + LIDAR_RANGE * math.cos(angle)],
                              [scan.origin[1], scan.origin[1] + LIDAR_RANGE * math.sin(angle)],
                              color="#dddddd", linewidth=0.4)
                continue
            end = (scan.origin[0] + distance * math.cos(angle),
                   scan.origin[1] + distance * math.sin(angle))
            color = colors.get(scan.category[index], "#999999")
            axis.plot([scan.origin[0], end[0]], [scan.origin[1], end[1]], color=color,
                      linewidth=0.6 if axis is axes[1] else 0.9,
                      alpha=0.75 if axis is axes[0] else 1.0)
            if axis is axes[1]:
                axis.plot([end[0]], [end[1]], marker="o", ms=1.4, color=color)
        for patch in patches:
            if patch.valid:
                array = np.asarray(patch.points_world)
                axis.plot(array[:, 0], array[:, 1], linestyle="none", marker="x", ms=3.0,
                          color="#ff2d95")
                _arrow(axis, (robots[robot_id][0], robots[robot_id][1]),
                       patch.normal_world, "#00a86b", "fitted n")
    figure.suptitle(f"{instance.name}: 72 rays, 1.20 m, one scan per 0.05 s control period; "
                    "class is an observable: object / static / teammate / no return", fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=110)
    plt.close(figure)
    return {"path": str(path), "valid_returns": int(sum(scan.valid)),
            "object_returns": int(sum(1 for i, v in enumerate(scan.valid)
                                      if v and scan.category[i] == lidar.CATEGORY_OBJECT)),
            "patches": [{"distance": patch.distance, "rays": patch.arc_rays,
                         "rms": patch.rms_m, "valid": patch.valid} for patch in patches]}


def _arrow(axis, start, vector, color, label=None, scale=0.35):
    norm = math.hypot(vector[0], vector[1]) or 1.0
    axis.annotate("", xy=(start[0] + vector[0] * scale, start[1] + vector[1] * scale),
                  xytext=start, arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.1))
    if label:
        axis.text(start[0] + vector[0] * scale, start[1] + vector[1] * scale, label,
                  color=color, fontsize=6)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--small", action="store_true", help="8/2/2 per split (smoke test)")
    args = parser.parse_args()
    out = Path(args.out)
    (out / "configs").mkdir(parents=True, exist_ok=True)
    (out / "results").mkdir(parents=True, exist_ok=True)
    counts = {"train": 2, "validation": 1, "test": 1} if args.small else None
    report: Dict[str, object] = {"maps": {}, "env_version": "random-start-reposition-v1",
                                 "split_sizes": SPLIT_SIZES}
    for map_name in ("L0", "L1"):
        report["maps"][map_name] = {}
        for split in ("train", "validation", "test"):
            count = None if counts is None else counts[split]
            instances, sampling = layout.build_split(map_name, split, count=count)
            payload = [instance.as_dict() for instance in instances]
            text = json.dumps(payload, indent=1, sort_keys=True)
            path = out / "configs" / f"instances_{map_name}_{split}.json"
            path.write_text(text)
            report["maps"][map_name][split] = {
                "sampling": sampling,
                "legality": layout.scene_legality_report(instances),
                "statistics": layout.instance_statistics(instances),
                "manifest": layout.split_manifest(instances),
                "config_path": str(path),
            }
            if split == "train":
                report["maps"][map_name]["atlas"] = atlas(
                    instances[:12], out / "videos" / f"atlas_{map_name}.png")
                coverage_figure(instances, report["maps"][map_name][split]["statistics"],
                                out / "videos" / f"coverage_{map_name}.png")
                report["maps"][map_name]["lidar"] = lidar_figure(
                    instances[0], out / "videos" / f"lidar_{map_name}.png")
    (out / "results" / "scene_sampling.json").write_text(json.dumps(report, indent=1))
    legality = {map_name: {split: report["maps"][map_name][split]["legality"]["ok"]
                           for split in ("train", "validation", "test")}
                for map_name in ("L0", "L1")}
    print(json.dumps({"legality_ok": legality,
                      "atlas": report["maps"]["L0"]["atlas"]["path"]}, indent=1))
    print("SCENES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
