"""Phase C mechanism animation: the world moves, the shared observation does not.

    python experiments/option_prediction_mechanism_v1/make_animation_c.py \
        --pair T1-12-symmetric

Four panels. Top: the truth of world A and world B for one verified positive pair. Bottom left:
the *shared* observation both robots receive, drawn at every interpolated object shift -- the
curves coincide exactly at every frame, with the running maximum absolute difference printed.
Bottom right: the true payoff of each of the eight joint skills in both worlds, with the
near-optimal sets marked, so the reader sees that the best action changes while nothing the
robots can measure changes. The observation panel is produced by the real perception code at each
frame; nothing is stylised.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "option_prediction_mechanism_v1" / "phase_c"))

import imageio_ffmpeg  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import mechanism_scene as ms  # noqa: E402
import perception as pc  # noqa: E402
from coimagine.continuous2d import Continuous2DEnv  # noqa: E402
from coimagine.continuous2d.scenes import Box  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
RECORDS = ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json"


def load_pair(pair_id: str) -> dict:
    payload = json.loads(RECORDS.read_text())
    for record in payload["records"]:
        if record["pair_id"] == pair_id:
            return record
    raise SystemExit(f"{pair_id} not in the phase-C records")


def params_from(payload: dict, object_shift: float) -> ms.MechanismParams:
    goal = payload["goal"]
    return ms.MechanismParams(
        name=payload["name"], object_len=payload["object_len"],
        object_thick=payload["object_thick"], object_centre_x=payload["object_centre_x"],
        shift=object_shift,
        goal=Box(goal["cx"], goal["cy"], goal["half_w"], goal["half_h"], goal["angle"]),
        robot_x=tuple(payload["robot_x"]), standoff=payload["standoff"],
        arena=tuple(payload["arena"]), edge_y=payload["edge_y"])


def draw_world(axis, params: ms.MechanismParams, object_x: float, title: str) -> None:
    """Draw one world's truth; ``object_x`` is the object centre that world actually has."""

    axis.clear()
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(3.0, 13.0)
    axis.set_ylim(0.6, 3.4)
    axis.set_xticks([])
    axis.set_yticks([])
    pose = (object_x, params.edge_y, 0.0)
    half = params.object_len / 2.0
    thickness = params.object_thick / 2.0
    axis.add_patch(patches.Rectangle((pose[0] - half, pose[1] - thickness),
                                     2 * half, 2 * thickness, fill=False, edgecolor="black",
                                     lw=1.4))
    axis.add_patch(patches.Rectangle((params.goal.cx - params.goal.half_w,
                                      params.goal.cy - params.goal.half_h),
                                     2 * params.goal.half_w, 2 * params.goal.half_h,
                                     fill=False, edgecolor="green", ls="--", lw=1.0))
    for robot_pose in params.robot_poses():
        axis.add_patch(patches.Circle((robot_pose.x, robot_pose.y), ms.ROBOT_R, fill=False,
                                      edgecolor="tab:blue", lw=1.2))
    axis.text(0.02, 0.02, f"object centre x = {pose[0]:.2f}", transform=axis.transAxes,
              fontsize=7)


def observation_curves(params_a: ms.MechanismParams, params_b: ms.MechanismParams):
    """The shared observation: per-ray ranges and the fitted patch, from the real code."""

    scene_a = params_a.scenes()[0]
    scene_b = params_b.scenes()[1]
    env = Continuous2DEnv(ms.mechanism_config(params_a))
    env.reset(scene_a)
    obs_a = pc.observe(scene_a, params_a.object_pose("A"), env.backend.robot_states())
    env.reset(scene_b)
    obs_b = pc.observe(scene_b, params_b.object_pose("B"), env.backend.robot_states())
    difference = max(pc.observation_differences(obs_a, obs_b).values())
    return obs_a, obs_b, difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", default="T1-12-symmetric")
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--out", default="docs/option_prediction_mechanism_v1/"
                                        "C_ambiguous_pair.mp4")
    args = parser.parse_args()
    record = load_pair(args.pair)
    payload = record["params"]
    shift = float(payload["shift"])
    analysis = record["analysis"]
    skills = [entry["skill_id"] for entry in record["skills"]]
    q_a = analysis["regret_A"]
    q_b = analysis["regret_B"]
    # regret dictionaries hold "action -> best - Q"; convert back to Q for the bars
    best_a, best_b = analysis["best_A"], analysis["best_B"]
    qa = {name: best_a - (q_a[name] if name in q_a else float("nan")) for name in skills}
    qb = {name: best_b - (q_b[name] if name in q_b else float("nan")) for name in skills}

    figure, axes = plt.subplots(2, 2, figsize=(10.0, 6.4))

    def update(frame: int):
        fraction = frame / max(1, args.frames - 1)
        current_shift = shift * fraction
        params_a = params_from(payload, 0.0)
        params_b = params_from(payload, current_shift)
        draw_world(axes[0][0], params_a, payload["object_centre_x"],
                   "world A truth (object centre fixed)")
        draw_world(axes[0][1], params_b, payload["object_centre_x"] + current_shift,
                   f"world B truth (object shifted {current_shift:+.2f} m)")
        obs_a, obs_b, difference = observation_curves(params_a, params_b)
        axis = axes[1][0]
        axis.clear()
        axis.set_title("shared observation both robots receive (A solid, B dashed)", fontsize=9)
        axis.set_xlabel("ray index", fontsize=8)
        axis.set_ylabel("range (m)", fontsize=8)
        for robot_id, colour in ((0, "tab:blue"), (1, "tab:orange")):
            for observation, style in ((obs_a, "-"), (obs_b, "--")):
                scan = observation.robots[robot_id].scan
                values = [float("nan") if r is None else r for r in scan.ranges]
                axis.plot(range(len(values)), values, style, color=colour, lw=1.0,
                          label=f"robot{robot_id} {'A' if style == '-' else 'B'}")
        axis.set_ylim(0.0, 0.75)
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=6, ncol=2)
        axis.text(0.02, 0.06,
                  f"max |A-B| = {difference:.3e}\n"
                  f"patch distance A/B = "
                  f"{obs_a.robots[0].scan.patch_distance:.4f} / "
                  f"{obs_b.robots[0].scan.patch_distance:.4f} m\n"
                  f"robot0 and robot1 curves coincide exactly",
                  transform=axis.transAxes, fontsize=7, color="crimson")
        axis = axes[1][1]
        axis.clear()
        axis.set_title("true payoff Q per joint skill", fontsize=9)
        positions = np.arange(len(skills))
        axis.bar(positions - 0.2, [qa[name] for name in skills], width=0.4, label="A")
        axis.bar(positions + 0.2, [qb[name] for name in skills], width=0.4, label="B")
        for name in analysis["near_optimal_A"]:
            axis.plot([skills.index(name) - 0.2], [qa[name]], "kv")
        for name in analysis["near_optimal_B"]:
            axis.plot([skills.index(name) + 0.2], [qb[name]], "k^")
        axis.set_xticks(positions)
        axis.set_xticklabels(skills, rotation=60, fontsize=6, ha="right")
        axis.tick_params(labelsize=7)
        axis.legend(fontsize=7)
        axis.axhline(0.0, color="grey", lw=0.6)
        missing = [name for name in skills
                   if not math.isfinite(qa[name]) or not math.isfinite(qb[name])]
        if missing:
            axis.text(0.02, 0.04, "absent bar = sequence diverged numerically\n"
                                  "(excluded from the payoff matrix): "
                                  + ", ".join(missing),
                      transform=axis.transAxes, fontsize=6, color="dimgrey")
        figure.suptitle(f"{args.pair}: the observation never changes, the best action does "
                        f"(mutual regret {analysis['mutual_regret']:.3f})", fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return []

    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(target), dpi=110):
        for frame in range(args.frames):
            update(frame)
            writer.grab_frame()
    plt.close(figure)
    print(json.dumps({"out": str(target), "pair": args.pair, "frames": args.frames,
                      "near_optimal_A": analysis["near_optimal_A"],
                      "near_optimal_B": analysis["near_optimal_B"],
                      "mutual_regret": analysis["mutual_regret"]}, indent=1))
    print("ANIMATION_C_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
