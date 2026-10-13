"""Phase A mechanism animation: the same initial state under the old and the new settle protocol.

    python experiments/option_prediction_mechanism_v1/make_animation_a.py \
        --root test/noise/... --variant noise \
        --old-tag w0_test --new-tag w020_test

Two rows, two columns. The top row is the world as the *truth audit* sees it (object, both robots
and the surface segment each robot actually has in its local scan). The bottom row is the settle
evidence over time: the on-board estimate each robot reads against the deployed 0.02 m/s line,
and, as a dashed line, the rotation-aware true surface normal speed that the audit uses. The grey
band marks the 0.30 s hold window that each protocol certified, so the difference between
"the robots' own evidence" and "ground truth" stays visible instead of being asserted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import imageio_ffmpeg  # noqa: E402
import matplotlib.animation as animation  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

OBJECT_HALF = (0.5, 0.125)
ROBOT_RADIUS = 0.18
SETTLE_SURFACE = 0.02
SETTLE_HOLD = 0.30
CONTROL_DT = 0.05


def load_rows(run_dir: Path, tag: str, root_id: str, variant: str) -> List[dict]:
    path = run_dir / "logs" / tag / f"{root_id.replace('/', '_')}__{variant}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.open() if json.loads(line).get("arm") == "V"]


def true_surface_speed(row: dict, robot_id: int) -> Optional[float]:
    observed = (row.get("observed") or {}).get(str(robot_id)) or {}
    normal, own = observed.get("patch_normal"), observed.get("own")
    if normal is None or own is None:
        return None
    segment = observed.get("segment_world")
    if not segment:
        return None
    points = [segment[0], segment[1],
              [0.5 * (segment[0][0] + segment[1][0]), 0.5 * (segment[0][1] + segment[1][1])]]
    obj = row["true"]["object"]
    speeds = []
    for point in points:
        rx, ry = point[0] - obj[0], point[1] - obj[1]
        speeds.append(abs((obj[3] - obj[5] * ry) * normal[0] + (obj[4] + obj[5] * rx) * normal[1]))
    speeds.sort()
    return speeds[1]


def active_robots(row: dict) -> List[int]:
    return [robot_id for robot_id in (0, 1)
            if ((row.get("observed") or {}).get(str(robot_id)) or {}).get("own") is not None]


def draw_world(axis, row: dict, title: str) -> None:
    axis.clear()
    axis.set_title(title, fontsize=9)
    axis.set_aspect("equal")
    axis.set_xlim(0.0, 3.2)
    axis.set_ylim(0.0, 2.2)
    axis.set_xticks([])
    axis.set_yticks([])
    object_state = row["true"]["object"]
    cos, sin = math.cos(object_state[2]), math.sin(object_state[2])
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            dx, dy = sx * OBJECT_HALF[0], sy * OBJECT_HALF[1]
            corners.append((object_state[0] + dx * cos - dy * sin,
                            object_state[1] + dx * sin + dy * cos))
    axis.add_patch(patches.Polygon(corners, closed=True, fill=False, edgecolor="black", lw=1.2))
    for robot_id, state in enumerate(row["true"]["robots"]):
        axis.add_patch(patches.Circle((state[0], state[1]), ROBOT_RADIUS, fill=False,
                                      edgecolor=("tab:blue" if robot_id == 0 else "tab:orange"),
                                      lw=1.2))
        axis.plot([state[0]], [state[1]], ".", color=("tab:blue" if robot_id == 0
                                                      else "tab:orange"), ms=3)
    for robot_id in (0, 1):
        observed = (row.get("observed") or {}).get(str(robot_id)) or {}
        segment = observed.get("segment_world")
        if segment:
            axis.plot([segment[0][0], segment[1][0]], [segment[0][1], segment[1][1]],
                      "-", color=("tab:blue" if robot_id == 0 else "tab:orange"), lw=2.5,
                      alpha=0.85)
    axis.text(0.02, 0.02, f"t={row['time']:.2f}s  stage={row.get('stage')}",
              transform=axis.transAxes, fontsize=7)


def draw_series(axis, rows: List[dict], title: str, verdict_time: Optional[float]) -> None:
    axis.clear()
    axis.set_title(title, fontsize=9)
    times = [row["time"] for row in rows]
    for robot_id in (0, 1):
        estimates = [(row.get("observed") or {}).get(str(robot_id), {}).get("surface_speed")
                     for row in rows]
        axis.plot(times, [float("nan") if v is None else v for v in estimates], "-",
                  color=("tab:blue" if robot_id == 0 else "tab:orange"), lw=1.2,
                  label=f"robot{robot_id} on-board estimate")
        truths = [true_surface_speed(row, robot_id) for row in rows]
        axis.plot(times, [float("nan") if v is None else v for v in truths], "--",
                  color=("tab:blue" if robot_id == 0 else "tab:orange"), lw=0.9, alpha=0.6,
                  label=f"robot{robot_id} true surface speed")
    axis.axhline(SETTLE_SURFACE, color="crimson", lw=1.0, ls=":")
    axis.text(times[0], SETTLE_SURFACE, " threshold 0.02 m/s", color="crimson", fontsize=7,
              va="bottom")
    if verdict_time is not None:
        axis.axvspan(verdict_time - SETTLE_HOLD, verdict_time, color="grey", alpha=0.18)
        axis.axvline(verdict_time, color="green", lw=1.2)
        axis.text(verdict_time, axis.get_ylim()[1], " certified", color="green", fontsize=7,
                  va="top")
    axis.set_xlabel("time (s)", fontsize=8)
    axis.set_ylabel("normal speed (m/s)", fontsize=8)
    axis.set_ylim(0.0, max(0.12, axis.get_ylim()[1]))
    axis.tick_params(labelsize=7)
    axis.legend(fontsize=6, loc="upper right")


def verdict_time(rows: List[dict], settled: bool) -> Optional[float]:
    if not settled or len(rows) < 2:
        return None
    return rows[-1]["time"] + CONTROL_DT


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="outputs/option_prediction_mechanism_v1/"
                                             "option-prediction-mechanism-v1")
    parser.add_argument("--root", required=True)
    parser.add_argument("--variant", default="noise")
    parser.add_argument("--old-tag", default="w0_test")
    parser.add_argument("--new-tag", default="w020_test")
    parser.add_argument("--out", default="docs/option_prediction_mechanism_v1/"
                                        "A_settle_fix_old_vs_new.mp4")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    run_dir = ROOT / args.run_root
    records = {}
    for tag in (args.old_tag, args.new_tag):
        path = run_dir / f"executions_{tag}.jsonl"
        records[tag] = {(r["root_id"], r["variant"]): r
                        for r in (json.loads(line) for line in path.open())}
    key = (args.root, args.variant)
    old_rows = [row for row in load_rows(run_dir, args.old_tag, args.root, args.variant)
                if row.get("stage") in ("push", "brake", "settle")]
    new_rows = [row for row in load_rows(run_dir, args.new_tag, args.root, args.variant)
                if row.get("stage") in ("push", "brake", "settle")]
    if not old_rows or not new_rows:
        raise SystemExit("no logged rows for this case")
    old_settled = bool(records[args.old_tag][key]["metrics"]["settled_observed"])
    new_settled = bool(records[args.new_tag][key]["metrics"]["settled_observed"])
    frame_count = max(len(old_rows), len(new_rows))
    figure, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))

    def update(frame: int):
        old_row = old_rows[min(frame, len(old_rows) - 1)]
        new_row = new_rows[min(frame, len(new_rows) - 1)]
        draw_world(axes[0][0], old_row, f"OLD protocol (w=0) - truth view   "
                                        f"[{records[args.old_tag][key]['metrics']['exit_reason']}]")
        draw_world(axes[0][1], new_row, f"NEW protocol (w=0.20) - truth view   "
                                        f"[{records[args.new_tag][key]['metrics']['exit_reason']}]")
        draw_series(axes[1][0], old_rows[: min(frame + 1, len(old_rows))], "OLD: on-board vs truth",
                    verdict_time(old_rows, old_settled))
        draw_series(axes[1][1], new_rows[: min(frame + 1, len(new_rows))], "NEW: on-board vs truth",
                    verdict_time(new_rows, new_settled))
        figure.suptitle(f"{args.root} / {args.variant}: same initial state, same commands, "
                        f"same controller", fontsize=10)
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        return []

    writer = animation.FFMpegWriter(fps=args.fps, bitrate=2400)
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, str(target), dpi=110):
        for frame in range(frame_count):
            update(frame)
            writer.grab_frame()
    plt.close(figure)
    print(json.dumps({"out": str(target), "frames": frame_count,
                      "old_exit": records[args.old_tag][key]["metrics"]["exit_reason"],
                      "new_exit": records[args.new_tag][key]["metrics"]["exit_reason"]}, indent=1))
    print("ANIMATION_A_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
