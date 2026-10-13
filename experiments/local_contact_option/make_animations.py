"""C: the animation package — 8 cases x 4 arms, single videos, quads and previews.

    python experiments/local_contact_option/make_animations.py

Every frame is drawn **only** from the per-control-period log of a real execution. Nothing is
re-simulated, so a video cannot show a different trajectory from the one that was measured. The
simulation runs at 1x: the log samples every 50 ms and the videos are written at 20 fps, so
playback time equals simulated time. Arms F and O stop after 2 s, so their quad cell freezes on
the exit frame and is labelled ``ENDED at t=...`` while the other arms keep playing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import imageio_ffmpeg  # noqa: E402

from coimagine.local_contact_option.roots import CLASSES  # noqa: E402
from coimagine.local_contact_option.runner import root_from_metadata  # noqa: E402

matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
matplotlib.rcParams["font.size"] = 7

ARMS = ("O", "P", "F", "N")
#: The first line inside every cell: which arm this subplot is. This is the only thing that
#: identifies the four cells, so it must be short enough to stay legible at GIF scale.
ARM_LABEL = {
    "O": "O  old ctrl / 2 s window",
    "P": "P  old ctrl / staged (oracle)",
    "F": "F  new ctrl / 2 s clock",
    "N": "N  new ctrl / staged",
}
#: one shared caption per figure, instead of repeating it inside every cell
FIGURE_NOTE = ("arrows: red = local normal | green = reference velocity (0.26 m/s scale) "
               "| black = commanded force (15 N scale)\n"
               "O: no reference of its own -> N/A.  P: green is the protocol ramp, not tracked by "
               "the old formula.  Object velocity is evaluator truth.")
assert len(ARM_LABEL) == 4
ARM_COLOUR = {"O": "#1f77b4", "P": "#d62728", "F": "#2ca02c", "N": "#9467bd"}
ROBOT_COLOUR = ("#1668c1", "#17a589")
OBJECT_COLOUR = "#e8871a"
VIDEO_FPS = 20          # the log samples every 50 ms -> exactly 1x
FINAL_HOLD_FRAMES = VIDEO_FPS  # one second of display-only freeze, not simulated time


# ---------------------------------------------------------------------------
# case selection
# ---------------------------------------------------------------------------


def _terminal_deviation(baseline, variant) -> float:
    a = baseline["metrics"]["relative_object"]
    b = variant["metrics"]["relative_object"]
    return math.hypot(a[0] - b[0], a[1] - b[1])


def select_cases(records: List[Dict[str, object]], *, roots_meta: Dict[str, dict]) -> List[dict]:
    by_root_arm_variant = {(row["root_id"], row["arm"], row["variant"]): row for row in records}
    used: set = set()
    cases: List[dict] = []

    for interaction_class in CLASSES:
        candidates = sorted({
            row["root_id"] for row in records
            if row["interaction_class"] == interaction_class and row["variant"] == "baseline"
            and row["arm"] == "N" and row["root_id"] not in used
        })
        if not candidates:
            continue
        root_id = candidates[0]
        used.add(root_id)
        cases.append({
            "case": f"{len(cases) + 1}_{interaction_class}", "root_id": root_id, "variant": "baseline",
            "selection_rule": f"first legal {interaction_class} root by root_id, baseline variant",
            "selection_values": {"candidates": candidates[:4]},
        })

    # delay stress case
    scored = []
    for row in records:
        if row["arm"] != "N" or row["variant"] != "delay":
            continue
        baseline = by_root_arm_variant.get((row["root_id"], "N", "baseline"))
        if baseline is None:
            continue
        scored.append((_terminal_deviation(baseline, row), row["root_id"]))
    scored.sort(reverse=True)
    for deviation, root_id in scored:
        if root_id in used:
            continue
        used.add(root_id)
        cases.append({
            "case": f"{len(cases) + 1}_delay_stress", "root_id": root_id, "variant": "delay",
            "selection_rule": "largest N delay-vs-baseline terminal position deviation "
                              "(result-driven selection; not a substitute for the overall table)",
            "selection_values": {"deviation_m": deviation,
                                 "ranking": [{"root_id": r, "deviation_m": d}
                                             for d, r in scored[:5]]},
        })
        break

    # failure / divergence case
    def n_aborted(root_id: str) -> bool:
        row = by_root_arm_variant.get((root_id, "N", "baseline"))
        return bool(row and row["metrics"]["exit_reason"] not in ("settled_observed",))

    def p_settled(root_id: str) -> bool:
        row = by_root_arm_variant.get((root_id, "P", "baseline"))
        return bool(row and row["metrics"]["exit_reason"] == "settled_observed")

    rule = None
    chosen = None
    candidates = sorted({row["root_id"] for row in records if row["variant"] == "baseline"})
    for rule_name, predicate in (
        ("N aborted while P completed", lambda r: n_aborted(r) and p_settled(r)),
        ("any N abort", n_aborted),
    ):
        pool = [r for r in candidates if predicate(r) and r not in used]
        if pool:
            chosen, rule = pool[0], rule_name
            break
    if chosen is None:
        diffs = []
        for root_id in candidates:
            if root_id in used:
                continue
            n = by_root_arm_variant.get((root_id, "N", "baseline"))
            p = by_root_arm_variant.get((root_id, "P", "baseline"))
            if not n or not p:
                continue
            diffs.append((abs(n["metrics"]["rotation"] - p["metrics"]["rotation"]), root_id))
        diffs.sort(reverse=True)
        if diffs:
            chosen, rule = diffs[0][1], "largest N/P terminal rotation difference"
            cases.append({
                "case": f"{len(cases) + 1}_divergence", "root_id": chosen, "variant": "baseline",
                "selection_rule": rule + " (result-driven selection)",
                "selection_values": {"difference_rad": diffs[0][0]},
            })
    if chosen is not None and chosen not in used:
        used.add(chosen)
        cases.append({
            "case": f"{len(cases) + 1}_failure_divergence", "root_id": chosen,
            "variant": "baseline",
            "selection_rule": rule + " (result-driven selection; not a substitute for the overall table)",
            "selection_values": {"interaction_class": roots_meta[chosen]["interaction_class"]},
        })
    return cases


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------


def _cell_size(spec, rows: int, cols: int, *, scale: float = 1.0) -> Tuple[float, float]:
    """A figure whose cells match the arena aspect, so the drawing fills the cell.

    A 4:3 cell around a 3.2 x 2.2 m arena leaves most of the cell as empty margin and shrinks
    the robots to a few pixels.
    """

    width, height = spec.arena
    arena_aspect = (width + 0.12) / (height + 0.12)
    cell_h = 4.6 * scale
    cell_w = cell_h * arena_aspect
    # the cell has no axes title (the first text line carries it), so only a hair of extra
    # height is needed for the frame and the shared caption below
    return (cols * cell_w + 0.3, rows * (cell_h * 1.02) + 0.95)


def _scene(root_meta: dict):
    spec = root_from_metadata(root_meta).scene()
    return spec


def _object_corners(state) -> List[Tuple[float, float]]:
    x, y, angle = state[0], state[1], state[2]
    cos, sin = math.cos(angle), math.sin(angle)
    return [(x + dx * cos - dy * sin, y + dx * sin + dy * cos)
            for dx, dy in ((-0.5, -0.125), (0.5, -0.125), (0.5, 0.125), (-0.5, 0.125))]


def draw_frame(axis, spec, rows_so_far, row, title: str, ended_at: Optional[float],
               trajectories: Dict[str, List[Tuple[float, float]]], initial_outline: List[Tuple[float, float]],
               scale: float, fontsize: float = 8.0):
    axis.clear()
    width, height = spec.arena
    axis.add_patch(patches.Rectangle((0, 0), width, height, fill=False, lw=1.0, ec="black"))
    for box in spec.obstacles:
        axis.add_patch(patches.Rectangle(
            (box.cx - box.half_w, box.cy - box.half_h), 2 * box.half_w, 2 * box.half_h,
            angle=math.degrees(box.angle), facecolor="0.65", edgecolor="black", lw=0.6))
    axis.add_patch(patches.Polygon(initial_outline, closed=True, fill=False, lw=0.8,
                                   ls="--", ec=OBJECT_COLOUR, alpha=0.45))
    object_state = row["true"]["object"]
    axis.add_patch(patches.Polygon(_object_corners(object_state), closed=True,
                                   facecolor=OBJECT_COLOUR, alpha=0.85, edgecolor="black", lw=0.7))
    centroid = [point[0] for point in trajectories["object"]], [point[1] for point in trajectories["object"]]
    axis.plot(centroid[0], centroid[1], color=OBJECT_COLOUR, lw=0.9, alpha=0.7)
    for index, robot in enumerate(row["true"]["robots"]):
        axis.plot([p[0] for p in trajectories[f"robot{index}"]],
                  [p[1] for p in trajectories[f"robot{index}"]],
                  color=ROBOT_COLOUR[index], lw=0.8, alpha=0.6)
        axis.add_patch(patches.Circle((robot[0], robot[1]), 0.18,
                                      facecolor=ROBOT_COLOUR[index], alpha=0.85, zorder=5))
        axis.annotate(f"r{index}", (robot[0], robot[1]), fontsize=6, ha="center", va="center",
                      color="white", zorder=6)
    observed = row.get("observed") or {}
    for key in ("0", "1"):
        entry = observed.get(key) or {}
        robot = row["true"]["robots"][int(key)]
        normal = entry.get("patch_normal")
        if normal:
            axis.arrow(robot[0], robot[1], normal[0] * 0.20, normal[1] * 0.20,
                       head_width=0.025, color="crimson", lw=0.9, zorder=7,
                       length_includes_head=True)
        reference = entry.get("reference")
        if reference and math.hypot(*reference) > 1e-6:
            axis.arrow(robot[0], robot[1], reference[0], reference[1],
                       head_width=0.025, color="darkgreen", lw=0.9, zorder=7,
                       length_includes_head=True)
        else:
            # only arms with no reference of their own keep the N/A label; the arm legend says which
            axis.annotate("v_ref N/A", (robot[0], robot[1] - 0.26), fontsize=5, color="darkgreen",
                          ha="center")
    for index, force in enumerate(row["force"]):
        robot = row["true"]["robots"][index]
        magnitude = math.hypot(*force)
        if magnitude > 1e-9:
            axis.arrow(robot[0], robot[1], force[0] / 15.0 * scale, force[1] / 15.0 * scale,
                       head_width=0.03, color="black", lw=1.0, zorder=8, length_includes_head=True)
    speed = math.hypot(object_state[3], object_state[4])
    axis.set_xlim(-0.06, width + 0.06)
    axis.set_ylim(-0.06, height + 0.06)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    stage = row.get("phase", row.get("stage"))
    contact = {key: bool((observed.get(key) or {}).get("contact")) for key in ("0", "1")}
    active = max((row.get("active_contact") or {}).get("0", 0.0),
                 (row.get("active_contact") or {}).get("1", 0.0))
    lines = [
        title,
        f"t={row['time']:.2f}s   {stage}   push={active:.2f}s",
        f"contact r0={int(contact['0'])} r1={int(contact['1'])}",
        f"|v|={speed:.4f}  |w|={abs(object_state[5]):.4f}",
    ]
    if ended_at is not None:
        lines.append(f"ENDED t={ended_at:.2f}s (frozen)")
    axis.text(0.012, 0.985, "\n".join(lines), transform=axis.transAxes, va="top", ha="left",
              fontsize=fontsize, linespacing=1.35, family="DejaVu Sans Mono",
              bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.6))


def render_case(case: dict, logs: Dict[str, List[dict]], root_meta: dict, out_dir: Path,
                figures_dir: Path, *, quads_only: bool = False) -> List[dict]:
    spec = _scene(root_meta)
    scale = 0.30
    widths = {arm: logs[arm][-1]["time"] for arm in ARMS if logs.get(arm)}
    total_time = max(widths.values()) if widths else 2.0
    initial_outline = _object_corners(logs["N"][0]["true"]["object"])
    entries = []

    def _series(arm):
        trajectories = {"object": [], "robot0": [], "robot1": []}
        frames = []
        for row in logs[arm]:
            trajectories["object"].append(tuple(row["true"]["object"][:2]))
            trajectories["robot0"].append(tuple(row["true"]["robots"][0][:2]))
            trajectories["robot1"].append(tuple(row["true"]["robots"][1][:2]))
            frames.append((row, {key: list(value) for key, value in trajectories.items()}))
        return frames

    # --- single-arm videos -------------------------------------------------
    for arm in ([] if quads_only else ARMS):
        if not logs.get(arm):
            continue
        frames = _series(arm)
        ended_at = None
        if logs[arm][-1]["time"] < total_time - 1e-9:
            ended_at = logs[arm][-1]["time"]
        cell = _cell_size(spec, 1, 1, scale=1.30)
        figure, axis = plt.subplots(figsize=cell, dpi=110)
        figure.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.115)
        figure.text(0.5, 0.018, FIGURE_NOTE, ha="center", va="bottom", fontsize=8.5,
                    color="0.25", linespacing=1.5)
        writer = animation.FFMpegWriter(fps=VIDEO_FPS,
                                        extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                    "-crf", "28"])
        path = out_dir / f"{case['case']}__{arm}.mp4"
        with writer.saving(figure, str(path), dpi=100):
            for row, trajectories in frames:
                draw_frame(axis, spec, None, row,
                           f"{case['case']}\n{ARM_LABEL[arm]}",
                           None, trajectories, initial_outline, scale, fontsize=10.5)
                writer.grab_frame()
            last_row, last_traj = frames[-1]
            for _ in range(FINAL_HOLD_FRAMES):
                draw_frame(axis, spec, None, last_row,
                           f"{case['case']}\n{ARM_LABEL[arm]}", ended_at, last_traj,
                           initial_outline, scale, fontsize=10.5)
                writer.grab_frame()
        plt.close(figure)
        entries.append({"case": case["case"], "arm": arm, "file": path.name, "kind": "single",
                        "seconds": last_row["time"]})

    # --- quad video --------------------------------------------------------
    quad_path = out_dir / f"{case['case']}__quad.mp4"
    cell = _cell_size(spec, 2, 2, scale=1.12)
    figure, axes = plt.subplots(2, 2, figsize=cell, dpi=110)
    figure.subplots_adjust(left=0.03, right=0.985, top=0.945, bottom=0.10,
                           hspace=0.14, wspace=0.06)
    figure.text(0.5, 0.012, FIGURE_NOTE, ha="center", va="bottom", fontsize=8.5,
                color="0.25", linespacing=1.5)
    writer = animation.FFMpegWriter(fps=VIDEO_FPS,
                                    extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                "-crf", "28"])
    series = {arm: _series(arm) for arm in ARMS if logs.get(arm)}
    n_frames = max(len(value) for value in series.values())
    with writer.saving(figure, str(quad_path), dpi=100):
        for index in range(n_frames + FINAL_HOLD_FRAMES):
            for axis, arm in zip(axes.ravel(), ARMS):
                if arm not in series:
                    axis.clear()
                    axis.axis("off")
                    continue
                frames = series[arm]
                position = min(index, len(frames) - 1)
                row, trajectories = frames[position]
                ended = frames[-1][1]["object"] and row["time"] >= frames[-1][0]["time"] - 1e-9
                ended_at = frames[-1][0]["time"] if ended and frames[-1][0]["time"] < total_time - 1e-9 else None
                draw_frame(axis, spec, None, row, ARM_LABEL[arm], ended_at, trajectories,
                           initial_outline, scale)
            writer.grab_frame()
    plt.close(figure)
    entries.append({"case": case["case"], "arm": "O/P/F/N", "file": quad_path.name, "kind": "quad",
                    "seconds": total_time})

    # --- preview GIF -------------------------------------------------------
    gif_path = figures_dir / f"{case['case']}__quad.gif"
    cell = _cell_size(spec, 2, 2, scale=1.02)
    figure, axes = plt.subplots(2, 2, figsize=cell, dpi=100)
    figure.subplots_adjust(left=0.03, right=0.985, top=0.945, bottom=0.10,
                           hspace=0.14, wspace=0.06)
    figure.text(0.5, 0.012, FIGURE_NOTE, ha="center", va="bottom", fontsize=8.5,
                color="0.25", linespacing=1.5)
    frames_out = []
    for index in range(0, n_frames, 2):   # 10 fps preview
        for axis, arm in zip(axes.ravel(), ARMS):
            if arm not in series:
                axis.clear()
                axis.axis("off")
                continue
            frames = series[arm]
            position = min(index, len(frames) - 1)
            row, trajectories = frames[position]
            ended_at = (frames[-1][0]["time"]
                        if row["time"] >= frames[-1][0]["time"] - 1e-9
                        and frames[-1][0]["time"] < total_time - 1e-9 else None)
            draw_frame(axis, spec, None, row, ARM_LABEL[arm], ended_at, trajectories,
                       initial_outline, scale)
        figure.canvas.draw()
        frames_out.append(_figure_to_array(figure))
    plt.close(figure)
    _write_gif(frames_out, gif_path, fps=10)
    entries.append({"case": case["case"], "arm": "O/P/F/N", "file": str(gif_path.name),
                    "kind": "preview_gif", "seconds": total_time})
    return entries


def _figure_to_array(figure):
    import numpy as np

    buffer = np.frombuffer(figure.canvas.buffer_rgba(), dtype="uint8")
    width, height = figure.canvas.get_width_height()
    return buffer.reshape((height, width, 4))[:, :, :3].copy()


def _write_gif(frames, path: Path, fps: int) -> None:
    """Write the preview GIF with Pillow, which matplotlib already depends on."""

    from PIL import Image

    # a preview must stay light: quantise to a small adaptive palette so the file does not grow
    # with the (now larger) frame size
    images = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=64)
              for frame in frames]
    if not images:
        return
    images[0].save(str(path), save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


# ---------------------------------------------------------------------------
# offline index
# ---------------------------------------------------------------------------


INDEX_TEMPLATE = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8">
<title>local_contact_option_repair_v2 animations</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
 .case {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 18px 0; padding: 12px; }}
 .case h2 {{ margin: 4px 0 8px; font-size: 16px; }}
 .meta {{ color: #555; font-size: 13px; margin-bottom: 8px; }}
 video {{ width: 100%; max-width: 960px; background: #000; }}
 .row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
 .row figure {{ margin: 0; }}
 .row video {{ max-width: 320px; }}
 table {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
 th, td {{ border: 1px solid #ddd; padding: 3px 6px; }}
</style></head><body>
<h1>local_contact_option_repair_v2 — 8 案例 × O/P/F/N</h1>
<p>每个案例同一 root、同一变体、同一初态，四臂同相机。视频为 1× 仿真时间（日志 50 ms 采样，20 fps 导出）。
F/O 的 2 s 窗口结束后终帧冻结并标注 ENDED；其它臂继续播放。物体速度是评估叠加，控制器不可见。</p>
{cards}
</body></html>
"""


def write_index(cases: List[dict], entries: List[dict], index_rows: List[dict], out_dir: Path) -> None:
    cards = []
    for case in cases:
        name = case["case"]
        quad = next((e for e in entries if e["case"] == name and e["kind"] == "quad"), None)
        gif = next((e for e in entries if e["case"] == name and e["kind"] == "preview_gif"), None)
        singles = [e for e in entries if e["case"] == name and e["kind"] == "single"]
        rows = [row for row in index_rows if row["case"] == name]
        table = ["<table><tr><th>arm</th><th>exit</th><th>displacement (m)</th>"
                 "<th>rotation (rad)</th><th>seconds</th><th>normal completion</th><th>video</th></tr>"]
        for row in sorted(rows, key=lambda r: ARMS.index(r["arm"])):
            table.append(
                f"<tr><td>{row['arm']}</td><td>{row['exit_reason']}</td>"
                f"<td>{row['displacement_m']:.4f}</td><td>{row['rotation_rad']:.4f}</td>"
                f"<td>{row['seconds']:.2f}</td><td>{row['normal_completion']}</td>"
                f"<td><a href=\"{row['single_video']}\">{row['arm']}.mp4</a></td></tr>")
        table.append("</table>")
        quad_rel = quad["file"].replace("figures/", "") if quad else None
        gif_rel = gif["file"].replace("figures/", "") if gif else None
        cards.append(
            f"<div class='case'><h2>{name}</h2>"
            f"<div class='meta'>root <code>{case['root_id']}</code> · variant <code>{case['variant']}</code> · "
            f"initial-state hash <code>{index_rows[0]['initial_state_hash']}</code><br>"
            f"选例规则：{case['selection_rule']}</div>"
            + (f"<video controls preload='none' src='figures/quads/{quad['file']}'></video>" if quad else "")
            + (f"<p><img src='figures/gif/{gif['file']}' alt='preview' style='max-width:640px'></p>" if gif else "")
            + "<div class='row'>" + "".join(
                f"<figure><figcaption>{e['arm']}</figcaption>"
                f"<video controls preload='none' src='figures/singles/{e['file']}'></video></figure>"
                for e in singles) + "</div>"
            + "".join(table) + "</div>")
    (out_dir / "index.html").write_text(INDEX_TEMPLATE.format(cards="\n".join(cards)))


GUIDE = """# 动画观看指南

## 四宫格布局（固定）

四张子图是**同一个案例**：同一 root、同一变体、同一初态、同一相机，只是控制方式不同。
每个格子第一行就是臂名，位置也固定：

| 位置 | 臂 | 控制方式 |
|---|---|---|
| 左上 | **O** | 旧控制 + 旧 2 s 窗口（历史路径，无制动概念） |
| 右上 | **P** | 旧控制 + 新分阶段协议（**oracle 基线**，公式读物体真值） |
| 左下 | **F** | 新局部控制 + 2 s 总墙钟（含接近，无终端稳定） |
| 右下 | **N** | 新局部控制 + 新分阶段协议 |

O/F 在 2 s 截止后终帧冻结（标 `ENDED`），P/N 继续播放；四格时间轴不同步是真实的，
没有把不同时长压缩成同一播放时长。

所有视频都是 **1× 仿真时间**（日志每 50 ms 采样，20 fps 导出）。物体速度、接触状态是评估
叠加，控制器看不到。F 与 O 在 2 s 窗口结束后冻结终帧并标 `ENDED at t=...`，Quad 里其它臂继续。

建议先看这 3 例：

1. **1_solo_central__quad.mp4** —— 单机器人中央正推。看四格的时间轴差异：O/F 在 2 s 截止，
   P/N 走完握手→2 s 有效推动→制动→判稳。N 的终帧物体速度明显低于 O。
2. **7_delay_stress__quad.mp4** —— 100 ms 观测延迟下 N 终态偏差最大的 root。看 N 与 F 在
   延迟下是否仍能建立并保持接触，以及 `settle_unobserved` 如何出现（延迟帧不能无限用于判稳）。
3. **8_failure_divergence__quad.mp4** —— 结果驱动的失败/分歧案例：N 中止而 P 完成的 root。
   看 N 是 `contact_lost` 还是 `settle_timeout`，以及 P 在同样初态下为什么没有中止。

其余案例：2 偏心推（转动与滑动）、3 双机器人同侧、4 双机器人旋转协作、5 不兼容输入
（预期失败）、6 受限场景。每个案例都给出 O/P/F/N 四个单臂视频，文件名 `<case>__<arm>.mp4`。

看的时候建议关注：phase 标签、有效接触推动时间 `push=`、两机器人接触布尔量、物体 `|v|`/`|w|`、
以及红/绿/黑三种箭头（局部法向 / 参考速度 / 实际施力；旧控制没有参考速度时标 `v_ref N/A`）。
机器人圆盘上的箭头不是 yaw 控制，本轮没有转向执行器。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-repair-v2")
    parser.add_argument("--out-root", default="outputs/local_contact_option_repair_v2")
    parser.add_argument("--docs-dir", default="docs/local_contact_option_repair_v2")
    parser.add_argument("--only", default=None, help="render only this case prefix")
    parser.add_argument("--quads-only", action="store_true",
                        help="skip the 32 single-arm videos and render only the quads + previews")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for name in ("singles", "quads", "gif"):
        (figures_dir / name).mkdir(exist_ok=True)

    records = [json.loads(line) for line in (run_dir / "executions_test.jsonl").open()]
    roots_meta = {row["root_id"]: row for row in json.loads((run_dir / "roots_test.json").read_text())["roots"]}
    cases = select_cases(records, roots_meta=roots_meta)
    if args.only:
        cases = [case for case in cases if case["case"].startswith(args.only)]
    print(f"selected {len(cases)} cases", flush=True)

    index_rows: List[dict] = []
    entries: List[dict] = []
    for case in cases:
        root_id = case["root_id"]
        variant = case["variant"]
        log_path = run_dir / "logs" / "test" / f"{root_id.replace('/', '_')}__{variant}.jsonl"
        logs: Dict[str, List[dict]] = {arm: [] for arm in ARMS}
        for line in log_path.open():
            row = json.loads(line)
            logs.setdefault(row["arm"], []).append(row)
        root_meta = roots_meta[root_id]
        produced = render_case(case, logs, root_meta, figures_dir / "singles", figures_dir / "gif",
                               quads_only=args.quads_only)
        # move the quad next to the singles for a self-contained folder
        for entry in produced:
            if entry["kind"] == "quad":
                source = figures_dir / "singles" / entry["file"]
                target = figures_dir / "quads" / entry["file"]
                if source.exists():
                    shutil.move(str(source), str(target))
        entries.extend(produced)
        initial_hash = hashlib.sha256(json.dumps(
            {"root": root_meta, "variant": variant}, sort_keys=True).encode()).hexdigest()[:16]
        for arm in ARMS:
            row = next((r for r in records
                        if r["root_id"] == root_id and r["arm"] == arm and r["variant"] == variant), None)
            if row is None:
                continue
            index_rows.append({
                "case": case["case"], "root_id": root_id, "variant": variant,
                "initial_state_hash": initial_hash, "arm": arm,
                "exit_reason": row["metrics"]["exit_reason"],
                "displacement_m": row["metrics"]["displacement"],
                "rotation_rad": row["metrics"]["rotation"],
                "seconds": row["metrics"]["wall_clock_seconds"],
                "normal_completion": row["metrics"].get("push_completed")
                and row["metrics"]["settled_observed"],
                "single_video": f"figures/singles/{case['case']}__{arm}.mp4",
                "quad_video": f"figures/quads/{case['case']}__quad.mp4",
                "preview_gif": f"figures/gif/{case['case']}__quad.gif",
                "log": f"logs/test/{root_id.replace('/', '_')}__{variant}.jsonl",
            })
        print(f"  {case['case']} done", flush=True)

    with (run_dir / "animation_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    (run_dir / "animation_index.json").write_text(json.dumps({
        "cases": cases, "rows": index_rows, "entries": entries,
        "fps": VIDEO_FPS, "frame_hold_frames": FINAL_HOLD_FRAMES,
        "real_time": "1x (50 ms log sampling, 20 fps export)",
        "single_videos": sum(1 for e in entries if e["kind"] == "single"),
        "quad_videos": sum(1 for e in entries if e["kind"] == "quad"),
        "preview_gifs": sum(1 for e in entries if e["kind"] == "preview_gif"),
    }, indent=1, sort_keys=True))
    write_index(cases, entries, index_rows, run_dir)
    docs = ROOT / args.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "ANIMATION_GUIDE.md").write_text(GUIDE)
    print(json.dumps({"cases": len(cases), "singles": sum(1 for e in entries if e["kind"] == "single"),
                      "quads": sum(1 for e in entries if e["kind"] == "quad"),
                      "gifs": sum(1 for e in entries if e["kind"] == "preview_gif")}, indent=1))
    print("ANIMATIONS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
