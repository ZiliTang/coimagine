"""Layered animation of a logged episode: truth, observation and decision geometry.

Three layers are drawn from the **same log** (the animation is never a re-simulation):

* **world truth** -- the real arena, obstacle, object, robots, trajectories and goal. Every
  quantity that the controller may not read is marked ``diagnostic truth`` in grey/dashed or
  confined to the task panel.
* **observation** -- what each robot's own scanner returned this period: the rays, the nearest
  hit, the no-return rays and the measured hit class. The teammate's shared scan is drawn
  separately, dimmer and dashed, and labelled as the teammate's.
* **decision geometry** -- the quantities the controller actually used this period: the fitted
  surface points, the estimated normal/tangent, the gap and its target, the push direction,
  the reposition direction and accumulated progress, the nearest measured obstacle, the
  avoidance direction *if it was enabled*, and the applied versus desired command.

The task/time panel repeats the clocks, the two skills, the goal distance and the low-speed
hold timer, with the truth-only quantities labelled as such.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

from .lidar import CATEGORY_NONE, CATEGORY_OBJECT, CATEGORY_STATIC, CATEGORY_TEAMMATE
from .spec import (
    CONTROL_DT,
    LIDAR_RANGE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    ROBOT_RADIUS,
    TIME_BUDGET_SECONDS,
    SKILL_BUDGET,
)

for _candidate in ("/home/tzl/.local/share/fonts/wqy.ttc", "/tmp/wqy.ttc", "/tmp/cjk.ttf"):
    if Path(_candidate).exists():
        font_manager.fontManager.addfont(_candidate)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=_candidate).get_name(), "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

COLORS = {
    "arena": "#333333", "obstacle": "#4a4a4a", "object": "#c98b4b", "object_edge": "#6d4c1b",
    "goal": "#bfe3c0", "goal_edge": "#2e7d32", "truth": "#9e9e9e",
    "r0": "#4c78c9", "r1": "#c94c6e",
    "ray_object": "#e08a2e", "ray_static": "#7a7a7a", "ray_teammate": "#a04cc9",
    "ray_none": "#d9d9d9",
    "patch": "#ff2d95", "normal": "#00a86b", "tangent": "#8b5cf6",
    "desired": "#1e88e5", "applied": "#d81b60", "avoid": "#ff9800",
}


def _object_polygon(pose: Sequence[float]) -> np.ndarray:
    half_width, half_height = OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0
    corners = np.asarray([[-half_width, -half_height], [half_width, -half_height],
                          [half_width, half_height], [-half_width, half_height]])
    angle = float(pose[2])
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    return corners @ rotation.T + np.asarray(pose[:2])


def _arrow(axis, start, vector, color, label=None, scale=1.0, linewidth=1.4, linestyle="-"):
    if vector is None:
        return
    norm = math.hypot(vector[0], vector[1])
    if norm < 1e-6:
        return
    axis.annotate("", xy=(start[0] + vector[0] * scale, start[1] + vector[1] * scale),
                  xytext=start,
                  arrowprops=dict(arrowstyle="-|>", color=color, linewidth=linewidth,
                                  linestyle=linestyle, shrinkA=0.0, shrinkB=0.0))
    if label:
        axis.text(start[0] + vector[0] * scale, start[1] + vector[1] * scale, label,
                  color=color, fontsize=6.5)


def _blend(first: str, second: str, weight: float = 0.35) -> Tuple[float, float, float]:
    """Blend two hex colours, used to tint a scan by the robot that owns it."""

    def rgb(value: str) -> Tuple[float, float, float]:
        value = value.lstrip("#")
        return tuple(int(value[index:index + 2], 16) / 255.0 for index in (0, 2, 4))  # type: ignore

    a, b = rgb(first), rgb(second)
    return tuple(a[index] * (1.0 - weight) + b[index] * weight for index in range(3))


def _ray_endpoints(row: Dict[str, object], robot_id: int) -> Tuple[List[Tuple[float, float]],
                                                                   List[Tuple[float, float]]]:
    """(hit segments, no-return segments) in world coordinates for one robot's own scan."""

    scan = row[f"r{robot_id}"]["scan"]
    origin = scan["origin"]
    heading = scan["heading"]
    hits: List[Tuple[float, float]] = []
    misses: List[Tuple[float, float]] = []
    for index, distance in enumerate(scan["distance"]):
        angle = heading + 2.0 * math.pi * index / len(scan["distance"])
        if scan["valid"][index]:
            hits.append((origin[0], origin[1], origin[0] + distance * math.cos(angle),
                         origin[1] + distance * math.sin(angle), scan["category"][index]))
        else:
            misses.append((origin[0], origin[1], origin[0] + LIDAR_RANGE * math.cos(angle),
                           origin[1] + LIDAR_RANGE * math.sin(angle), CATEGORY_NONE))
    return hits, misses


#: A clip is capped at this many frames: the log stays complete, the video is sampled.
MAX_CLIP_FRAMES = 180


def render_frame(rows: Sequence[Dict[str, object]], index: int, meta: Dict[str, object],
                 layers: Tuple[str, ...] = ("world", "observation", "decision")) -> np.ndarray:
    """One RGB frame of the animation, drawn from the log rows only."""

    row = rows[index]
    figure = plt.figure(figsize=(11.2, 4.8), dpi=95)
    axis = figure.add_axes([0.02, 0.06, 0.58, 0.88])
    panel = figure.add_axes([0.63, 0.06, 0.35, 0.88])
    panel.axis("off")

    width, height = meta["arena"]
    # ---- world truth layer ------------------------------------------------
    axis.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor=COLORS["arena"],
                             linewidth=1.4))
    for corner in ((0.0, 0.0),):
        axis.plot([corner[0]], [corner[1]], marker=".", color=COLORS["arena"], ms=2)
    for box in meta["obstacles"]:
        axis.add_patch(Rectangle((box["cx"] - box["half_w"], box["cy"] - box["half_h"]),
                                 2 * box["half_w"], 2 * box["half_h"],
                                 color=COLORS["obstacle"], zorder=1.5))
    goal = row["goal"]
    axis.add_patch(Circle((goal[0], goal[1]), row["goal_radius"],
                          facecolor=COLORS["goal"], edgecolor=COLORS["goal_edge"],
                          alpha=0.85, linewidth=1.3, zorder=1))
    axis.text(goal[0], goal[1], "goal", ha="center", va="center", fontsize=7,
              color=COLORS["goal_edge"], zorder=1)
    true_object = row["true"]["object"]
    axis.add_patch(Polygon(_object_polygon(true_object), closed=True,
                           facecolor=COLORS["object"], edgecolor=COLORS["object_edge"],
                           linewidth=1.1, zorder=3))
    # true trajectories up to now, dashed and grey: diagnostic truth
    for robot_id in (0, 1):
        path = np.asarray([rows[i]["true"]["robots"][robot_id][:2] for i in range(index + 1)])
        if len(path) > 1:
            axis.plot(path[:, 0], path[:, 1], linestyle="--", linewidth=0.9,
                      color=COLORS["truth"], zorder=2)
    for robot_id in (0, 1):
        pose = row["true"]["robots"][robot_id]
        color = COLORS["r0"] if robot_id == 0 else COLORS["r1"]
        axis.add_patch(Circle((pose[0], pose[1]), ROBOT_RADIUS, facecolor=color,
                              edgecolor="#1b2b4c", linewidth=0.9, zorder=5, alpha=0.85))
        axis.plot([pose[0], pose[0] + 0.18 * math.cos(pose[2])],
                  [pose[1], pose[1] + 0.18 * math.sin(pose[2])], color="#1b2b4c",
                  linewidth=1.0, zorder=6)
        axis.text(pose[0] - 0.10, pose[1] + 0.22, f"r{robot_id}", fontsize=7, color=color,
                  zorder=6)
    object_velocity = (true_object[3], true_object[4])
    _arrow(axis, (true_object[0], true_object[1]), object_velocity, COLORS["truth"],
           "v_true", scale=1.0, linestyle="--")

    # ---- observation layer ------------------------------------------------
    if "observation" in layers:
        for robot_id in (0, 1):
            hits, misses = _ray_endpoints(row, robot_id)
            owner = COLORS["r0"] if robot_id == 0 else COLORS["r1"]
            for x0, y0, x1, y1, category in misses:
                axis.plot([x0, x1], [y0, y1],
                          color=_blend(COLORS["ray_none"], owner, 0.30),
                          linewidth=0.30, zorder=1.2)
            for x0, y0, x1, y1, category in hits:
                color = {CATEGORY_OBJECT: COLORS["ray_object"],
                         CATEGORY_STATIC: COLORS["ray_static"],
                         CATEGORY_TEAMMATE: COLORS["ray_teammate"]}.get(
                             category, COLORS["ray_none"])
                axis.plot([x0, x1], [y0, y1], color=_blend(color, owner, 0.35),
                          linewidth=0.55, alpha=0.9, zorder=4)
                axis.plot([x1], [y1], marker="o", ms=1.3,
                          color=_blend(color, owner, 0.35), zorder=4.5)

    # ---- decision geometry layer -----------------------------------------
    if "decision" in layers:
        for robot_id in (0, 1):
            info = row[f"r{robot_id}"]
            color = COLORS["r0"] if robot_id == 0 else COLORS["r1"]
            points = info["patch_points_world"]
            if points:
                array = np.asarray(points)
                axis.plot(array[:, 0], array[:, 1], linestyle="none", marker="x", ms=2.6,
                          color=COLORS["patch"], zorder=7)
            pose = row["true"]["robots"][robot_id]
            origin = (pose[0], pose[1])
            normal = info.get("outward_normal") or info.get("normal_world")
            if normal is not None and info.get("normal_valid"):
                length = 0.35
                _arrow(axis, origin, normal, COLORS["normal"], "n_out", scale=length)
            tangent = info.get("tangent_world")
            if tangent is not None:
                _arrow(axis, origin, tangent, COLORS["tangent"], "tangent", scale=0.35)
            desired = info.get("desired_velocity")
            if desired is not None:
                _arrow(axis, origin, desired, COLORS["desired"], "cmd", scale=1.1)
            force = info.get("force")
            if force is not None:
                norm = math.hypot(force[0], force[1]) or 1.0
                _arrow(axis, origin, (force[0] / norm, force[1] / norm), COLORS["applied"],
                       "F", scale=0.45)
            if info.get("avoidance_active") and info.get("avoidance_direction"):
                _arrow(axis, origin, info["avoidance_direction"], COLORS["avoid"],
                       "avoid", scale=0.30)
            nearest = info.get("obstacle_nearest_point")
            if nearest is not None:
                axis.plot([nearest[0]], [nearest[1]], marker="v", ms=3.2,
                          color=COLORS["ray_static"], zorder=6)
            if info.get("contacting_object"):
                axis.plot([pose[0]], [pose[1]], marker="o", ms=12.5, mfc="none",
                          mec="#111111", mew=1.0, zorder=6)
    axis.set_xlim(-0.06, width + 0.06)
    axis.set_ylim(-0.06, height + 0.06)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{meta['name']}  |  {meta['map']}  |  {meta['title']}", fontsize=8.5)
    # a scale bar, so distances in the picture are readable
    axis.plot([0.15, 0.65], [-0.02, -0.02], color="#111111", linewidth=2.0,
              clip_on=False)
    axis.text(0.40, -0.055, "0.5 m", ha="center", fontsize=6.5, clip_on=False)

    _draw_panel(panel, rows, index, meta, layers)
    figure.canvas.draw()
    frame = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(figure)
    return frame


def _draw_panel(panel, rows: Sequence[Dict[str, object]], index: int, meta: Dict[str, object],
                layers: Tuple[str, ...]) -> None:
    row = rows[index]
    lines: List[str] = []
    total = len(rows)
    elapsed = row["time"]
    lines.append(f"t = {elapsed:6.2f} s   tick {row['tick']}   period {row['period']}/{total}")
    lines.append(f"joint skill {row['skill_index']}   budget left "
                 f"{max(0.0, TIME_BUDGET_SECONDS - elapsed):5.1f} s   skills "
                 f"{row['skill_index']}/{SKILL_BUDGET}")
    lines.append(f"joint action: {row['joint_action_label']}")
    lines.append("")
    for robot_id in (0, 1):
        info = row[f"r{robot_id}"]
        color = COLORS["r0"] if robot_id == 0 else COLORS["r1"]
        lines.append(f"r{robot_id}  {info['skill']}")
        lines.append(f"   stage={info['stage']}  exit={info['exit_reason']}  "
                     f"t_skill={_fmt(info['skill_time'])}s")
        if info.get("progress") is not None:
            lines.append(f"   progress={_fmt(info['progress'])}/"
                         f"{_fmt(info['progress_target'])}  "
                         f"no_progress={_fmt(info['no_progress_seconds'])}s")
        if info.get("effective_contact_seconds") is not None:
            lines.append(f"   effective contact={_fmt(info['effective_contact_seconds'])}s "
                         f"(target 2.00)  contact={int(bool(info['contacting_object']))}")
        if info.get("gap") is not None:
            lines.append(f"   gap={_fmt(info['gap'])}  target={_fmt(info['gap_target'])}  "
                         f"err={_fmt(info['gap_error'])}")
        lines.append(f"   patch: valid={int(bool(info['patch_valid']))} "
                     f"rays={info['patch_arc_rays']} rms={_fmt(info['patch_rms'],4)} "
                     f"conf={_fmt(info['patch_confidence'],2)}")
        if info.get("tangent_direction"):
            lines.append(f"   direction={info['tangent_direction']}  "
                         f"corner_events={info['corner_events']}")
        lines.append(f"   cmd={_vec(info.get('desired_velocity'))}  "
                     f"F={_vec(info.get('force'))} sat={int(bool(info.get('force_saturated')))}")
        if info.get("avoidance_active"):
            lines.append(f"   AVOIDING: dir={_vec(info.get('avoidance_direction'))}  "
                         f"nearest static={_fmt(info.get('obstacle_nearest_distance'))} m")
        lines.append("")
    lines.append("TASK (diagnostic truth, not controller input)")
    lines.append(f"   object->goal distance {row['true_goal_distance']:.3f} m   "
                 f"hold {row['hold_seconds']:.2f}/0.50 s   "
                 f"linear {row['success_conditions']['linear']:.4f} m/s   "
                 f"angular {row['success_conditions']['angular']:.4f} rad/s")
    lines.append(f"   penetration {row['penetration']:.5f} m")
    lines.append("")
    lines.append("grey dashed = true trajectory / true object velocity")
    if "observation" in layers:
        lines.append("rays are each robot's OWN scan, tinted by owner; colour = hit class")
    if "decision" in layers:
        lines.append("pink x = fitted surface returns; green = n_out; violet = tangent")
    panel.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.0,
               family="monospace", transform=panel.transAxes)

    curve = panel.inset_axes([0.02, 0.02, 0.96, 0.20])
    distances = [r["true_goal_distance"] for r in rows[: index + 1]]
    curve.plot(np.arange(len(distances)) * CONTROL_DT, distances, color="#c2185b",
               linewidth=1.0)
    curve.axhline(0.20, color="#2e7d32", linewidth=0.8, linestyle=":")
    curve.set_xlim(0, max(5.0, len(rows) * CONTROL_DT))
    curve.set_ylim(0, max(0.6, max(distances) * 1.1))
    curve.tick_params(labelsize=5.5)
    curve.set_title("true object-goal distance (m), dotted = 0.20 m success band",
                    fontsize=6)


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _vec(value) -> str:
    if value is None:
        return "-"
    return f"({float(value[0]):+.3f},{float(value[1]):+.3f})"


def keyframe_report(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Fixed-rule key frames: initial state, first sighting, first contact, corner, avoidance,
    every skill exit, and the last frame."""

    picks: List[Dict[str, object]] = []

    def add(index: int, label: str, diagnosis: str) -> None:
        if 0 <= index < len(rows) and all(pick["period"] != rows[index]["period"] for pick in picks):
            picks.append({"period": rows[index]["period"], "index": index, "label": label,
                          "diagnosis": diagnosis})

    add(0, "initial_state", "random start: object, both robots, goal and obstacle off to work")
    seen = next((i for i, row in enumerate(rows)
                 if row["r0"]["patch_valid"] or row["r1"]["patch_valid"]), None)
    if seen is not None:
        add(seen, "first_object_sighting",
            "the scanner first returns object-class hits and a surface fragment is fitted"
            + (" (already visible in the initial observation)"
               if seen == 0 else ""))
    contact = next((i for i, row in enumerate(rows)
                    if row["r0"]["contacting_object"] or row["r1"]["contacting_object"]), None)
    if contact is not None:
        add(contact, "first_contact", "the contact sensor first reports a robot-object touch")
    corner = next((i for i, row in enumerate(rows)
                   if row["r0"]["corner_event"] or row["r1"]["corner_event"]), None)
    if corner is not None:
        add(corner, "corner_transition", "the fitted normal jumped: the robot is passing a corner "
                                         "and progress accumulation pauses for that period")
    avoid = next((i for i, row in enumerate(rows)
                  if row["r0"]["avoidance_active"] or row["r1"]["avoidance_active"]), None)
    if avoid is not None:
        add(avoid, "avoidance_enabled", "a measured obstacle/teammate is inside the avoidance "
                                        "margin and the repulsion is actually applied")
    last_skill = None
    for i, row in enumerate(rows):
        if row["skill_index"] != last_skill:
            last_skill = row["skill_index"]
            if i > 0:
                previous = rows[i - 1]
                reasons = [f"r0={previous['r0']['exit_reason']}", f"r1={previous['r1']['exit_reason']}"]
                add(i - 1, f"skill_{row['skill_index']}_boundary",
                    f"joint boundary after skill {row['skill_index'] - 1} (" + ", ".join(reasons) + ")")
    add(len(rows) - 1, "final_frame",
        "last logged legal frame; the episode's terminal state is this frame")
    picks.sort(key=lambda pick: pick["index"])
    return picks


def frame_indices(row_count: int, step: int) -> List[int]:
    """The exact frame list a clip uses, so a log can be written against the same frames."""

    step = max(step, int(math.ceil(row_count / MAX_CLIP_FRAMES)))
    indices = list(range(0, row_count, step))
    if indices[-1] != row_count - 1:
        indices.append(row_count - 1)
    return indices


def write_frame_log(rows: Sequence[Dict[str, object]], indices: Sequence[int], out_dir: Path,
                    stem: str) -> Path:
    """Per-frame measurement/control/event log for the frames the video actually shows."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = []
    for index in indices:
        row = rows[index]
        payload.append({
            "tick": row["tick"], "time": row["time"], "period": row["period"],
            "skill_index": row["skill_index"], "joint_action": row["joint_action_label"],
            "forces": row["forces"],
            "true_goal_distance": row["true_goal_distance"],
            "hold_seconds": row["hold_seconds"],
            "robots": {
                str(robot_id): {
                    "skill": row[f"r{robot_id}"]["skill"], "stage": row[f"r{robot_id}"]["stage"],
                    "exit_reason": row[f"r{robot_id}"]["exit_reason"],
                    "desired_velocity": row[f"r{robot_id}"]["desired_velocity"],
                    "reference_velocity": row[f"r{robot_id}"]["reference_velocity"],
                    "force": row[f"r{robot_id}"]["force"],
                    "force_saturated": row[f"r{robot_id}"]["force_saturated"],
                    "gap": row[f"r{robot_id}"]["gap"],
                    "gap_target": row[f"r{robot_id}"]["gap_target"],
                    "progress": row[f"r{robot_id}"]["progress"],
                    "progress_target": row[f"r{robot_id}"]["progress_target"],
                    "patch_valid": row[f"r{robot_id}"]["patch_valid"],
                    "patch_arc_rays": row[f"r{robot_id}"]["patch_arc_rays"],
                    "contacting_object": row[f"r{robot_id}"]["contacting_object"],
                    "avoidance_active": row[f"r{robot_id}"]["avoidance_active"],
                    "tangent_direction": row[f"r{robot_id}"]["tangent_direction"],
                    "scan_distance": row[f"r{robot_id}"]["scan"]["distance"],
                    "scan_valid": row[f"r{robot_id}"]["scan"]["valid"],
                    "scan_category": row[f"r{robot_id}"]["scan"]["category"],
                } for robot_id in (0, 1)
            },
        })
    path = out_dir / f"{stem}_frames.json"
    path.write_text(json.dumps({"frames": payload, "count": len(payload)}, indent=0))
    return path


def write_html_replay(out_dir: Path, stem: str, title: str,
                      keyframes: Optional[Sequence[Dict[str, object]]] = None) -> Path:
    """A single-file replay: the physics clip and the overlay clip, synchronised, plus a table."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    physics = f"{stem}_physics.mp4"
    overlay = f"{stem}_overlay.mp4"
    frames_file = f"{stem}_frames.json"
    keyframe_html = "".join(
        f'<figure><img src="{Path(str(entry["path"])).name}" width="320">'
        f'<figcaption>t={entry["period"]}: {entry["label"]} — {entry["diagnosis"]}</figcaption>'
        f'</figure>' for entry in (keyframes or []) if entry.get("path"))
    html = f"""<!doctype html>
<meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 1rem; }}
 video {{ width: 48%; vertical-align: top; background: #000; }}
 table {{ border-collapse: collapse; font-size: 11px; }}
 th, td {{ border: 1px solid #ccc; padding: 2px 4px; }}
 figure {{ display: inline-block; margin: 4px; font-size: 11px; }}
</style>
<h1 style="font-size:16px">{title}</h1>
<p style="font-size:12px">
 <button onclick="setPlay(true)">play both</button>
 <button onclick="setPlay(false)">pause both</button>
 <label><input type="checkbox" id="showPhysics" checked> physics clip</label>
 <label><input type="checkbox" id="showOverlay" checked> observation+decision overlay</label>
 <input type="range" id="seek" min="0" max="1000" value="0" style="width:40%">
 <span id="clock" style="font-size:12px"></span>
</p>
<video id="physics" src="{physics}" muted loop></video>
<video id="overlay" src="{overlay}" muted loop></video>
<p style="font-size:12px">The overlay adds the measured rays, the fitted surface, the estimated
normal/tangent, the gap, the commands actually applied, the avoidance direction when it is
enabled, and the skill stages — all read from <code>{frames_file}</code>, one row per shown
frame. Grey dashed geometry is diagnostic truth and is not a controller input.</p>
<h2 style="font-size:14px">key frames</h2>
{keyframe_html}
<h2 style="font-size:14px">per-frame log</h2>
<div id="table" style="max-height:40vh; overflow:auto"></div>
<script>
const physics = document.getElementById('physics');
const overlay = document.getElementById('overlay');
const seek = document.getElementById('seek');
const clock = document.getElementById('clock');
const table = document.getElementById('table');
function setPlay(state) {{ physics.paused === state && (state ? physics.play() : physics.pause());
                           state ? overlay.play() : overlay.pause(); }}
document.getElementById('showPhysics').onchange = e => {{ physics.style.display = e.target.checked ? '' : 'none'; }};
document.getElementById('showOverlay').onchange = e => {{ overlay.style.display = e.target.checked ? '' : 'none'; }};
function sync(source, target) {{ source.addEventListener('timeupdate', () => {{
    if (Math.abs(target.currentTime - source.currentTime) > 0.08) target.currentTime = source.currentTime;
    seek.value = 1000 * (source.currentTime / (source.duration || 1));
    clock.textContent = source.currentTime.toFixed(2) + ' s'; }}); }}
sync(physics, overlay); sync(overlay, physics);
seek.oninput = () => {{ const t = seek.value / 1000 * (physics.duration || 0);
                        physics.currentTime = t; overlay.currentTime = t; }};
fetch('{frames_file}').then(r => r.json()).then(data => {{
  const head = ['time','skill','joint action','r0 gap','r0 progress','r0 contact','r1 gap','r1 progress','r1 contact','goal distance (truth)'];
  const rows = data.frames.map(f => [f.time.toFixed(2), f.skill_index, f.joint_action,
      (f.robots['0'].gap ?? 0).toFixed(3), (f.robots['0'].progress ?? 0).toFixed(3), f.robots['0'].contacting_object,
      (f.robots['1'].gap ?? 0).toFixed(3), (f.robots['1'].progress ?? 0).toFixed(3), f.robots['1'].contacting_object,
      f.true_goal_distance.toFixed(3)]);
  table.innerHTML = '<table><tr>' + head.map(h => '<th>' + h + '</th>').join('') + '</tr>'
    + rows.map(r => '<tr>' + r.map(c => '<td>' + c + '</td>').join('') + '</tr>').join('') + '</table>';
}});
</script>
"""
    path = out_dir / f"{stem}_replay.html"
    path.write_text(html)
    return path


def render_episode(rows: Sequence[Dict[str, object]], meta: Dict[str, object], out_dir: Path,
                   stem: str, *, fps: int = 20, step: int = 1, gif: bool = True) -> Dict[str, object]:
    """Write the physics-only clip, the observation+decision clip, key frames and a GIF."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    indices = frame_indices(len(rows), step)
    outputs: Dict[str, object] = {"episode": meta["name"], "frames": len(indices)}
    outputs["frames_log"] = str(write_frame_log(rows, indices, out_dir, stem))
    outputs["replay_html"] = str(write_html_replay(out_dir, stem, str(meta.get("title", stem))))

    for mode, layers in (("physics", ("world",)),
                         ("overlay", ("world", "observation", "decision"))):
        frames = [render_frame(rows, index, meta, layers) for index in indices]
        path = out_dir / f"{stem}_{mode}.mp4"
        try:
            import imageio_ffmpeg
            writer = imageio_ffmpeg.write_frames(str(path), (frames[0].shape[1],
                                                             frames[0].shape[0]), fps=fps)
            writer.send(None)
            for frame in frames:
                writer.send(frame.tobytes())
            writer.close()
            outputs[f"{mode}_mp4"] = str(path)
        except Exception as error:  # pragma: no cover - ffmpeg absence is reported, not hidden
            outputs[f"{mode}_mp4"] = None
            outputs[f"{mode}_error"] = repr(error)
        if mode == "overlay" and gif:
            from PIL import Image
            preview = [Image.fromarray(frame).resize((frames[0].shape[1] // 2,
                                                      frames[0].shape[0] // 2))
                       for frame in frames[:: max(1, len(frames) // 60)]]
            gif_path = out_dir / f"{stem}_overlay.gif"
            preview[0].save(gif_path, save_all=True, append_images=preview[1:],
                            duration=int(1000 / max(1, fps // max(1, len(frames) // 60))), loop=0)
            outputs["gif"] = str(gif_path)
    return outputs


def write_keyframes(rows: Sequence[Dict[str, object]], meta: Dict[str, object], out_dir: Path,
                    stem: str) -> List[Dict[str, object]]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    picks = keyframe_report(rows)
    from PIL import Image

    for pick in picks:
        frame = render_frame(rows, pick["index"], meta,
                             ("world", "observation", "decision"))
        path = out_dir / f"{stem}_key_{pick['period']:05d}_{pick['label']}.png"
        Image.fromarray(frame).save(path)
        pick["path"] = str(path)
    (out_dir / f"{stem}_keyframes.json").write_text(json.dumps(picks, indent=1))
    # the replay page embeds the key frames, so it is rewritten now that they exist
    write_html_replay(out_dir, stem, stem, picks)
    return picks
