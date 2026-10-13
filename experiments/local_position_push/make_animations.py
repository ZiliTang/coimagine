"""D: V/X animations — 16 single-arm videos, 8 side-by-side videos, 8 preview GIFs, offline index.

    python experiments/local_position_push/make_animations.py

Every frame comes from the per-control-period log of a recorded execution; nothing is
re-simulated. Playback is 1x: the log samples every 50 ms and the videos are written at 20 fps.

Arm X additionally draws its reference geometry -- the surface point ``q``, the contact-centre
reference ``p_contact``, the advance reference ``p_ref`` and the robot centre ``p`` -- with a thin
line for the position error. ``p_ref`` sits inside the object by construction, so it is drawn as
an open marker with a dotted connector and the legend says it is a virtual reference, not a
physical robot. Arm V has no position target of its own and is labelled N/A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
matplotlib.rcParams["font.size"] = 8

ARMS = ("V", "X")
ARM_LABEL = {
    "V": "V  velocity push (frozen v3)",
    "X": "X  surface-position push",
}
ROBOT_COLOUR = ("#1668c1", "#17a589")
OBJECT_COLOUR = "#e8871a"
FPS = 20
HOLD_FRAMES = FPS
FORCE_SCALE = 0.30      # 15 N -> 0.30 m, fixed for every frame
REFERENCE_SCALE = 0.26  # 0.26 m/s -> 0.26 m, fixed for every frame
NOTE = ("arrows: red = local normal | green = reference velocity (0.26 m/s scale) | "
        "black = commanded force (15 N scale)\n"
        "X markers: q = surface point, p_contact = no-penetration contact centre, "
        "p_ref = advance reference (virtual, inside the object), thin line = position error. "
        "V has no position target of its own: N/A.")


def _corners(state) -> List[Tuple[float, float]]:
    x, y, angle = state[0], state[1], state[2]
    cos, sin = math.cos(angle), math.sin(angle)
    return [(x + dx * cos - dy * sin, y + dx * sin + dy * cos)
            for dx, dy in ((-0.5, -0.125), (0.5, -0.125), (0.5, 0.125), (-0.5, 0.125))]


def select_cases(records: List[dict], split: str) -> List[dict]:
    baseline = [row for row in records if row["variant"] == "baseline"]
    by_key = {(row["root_id"], row["arm"]): row for row in baseline}
    used: set = set()
    cases: List[dict] = []
    for interaction_class in CLASSES:
        candidates = sorted({row["root_id"] for row in baseline
                             if row["interaction_class"] == interaction_class
                             and row["root_id"] not in used})
        if not candidates:
            continue
        used.add(candidates[0])
        cases.append({"case": f"{len(cases) + 1}_{interaction_class}",
                      "root_id": candidates[0], "variant": "baseline",
                      "rule": f"first {interaction_class} root by root_id (baseline variant)"})
    # largest V/X difference
    scored = []
    for root_id in sorted({row["root_id"] for row in baseline}):
        v, x = by_key.get((root_id, "V")), by_key.get((root_id, "X"))
        if v is None or x is None:
            continue
        scored.append((abs(x["metrics"]["displacement"] - v["metrics"]["displacement"]), root_id))
    scored.sort(reverse=True)
    for difference, root_id in scored:
        if root_id in used:
            continue
        used.add(root_id)
        cases.append({"case": f"{len(cases) + 1}_max_difference", "root_id": root_id,
                      "variant": "baseline",
                      "rule": "largest |X - V| baseline displacement (result-driven, for display "
                              "only; it does not replace the aggregate tables)",
                      "values": {"difference_m": difference}})
        break
    # failure / abort
    def failed(arm: str, root_id: str) -> bool:
        row = by_key.get((root_id, arm))
        return bool(row and row["metrics"]["exit_reason"] != "settled_observed")

    chosen = None
    for rule, predicate in (
        ("X aborted while V settled", lambda root: failed("X", root) and not failed("V", root)),
        ("any X abort", lambda root: failed("X", root)),
    ):
        pool = [root_id for _difference, root_id in scored
                if root_id not in used and predicate(root_id)]
        if pool:
            chosen = sorted(pool)[0]
            cases.append({"case": f"{len(cases) + 1}_failure", "root_id": chosen,
                          "variant": "baseline", "rule": f"{rule} (result-driven)"})
            used.add(chosen)
            break
    return cases


def draw(axis, spec, row, title, ended_at, trajectories, initial_outline, fontsize=9.0):
    axis.clear()
    width, height = spec.arena
    axis.add_patch(patches.Rectangle((0, 0), width, height, fill=False, lw=1.0, ec="black"))
    for box in spec.obstacles:
        axis.add_patch(patches.Rectangle(
            (box.cx - box.half_w, box.cy - box.half_h), 2 * box.half_w, 2 * box.half_h,
            angle=math.degrees(box.angle), facecolor="0.65", edgecolor="black", lw=0.6))
    axis.add_patch(patches.Polygon(initial_outline, closed=True, fill=False, lw=0.8, ls="--",
                                   ec=OBJECT_COLOUR, alpha=0.45))
    object_state = row["true"]["object"]
    axis.add_patch(patches.Polygon(_corners(object_state), closed=True, facecolor=OBJECT_COLOUR,
                                   alpha=0.85, edgecolor="black", lw=0.7))
    axis.plot([p[0] for p in trajectories["object"]], [p[1] for p in trajectories["object"]],
              color=OBJECT_COLOUR, lw=0.9, alpha=0.7)
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
            axis.arrow(robot[0], robot[1], normal[0] * 0.20, normal[1] * 0.20, head_width=0.025,
                       color="crimson", lw=0.9, zorder=7, length_includes_head=True)
        reference = entry.get("reference")
        if reference and math.hypot(*reference) > 1e-6 and title.startswith("V"):
            axis.arrow(robot[0], robot[1], reference[0] * REFERENCE_SCALE / 0.26,
                       reference[1] * REFERENCE_SCALE / 0.26, head_width=0.025,
                       color="darkgreen", lw=0.9, zorder=7, length_includes_head=True)
        # arm X reference geometry
        p_contact, p_ref, q = entry.get("p_contact"), entry.get("p_ref"), entry.get("q")
        if q:
            axis.plot([q[0]], [q[1]], marker="s", ms=6.0, color="tab:red", zorder=10)
        if p_contact:
            axis.plot([p_contact[0]], [p_contact[1]], marker="o", ms=6.5, mfc="none",
                      mec="tab:red", mew=1.2, zorder=10)
        if p_contact and p_ref:
            axis.plot([p_contact[0], p_ref[0]], [p_contact[1], p_ref[1]], color="tab:green",
                      lw=1.4, ls=":", zorder=9)
        if p_ref:
            # the advance reference is virtual and lies inside the object: open marker
            axis.plot([p_ref[0]], [p_ref[1]], marker="o", ms=9.0, mfc="none", mec="tab:green",
                      mew=1.4, zorder=10)
            axis.plot([robot[0], p_ref[0]], [robot[1], p_ref[1]], color="tab:blue", lw=1.1,
                      alpha=0.9, zorder=8)
    for index, force in enumerate(row["force"]):
        robot = row["true"]["robots"][index]
        magnitude = math.hypot(*force)
        if magnitude > 1e-9:
            axis.arrow(robot[0], robot[1], force[0] / 15.0 * FORCE_SCALE,
                       force[1] / 15.0 * FORCE_SCALE, head_width=0.03, color="black", lw=1.0,
                       zorder=8, length_includes_head=True)
    axis.set_xlim(-0.06, width + 0.06)
    axis.set_ylim(-0.06, height + 0.06)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    entry = observed.get("0") or {}
    contact = {key: bool((observed.get(key) or {}).get("contact")) for key in ("0", "1")}
    lines = [
        title,
        f"t={row['time']:.2f}s  {row.get('phase') or row.get('stage')}  "
        f"push={max((row.get('active_contact') or {}).values(), default=0.0):.2f}s  "
        f"beta={entry.get('beta'):.2f}" if entry.get("beta") is not None else
        f"t={row['time']:.2f}s  {row.get('phase') or row.get('stage')}",
        f"contact r0={int(contact['0'])} r1={int(contact['1'])}",
        f"|v_obj|={math.hypot(object_state[3], object_state[4]):.4f}  "
        f"|w_obj|={abs(object_state[5]):.4f}   |F|={math.hypot(*row['force'][0]):.2f} N",
    ]
    if ended_at is not None:
        lines.append(f"ENDED t={ended_at:.2f}s (frozen)")
    axis.text(0.012, 0.985, "\n".join(lines), transform=axis.transAxes, va="top", ha="left",
              fontsize=fontsize, linespacing=1.35, family="DejaVu Sans Mono",
              bbox=dict(facecolor="white", alpha=0.72, edgecolor="none", pad=1.6))


def _cell(spec, rows, cols, scale=1.0):
    width, height = spec.arena
    aspect = (width + 0.12) / (height + 0.12)
    cell_h = 4.6 * scale
    return (cols * cell_h * aspect + 0.3, rows * cell_h * 1.02 + 0.95)


def _series(rows):
    trajectories = {"object": [], "robot0": [], "robot1": []}
    frames = []
    for row in rows:
        trajectories["object"].append(tuple(row["true"]["object"][:2]))
        trajectories["robot0"].append(tuple(row["true"]["robots"][0][:2]))
        trajectories["robot1"].append(tuple(row["true"]["robots"][1][:2]))
        frames.append((row, {key: list(value) for key, value in trajectories.items()}))
    return frames


def render(case, logs, root_meta, out_singles, out_pairs, out_gifs,
           docs_dir: Optional[Path] = None):
    spec = root_from_metadata(root_meta).scene()
    initial = _corners(logs["V"][0]["true"]["object"])
    total = max(logs[arm][-1]["time"] for arm in ARMS)
    series = {arm: _series(logs[arm]) for arm in ARMS}
    entries = []

    # single-arm videos
    for arm in ARMS:
        frames = series[arm]
        ended = frames[-1][0]["time"] if frames[-1][0]["time"] < total - 1e-9 else None
        figure, axis = plt.subplots(figsize=_cell(spec, 1, 1, 1.30), dpi=110)
        figure.subplots_adjust(left=0.035, right=0.985, top=0.95, bottom=0.115)
        figure.text(0.5, 0.018, NOTE, ha="center", va="bottom", fontsize=8.5, color="0.25",
                    linespacing=1.5)
        writer = animation.FFMpegWriter(
            fps=FPS, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "28"])
        path = out_singles / f"{case['case']}__{arm}.mp4"
        with writer.saving(figure, str(path), dpi=110):
            for row, trajectory in frames:
                draw(axis, spec, row, ARM_LABEL[arm], None, trajectory, initial, fontsize=10.5)
                writer.grab_frame()
            for _ in range(HOLD_FRAMES):
                draw(axis, spec, frames[-1][0], ARM_LABEL[arm], ended, frames[-1][1], initial,
                     fontsize=10.5)
                writer.grab_frame()
        plt.close(figure)
        entries.append({"kind": "single", "arm": arm, "file": path.name})

    # side-by-side video
    figure, axes = plt.subplots(1, 2, figsize=_cell(spec, 1, 2, 1.10), dpi=110)
    figure.subplots_adjust(left=0.03, right=0.985, top=0.95, bottom=0.14, wspace=0.06)
    figure.text(0.5, 0.02, NOTE, ha="center", va="bottom", fontsize=8.5, color="0.25",
                linespacing=1.5)
    n_frames = max(len(value) for value in series.values())
    writer = animation.FFMpegWriter(
        fps=FPS, extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "28"])
    pair_path = out_pairs / f"{case['case']}__VX.mp4"
    with writer.saving(figure, str(pair_path), dpi=110):
        for index in range(n_frames + HOLD_FRAMES):
            for axis, arm in zip(axes, ARMS):
                frames = series[arm]
                position = min(index, len(frames) - 1)
                row, trajectory = frames[position]
                ended = (frames[-1][0]["time"]
                         if row["time"] >= frames[-1][0]["time"] - 1e-9
                         and frames[-1][0]["time"] < total - 1e-9 else None)
                draw(axis, spec, row, ARM_LABEL[arm], ended, trajectory, initial, fontsize=9.0)
            writer.grab_frame()
    plt.close(figure)
    entries.append({"kind": "pair", "arm": "V/X", "file": pair_path.name})

    # preview gif
    figure, axes = plt.subplots(1, 2, figsize=_cell(spec, 1, 2, 0.95), dpi=100)
    figure.subplots_adjust(left=0.03, right=0.985, top=0.95, bottom=0.15, wspace=0.06)
    figure.text(0.5, 0.02, NOTE, ha="center", va="bottom", fontsize=8, color="0.25",
                linespacing=1.5)
    frames_out = []
    for index in range(0, n_frames, 2):
        for axis, arm in zip(axes, ARMS):
            frames = series[arm]
            position = min(index, len(frames) - 1)
            row, trajectory = frames[position]
            ended = (frames[-1][0]["time"]
                     if row["time"] >= frames[-1][0]["time"] - 1e-9
                     and frames[-1][0]["time"] < total - 1e-9 else None)
            draw(axis, spec, row, ARM_LABEL[arm], ended, trajectory, initial, fontsize=8.5)
        figure.canvas.draw()
        import numpy as np

        buffer = np.frombuffer(figure.canvas.buffer_rgba(), dtype="uint8")
        width, height = figure.canvas.get_width_height()
        frames_out.append(buffer.reshape((height, width, 4))[:, :, :3].copy())
    plt.close(figure)
    from PIL import Image

    images = [Image.fromarray(frame).convert("P", palette=Image.ADAPTIVE, colors=64)
              for frame in frames_out]
    if images:
        gif_path = out_gifs / f"{case['case']}__VX.gif"
        images[0].save(str(gif_path), save_all=True, append_images=images[1:],
                       duration=int(1000 / 10), loop=0, optimize=True)
        entries.append({"kind": "preview_gif", "arm": "V/X", "file": gif_path.name})
    return entries


INDEX = """<!doctype html>
<html lang="zh"><head><meta charset="utf-8"><title>local_position_push_v1</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 16px; background: #fafafa; color: #222; }}
 .case {{ background: #fff; border: 1px solid #ddd; border-radius: 6px; margin: 18px 0; padding: 12px; }}
 video {{ width: 100%; max-width: 1100px; background: #000; }}
 .row {{ display: flex; flex-wrap: wrap; gap: 12px; }}
 .row figure {{ margin: 0; }}
 .row video {{ max-width: 380px; }}
 table {{ border-collapse: collapse; font-size: 12px; margin-top: 8px; }}
 th, td {{ border: 1px solid #ddd; padding: 3px 6px; }}
</style></head><body>
<h1>local_position_push_v1 — V (velocity) vs X (surface position), 8 cases</h1>
<p>1× 仿真时间（50 ms 采样，20 fps 导出）。X 叠加 q / p_contact / p_ref 与位置误差细线；
p_ref 是虚拟参考、位于物体内部，用空心点与虚线区分。V 没有位置目标，标 N/A。</p>
{cards}
</body></html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--docs-dir", default="docs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--only", default=None)
    parser.add_argument("--index-only", action="store_true",
                        help="rebuild index.html from the existing animation_index.json")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    records = [json.loads(line) for line in (run_dir / f"executions_{args.split}.jsonl").open()]
    roots = {row["root_id"]: row for row in json.loads(
        (run_dir / f"roots_{args.split}.json").read_text())["roots"]}
    if args.index_only:
        existing = json.loads((run_dir / "animation_index.json").read_text())
        cases = existing["cases"]
        index_rows = existing["rows"]
        entries = existing["entries"]
        _write_index(cases, index_rows, entries, run_dir)
        print(json.dumps({"cases": len(cases), "index_only": True}, indent=1))
        print("ANIMATIONS_OK")
        return 0
    cases = select_cases(records, args.split)
    if args.only:
        cases = [case for case in cases if case["case"].startswith(args.only)]
    figures = run_dir / "figures"
    for name in ("singles", "pairs", "gif"):
        (figures / name).mkdir(parents=True, exist_ok=True)
    index_rows, entries = [], []
    for case in cases:
        logs = {}
        for arm in ARMS:
            path = run_dir / "logs" / args.split / f"{case['root_id'].replace('/', '_')}__baseline.jsonl"
            logs[arm] = [row for row in (json.loads(line) for line in path.open())
                         if row.get("arm") == arm]
        produced = render(case, logs, roots[case["root_id"]], figures / "singles",
                          figures / "pairs", figures / "gif")
        entries.extend([{**entry, "case": case["case"]} for entry in produced])
        digest = hashlib.sha256(json.dumps(roots[case["root_id"]], sort_keys=True).encode()).hexdigest()[:16]
        for arm in ARMS:
            row = next((item for item in records if item["root_id"] == case["root_id"]
                        and item["arm"] == arm and item["variant"] == "baseline"), None)
            if row is None:
                continue
            index_rows.append({
                "case": case["case"], "root_id": case["root_id"], "variant": "baseline",
                "initial_state_hash": digest, "arm": arm,
                "exit_reason": row["metrics"]["exit_reason"],
                "displacement_m": row["metrics"]["displacement"],
                "rotation_rad": row["metrics"]["rotation"],
                "seconds": row["metrics"]["wall_clock_seconds"],
                "normal_completion": bool(row["metrics"].get("push_completed")
                                          and row["metrics"]["settled_observed"]),
                "single_video": f"figures/singles/{case['case']}__{arm}.mp4",
                "pair_video": f"figures/pairs/{case['case']}__VX.mp4",
                "preview_gif": f"figures/gif/{case['case']}__VX.gif",
                "log": f"logs/{args.split}/{case['root_id'].replace('/', '_')}__baseline.jsonl",
            })
        print(f"  {case['case']} done", flush=True)
    with (run_dir / "animation_index.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)
    (run_dir / "animation_index.json").write_text(json.dumps({
        "cases": cases, "rows": index_rows, "entries": entries, "fps": FPS,
        "real_time": "1x (50 ms log sampling, 20 fps export)",
        "single_videos": sum(1 for entry in entries if entry["kind"] == "single"),
        "pair_videos": sum(1 for entry in entries if entry["kind"] == "pair"),
        "preview_gifs": sum(1 for entry in entries if entry["kind"] == "preview_gif"),
    }, indent=1, sort_keys=True))
    _write_index(cases, index_rows, entries, run_dir)
    docs = ROOT / args.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"cases": len(cases),
                      "singles": sum(1 for entry in entries if entry["kind"] == "single"),
                      "pairs": sum(1 for entry in entries if entry["kind"] == "pair"),
                      "gifs": sum(1 for entry in entries if entry["kind"] == "preview_gif")}, indent=1))
    print("ANIMATIONS_OK")
    return 0


def _write_index(cases, index_rows, entries, run_dir: Path) -> None:
    cards = []
    for case in cases:
        quad = f"figures/pairs/{case['case']}__VX.mp4"
        gif = f"figures/gif/{case['case']}__VX.gif"
        rows = [row for row in index_rows if row["case"] == case["case"]]
        table = ["<table><tr><th>arm</th><th>exit</th><th>disp (m)</th><th>rot (rad)</th>"
                 "<th>s</th><th>normal</th><th>video</th></tr>"]
        for row in sorted(rows, key=lambda item: ARMS.index(item["arm"])):
            table.append(f"<tr><td>{row['arm']}</td><td>{row['exit_reason']}</td>"
                         f"<td>{row['displacement_m']:.4f}</td><td>{row['rotation_rad']:.4f}</td>"
                         f"<td>{row['seconds']:.2f}</td><td>{row['normal_completion']}</td>"
                         f"<td><a href='{row['single_video']}'>{row['arm']}.mp4</a></td></tr>")
        table.append("</table>")
        timing = ""
        for interaction_class in ("solo_central", "dual_same_side", "dual_rotate"):
            path = run_dir / "timing" / f"{interaction_class}__diagnostic.png"
            if path.exists() and interaction_class in case["root_id"]:
                timing = (f"<p>逐周期时序诊断（同一 root）："
                          f"<a href='timing/{interaction_class}__diagnostic.png'>PNG</a> · "
                          f"<a href='timing/{interaction_class}__V.csv'>V.csv</a> · "
                          f"<a href='timing/{interaction_class}__X.csv'>X.csv</a></p>")
        cards.append(f"<div class='case'><h2>{case['case']}</h2>"
                     f"<div>root <code>{case['root_id']}</code> · {case['rule']}</div>"
                     f"<video controls preload='none' src='{quad}'></video>"
                     f"<p><img src='{gif}' style='max-width:760px'></p>"
                     + "".join(table) + timing + "</div>")
    # every case links to the three timing diagnostics, whichever root they use
    cards.append("<div class='case'><h2>逐周期时序诊断（三个必交案例）</h2><div class='row'>"
                 + "".join(f"<figure><figcaption>{name}</figcaption>"
                           f"<a href='timing/{name}__diagnostic.png'>"
                           f"<img src='timing/{name}__diagnostic.png' style='max-width:520px'></a>"
                           f"</figure>"
                           for name in ("solo_central", "dual_same_side", "dual_rotate")
                           if (run_dir / "timing" / f"{name}__diagnostic.png").exists())
                 + "</div></div>")
    (run_dir / "index.html").write_text(INDEX.format(cards="\n".join(cards)))


if __name__ == "__main__":
    raise SystemExit(main())
