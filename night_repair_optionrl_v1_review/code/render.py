"""Behaviour clips for the RL episodes.

Draws the arena, obstacles, object, both robots and the **public** goal marker, then writes a GIF and
a small MP4 per clip. Clips are chosen by a fixed rule (best, median, worst by net progress), never
by hand, and a failure clip is delivered even when no success exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

for candidate in ("/home/tzl/.local/share/fonts/wqy.ttc", "/tmp/wqy.ttc", "/tmp/cjk.ttf"):
    if Path(candidate).exists():
        font_manager.fontManager.addfont(candidate)
        matplotlib.rcParams["font.sans-serif"] = [
            font_manager.FontProperties(fname=candidate).get_name(), "DejaVu Sans"]
        break
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = ROOT / "outputs/night_repair_optionrl_v1/collection"
CLIPS = ROOT / "outputs/night_repair_optionrl_v1/report/clips"


def _object_polygon(pose: Sequence[float], size: Tuple[float, float]) -> np.ndarray:
    import math
    half_width, half_height = size[0] / 2.0, size[1] / 2.0
    corners = np.asarray([[-half_width, -half_height], [half_width, -half_height],
                          [half_width, half_height], [-half_width, half_height]])
    angle = float(pose[2])
    rotation = np.asarray([[math.cos(angle), -math.sin(angle)],
                           [math.sin(angle), math.cos(angle)]])
    return corners @ rotation.T + np.asarray(pose[:2])


def draw(axis, scene, goal, object_pose, object_size, robots, step_label: str) -> None:
    width, height = scene.arena
    axis.add_patch(Rectangle((0, 0), width, height, fill=False, edgecolor="#333333", linewidth=1.2))
    for box in scene.obstacles:
        axis.add_patch(Rectangle((box.cx - box.half_w, box.cy - box.half_h),
                                 2 * box.half_w, 2 * box.half_h, color="#4a4a4a"))
    axis.add_patch(Circle((goal[0], goal[1]), 0.10, facecolor="#bfe3c0",
                          edgecolor="#2e7d32", alpha=0.75, linewidth=1.4, zorder=1))
    axis.add_patch(Polygon(_object_polygon(object_pose, object_size), closed=True,
                           facecolor="#c98b4b", edgecolor="#6d4c1b", linewidth=1.0, zorder=2))
    for robot_id, robot in enumerate(robots):
        axis.add_patch(Circle((robot[0], robot[1]), 0.18,
                              facecolor="#4c78c9" if robot_id == 0 else "#c94c6e",
                              edgecolor="#1b2b4c", linewidth=0.8, zorder=3))
    axis.set_xlim(-0.05, width + 0.05)
    axis.set_ylim(-0.05, height + 0.05)
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(step_label, fontsize=9)


def render_episode_shard(shard: Path, scene_lookup, *, stem: str, out_dir: Path = CLIPS,
                         every: int = 3, fps: int = 12) -> Dict[str, object]:
    """One shard -> one GIF plus one MP4. Requires the shard to carry its own state trace."""

    data = dict(np.load(shard, allow_pickle=True))
    meta = json.loads(str(data["meta"]))
    if "trace" not in data:
        return {"shard": str(shard), "status": "no trace stored"}
    # the trace is stored as a JSON string inside a 0-d object array
    trace = json.loads(str(data["trace"]))
    if not trace:
        return {"shard": str(shard), "status": "empty trace"}
    scene = scene_lookup(meta)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[np.ndarray] = []
    for index in range(0, len(trace), every):
        row = trace[index]
        figure, axis = plt.subplots(figsize=(4.2, 4.2))
        draw(axis, scene, meta["goal"], row["object"],
             (scene.object_width or 1.0, scene.object_height or 0.25), row["robots"],
             f"{meta['behaviour_source']} skill {row['skill_index']} t={row['time']:.2f}s")
        figure.tight_layout()
        figure.canvas.draw()
        buffer = np.asarray(figure.canvas.buffer_rgba())[:, :, :3].copy()
        frames.append(buffer)
        plt.close(figure)
    if not frames:
        return {"shard": str(shard), "status": "empty trace"}
    gif_path = out_dir / f"{stem}.gif"
    imageio_gif = __import__("PIL.Image", fromlist=["Image"])
    images = [imageio_gif.fromarray(frame) for frame in frames]
    images[0].save(gif_path, save_all=True, append_images=images[1:], duration=int(1000 / fps),
                   loop=0)
    mp4_path = out_dir / f"{stem}.mp4"
    try:
        import imageio_ffmpeg
        writer = imageio_ffmpeg.write_frames(str(mp4_path), (frames[0].shape[1],
                                                             frames[0].shape[0]), fps=fps)
        writer.send(None)
        for frame in frames:
            writer.send(frame.tobytes())
        writer.close()
    except Exception as error:
        mp4_path = None
        print(f"    mp4 skipped for {stem}: {error!r}")
    return {"shard": str(shard), "gif": str(gif_path), "mp4": str(mp4_path) if mp4_path else None,
            "frames": len(frames), "meta": {k: meta[k] for k in
                                            ("episode_id", "root_id", "behaviour_source", "success",
                                             "net_progress_m", "skills")}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(CLIPS))
    args = parser.parse_args()
    print("render module ready; clips are produced by the evaluation driver")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
