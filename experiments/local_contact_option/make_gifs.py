"""Render 8-12 representative GIFs from the per-control-period logs.

    python experiments/local_contact_option/make_gifs.py
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
from matplotlib.animation import PillowWriter  # noqa: E402

SELECTION = (
    ("solo_central", "N", "central push, new law + staged protocol"),
    ("solo_central", "O", "central push, old window"),
    ("solo_offset", "N", "offset push, new law"),
    ("dual_same_side", "N", "same-side co-translation, new law"),
    ("dual_rotate", "P", "rotation, oracle formula + staged protocol"),
    ("dual_rotate", "N", "rotation, new law"),
    ("incompatible", "N", "opposing push (expected to fail), new law"),
    ("constrained", "N", "passage constraint, new law"),
)


def load_log(path: Path):
    rows = [json.loads(line) for line in path.open()]
    by_arm = {}
    for row in rows:
        by_arm.setdefault(row["arm"], []).append(row)
    return by_arm


def render(rows, scene, title: str, out: Path, stride: int = 2):
    frames = rows[::stride]
    figure, axis = plt.subplots(figsize=(4.2, 3.0))
    writer = PillowWriter(fps=20)
    with writer.saving(figure, str(out), dpi=90):
        for row in frames:
            axis.clear()
            width, height = scene["arena"]
            axis.add_patch(patches.Rectangle((0, 0), width, height, fill=False, lw=0.8, ec="black"))
            for box in scene["obstacles"]:
                axis.add_patch(patches.Rectangle(
                    (box["cx"] - box["half_w"], box["cy"] - box["half_h"]),
                    2 * box["half_w"], 2 * box["half_h"], facecolor="0.7", edgecolor="black", lw=0.5))
            obj = row["true"]["object"]
            axis.add_patch(patches.Polygon(
                [(obj[0] + dx * math.cos(obj[2]) - dy * math.sin(obj[2]),
                  obj[1] + dx * math.sin(obj[2]) + dy * math.cos(obj[2]))
                 for dx, dy in ((-0.5, -0.125), (0.5, -0.125), (0.5, 0.125), (-0.5, 0.125))],
                closed=True, facecolor="tab:orange", alpha=0.8, edgecolor="black", lw=0.6))
            for index, robot in enumerate(row["true"]["robots"]):
                colour = "tab:blue" if index == 0 else "tab:green"
                axis.add_patch(patches.Circle((robot[0], robot[1]), 0.18, facecolor=colour, alpha=0.8))
                axis.annotate(f"r{index}", (robot[0], robot[1]), fontsize=6, ha="center", va="center")
            observed = row.get("observed", {})
            for index in ("0", "1"):
                entry = observed.get(index, {})
                normal = entry.get("patch_normal")
                robot = row["true"]["robots"][int(index)]
                if normal:
                    axis.arrow(robot[0], robot[1], normal[0] * 0.18, normal[1] * 0.18,
                               head_width=0.03, color="crimson", lw=0.8)
            axis.set_xlim(-0.05, width + 0.05)
            axis.set_ylim(-0.05, height + 0.05)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_title(f"{title}\nt={row['time']:.2f}s stage={row['stage']}", fontsize=7)
            writer.grab_frame()
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-v1")
    parser.add_argument("--out-root", default="outputs/local_contact_option_v1")
    args = parser.parse_args()
    out_dir = ROOT / args.out_root / args.run_id
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    roots = {row["root_id"]: row for row in json.loads((out_dir / "roots_test.json").read_text())["roots"]}
    log_dir = out_dir / "logs" / "test"
    made = []
    for interaction_class, arm, description in SELECTION:
        candidates = [root_id for root_id in roots if root_id.startswith(f"test/{interaction_class}/")]
        for root_id in sorted(candidates):
            path = log_dir / f"{root_id.replace('/', '_')}__baseline.jsonl"
            if not path.exists():
                continue
            by_arm = load_log(path)
            if arm not in by_arm:
                continue
            meta = roots[root_id]
            from coimagine.continuous2d.scenes import S0_DEVELOPMENT  # noqa: F401
            from coimagine.local_contact_option.roots import TEMPLATES

            template = TEMPLATES[meta["template"]]
            scene = {
                "arena": list(template.arena),
                "obstacles": [
                    {"cx": box.cx, "cy": box.cy, "half_w": box.half_w, "half_h": box.half_h}
                    for box in template.obstacles
                ],
            }
            out = figures / f"{interaction_class}_{arm}.gif"
            render(by_arm[arm], scene,
                   f"{description}  [{root_id.split('/')[-1]}]  alpha={meta['alpha_deg'][0]} "
                   f"v={meta['speed'][0]}", out)
            made.append(str(out.relative_to(ROOT)))
            break
    print(json.dumps({"gifs": made, "count": len(made)}, indent=1))
    print("GIFS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
