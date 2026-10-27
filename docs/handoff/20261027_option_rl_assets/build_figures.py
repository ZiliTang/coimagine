"""Build advisor figures from a frozen history and two focused option replays."""
from pathlib import Path
import json
import math
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
font_path = Path.home() / ".local/share/fonts/NotoSansSC-Regular.otf"
if font_path.exists():
    font_manager.fontManager.addfont(str(font_path))
plt.rcParams.update({"font.family": ["Noto Sans SC", "sans-serif"], "font.size": 10,
                     "axes.unicode_minus": False, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
BLUE, ORANGE, TEAL = "#2864ad", "#d87929", "#238976"


def collect_examples():
    from coimagine.continuous2d.scenes import Pose
    from coimagine.random_start import env as E, layout, spec
    instance = layout.InstanceSpec(
        name="advisor_coop_v46", map="L0", split="dev", index=0, seed=0,
        arena=(4., 3.), obstacles=(), object_pose=Pose(2., 1.5, 0.),
        robot_poses=(Pose(1.8, 1.075, -math.pi / 2),
                     Pose(2.2, .745, -math.pi / 2)),
        goal_centre=(2., 2.65), goal_radius=spec.GOAL_RADIUS,
        env_version=spec.ENV_VERSION)
    examples = []
    for name, instance, actions in [
            ("协同：先到者等待，双方就绪后同步推", instance, (6, 6)),
            ("短面：建立接触后完成 0.50 m 行程", layout.build_split("L1", "train", count=3)[0][2], (6, 20))]:
        env = E.RandomStartEnv(instance)
        initial = {"time": 0., "true": {
            "object": list(env.object_state().as_tuple()),
            "robots": [list(r.as_tuple()) for r in env.robot_states()]}}
        boundary = env.joint_step(*actions)
        rows = env.rows
        if actions == (6, 6):
            middle = next(r for r in rows if r["r0"].get("ready") and r["r1"].get("ready"))
            labels = ["接近物体", "双方就绪，等待同步释放", "执行结果"]
        else:
            middle = next(r for r in rows if r["r0"]["contacting_object"])
            labels = ["面对短侧面", "建立接触", "执行结果"]
        examples.append({"title": name, "instance": instance.as_dict(), "actions": actions,
                         "labels": labels, "frames": [initial, middle, rows[-1]],
                         "boundary": boundary,
                         "object_displacement": math.hypot(
                             rows[-1]["true"]["object"][0] - initial["true"]["object"][0],
                             rows[-1]["true"]["object"][1] - initial["true"]["object"][1])})
    (HERE / "option_replay_snapshots.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n")
    return examples


def corners(state, hx, hy):
    x, y, a = state[:3]
    c, s = math.cos(a), math.sin(a)
    return [(x + c * dx - s * dy, y + s * dx + c * dy)
            for dx, dy in [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]]


def option_figure(examples):
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for i, example in enumerate(examples):
        positions = [p for f in example["frames"]
                     for p in [f["true"]["object"], *f["true"]["robots"]]]
        xmin, xmax = min(p[0] for p in positions) - .65, max(p[0] for p in positions) + .65
        ymin, ymax = min(p[1] for p in positions) - .65, max(p[1] for p in positions) + .65
        for j, (frame, label) in enumerate(zip(example["frames"], example["labels"])):
            ax = axes[i, j]
            for box in example["instance"]["obstacles"]:
                ax.add_patch(Polygon(corners((box["cx"], box["cy"], box["angle"]),
                                             box["half_w"], box["half_h"]),
                                     facecolor="#eedccf", edgecolor="#9b7658", hatch="///"))
            width, height = example["instance"]["arena"]
            for xwall in (0, width):
                ax.axvline(xwall, color="#596976", lw=1)
            for ywall in (0, height):
                ax.axhline(ywall, color="#596976", lw=1)
            ax.add_patch(Polygon(corners(frame["true"]["object"], .5, .125),
                                 facecolor="#d8dfe5", edgecolor="#596976", linewidth=1.8))
            for rid, state in enumerate(frame["true"]["robots"]):
                color = [BLUE, ORANGE][rid]
                ax.add_patch(Circle(state[:2], .18, facecolor=color, edgecolor="white", linewidth=1.5))
                ax.text(state[0], state[1], str(rid), color="white", ha="center", va="center", fontweight="bold")
                ax.arrow(state[0], state[1], .23 * math.cos(state[2]), .23 * math.sin(state[2]),
                         width=.012, head_width=.07, color=color, length_includes_head=True)
            ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax), aspect="equal")
            ax.set_title(f"{label}  |  t={frame['time']:.2f} s", fontsize=10)
            ax.set_xlabel("x / m")
            if j == 0:
                ax.set_ylabel("y / m")
            ax.grid(alpha=.12)
        axes[i, 0].text(0, 1.22, example["title"], transform=axes[i, 0].transAxes,
                        fontsize=12, fontweight="bold", color="#203448")
    fig.text(.5, .015, "当前 v4.6 的两条代表性实际回放；灰色为物体，蓝/橙为机器人。真值位置仅用于离线展示。",
             ha="center", fontsize=10, color="#596976")
    fig.subplots_adjust(left=.07, right=.98, top=.90, bottom=.09, hspace=.62, wspace=.26)
    fig.savefig(HERE / "option_examples.png", dpi=170)
    plt.close(fig)


def rolling(values, window=20):
    values = np.asarray(values, dtype=float)
    return np.asarray([values[max(0, i - window + 1):i + 1].mean() for i in range(len(values))])


def rl_figure():
    snapshot = json.loads((HERE / "run9_history_snapshot.json").read_text())
    h = snapshot["history"]
    x = np.array([r["iteration"] for r in h])
    ev = [r for r in h if r.get("success_rate") is not None]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    panels = [("explore_return", "训练回报", "回报"),
              ("explore_success", "成功率", "成功率 / %"),
              ("explore_progress", "向目标的相对进度", "progress")]
    for ax, (key, title, ylabel) in zip(axes, panels):
        scale = 100 if key == "explore_success" else 1
        y = np.array([r[key] for r in h]) * scale
        ax.plot(x, y, color=TEAL, alpha=.25, lw=.8, label="训练单轮")
        ax.plot(x, rolling(y), color=TEAL, lw=2.1, label="训练近20轮均值")
        if key != "explore_return":
            vk = "success_rate" if key == "explore_success" else "mean_progress"
            ax.plot([r["iteration"] for r in ev], [r[vk] * scale for r in ev],
                    color=ORANGE, marker="o", markersize=2.5, lw=1.4, label="固定16根验证")
            ax.axhline(0, color="#64748b", lw=.6, linestyle="--")
        ax.set(title=title, xlabel="训练迭代", ylabel=ylabel, xlim=(1, int(x[-1])))
        ax.grid(alpha=.15)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"run9：因子化动作头，随机全目标场景（截至第 {x[-1]} 轮）", fontsize=13, y=1.02)
    fig.text(.5, -.01, "progress = (初始目标距离 − 最终目标距离) / 初始目标距离，截断至 [-1, 1]；正值表示接近目标。",
             ha="center", fontsize=9, color="#596976")
    fig.tight_layout()
    fig.savefig(HERE / "run9_training.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    path = HERE / "option_replay_snapshots.json"
    examples = json.loads(path.read_text()) if path.exists() else collect_examples()
    option_figure(examples)
    rl_figure()
    print(json.dumps([{ "name": e["instance"]["name"],
                       "object_displacement": e["object_displacement"],
                       "exits": [v["exit_reason"] for v in e["boundary"]["skill_ends"].values()]}
                      for e in examples], ensure_ascii=False))
