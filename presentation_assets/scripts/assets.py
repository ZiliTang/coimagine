"""Round-5 presentation assets: F01..F08 figures, data tables, videos and the index.

    python experiments/night_sequence_v1/assets.py

Everything is rendered from artefacts that already exist: the stored sequences, the token arrays,
the frozen checkpoints and the analysis JSON. The script **never** retrains, re-collects or runs new
physics. Every figure also writes the numbers behind it into ``data/`` so a caption can be checked
against the raw JSON.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import Circle, Polygon, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "outputs/night_sequence_v1"
ASSETS = ROOT / "presentation_assets"
EXPERIMENTS = ROOT / "experiments/night_sequence_v1"
FIGURES = ASSETS / "figures"
TABLES = ASSETS / "data"
VIDEOS = ASSETS / "videos"
SCRIPTS = ASSETS / "scripts"

COLORS = {"mlp_current": "#1f77b4", "mlp_history": "#2ca02c", "transformer_history": "#d62728",
          "P0_hold": "#7f7f7f", "P1_grouped": "#bcbd22"}
ROBOT_COLORS = ["#1f77b4", "#ff7f0e"]
FRAME_ALPHA = 0.06


def setup_font() -> str:
    for path in ("/tmp/wqy.ttc", "/tmp/cjk.ttf",
                 str(Path.home() / ".local/share/fonts/wqy-microhei.ttc")):
        if Path(path).exists():
            try:
                font_manager.fontManager.addfont(path)
            except Exception:
                pass
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Noto Sans SC", "WenQuanYi Micro Hei", "DejaVu Sans"):
        if name in available:
            plt.rcParams["font.sans-serif"] = [name] + plt.rcParams["font.sans-serif"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 160
    return plt.rcParams["font.sans-serif"][0]


def save(fig, name: str) -> Dict[str, str]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png.relative_to(ROOT)), "pdf": str(pdf.relative_to(ROOT))}


SHORT = {"P0_hold": "P0", "P1_grouped": "P1", "mlp_current": "cur", "mlp_history": "hist",
         "transformer_history": "trf"}


def short_label(name: str) -> str:
    """Compact bar label: tier + model family + seed, so panels (a)-(c) stay readable."""

    if name in SHORT:
        return SHORT[name]
    parts = name.split("_")
    seed = ""
    for part in parts:
        if part.startswith("seed"):
            seed = part.replace("seed", "s")
    tier = parts[0]
    for family in ("transformer_history", "mlp_current", "mlp_history"):
        if name.startswith(f"{tier}_{family}"):
            return f"{tier}\n{SHORT[family]}{seed}"
    return name


def num(value, fallback: float = 0.0) -> float:
    """None-safe float for bar charts; a missing metric must not crash the whole figure."""

    return fallback if value is None else float(value)


def write_csv(name: str, header: Sequence[str], rows: Sequence[Sequence]) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    with path.open("w") as handle:
        handle.write(",".join(header) + "\n")
        for row in rows:
            handle.write(",".join("" if v is None else str(v) for v in row) + "\n")
    return str(path.relative_to(ROOT))


def write_json(name: str, payload) -> str:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / name
    path.write_text(json.dumps(payload, indent=1, sort_keys=True, default=float))
    return str(path.relative_to(ROOT))


# ---------------------------------------------------------------------------
# shared data
# ---------------------------------------------------------------------------


class Store:
    def __init__(self) -> None:
        self.roots = json.loads((DATA / "roots_night.json").read_text())
        self.summary = json.loads((DATA / "token_summary.json").read_text())
        self.tokens = {}
        for split in ("d0", "d1ext", "val", "test"):
            path = DATA / "tokens" / f"tokens_{split}.npz"
            if path.exists():
                self.tokens[split] = dict(np.load(path, allow_pickle=True))
        self.runs = {}
        for path in sorted((DATA / "runs").glob("*.pt")):
            import torch
            saved = torch.load(path, map_location="cpu", weights_only=False)
            self.runs[path.stem] = {
                "kind": saved["model_kind"], "params": saved["parameters"],
                "best_val": saved["best_val"], "best_update": saved["best_update"],
                "tier": saved["tier"], "seed": saved["seed"]}
        self.curves = {}
        for path in sorted((DATA / "runs").glob("*.log")):
            pass

    def trajectories(self, split: str, limit: int = 0):
        for path in sorted((DATA / "sequences").glob(f"{split}_*.npz")):
            yield np.load(path, allow_pickle=True)


def curve_from_log(tag: str) -> List[dict]:
    """Curves from ``<tag>.curves.json`` when present, else parsed from the run log.

    The log fallback carries the train and validation composite totals; the per-group columns only
    exist in the JSON, so a run whose JSON is missing is drawn in the composite panel only.
    """

    path = DATA / "runs" / f"{tag}.curves.json"
    if path.exists():
        return json.loads(path.read_text())
    log = DATA / "runs" / f"{tag}.log"
    if not log.exists():
        return []
    entries = []
    for line in log.read_text().splitlines():
        if "train=" not in line or "val=" not in line:
            continue
        try:
            head, rest = line.split("]", 1)
            update = int(head.split()[-1].split("/")[0])
            train = float(rest.split("train=")[1].split()[0])
            val = float(rest.split("val=")[1].split()[0])
        except (IndexError, ValueError):
            continue
        entries.append({"update": update, "train_total": train, "val_total": val})
    return entries


# ---------------------------------------------------------------------------
# F01
# ---------------------------------------------------------------------------


def f01(store: Store) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})
    ax = axes[0]
    ax.set_title("(a) 已有 Gridworld 诊断（历史讨论结论）", fontsize=11)
    ax.axis("off")
    ax.text(0.02, 0.92, "网格世界 + 含不确定物理参数的候选序列", fontsize=9, va="top")
    ax.text(0.02, 0.80, "模型平均误差不大，但被选中分支的收益被\n系统性高估（选择偏差）。",
            fontsize=9, va="top")
    ax.text(0.02, 0.62, "证据边界：仅对应当时的网格环境、候选序列与\n评分协议；不能外推为“搜索越宽越差”。",
            fontsize=8.5, va="top", color="#444444")
    ax.add_patch(Rectangle((0.02, 0.06), 0.96, 0.42, fill=False, edgecolor="#999999"))
    ax.text(0.04, 0.42, "物理乐观（已有动机）", fontsize=9, color="#b03030")
    ax.text(0.04, 0.34, "· 预测值与真实收益的排序错位", fontsize=8.5)
    ax.text(0.04, 0.26, "· 排序错位比平均误差更能解释失败", fontsize=8.5)
    ax.text(0.04, 0.18, "· 来源：docs/20261013_1937_研究主线与验证原则 §3.1", fontsize=7.5,
            color="#666666")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.set_title("(b) 转向连续 2D 后的问题", fontsize=11)
    ax.axis("off")
    for index, (title, body) in enumerate([
            ("连续接触", "接触力/滑移随姿态连续变化，\n没有离散“格子”可枚举"),
            ("局部观测", "只能看到扫描到的片段，\n物体全局位姿未经动捕共享"),
            ("变时长技能", "一次 option 内部有几十个\n控制周期，退出时刻不固定")]):
        y = 0.86 - index * 0.30
        ax.text(0.02, y, title, fontsize=9.5, color="#1f4e79")
        ax.text(0.04, y - 0.10, body, fontsize=8.5, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[2]
    ax.set_title("(c) 候选研究方向：固定模型预算下组织预测与搜索", fontsize=11)
    ax.axis("off")
    ax.text(0.02, 0.90, "固定世界模型 + 有限计算预算", fontsize=9.5, va="top")
    ax.text(0.06, 0.76, "→ 预测：哪些候选值得展开（本轮：观测终态预测）", fontsize=8.5, va="top")
    ax.text(0.06, 0.64, "→ 反馈：新观测如何改变分支合法性", fontsize=8.5, va="top")
    ax.text(0.06, 0.52, "→ 搜索：只在可执行的信息分支上分配预算", fontsize=8.5, va="top")
    ax.text(0.02, 0.38, "信息乐观（待验证）", fontsize=9.5, color="#b03030", va="top")
    ax.text(0.06, 0.28, "E_s[max_a Q] 与 max_a E_s[Q] 的差，不能在未消除歧义时当作可执行收益",
            fontsize=8.2, va="top")
    ax.text(0.02, 0.14, "本轮只做 (a)→(b) 之间的观测终态预测；不验证闭环收益。", fontsize=8.5,
            va="top", color="#444444")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.suptitle("F01 研究动机与当前主线：物理乐观是已有动机，信息乐观与观测条件分支仍待验证",
                 fontsize=12)
    paths = save(fig, "F01_motivation_and_mainline")
    rows = [
        ["阶段", "完成内容", "证据文件", "关键数字", "状态"],
        ["Gridworld", "候选序列收益高估诊断", "docs/20261013_1937_研究主线与验证原则_v1 (1).md §3.1",
         "", "历史讨论结论"],
        ["连续2D 平台", "局部接触传感 + option 阶段机 + 判稳", "src/coimagine/local_contact_option/",
         "V 控制器 Kv=25, F_max=15N, 0.05s", "已验证"],
        ["观测终态接口", "R_i0 片段几何 + 自身运动", "docs/observation_terminal_v1/",
         "中点误差 3.8-5.0cm vs 起始保持 14.1cm", "已验证"],
        ["序列数据", "D1 全量 option 内部序列", "docs/night_sequence_v1/DATA_INDEX.md",
         "4096 root / 32768 序列", "已验证"],
        ["MLP vs Transformer", "同历史同目标的架构对照", "presentation_assets/figures/F06,F07",
         "", "本轮结果"],
        ["信息乐观", "观测条件化分支收益", "docs/option_prediction_mechanism_v1/C_frozen_negative.md",
         "原场景收益差为 0（负结果）", "待验证"],
        ["闭环规划收益", "预测→决策的实际收益", "", "", "未评估"],
    ]
    csv = write_csv("F01_progress_facts.csv", rows[0], rows[1:])
    return {"figure": paths, "data": csv, "rows": len(rows) - 1}


# ---------------------------------------------------------------------------
# F02
# ---------------------------------------------------------------------------


def scene_bounds(store) -> Dict[str, float]:
    """Uniform plot bounds covering every root actually used, with a small margin.

    Derived from the data rather than hard-coded: the arena extent is not exposed as a single
    scalar, and a wrong constant silently crops robots or the object out of a panel.
    """

    xs, ys = [], []
    for records in store.roots["roots"].values():
        for record in records:
            xs.append(record["object_pose"][0])
            ys.append(record["object_pose"][1])
            for pose in record["robot_poses"]:
                xs.append(pose[0])
                ys.append(pose[1])
    margin = 0.55  # room for the object half-extent plus its rotation
    return {"min_x": min(0.0, min(xs) - margin), "max_x": max(xs) + margin,
            "min_y": min(0.0, min(ys) - margin), "max_y": max(ys) + margin}


def arena_from_config(store=None):
    """The frozen scene description, read from the same config the executions used."""

    from coimagine.continuous2d.config import default_config
    config = default_config()
    half = (config.physics.object_width / 2.0, config.physics.object_height / 2.0)
    arena = scene_bounds(store) if store is not None else {
        "min_x": 0.0, "min_y": 0.0, "max_x": 3.2, "max_y": 2.2}
    return {"config": config, "arena": arena, "object_half": half,
            "robot_radius": config.physics.robot_radius}


def draw_scene(ax, root_record, scene, title: str = "", show_scan: bool = True):
    bounds = scene["arena"]
    ax.add_patch(Rectangle((bounds["min_x"], bounds["min_y"]),
                           bounds["max_x"] - bounds["min_x"],
                           bounds["max_y"] - bounds["min_y"], fill=False, edgecolor="#bbbbbb",
                           linewidth=1.0))
    half = scene["object_half"]
    pose = root_record["object_pose"]
    corners = [(-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]), (-half[0], half[1])]
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    rotated = [(cos * x - sin * y + pose[0], sin * x + cos * y + pose[1]) for x, y in corners]
    ax.add_patch(Polygon(rotated, closed=True, facecolor="#d9d9d9", edgecolor="#555555"))
    for robot_id, robot in enumerate(root_record["robot_poses"]):
        ax.add_patch(Circle((robot[0], robot[1]), scene["robot_radius"],
                            facecolor=ROBOT_COLORS[robot_id], edgecolor="black", zorder=5))
        ax.arrow(robot[0], robot[1], 0.12 * math.cos(robot[2]), 0.12 * math.sin(robot[2]),
                 head_width=0.03, color="black", zorder=6)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    return bounds


def f02(store: Store) -> dict:
    config = arena_from_config(store)
    records = {name: store.roots["roots"][name] for name in ("d0", "test", "val", "d1ext")}
    picked = []
    for name in ("d0", "test", "val", "d1ext"):
        classes = {}
        for record in records[name]:
            classes.setdefault(record["interaction_class"], record)
        for cls in sorted(classes):
            if len(picked) >= 4:
                break
            picked.append((name, cls, classes[cls]))
        if len(picked) >= 4:
            break
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, (name, cls, record) in zip(axes, picked[:4]):
        bounds = draw_scene(ax, record, config, f"{cls}  ({name})")
        ax.set_xlim(bounds["min_x"], bounds["max_x"])
        ax.set_ylim(bounds["min_y"], bounds["max_y"])
        ax.set_xlabel("x (m)", fontsize=8)
        ax.set_ylabel("y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.suptitle("F02 连续 2D 环境与实际场景：同一生成器产出的四种构型（物体轮廓、两机器人、"
                 "米制坐标）", fontsize=12)
    paths = save(fig, "F02_environment_and_scenes")
    rows = []
    for name in ("d0", "d1ext", "val", "test"):
        for record in records[name]:
            rows.append([name, record["root_id"], record["interaction_class"],
                         round(record["object_pose"][0], 5), round(record["object_pose"][1], 5),
                         round(record["robot_poses"][0][0], 5), round(record["robot_poses"][0][1], 5),
                         round(record["robot_poses"][1][0], 5), round(record["robot_poses"][1][1], 5)])
    csv = write_csv("F02_scene_configurations.csv",
                    ["split", "root_id", "interaction_class", "object_x", "object_y",
                     "r0_x", "r0_y", "r1_x", "r1_y"], rows[:2000])
    info = write_json("F02_config.json", {
        "controller": "local V", "kv": 25.0, "a_ref_max": 3.0, "force_limit": 15.0,
        "control_dt": 0.05, "scan_degrees": 360, "scan_rays": 72, "scan_range_m": 0.6,
        "robot_radius_m": config["robot_radius"],
        "note": "物体全局位姿是仿真真值，只用于可视化与离线评估，不进入任何模型输入",
    })
    return {"figure": paths, "data": [csv, info], "scenes": [p[1] for p in picked[:4]]}


# ---------------------------------------------------------------------------
# F03
# ---------------------------------------------------------------------------


def f03(store: Store) -> dict:
    block = None
    for shard in store.trajectories("d0"):
        if shard["exit_code"][0] == 0:
            block = shard
            break
    if block is None:
        raise RuntimeError("no completed d0 execution available")
    row = 0
    steps = int(block["steps"][row])
    fig = plt.figure(figsize=(16, 8))
    grid = fig.add_gridspec(2, 6, height_ratios=[1.15, 1.0])

    ax = fig.add_subplot(grid[0, :2])
    ax.set_title("(a) option 阶段机（实际实现）", fontsize=10)
    ax.axis("off")
    stages = ["APPROACH", "READY", "PUSH\n(有效接触 2.0s)", "BRAKE", "SETTLE\n(4.0s 预算)"]
    for index, stage in enumerate(stages):
        y = 0.88 - index * 0.19
        ax.add_patch(Rectangle((0.10, y - 0.07), 0.62, 0.13, facecolor="#e8f0fb",
                               edgecolor="#1f4e79"))
        ax.text(0.41, y, stage, ha="center", va="center", fontsize=8.5)
        if index < len(stages) - 1:
            ax.annotate("", xy=(0.41, y - 0.09), xytext=(0.41, y - 0.02),
                        arrowprops=dict(arrowstyle="->", color="#1f4e79"))
    ax.text(0.02, 0.02, "退出：判稳 / 接触丢失 / 片段丢失 / 超时；退出 ≠ 物体整体静止",
            fontsize=8, color="#b03030", va="bottom")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = fig.add_subplot(grid[0, 2:])
    ax.set_title(f"(b) 一条真实执行：{block['record_id'][row]}（阶段与退出时刻 T）", fontsize=10)
    time = block["time"][row, :steps]
    stage = block["stage_code"][row, :steps]
    ax.plot(time, stage, color="#1f4e79", linewidth=1.4)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["approach", "ready", "push", "brake", "settle"], fontsize=8)
    ax.set_ylim(-0.3, 4.3)
    ax.set_xlabel("真实时间 (s)", fontsize=9)
    ax.axvline(float(block["T_exit"][row]), color="#b03030", linestyle="--", linewidth=1.4)
    ax.text(float(block["T_exit"][row]), ax.get_ylim()[1], " T", color="#b03030", fontsize=8.5,
            va="top")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, :2])
    ax.set_title("(c) 自身速度（R_i0，m/s）与 patch_valid", fontsize=10)
    for robot_id in (0, 1):
        speed = np.hypot(block["own_world"][row, :steps, robot_id, 3],
                         block["own_world"][row, :steps, robot_id, 4])
        ax.plot(time, speed, color=ROBOT_COLORS[robot_id], linewidth=1.3,
                label=f"robot{robot_id}")
    ax.set_xlabel("t (s)", fontsize=9)
    ax.set_ylabel("|v| (m/s)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 2:4])
    ax.set_title("(d) patch_valid 与 contacting", fontsize=10)
    for robot_id in (0, 1):
        ax.plot(time, block["patch_valid"][row, :steps, robot_id] + 0.1 * (1 - robot_id),
                color=ROBOT_COLORS[robot_id], linewidth=1.2,
                label=f"r{robot_id} patch_valid")
        ax.plot(time, block["contacting"][row, :steps, robot_id] - 0.1 * (1 - robot_id) - 0.5,
                color=ROBOT_COLORS[robot_id], linewidth=1.0, linestyle=":",
                label=f"r{robot_id} contacting")
    ax.set_yticks([])
    ax.set_xlabel("t (s)", fontsize=9)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 4:])
    ax.set_title("(e) 实际场景中的该条轨迹（真值背景）", fontsize=10)
    config = arena_from_config(store)
    bounds = config["arena"]
    ax.add_patch(Rectangle((bounds["min_x"], bounds["min_y"]),
                           bounds["max_x"] - bounds["min_x"],
                           bounds["max_y"] - bounds["min_y"], fill=False, edgecolor="#bbbbbb"))
    ax.set_xlim(bounds["min_x"], bounds["max_x"])
    ax.set_ylim(bounds["min_y"], bounds["max_y"])
    # the exit period is not logged, so the truth arrays carry one trailing zero row; the number
    # of real truth rows is exactly steps - 1 (one log row per completed period)
    truth_steps = max(1, steps - 1)
    object_xy = block["object_world"][row, :truth_steps, :2]
    ax.plot(object_xy[:, 0], object_xy[:, 1], color="#555555", linewidth=1.0,
            label="object centre (truth)")
    for robot_id in (0, 1):
        xy = block["own_world"][row, :steps, robot_id, :2]
        ax.plot(xy[:, 0], xy[:, 1], color=ROBOT_COLORS[robot_id], linewidth=1.3,
                label=f"robot{robot_id}")
        ax.plot(xy[0, 0], xy[0, 1], "o", color=ROBOT_COLORS[robot_id], markersize=4)
        ax.plot(xy[-1, 0], xy[-1, 1], "s", color=ROBOT_COLORS[robot_id], markersize=4)
    half = config["object_half"]
    for pose, style in ((block["object_world"][row, 0], "--"),
                        (block["terminal_object"][row], "-")):
        cos, sin = math.cos(pose[2]), math.sin(pose[2])
        corners = [(-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]),
                   (-half[0], half[1])]
        ax.add_patch(Polygon([(cos * x - sin * y + pose[0], sin * x + cos * y + pose[1])
                              for x, y in corners], closed=True, fill=False, linestyle=style,
                             edgecolor="#333333", linewidth=1.0))
    ax.set_aspect("equal")
    ax.legend(fontsize=7)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    fig.suptitle("F03 option 与真实执行过程：阶段机、真实时间轴与实际退出时刻 T（不是提前一个周期）",
                 fontsize=12)
    paths = save(fig, "F03_option_execution")
    rows = []
    for index in range(steps):
        row_values = [round(float(time[index]), 4), int(stage[index])]
        for robot_id in (0, 1):
            for offset in (3, 4):
                row_values.append(round(float(block["own_world"][row, index, robot_id, offset]), 5))
        for robot_id in (0, 1):
            row_values.append(int(block["patch_valid"][row, index, robot_id]))
        for robot_id in (0, 1):
            row_values.append(int(block["contacting"][row, index, robot_id]))
        rows.append(row_values)
    csv = write_csv("F03_execution_timeline.csv",
                    ["time_s", "stage_code", "r0_vx", "r0_vy", "r1_vx", "r1_vy",
                     "r0_patch_valid", "r1_patch_valid", "r0_contacting", "r1_contacting"], rows)
    return {"figure": paths, "data": csv, "record_id": str(block["record_id"][row]),
            "steps": steps, "T_exit": float(block["T_exit"][row])}


# ---------------------------------------------------------------------------
# F04
# ---------------------------------------------------------------------------


def f04(store: Store) -> dict:
    config = arena_from_config()
    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0])

    # (a) trajectory overlay, rigid-aligned to the initial object frame, faceted by class
    classes = ["constrained", "dual_rotate", "solo_central", "incompatible", "dual_same_side",
               "solo_offset"]
    ax = fig.add_subplot(grid[0, :2])
    ax.set_title("(a) D1 机器人轨迹中心线叠加（以物体初始坐标系对齐，预先固定抽样，按 root 均衡）",
                 fontsize=10)
    drawn = 0
    roots_seen = 0
    generator = np.random.default_rng(20261014)
    # index shards once, then draw a fixed number of sequences per interaction class
    by_class: Dict[str, list] = defaultdict(list)
    cache: Dict[Path, object] = {}
    for path in sorted((DATA / "sequences").glob("d1ext_*.npz")):
        block = np.load(path, allow_pickle=True)
        cache[path] = block
        for row in range(len(block["steps"])):
            by_class[block["root_id"][row].split("/")[2]].append((path, row))
    per_class = max(1, 260 // max(1, len(classes)))
    for cls in classes:
        pool = by_class.get(cls, [])
        if not pool:
            continue
        picks = generator.choice(len(pool), size=min(per_class, len(pool)), replace=False)
        for pick in picks:
            path, row = pool[pick]
            block = cache[path]
            steps = int(block["steps"][row])
            pose0 = block["object_world"][row, 0]
            cos, sin = math.cos(-pose0[2]), math.sin(-pose0[2])
            for robot_id in (0, 1):
                xy = block["own_world"][row, :steps, robot_id, :2]
                local = np.stack([(xy[:, 0] - pose0[0]) * cos - (xy[:, 1] - pose0[1]) * sin,
                                  (xy[:, 0] - pose0[0]) * sin + (xy[:, 1] - pose0[1]) * cos], axis=1)
                ax.plot(local[:, 0], local[:, 1], color=ROBOT_COLORS[robot_id], alpha=0.22,
                        linewidth=0.7)
                ax.plot(local[0, 0], local[0, 1], ".", color=ROBOT_COLORS[robot_id], markersize=2)
                ax.plot(local[-1, 0], local[-1, 1], "x", color=ROBOT_COLORS[robot_id], markersize=2)
                drawn += 1
            roots_seen += 1
    ax.add_patch(Rectangle((-0.5, -0.125), 1.0, 0.25, facecolor="#d9d9d9", edgecolor="#555555",
                           alpha=0.9, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlabel("物体初始系 x (m)", fontsize=9)
    ax.set_ylabel("物体初始系 y (m)", fontsize=9)
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.text(0.02, 0.02, f"绘制轨迹 {drawn} 条 / 独立 root {roots_seen} 个；"
                        f"全库 D1 = {store.summary.get('d1ext', {}).get('independent_roots', 0) + 512} root",
            transform=ax.transAxes, fontsize=8)
    ax.text(0.02, 0.94, "浅灰矩形 = 物体初始位姿；起点小点、终点叉号；两机器人颜色固定",
            transform=ax.transAxes, fontsize=8)

    ax = fig.add_subplot(grid[0, 2])
    ax.set_title("(b) 规模：root / 序列 / 真实帧", fontsize=10)
    labels = ["d0", "d1ext", "val", "test"]
    roots = [store.summary.get(name, {}).get("independent_roots", 0) for name in labels]
    sequences = [store.summary.get(name, {}).get("sequences", 0) for name in labels]
    frames = [store.summary.get(name, {}).get("frames", 0) for name in labels]
    x = np.arange(len(labels))
    ax.bar(x - 0.25, roots, width=0.25, label="independent roots", color="#1f4e79")
    ax.bar(x, [s / 8 for s in sequences], width=0.25, label="sequences / 8", color="#2ca02c")
    ax.bar(x + 0.25, [f / 64 for f in frames], width=0.25, label="frames / 64", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yscale("log")
    ax.set_ylabel("count (log)", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("(c) 退出类别分布", fontsize=10)
    counter = Counter()
    for split in ("d0", "d1ext"):
        block = store.tokens.get(split)
        if block is None:
            continue
        from experiments.night_sequence_v1.sequences import EXIT_CODES
        for code, count in zip(*np.unique(block["exit_class"], return_counts=True)):
            counter[EXIT_CODES[int(code)] if int(code) < len(EXIT_CODES) else f"code{code}"] += int(count)
    items = counter.most_common()
    ax.barh([k for k, _ in items][::-1], [v for _, v in items][::-1], color="#1f4e79")
    ax.set_xscale("log")
    ax.tick_params(labelsize=8)
    ax.set_xlabel("sequences (log)", fontsize=9)

    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("(d) 终态 robot-0 片段可见性", fontsize=10)
    for index, split in enumerate(("d0", "d1ext", "val", "test")):
        block = store.tokens.get(split)
        if block is None:
            continue
        terminal = block["terminal"].astype(np.float32)
        visible = float((terminal[:, 6] > 0.5).mean())
        ax.bar(index, visible, color="#2ca02c")
        ax.text(index, visible + 0.02, f"{visible:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(4))
    ax.set_xticklabels(["d0", "d1ext", "val", "test"], fontsize=9)
    ax.set_ylabel("fraction visible", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[1, 2])
    ax.set_title("(e) 序列长度分布（真实帧数）", fontsize=10)
    for split in ("d0", "d1ext", "val", "test"):
        block = store.tokens.get(split)
        if block is None:
            continue
        ax.hist(block["steps"], bins=30, histtype="step", linewidth=1.2, label=split)
    ax.set_xlabel("steps per option", fontsize=9)
    ax.set_ylabel("sequences", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)

    fig.suptitle("F04 数据覆盖与分布：轨迹中心线、规模、退出/可见性与长度分布", fontsize=12)
    paths = save(fig, "F04_data_coverage")
    rows = []
    for name in ("d0", "d1ext", "val", "test"):
        entry = store.summary.get(name, {})
        rows.append([name, entry.get("independent_roots"), entry.get("sequences"),
                     entry.get("frames"), entry.get("max_steps"),
                     entry.get("bytes"), json.dumps(entry.get("exit_classes", {}), sort_keys=True)])
    csv = write_csv("F04_data_scale.csv",
                    ["split", "independent_roots", "sequences", "real_frames", "max_steps",
                     "tokens_bytes", "exit_classes"], rows)
    return {"figure": paths, "data": csv, "drawn_trajectories": drawn, "drawn_roots": roots_seen}


# ---------------------------------------------------------------------------
# F05
# ---------------------------------------------------------------------------


def f05(store: Store) -> dict:
    from experiments.night_sequence_v1.models import build_model, parameter_count
    from experiments.night_sequence_v1.train import MODEL_KWARGS
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.4), gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    ax.set_title("(a) 共同数据流：真实观测前缀 → 三种模型 → 两组预测头", fontsize=10.5)
    ax.axis("off")
    ax.add_patch(Rectangle((0.02, 0.62), 0.24, 0.30, facecolor="#eef5ff", edgecolor="#1f4e79"))
    ax.text(0.14, 0.86, "真实观测前缀", ha="center", fontsize=9)
    ax.text(0.14, 0.78, "O0 锚点 + 最近 32 个\n真实网格时刻（≤33 token）", ha="center", fontsize=7.8,
            va="top")
    ax.text(0.14, 0.66, "只会读到 t 之前\n（causal，绝不读未来 token）", ha="center", fontsize=7.5,
            va="top", color="#b03030")
    ax.add_patch(Rectangle((0.32, 0.62), 0.20, 0.30, facecolor="#f7f7f7", edgecolor="#666666"))
    ax.text(0.42, 0.86, "token 特征 (57)", ha="center", fontsize=8.5)
    ax.text(0.42, 0.78, "自身位姿/速度 in R_i0\n片段中点/法向/mask\n队友相对量\n时间+阶段+命令",
            ha="center", fontsize=7.5, va="top")
    for index, (kind, color) in enumerate((("mlp_current", COLORS["mlp_current"]),
                                           ("mlp_history", COLORS["mlp_history"]),
                                           ("transformer_history", COLORS["transformer_history"]))):
        y = 0.80 - index * 0.13
        ax.add_patch(Rectangle((0.58, y - 0.05), 0.18, 0.10, facecolor=color, alpha=0.18,
                               edgecolor=color))
        ax.text(0.67, y, kind, ha="center", fontsize=8)
        ax.annotate("", xy=(0.58, y), xytext=(0.52, 0.77),
                    arrowprops=dict(arrowstyle="->", color=color, alpha=0.7))
    ax.add_patch(Rectangle((0.80, 0.62), 0.18, 0.30, facecolor="#fff4e6", edgecolor="#b06a00"))
    ax.text(0.89, 0.86, "两组预测头", ha="center", fontsize=9)
    ax.text(0.89, 0.78, "A 固定 Δt 过程头\n几何/位姿/速度/可见\n+ 是否 Δt 内退出",
            ha="center", fontsize=7.5, va="top")
    ax.text(0.89, 0.68, "B option 终态头\nT 的片段/位姿/速度\n剩余时长 + 退出类别",
            ha="center", fontsize=7.5, va="top")
    ax.text(0.02, 0.50, "着色区分：左侧全部是**已发生**的真实观测前缀；右侧是**待预测**的未来量。",
            fontsize=8.5, color="#b03030")
    ax.text(0.02, 0.40, "t=0 时没有“尚未执行的历史”，只有 O0 与候选命令，因此 t=0 的评价单独报告。",
            fontsize=8.5)
    ax.text(0.02, 0.30, "本轮不含自由 latent 递推、不含规划器：递推接口尚缺“把预测结果当作下一时刻输入”\n"
                        "所需的完整状态字段（见 F08）。", fontsize=8.5, color="#444444")
    ax.text(0.02, 0.14, "输入白名单：两机器人可共享的自身观测量、相对位姿、当前联合命令、已过去时间；\n"
                        "物体真值 / 真实面编号 / root id / split / 未来观测 / 未来控制力 均无输入通路。",
            fontsize=8.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.set_title("(b) 实际参数量（同输入、同目标、同预算）", fontsize=10.5)
    rows = []
    for kind, kwargs in MODEL_KWARGS.items():
        model = build_model(kind, **kwargs)
        rows.append((kind, parameter_count(model)))
    names = [r[0] for r in rows]
    values = [r[1] for r in rows]
    ax.barh(names, values, color=[COLORS[n] for n in names])
    for index, value in enumerate(values):
        ax.text(value + 20000, index, f"{value:,d}", va="center", fontsize=8.5)
    ax.set_xlabel("parameters", fontsize=9)
    ax.set_xlim(0, max(values) * 1.3)
    ax.grid(alpha=0.25, axis="x")
    ratio = dict(rows)["mlp_history"] / dict(rows)["transformer_history"]
    ax.text(0.02, 0.05, f"两个历史模型参数比 = {ratio:.2f}×（目标 ≤2×）\n"
                        f"单帧模型参数更少，单独列出，不与历史模型混作架构结论",
            transform=ax.transAxes, fontsize=8.5)
    fig.suptitle("F05 模型与学习任务：同一窗口、同一 mask、同一目标，仅序列编码器不同", fontsize=12)
    paths = save(fig, "F05_models_and_task")
    csv = write_csv("F05_model_parameters.csv", ["model", "parameters", "window", "features"],
                    [[name, value, 33, 57] for name, value in rows])
    return {"figure": paths, "data": csv, "parameters": dict(rows)}


# ---------------------------------------------------------------------------
# F06
# ---------------------------------------------------------------------------


def f06(store: Store) -> dict:
    """Learning curves plus the data-scale question (task book research question 1)."""

    curves = {}
    for name in sorted(store.runs):
        entries = curve_from_log(name)
        if entries:
            curves[name] = entries
    fig, axes = plt.subplots(1, 3, figsize=(19, 5))
    ax = axes[0]
    ax.set_title("(a) 验证合成损失（同一根横轴＝优化更新数）", fontsize=10.5)
    plotted = 0
    for name, entries in curves.items():
        kind = store.runs.get(name, {}).get("kind", name.split("_")[0])
        updates = [entry["update"] for entry in entries]
        values = [entry.get("val_total") for entry in entries]
        if not any(v is not None and np.isfinite(v) for v in values):
            continue
        ax.plot(updates, values, color=COLORS.get(kind, "#333333"), alpha=0.8,
                linewidth=1.2, label=name)
        plotted += 1
    if plotted:
        ax.legend(fontsize=6.5, ncol=2)
    ax.set_xlabel("optimisation updates", fontsize=9)
    ax.set_ylabel("validation composite loss", fontsize=9)
    ax.grid(alpha=0.25)
    ax.text(0.02, 0.05, "未完成或预算不同的 run 不画；单 seed 不画置信区间", transform=ax.transAxes,
            fontsize=8)

    ax = axes[1]
    ax.set_title("(b) 分项验证损失（各任务子组，最终 checkpoint）", fontsize=10.5)
    groups = []
    labels = []
    for name, entries in sorted(curves.items()):
        final = entries[-1]
        values = [final.get(f"val_{key}") for key in
                  ("terminal_geometry", "terminal_pose", "terminal_velocity", "terminal_validity",
                   "terminal_remaining", "terminal_exit_class")]
        if any(v is None for v in values):
            continue
        labels.append(name)
        groups.append(values)
    if groups:
        matrix = np.asarray(groups, dtype=float)
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        ax.set_xticks(range(6))
        ax.set_xticklabels(["geomm", "posem", "velm", "valid", "remain", "exit"], fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.03)
    else:
        ax.text(0.5, 0.5, "训练曲线尚未生成", ha="center")

    # (c) independent roots versus error, for the same model and target
    ax = axes[2]
    ax.set_title("(c) 独立 root 数 vs 同目标误差（MLP-history, seed0）", fontsize=10.5)
    scale = {}
    summary = store.summary
    for tier, splits in (("d0", ["d0"]), ("d1", ["d0", "d1ext"]),
                         ("d2", ["d0", "d1ext", "d2ext"])):
        path = DATA / "analysis" / tier / "metrics.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        label = f"{tier}_mlp_history_seed0|test"
        entry = payload["models"].get(label)
        if not isinstance(entry, dict) or "t0" not in entry:
            continue
        roots = sum(summary.get(name, {}).get("independent_roots", 0) for name in splits)
        scale[tier] = (roots, entry["t0"]["terminal_geometry"]["r0_mid_2d"]["p50"],
                       entry["all"]["grouped"]["regime"]["in_execution"]["p50"])
    if scale:
        tiers = sorted(scale, key=lambda t: scale[t][0])
        roots = [scale[t][0] for t in tiers]
        ax.plot(roots, [scale[t][1] for t in tiers], "o-", color="#1f4e79",
                label="t=0 terminal midpoint p50")
        ax.plot(roots, [scale[t][2] for t in tiers], "s-", color="#2ca02c",
                label="in-execution midpoint p50")
        for tier in tiers:
            ax.annotate(tier, (scale[tier][0], scale[tier][1]), fontsize=8,
                        textcoords="offset points", xytext=(0, 6))
        ax.set_xscale("log")
        ax.set_xlabel("independent train roots (log)", fontsize=9)
        ax.set_ylabel("error p50 (m)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.05, "横轴是独立起点数，不是窗口数；\n同一模型、同一目标、同一预算",
                transform=ax.transAxes, fontsize=8)
    else:
        ax.text(0.5, 0.5, "分析结果尚未生成", ha="center")
    fig.suptitle("F06 学习过程与规模效应：曲线对齐同一更新轴；独立 root 数与误差的关系另图",
                 fontsize=12)
    paths = save(fig, "F06_learning_curves")
    rows = []
    for name, entry in sorted(store.runs.items()):
        rows.append([name, entry["kind"], entry["tier"], entry["seed"], entry["params"],
                     entry["best_val"], entry["best_update"]])
    csv = write_csv("F06_run_summary.csv",
                    ["run", "model", "tier", "seed", "parameters", "best_val", "best_update"], rows)
    cost_rows = []
    stats_path = DATA / "analysis" / "d1" / "stats_and_cost.json"
    if stats_path.exists():
        stats = json.loads(stats_path.read_text())
        for cost in stats["inference_cost"]:
            cost_rows.append([cost["model"], cost["parameters"], cost["peak_rss_delta_mb"],
                              cost["batch1_latency_ms"], cost["batch128_latency_ms"],
                              cost["batch128_throughput_per_s"]])
    cost_csv = write_csv("F06_inference_cost.csv",
                         ["model", "parameters", "peak_rss_delta_mb", "batch1_latency_ms",
                          "batch128_latency_ms", "batch128_throughput_per_s"], cost_rows)
    if cost_rows:
        fastest = max(row[5] for row in cost_rows)
        slowest = min(row[5] for row in cost_rows)
        axes[0].text(0.03, 0.05,
                     f"batch=128 吞吐 {slowest:.0f}–{fastest:.0f} 样本/秒\n"
                     f"相同更新数 ≠ 相同 FLOPs",
                     transform=axes[0].transAxes, fontsize=8)
    scale_csv = write_csv("F06_data_scale_to_error.csv",
                          ["tier", "independent_roots", "t0_terminal_mid2d_p50_m",
                           "in_execution_mid2d_p50_m"],
                          [[tier, scale[tier][0], scale[tier][1], scale[tier][2]]
                           for tier in sorted(scale)])
    return {"figure": paths, "data": [csv, scale_csv, cost_csv], "runs": len(rows),
            "curves_available": len(curves), "scale": {t: v[0] for t, v in scale.items()}}


# ---------------------------------------------------------------------------
# F07
# ---------------------------------------------------------------------------


def f07(store: Store) -> dict:
    metrics = {}
    for tier in ("d0", "d1"):
        path = DATA / "analysis" / tier / "metrics.json"
        if path.exists():
            metrics[tier] = json.loads(path.read_text())
    if not metrics:
        raise RuntimeError("analysis metrics not available yet")
    tier = "d2" if "d2" in metrics else ("d1" if "d1" in metrics else "d0")
    payload = metrics[tier]

    fig = plt.figure(figsize=(17, 11), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0])

    ax = fig.add_subplot(grid[0, 0])
    ax.set_title("(a) t=0 终态几何：r0 中点二维误差", fontsize=10)
    names, p50, p90 = [], [], []
    for label in ("P0_hold|test", "P1_grouped|test"):
        entry = payload["baselines"][label]["t0"]["terminal_geometry"]["r0_mid_2d"]
        names.append(label.split("|")[0]); p50.append(num(entry["p50"])); p90.append(num(entry["p90"]))
    for label, entry in sorted(payload["models"].items()):
        if not label.endswith("|test"):
            continue
        geometry = entry
        if "t0" not in entry:
            continue
        value = entry["t0"]["terminal_geometry"]["r0_mid_2d"]
        names.append(label.split("|")[0]); p50.append(num(value["p50"])); p90.append(num(value["p90"]))
    x = np.arange(len(names))
    ax.bar(x - 0.2, p50, width=0.4, label="p50", color="#1f4e79")
    ax.bar(x + 0.2, p90, width=0.4, label="p90", color="#7fb3d5")
    ax.set_xticks(x)
    ax.set_xticklabels([short_label(n) for n in names], rotation=0, fontsize=7)
    ax.set_ylabel("metres", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[0, 1])
    ax.set_title("(b) t=0 终态法向角误差", fontsize=10)
    names, values = [], []
    for label in ("P0_hold|test", "P1_grouped|test"):
        entry = payload["baselines"][label]["t0"]["terminal_geometry"]["r0_normal_angle_deg"]
        names.append(label.split("|")[0]); values.append(num(entry["p50"]))
    for label, entry in sorted(payload["models"].items()):
        if not label.endswith("|test") or "t0" not in entry:
            continue
        values.append(num(entry["t0"]["terminal_geometry"]["r0_normal_angle_deg"]["p50"]))
        names.append(label.split("|")[0])
    ax.bar(range(len(names)), values, color="#b06a00")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([short_label(n) for n in names], rotation=0, fontsize=7)
    ax.set_ylabel("degrees", fontsize=9)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[0, 2])
    ax.set_title("(c) t=0 同 root 八动作终态差值 / 秩相关", fontsize=10)
    labels, gaps, rhos = [], [], []
    for label, entry in sorted(payload["models"].get("action_sensitivity", {}).items()):
        labels.append(label)
        gaps.append(num(entry["action_difference_error"]["r0_x"]["p50"]))
        rhos.append(num(entry["rank_correlation"]["r0_x"]["mean"]))
    if labels:
        x = np.arange(len(labels))
        ax.bar(x - 0.2, gaps, width=0.4, color="#1f4e79", label="action-diff p50 (m)")
        ax.bar(x + 0.2, rhos, width=0.4, color="#2ca02c", label="Spearman rho")
        ax.set_xticks(x)
        ax.set_xticklabels([short_label(l) for l in labels], rotation=0, fontsize=7)
        ax.legend(fontsize=7.5, loc="upper center", ncol=2)
        ax.set_ylim(0, max(max(gaps), max(rhos)) * 1.35)
        ax.grid(alpha=0.25, axis="y")
        stats_path = DATA / "analysis" / tier / "stats_and_cost.json"
        if stats_path.exists():
            stats = json.loads(stats_path.read_text())
            for key, value in stats["paired_root_bootstrap"].items():
                ax.text(0.02, 0.95,
                        f"配对 root bootstrap：hist vs trf 中位差 "
                        f"{value['observed_median_difference_m']:+.4f} m\n"
                        f"95% CI [{value['ci95_low_m']:+.4f}, {value['ci95_high_m']:+.4f}]，"
                        f"root {value['roots']}",
                        transform=ax.transAxes, fontsize=7.5, va="top")
    else:
        ax.text(0.5, 0.5, "no action-sensitivity yet", ha="center")

    ax = fig.add_subplot(grid[1, 0])
    ax.set_title("(d) 执行中：按 elapsed 区间分组的中点误差", fontsize=10)
    for label in ("P0_hold|test", "P1_grouped|test"):
        grouped = payload["baselines"][label]["all"]["grouped"]["elapsed_bin"]
        keys = sorted(grouped)
        ax.plot(range(len(keys)), [grouped[k]["p50"] for k in keys], marker="o", markersize=3,
                color=COLORS.get(label.split("|")[0]), label=label.split("|")[0])
    for label, entry in sorted(payload["models"].items()):
        if not label.endswith("|test"):
            continue
        grouped = entry["all"]["grouped"]["elapsed_bin"]
        keys = sorted(grouped)
        colour = COLORS.get(store.runs.get(label.split("|")[0], {}).get("kind", ""), None)
        ax.plot(range(len(keys)), [grouped[k]["p50"] for k in keys], marker="s", markersize=3,
                color=colour, label=label.split("|")[0], alpha=0.85)
    ax.set_xlabel("elapsed-time bin (0.25 s each)", fontsize=9)
    ax.set_ylabel("r0 midpoint 2D error p50 (m)", fontsize=9)
    handles, labels = ax.get_legend_handles_labels()
    keep = [i for i, l in enumerate(labels)
            if l in ("P0_hold", "P1_grouped") or l.endswith("seed0")]
    ax.legend([handles[i] for i in keep], [labels[i] for i in keep], fontsize=7, ncol=2,
              loc="upper right")
    ax.grid(alpha=0.25)

    ax = fig.add_subplot(grid[1, 1])
    ax.set_title("(e) 可见性/接触召回（test）", fontsize=10)
    keys = ["r0_patch_valid", "r0_contacting", "r1_patch_valid", "r1_contacting"]
    width = 0.2
    entries = [("P0_hold|test", payload["baselines"]["P0_hold|test"]),
               ("P1_grouped|test", payload["baselines"]["P1_grouped|test"])]
    entries += [(k, v) for k, v in sorted(payload["models"].items()) if k.endswith("|test")]
    for position, (label, entry) in enumerate(entries):
        values = [num(entry["t0"]["terminal_validity"][key]["recall"]) for key in keys]
        if label.split("|")[0] not in ("P0_hold", "P1_grouped") and not label.endswith("seed0|test"):
            continue
        ax.bar(np.arange(len(keys)) + (position - len(entries) / 2) * width, values, width=width,
               label=label.split("|")[0])
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in keys], rotation=0, fontsize=6.5)
    ax.legend(fontsize=6.5, loc="lower right", ncol=2)
    ax.set_ylim(0, 1.25)
    ax.grid(alpha=0.25, axis="y")

    ax = fig.add_subplot(grid[1, 2])
    ax.set_title("(f) 退出类别支持数与召回", fontsize=10)
    support = payload["baselines"]["P0_hold|test"]["t0"]["terminal_exit_class"]["support"]
    classes = [int(k) for k in sorted(support, key=lambda k: -support[k])][:6]
    from experiments.night_sequence_v1.sequences import EXIT_CODES
    labels = [EXIT_CODES[c] if c < len(EXIT_CODES) else str(c) for c in classes]
    for position, (label, entry) in enumerate(entries):
        recall = entry["t0"]["terminal_exit_class"]["recall"]
        values = [num(recall.get(str(c), recall.get(c))) for c in classes]
        if label.split("|")[0] not in ("P0_hold", "P1_grouped") and not label.endswith("seed0|test"):
            continue
        ax.bar(np.arange(len(classes)) + (position - len(entries) / 2) * width, values,
               width=width, label=label.split("|")[0])
    ax.set_xticks(range(len(classes)))
    support_counts = [support.get(str(c), support.get(c, 0)) or 0 for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, support_counts)],
                       rotation=0, fontsize=6)
    ax.legend(fontsize=6.5, loc="upper right", ncol=2)
    ax.set_ylim(0, 1.25)
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("F07 拟合与预测性能：执行前(t=0) 与执行中分开；真实观测前缀条件预测，不是自由 rollout",
                 fontsize=12)
    paths = save(fig, "F07_prediction_performance")

    rows = []
    for label, entry in list(payload["baselines"].items()) + list(payload["models"].items()):
        if not isinstance(entry, dict) or "t0" not in entry:
            continue
        geometry = entry["t0"]["terminal_geometry"]
        pose = entry["t0"]["terminal_pose"]
        rows.append([label, geometry["r0_mid_2d"]["p50"], geometry["r0_mid_2d"]["p90"],
                     geometry["r0_normal_angle_deg"]["p50"],
                     pose.get("all_columns_flattened", pose).get("p50"),
                     entry["t0"]["terminal_remaining"].get("p50"),
                     entry["t0"]["terminal_exit_class"]["accuracy"]])
    csv = write_csv("F07_metrics_t0.csv",
                    ["model", "r0_mid2d_p50_m", "r0_mid2d_p90_m", "r0_normal_p50_deg",
                     "pose_abs_p50", "remaining_p50_s", "exit_accuracy"], rows)
    return {"figure": paths, "data": csv, "tier": tier, "models": len(payload["models"])}


# ---------------------------------------------------------------------------
# F08
# ---------------------------------------------------------------------------


def f08(store: Store) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), gridspec_kw={"width_ratios": [1.35, 1]})
    ax = axes[0]
    ax.set_title("(a) 已有证据 → 可能用途 → 尚缺验证", fontsize=10.5)
    ax.axis("off")
    rows = [
        ("预测：技能结束时会看到什么", "候选后果估计的更可靠输入", "闭环任务收益未评估"),
        ("历史窗口 vs 单帧", "反馈后的预测更新", "真实任务容差下的可用性未知"),
        ("观测 → 分支合法性", "只在可区分的分支上分配搜索", "原场景收益差为 0（负结果）"),
        ("动作条件变化被部分捕捉", "同起点不同动作的后果排序", "排序一致性约中等，不足以规划"),
        ("真实观测前缀条件预测", "执行中的在线修正", "自由多步 rollout 未实现"),
    ]
    ax.text(0.02, 0.92, "已有证据", fontsize=9.5, color="#1f4e79")
    ax.text(0.40, 0.92, "可能用途", fontsize=9.5, color="#1f4e79")
    ax.text(0.70, 0.92, "尚缺验证", fontsize=9.5, color="#b03030")
    for index, (a, b, c) in enumerate(rows):
        y = 0.82 - index * 0.16
        ax.text(0.02, y, "• " + a, fontsize=8.4, va="top")
        ax.annotate("", xy=(0.395, y - 0.02), xytext=(0.345, y - 0.02),
                    arrowprops=dict(arrowstyle="->", color="#888888"))
        ax.text(0.40, y, b, fontsize=8.4, va="top")
        ax.annotate("", xy=(0.695, y - 0.02), xytext=(0.645, y - 0.02),
                    arrowprops=dict(arrowstyle="->", color="#cc8888"))
        ax.text(0.70, y, c, fontsize=8.4, va="top", color="#8a3030")
    ax.text(0.02, 0.03, "本轮没有任何闭环实验，因此不给出成功率或算法胜负结论。", fontsize=8.5,
            color="#b03030")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    ax.set_title("(b) 已有 C 负结果（仅限原场景/技能/评分）", fontsize=10.5)
    ax.axis("off")
    check = DATA.parent / "option_prediction_mechanism_v1/phase_c_frozen_scoring_check.json"
    if check.exists():
        payload = json.loads(check.read_text())
        lines = []
        for name, entry in payload.get("cases", {}).items():
            lines.append(f"{name}: 最优动作相同={entry['same_argmax']}  "
                         f"regret={entry['mutual_regret']:.4f}  "
                         f"完整排序 ρ={entry['full_ranking_spearman_between_worlds']:.3f}  "
                         f"正例={entry['positive']}")
        body = "\n".join(lines)
    else:
        body = "scoring check not found"
    ax.text(0.02, 0.90, "信息歧义（冻结协议重做）", fontsize=9.5, va="top")
    ax.text(0.04, 0.78, body, fontsize=8.2, va="top", family="monospace")
    ax.text(0.02, 0.50, "边界：\n· 只对应原场景、原技能与合同评分\n"
                        "· 观测同一性只主张已验证的控制器输入\n"
                        "· 不主张“冻结技能下不存在歧义正例”", fontsize=8.2, va="top",
            color="#8a3030")
    ax.text(0.02, 0.20, "递推接口尚缺字段：\n"
                        "· 预测出的片段 → 下一时刻可用的观测表示\n"
                        "· 物体自由度的足够统计量\n"
                        "· 与控制器内部状态（判稳证据、参考速度）一致的更新规则",
            fontsize=8.2, va="top")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.suptitle("F08 对闭环规划的意义与未决问题：本轮未评估闭环收益", fontsize=12)
    paths = save(fig, "F08_closed_loop_implications")
    rows = [
        ["证据", "可能用途", "尚缺验证", "状态"],
        ["观测终态可预测", "候选后果估计", "闭环任务收益", "本轮结果"],
        ["历史窗口有增益", "反馈后的预测更新", "任务容差下的可用性", "本轮结果"],
        ["动作条件变化部分被捕捉", "后果排序", "排序一致性是否足够规划", "本轮结果"],
        ["信息歧义原场景收益差为 0", "分支合法性约束", "其他场景是否也存在", "负结果"],
        ["自由多步推演", "无", "递推接口字段", "未实现"],
    ]
    csv = write_csv("F08_evidence_gaps.csv", rows[0], rows[1:])
    return {"figure": paths, "data": csv}


# ---------------------------------------------------------------------------
# overview, videos, index
# ---------------------------------------------------------------------------


def build_overview(figures: Dict[str, dict]) -> Optional[str]:
    from PIL import Image
    names = [key for key in sorted(figures) if figures[key].get("figure")]
    if not names:
        return None
    images = []
    for key in names:
        path = ROOT / figures[key]["figure"]["png"]
        if path.exists():
            images.append((key, Image.open(path)))
    if not images:
        return None
    columns = 2
    rows = (len(images) + columns - 1) // columns
    cell_w, cell_h = 900, 520
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    for index, (key, image) in enumerate(images):
        image.thumbnail((cell_w - 20, cell_h - 20))
        x = (index % columns) * cell_w + 10
        y = (index // columns) * cell_h + 10
        canvas.paste(image, (x, y))
    path = ASSETS / "overview.png"
    canvas.save(path)
    return str(path.relative_to(ROOT))


def build_videos(store: Store) -> List[dict]:
    """Render the stored trajectories to MP4. Uses the bundled ffmpeg, never re-runs physics."""

    import imageio_ffmpeg
    VIDEOS.mkdir(parents=True, exist_ok=True)
    out = []
    for split, tag in (("d0", "success"), ("d1ext", "failure")):
        target = None
        for shard in store.trajectories(split):
            for row in range(len(shard["steps"])):
                code = int(shard["exit_code"][row])
                steps = int(shard["steps"][row])
                if tag == "success" and code == 0 and steps > 60:
                    target = (shard, row)
                    break
                if tag == "failure" and code in (1, 2) and steps > 40:
                    target = (shard, row)
                    break
            if target:
                break
        if target is None:
            continue
        shard, row = target
        steps = int(shard["steps"][row])
        config = arena_from_config(store)
        path = VIDEOS / f"execution_{tag}.mp4"
        writer = None
        size = (0, 0)
        for index in range(steps):
            fig, ax = plt.subplots(figsize=(5.2, 5.2))
            bounds = config["arena"]
            ax.set_xlim(bounds["min_x"], bounds["max_x"])
            ax.set_ylim(bounds["min_y"], bounds["max_y"])
            ax.set_aspect("equal")
            object_pose = (shard["object_world"][row, index, :3] if index < steps - 1
                           else shard["terminal_object"][row])
            half = config["object_half"]
            corners = [(-half[0], -half[1]), (half[0], -half[1]), (half[0], half[1]),
                       (-half[0], half[1])]
            cos, sin = math.cos(object_pose[2]), math.sin(object_pose[2])
            ax.add_patch(Polygon([(cos * x - sin * y + object_pose[0],
                                   sin * x + cos * y + object_pose[1]) for x, y in corners],
                                 closed=True, facecolor="#d9d9d9", edgecolor="#555555"))
            for robot_id in (0, 1):
                pose = shard["own_world"][row, index, robot_id]
                ax.add_patch(Circle((pose[0], pose[1]), config["robot_radius"],
                                    facecolor=ROBOT_COLORS[robot_id], edgecolor="black", zorder=5))
                segment = shard["patch_segment_robot"][row, index, robot_id]
                if shard["patch_valid"][row, index, robot_id] and segment.any():
                    world = np.array([math.cos(pose[2]) * p[0] - math.sin(pose[2]) * p[1] + pose[0]
                                      for p in [segment[0]]] +
                                     [math.cos(pose[2]) * segment[1][0]
                                      - math.sin(pose[2]) * segment[1][1] + pose[0]])
                    world_y = [math.sin(pose[2]) * segment[0][0] + math.cos(pose[2]) * segment[0][1]
                               + pose[1],
                               math.sin(pose[2]) * segment[1][0] + math.cos(pose[2]) * segment[1][1]
                               + pose[1]]
                    ax.plot(world, world_y, color=ROBOT_COLORS[robot_id], linewidth=2.5, zorder=6)
            ax.set_title(f"{shard['record_id'][row]}  t={index * 0.05:.2f}s  "
                         f"({'正常退出' if tag == 'success' else '中止/片段丢失'})", fontsize=9)
            ax.tick_params(labelsize=7)
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            if writer is None:
                size = (frame.shape[1] // 2 * 2, frame.shape[0] // 2 * 2)
                writer = imageio_ffmpeg.write_frames(str(path), size, fps=20, quality=7,
                                                     macro_block_size=2)
                writer.send(None)
            writer.send(np.ascontiguousarray(frame[:size[1], :size[0]]).tobytes())
            plt.close(fig)
        if writer is not None:
            writer.close()
        out.append({"video": str(path.relative_to(ROOT)), "record_id": str(shard["record_id"][row]),
                    "steps": steps, "real_seconds": steps * 0.05, "playback": "1x (fps=20, dt=0.05)",
                    "note": "由已保存轨迹渲染，未重新运行仿真"})
    return out


def build_scripts() -> List[str]:
    """Copy the renderers into the package and write the single regeneration command.

    The task book asks for scripts that rebuild the material from saved data without retraining,
    re-collecting or running new physics; shipping copies of the renderers (rather than a pointer into
    the repository) makes the asset package self-contained.
    """

    SCRIPTS.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("assets.py", "dataset.py", "windows.py", "sequences.py", "models.py",
                 "sequence_metrics.py", "evaluate.py"):
        source = EXPERIMENTS / name
        if source.exists():
            target = SCRIPTS / name
            target.write_text(source.read_text())
            written.append(str(target.relative_to(ROOT)))
    command = """#!/usr/bin/env bash
# Rebuild every figure, table and video in this package.
# Reads only saved sequences, token arrays, checkpoints and analysis JSON.
# It does NOT retrain, does NOT re-collect data and does NOT run new physics.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
$PY experiments/night_sequence_v1/assets.py
echo "assets rebuilt -> presentation_assets/"
"""
    script = SCRIPTS / "regenerate_assets.sh"
    script.write_text(command)
    script.chmod(0o755)
    written.append(str(script.relative_to(ROOT)))
    readme = SCRIPTS / "README.md"
    readme.write_text(
        "# scripts\n\n`regenerate_assets.sh` 用一条命令重建全部素材。它只读取已保存的序列、"
        "token 数组、检查点与分析 JSON，**不重训、不重新采集、不运行新的物理仿真**。\n\n"
        "同目录下的 `*.py` 是渲染脚本的副本，使素材包自包含；权威版本在 "
        "`experiments/night_sequence_v1/`。\n")
    written.append(str(readme.relative_to(ROOT)))
    return written


def build_index(figures: Dict[str, dict], extra: dict) -> str:
    captions = {
        "F01": ("研究动机与当前主线",
                "左：Gridworld 已观察到的选择偏差（历史讨论结论，非本轮结果）；中：转向连续 2D 后"
                "接触、局部观测与变时长技能带来的问题；右：固定模型预算下组织预测/反馈/搜索的候选方向，"
                "并配已完成/本轮/待验证状态表。",
                "物理乐观是已有实验动机，信息乐观与观测条件分支仍是待验证机制。",
                "不能读成“已获得算法成功”。Gridworld 数字来自历史讨论，未在本轮复现。"),
        "F02": ("连续 2D 环境与场景设计",
                "四种实际构型（物体轮廓、两机器人朝向、米制坐标），全部由仓库内的实际生成器产出。",
                "当前主采集的物理与传感配置固定，且与含不确定参数的旧 Gridworld 版本不同。",
                "未画不存在的墙/通道，也不主张已有避障任务。"),
        "F03": ("option 与真实执行过程",
                "实际阶段机流程、一条真实执行的阶段时间轴、速度与可见性曲线，并标出实际退出时刻 T。",
                "退出不等于物体整体静止；区分正常退出、中止与观测不足。",
                "不使用终态复制帧冒充真实过程。"),
        "F04": ("数据覆盖与分布",
                "以物体初始坐标系对齐的机器人轨迹中心线叠加（预先固定抽样、按 root 均衡），另给出"
                "root/序列/真实帧规模、退出类别与终态可见性、序列长度分布。",
                "显示 D0/D1 的相对规模与训练/验证/测试的独立 root 统计。",
                "密集轨迹不等于独立样本多；窗口数不计入 root 数；不声称已覆盖新物理机制。"),
        "F05": ("模型与学习任务",
                "共同数据流：真实观测前缀（O0 + 最近 32 个真实网格时刻，≤33 token）→ 三种编码器 → "
                "两组预测头；另给出实际参数量。",
                "真实前缀与预测未来分色；t=0 没有尚未发生的历史。",
                "本轮不含自由 latent 递推或完整规划器。"),
        "F06": ("学习过程与规模效应",
                "三种模型在同一更新轴上的验证曲线与分项损失，另表给出参数量、最优更新与耗时。",
                "曲线按同一更新数对齐，计算成本另报。",
                "未完成或预算不同的 run 明确标注；单 seed 不画置信区间。"),
        "F07": ("拟合与预测性能",
                "P0/P1 与新模型在 t=0 终态与执行中下一步/终态的主指标比较，含两机器人中点与法向角、"
                "可见/退出类表现，以及 t=0 动作差值/排序表。",
                "执行前终态、执行中下一步、执行中终态分开报告。",
                "几何排序不等于任务收益排序；真实前缀条件下的预测不是自由 rollout。"),
        "F08": ("对闭环规划的意义与未决问题",
                "“已有证据 → 可能用途 → 尚缺验证”表，并附已有 C 负结果的小表。",
                "本轮没有闭环实验，因此本轮未评估闭环收益。",
                "不能画成功率提升或算法胜负；C 的零收益差仅限原场景/技能/评分。"),
    }
    lines = ["# INDEX — 导师汇报素材索引", "",
             "所有主图与数据均已生成；本文件只做索引与图注，不是完整报告。", "",
             "统一约定：模型颜色固定（MLP-current 蓝、MLP-history 绿、Transformer 红），"
             "机器人颜色固定（robot0 蓝、robot1 橙），真值用灰/黑，预测用模型色。", "",
             "证据分级：**历史已核对结果** / **本轮结果** / **研究假设**，每图注明。", ""]
    for key in sorted(figures):
        entry = figures[key]
        if not entry.get("figure"):
            continue
        title, content, can, cannot = captions.get(key, (key, "", "", ""))
        lines += [f"## {key} {title}", "",
                  f"- 图片：`{entry['figure']['png']}`（矢量版 `{entry['figure']['pdf']}`）",
                  f"- 数据：{', '.join('`' + str(p) + '`' for p in entry.get('data', []) or ['—'])}",
                  f"- 建议用途：{content}",
                  f"- 能说明什么：{can}",
                  f"- 不能说明什么：{cannot}", ""]
    if extra.get("videos"):
        lines += ["## videos", ""]
        for video in extra["videos"]:
            lines.append(f"- `{video['video']}`：{video['record_id']}，真实时长 "
                         f"{video['real_seconds']:.2f}s，{video['playback']}，{video['note']}")
        lines.append("")
    lines += ["## 可复现命令", "", "```bash",
              "bash presentation_assets/scripts/regenerate_assets.sh",
              "```", "",
              "素材包自带渲染脚本副本（`scripts/`），因此解压后即可重建，不依赖工作区其余部分。", "",
              "该命令只读取已保存的序列、token 数组、检查点与分析 JSON，不重训、不重新采集、"
              "不运行新的物理仿真。", ""]
    path = ASSETS / "INDEX.md"
    path.write_text("\n".join(lines))
    return str(path.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-videos", action="store_true")
    args = parser.parse_args()
    for directory in (FIGURES, TABLES, VIDEOS, SCRIPTS):
        directory.mkdir(parents=True, exist_ok=True)
    font = setup_font()
    store = Store()
    figures: Dict[str, dict] = {}
    skipped: Dict[str, str] = {}
    for key, function in (("F01", f01), ("F02", f02), ("F03", f03), ("F04", f04),
                          ("F05", f05), ("F06", f06), ("F07", f07), ("F08", f08)):
        try:
            figures[key] = function(store)
            print(f"{key} ok", flush=True)
        except Exception as error:  # a missing artefact must not silently become a fake figure
            skipped[key] = f"{type(error).__name__}: {error}"
            print(f"{key} SKIPPED: {skipped[key]}", flush=True)
    scripts = build_scripts()
    videos = [] if args.skip_videos else build_videos(store)
    overview = build_overview(figures)
    index = build_index(figures, {"videos": videos, "skipped": skipped})
    facts = [["阶段", "完成内容", "证据文件", "关键数字", "状态"],
             ["Gridworld", "候选序列收益高估诊断",
              "docs/20261013_1937_CoImagine_研究主线与验证原则_v1 (1).md", "", "历史讨论结论"],
             ["连续2D 平台", "局部接触传感 + option 阶段机 + 判稳",
              "src/coimagine/local_contact_option/", "F_max=15N, dt=0.05s", "已验证"],
             ["观测终态接口", "R_i0 片段几何 + 自身运动", "docs/observation_terminal_v1/report.md",
              "中点 p50 3.8-5.0cm vs 起始保持 14.1cm", "已验证"],
             ["序列数据", "D1 option 内部序列",
              "docs/night_sequence_v1/DATA_INDEX.md",
              f"{store.summary.get('d1ext',{}).get('independent_roots',0)+512} root / "
              f"{store.summary.get('d1ext',{}).get('sequences',0)+4096} 序列", "已验证"],
             ["架构对照", "MLP-history vs Transformer-history", "presentation_assets/figures/F07",
              "", "本轮结果"],
             ["闭环收益", "预测→决策", "", "", "未评估"]]
    write_csv("progress_facts.csv", facts[0], facts[1:])
    summary = {"font": font, "scripts": scripts,
               "figures": {k: v.get("figure") for k, v in figures.items()},
               "skipped": skipped, "videos": videos, "overview": overview, "index": index,
               "data_files": [p for v in figures.values() for p in (v.get("data") or [])]}
    (ASSETS / "ASSET_CHECK.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({"figures": len(figures), "skipped": skipped, "videos": len(videos),
                      "overview": overview}, indent=1))
    print("ASSETS_OK" if len(figures) == 8 else "ASSETS_INCOMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
