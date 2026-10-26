"""PUSH continuity / corner contact-point visual evaluation.

This is a *presentation round* on top of the frozen option testbench.  It uses the same
panel -> request -> isolated branch -> raw log pipeline as ``cli.py run`` but asks a narrower
question:

* under several random-start distributions, does a PUSH keep real contact for its whole
  execution, and where does it lose it?
* when the robot starts near an object corner, does the approach shift its lateral
  projection to a non-corner contact point (the visible-segment centre), and does the
  contact point stay away from the corner?

The output is a self-contained local HTML page (``index.html``) with per-frame geometry,
a contact timeline, and a small metric table.  No controller or reward change.

Usage:
    .venv/bin/python experiments/option_testbench/continuity_eval.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panel as P  # noqa: E402
import runner  # noqa: E402
from coimagine.continuous2d.scenes import Pose  # noqa: E402
from coimagine.random_start import layout, spec  # noqa: E402

DT = spec.CONTROL_DT
ROUND = "round_035_pusheval"
PUSH_039_100 = 11      # PUSH a=0, v=0.39 m/s, requested 1.00 m
PUSH_024_050 = 6       # PUSH a=0, v=0.24 m/s, requested 0.50 m
PUSH_012_050 = 2       # PUSH a=0, v=0.12 m/s, requested 0.50 m
PUSH_012_100 = 3       # PUSH a=0, v=0.12 m/s, requested 1.00 m
PUSH_024_100 = 7       # PUSH a=0, v=0.24 m/s, requested 1.00 m
PUSH_039_050 = 10      # PUSH a=0, v=0.39 m/s, requested 0.50 m
WAIT_BRAKE = 20        # WAIT_BRAKE 2 s
PUSH_PAIR = (PUSH_024_050, PUSH_024_050)
CURATED = ("face", "corner_minus", "corner_plus", "edge_turn", "teammate")
CORNER_EXTRA = ("corner_minus", "corner_plus", "edge_turn",
                "corner_out_002", "corner_out_006", "corner_out_014", "corner_in_004")
COOP_ARENA = (4.0, 3.0)
COOP_OBJECT = (2.0, 1.5, 0.0)
COOP_GOAL = (2.0, 2.65)
#: name, r0 pose, r1 pose, short label.  Both robots approach the same long face from below;
#: no obstacles, only the 4.0 x 3.0 m boundary.  The goal is +y, so a clean pair push moves up.
COOP_SCENES = (
    ("coop_sym_close", (1.75, 1.075, -math.pi / 2), (2.25, 1.075, -math.pi / 2),
     "对称近距离"),
    ("coop_sym_wide", (1.65, 1.075, -math.pi / 2), (2.35, 1.075, -math.pi / 2),
     "对称远距离"),
    ("coop_sym_ends", (1.55, 1.075, -math.pi / 2), (2.45, 1.075, -math.pi / 2),
     "对称近两端"),
    ("coop_asym", (1.60, 1.075, -math.pi / 2), (2.30, 1.075, -math.pi / 2),
     "非对称位置"),
    ("coop_uneven_gap", (1.70, 0.995, -math.pi / 2), (2.30, 1.075, -math.pi / 2),
     "不等起始间隙"),
    ("coop_one_far", (1.80, 1.075, -math.pi / 2), (2.20, 0.745, -math.pi / 2),
     "一侧远距离接近"),
    ("coop_corner_left", (1.55, 1.075, -math.pi / 2), (2.10, 1.075, -math.pi / 2),
     "左侧近角点"),
    ("coop_corner_right", (2.45, 1.075, -math.pi / 2), (1.90, 1.075, -math.pi / 2),
     "右侧近角点"),
    ("coop_adjacent", (1.80, 1.075, -math.pi / 2), (2.83, 1.50, math.pi),
     "相邻侧（下 + 右）"),
)


def _labels(family: str, title_zh: str, title_en: str, problem_zh: str, problem_en: str,
            geometry_zh: str, geometry_en: str, expect_zh: str, expect_en: str,
            watch_zh: str, watch_en: str) -> Dict[str, str]:
    return {"family": family, "title_zh": title_zh, "title_en": title_en,
            "problem_zh": problem_zh, "problem_en": problem_en,
            "geometry_zh": geometry_zh, "geometry_en": geometry_en,
            "expectation_zh": expect_zh, "expectation_en": expect_en,
            "watch_zh": watch_zh, "watch_en": watch_en}


def random_labels(instance, group: str) -> Dict[str, str]:
    obj, goal = instance.object_pose, instance.goal_centre
    r0, r1 = instance.robot_poses
    distance = math.hypot(goal[0] - obj.x, goal[1] - obj.y)
    return _labels(
        group,
        f"随机起点 {instance.name}", f"Random start {instance.name}",
        f"{group} 采样得到的随机合法初始状态；用于观察一般起点下的 PUSH 连续性。",
        f"A random legal start sampled from {group}; used to inspect PUSH continuity on a "
        f"general start.",
        f"物体 ({obj.x:.2f},{obj.y:.2f}) 朝向 {math.degrees(obj.angle):+.1f}°；"
        f"r0 ({r0.x:.2f},{r0.y:.2f})、r1 ({r1.x:.2f},{r1.y:.2f})；"
        f"目标 ({goal[0]:.2f},{goal[1]:.2f})，物体到目标 {distance:.2f} m。",
        f"Object ({obj.x:.2f},{obj.y:.2f}) heading {math.degrees(obj.angle):+.1f} deg; "
        f"r0 ({r0.x:.2f},{r0.y:.2f}), r1 ({r1.x:.2f},{r1.y:.2f}); goal "
        f"({goal[0]:.2f},{goal[1]:.2f}), object-to-goal {distance:.2f} m.",
        "不预设成功；观察接触是否连续、是否在角点附近丢接触、以及最终位移方向。",
        "No success is assumed; watch contact continuity, corner-adjacent contact loss, and the "
        "final displacement direction.",
        "接触时间条、最长丢失、face switch、角点拒绝、signed progress。",
        "Contact timeline, longest loss, face switches, corner rejection, signed progress.")


def corner_labels(name: str, dx: float) -> Dict[str, str]:
    outside = dx - 2.10
    side = "外侧" if outside > 0 else "内侧"
    side_en = "outside" if outside > 0 else "inside"
    return _labels(
        "corner_sweep",
        f"角点偏移扫描 {name}（{outside:+.3f} m）",
        f"Corner-offset sweep {name} ({outside:+.3f} m)",
        "物体、目标与另一台机器人保持不变，只把 r0 沿 -y 面移到 +x 角点的不同偏移量；"
        "检验 approach 是否把接触点从角点移到可见面中心。",
        "Same object, goal and teammate; only r0 is shifted along the -y face to a different "
        "offset from the +x corner. Tests whether the approach moves the contact point from the "
        "corner towards the visible-face centre.",
        f"物体中心 (1.60,1.10)，朝向 0°；r0 (x={dx:.3f}, y=0.545) 朝 +y，距 +x 角点 "
        f"{abs(outside):.3f} m {side}；r1 (1.60,1.655) 朝 -y；目标 (1.60,1.95)。",
        f"Object centre (1.60,1.10), heading 0; r0 at (x={dx:.3f}, y=0.545) facing +y, "
        f"{abs(outside):.3f} m {side_en} the +x corner; r1 at (1.60,1.655) facing -y; goal "
        f"(1.60,1.95).",
        "如果角点处理正确，approach 阶段应沿切向把 r0 的投影拉向面中心，首次接触点不应停在角点；"
        "偏移越大，需要的切向修正越长。",
        "With correct corner handling the approach should pull r0's projection towards the face "
        "centre; the first contact should not sit on the corner. A larger offset needs a longer "
        "tangential correction.",
        "corner_rejected 时长、投影移动量、首次接触点相对面中心的偏移、接触是否连续。",
        "corner_rejected duration, projection shift, first-contact offset from the face centre, "
        "and contact continuity.")


def short_side_labels(instance) -> Dict[str, str]:
    obj, goal = instance.object_pose, instance.goal_centre
    r0 = instance.robot_poses[0]
    return _labels(
        "short_side",
        f"短侧面诊断 {instance.name}", f"Short-side diagnosis {instance.name}",
        "短面长 0.25 m，小于机器人直径 0.36 m。v4.5 的硬 guard 要求可见线段 >= 0.40 m，"
        "因此旧版本在这里停在 gap≈0.125 m、报 insufficient_visible_segment。v4.6 改为软内缩："
        "允许窄面接触，宽面才把接触点向内收；本分支用于检查现在是否真的能中心接触并推动。",
        "The short face is 0.25 m, shorter than the 0.36 m robot diameter. The old v4.5 hard "
        "guard required a 0.40 m visible segment and therefore stopped at gap about 0.125 m with "
        "insufficient_visible_segment. v4.6 uses a soft inset: narrow faces are allowed and only "
        "wide faces move the contact point inward. This branch checks whether the centre contact "
        "and push now actually happen.",
        f"物体 ({obj.x:.2f},{obj.y:.2f}) 朝向 {math.degrees(obj.angle):+.1f}°；r0 "
        f"({r0.x:.2f},{r0.y:.2f})；目标 ({goal[0]:.2f},{goal[1]:.2f})。可见线段约 0.22 m；"
        "旧 v4.5 hard guard 会停在 gap≈0.125 m 并报 insufficient_visible_segment，"
        "v4.6 soft inset 应允许中心接触并进入 push。",
        f"Object ({obj.x:.2f},{obj.y:.2f}) heading {math.degrees(obj.angle):+.1f} deg; r0 "
        f"({r0.x:.2f},{r0.y:.2f}); goal ({goal[0]:.2f},{goal[1]:.2f}). Visible segment about "
        "0.22 m; the old v4.5 hard guard stopped at gap about 0.125 m with "
        "insufficient_visible_segment, while the v4.6 soft inset should allow a centre contact "
        "and enter push.",
        "几何上，圆盘在短面中心接触时到两个角点的距离是 sqrt(0.18²+0.125²)=0.219 m > "
        "R=0.18 m，并不碰角；旧 guard 的 0.40 m 阈值是保守策略。新软内缩允许中心接触，"
        "但短面比机器人窄，接触点偏一点就会产生力矩，因此本分支要同时看转角/侧向漂移。",
        "This is not a lost approach target. Geometrically, a disc centred on the short face is "
        "0.219 m from either corner, larger than R=0.18 m, so it does not touch the corners; the "
        "old 0.40 m guard was a conservative policy. The v4.6 soft inset targets the centre; "
        "watch the contact point and torque because the face is narrower than the disc.",
        "观察是否进入 push、首次接触点相对中心的偏移、signed progress 和转角；"
        "不应再预期 insufficient_visible_segment。",
        "Watch whether it enters push, the first-contact offset from the centre, signed progress "
        "and rotation; insufficient_visible_segment is no longer the expected outcome.")


def coop_labels(name: str, r0, r1, tag: str) -> Dict[str, str]:
    gap0 = 1.375 - (r0[1] + spec.ROBOT_RADIUS)
    gap1 = 1.375 - (r1[1] + spec.ROBOT_RADIUS)
    if r1[0] > 2.6:
        gap1 = r1[0] - spec.ROBOT_RADIUS - 2.50
    return _labels(
        "cooperative",
        f"协同 PUSH：{tag}（{name}）", f"Cooperative PUSH: {tag} ({name})",
        "4.0 x 3.0 m 大空场，无额外障碍物，只有边界；两台机器人从不同初始位置接近同一长面。"
        "观察双方是否都完成 approach、都进入 push、等待/同步后一起推，以及推动中丢接触多少。",
        "4.0 x 3.0 m empty arena, no extra obstacles, boundary only; both robots approach the "
        "same long face from different starts. Watch whether both complete approach, both enter "
        "push, hold/synchronise before pushing, and how much contact is lost while pushing.",
        f"物体 (2.00,1.50) 朝向 0°；目标 (2.00,2.65)；r0 ({r0[0]:.2f},{r0[1]:.2f}) gap≈"
        f"{gap0:.2f} m，r1 ({r1[0]:.2f},{r1[1]:.2f}) gap≈{gap1:.2f} m；无静态障碍物。",
        f"Object (2.00,1.50) heading 0; goal (2.00,2.65); r0 ({r0[0]:.2f},{r0[1]:.2f}) gap "
        f"about {gap0:.2f} m, r1 ({r1[0]:.2f},{r1[1]:.2f}) gap about {gap1:.2f} m; no static "
        "obstacles.",
        "v4.6 协同 staging：两侧各自到约 3 cm、朝向对齐后零力等待；双方都 ready 才一起 push。"
        "观察 ready wait、staging gap、两侧 push 是否同步、first force 与 first both-contact "
        "的先后，以及推动中接触丢失。",
        "v4.6 cooperative staging: each robot approaches to about 3 cm, aligns its heading, then "
        "waits at zero force; both are released into push together. Watch ready wait, staging "
        "gaps, push synchrony, whether first force precedes first both-contact, and contact loss.",
        "sync delay、ready wait、staging gap、both contact fraction、r0/r1 contact fraction、"
        "push_complete、最终转角/位移。",
        "Sync delay, ready wait, staging gaps, both-contact fraction, per-robot contact fraction, "
        "push_complete, final rotation/displacement.")


def make_panel(out_dir: Path) -> Tuple[Dict, Dict, List[str]]:
    P.prepare(out_dir)
    source = json.loads((out_dir / "panel.json").read_text())
    snapshots: Dict[str, Dict] = {}
    fixtures: Dict[str, Dict] = {}
    order: List[str] = []
    for name in CURATED:
        snapshots[name] = source["snapshots"][name]
        fixtures[name] = source["fixtures"][name]
        order.append(name)

    for group, map_name, split, count, offset in (
            ("random_L0_train", "L0", "train", 3, 0),
            ("random_L1_train", "L1", "train", 3, 0),
            ("random_L0_train_more", "L0", "train", 9, 3)):
        instances, _ = layout.build_split(map_name, split, count=count)
        instances = instances[offset:offset + 3]
        for instance in instances:
            if instance.name in snapshots:
                continue
            snapshots[instance.name] = {"kind": "static", "instance": instance.as_dict(),
                                        "sha256": P.snapshot_hash(instance)}
            labels = (short_side_labels(instance) if instance.name == "L1_train_0002"
                      else random_labels(instance, group))
            fixtures[instance.name] = {"labels": labels, "legal": True,
                                       "problems": [],
                                       "content_sha256": layout.content_hash(instance),
                                       "instance": instance.as_dict()}
            order.append(instance.name)

    # controlled corner sweep: only r0's position along the -y face changes
    half = math.pi / 2.0
    for name, dx in (("corner_out_002", 2.12), ("corner_out_006", 2.16),
                     ("corner_out_014", 2.24), ("corner_in_004", 2.06)):
        instance, _ = P._instance(
            name, (1.60, 1.10, 0.0),
            [(dx, 0.545, -half), (1.60, 1.655, half)],
            (1.60, 1.95), corner_labels(name, dx))
        legality = layout.scene_legality_report([instance])
        if not legality["ok"]:
            raise ValueError(f"illegal corner fixture {name}: {legality['problems']}")
        snapshots[name] = {"kind": "static", "instance": instance.as_dict(),
                           "sha256": P.snapshot_hash(instance)}
        fixtures[name] = {"labels": corner_labels(name, dx), "legal": True,
                          "problems": [], "content_sha256": layout.content_hash(instance),
                          "instance": instance.as_dict()}
        order.append(name)

    # cooperative scenes: larger empty arena, no static obstacles, only the boundary
    for name, r0, r1, tag in COOP_SCENES:
        instance = layout.InstanceSpec(
            name=name, map="L0", split="dev", index=0, seed=0, arena=COOP_ARENA,
            obstacles=(), object_pose=Pose(*COOP_OBJECT),
            robot_poses=(Pose(*r0), Pose(*r1)), goal_centre=COOP_GOAL,
            goal_radius=spec.GOAL_RADIUS, env_version=spec.ENV_VERSION)
        legality = layout.scene_legality_report([instance])
        if not legality["ok"]:
            raise ValueError(f"illegal cooperative fixture {name}: {legality['problems']}")
        labels = coop_labels(name, r0, r1, tag)
        snapshots[name] = {"kind": "static", "instance": instance.as_dict(),
                           "sha256": P.snapshot_hash(instance)}
        fixtures[name] = {"labels": labels, "legal": True, "problems": [],
                          "content_sha256": layout.content_hash(instance),
                          "instance": instance.as_dict()}
        order.append(name)

    panel = {"panel_version": "continuity-eval-v2", "env_version": spec.ENV_VERSION,
             "snapshot_count": len(order), "fixtures": fixtures, "snapshots": snapshots,
             "note": "PUSH continuity / corner / cooperative presentation round; no controller change"}
    (out_dir / "panel.json").write_text(json.dumps(panel, indent=1, ensure_ascii=False) + "\n")
    return panel, fixtures, order


def make_requests(order: Sequence[str]) -> List[Dict]:
    requests: List[Dict] = []

    def add(scene: str, suffix: str, actions, focus: str, focus_zh: str, focus_en: str,
            detail_zh: str, detail_en: str) -> None:
        requests.append({
            "branch_id": f"{scene}__{suffix}", "snapshot": scene, "actions": actions,
            "focus": focus, "focus_zh": focus_zh, "focus_en": focus_en,
            "focus_detail_zh": detail_zh, "focus_detail_en": detail_en,
            "action_labels": [[spec.ROBOT_ACTIONS[index].label for index in pair]
                              for pair in actions],
            "recovery": False, "teammate_protocol": "shared_clock_joint",
            "controller_version": "baseline",
            "requested_wall_seconds": 40.0, "requested_physical_seconds": 30.0})

    for scene in order:
        add(scene, "push_039_100", [[PUSH_039_100, WAIT_BRAKE]], "push_long_fast",
            "快推 1.00 m / 0.39 m/s", "Fast push 1.00 m / 0.39 m/s",
            "r0 单侧 PUSH(0°,0.39 m/s, 1.00 m)，r1 制动；用于观察长推的接触连续性。",
            "r0 single PUSH (0 deg, 0.39 m/s, 1.00 m), r1 brakes; long-push contact continuity.")
        add(scene, "push_024_050", [[PUSH_024_050, WAIT_BRAKE]], "push_mid",
            "中速推 0.50 m / 0.24 m/s", "Medium push 0.50 m / 0.24 m/s",
            "r0 单侧 PUSH(0°,0.24 m/s, 0.50 m)，r1 制动；与快推对照。",
            "r0 single PUSH (0 deg, 0.24 m/s, 0.50 m), r1 brakes; contrast with the fast push.")
        if scene in CORNER_EXTRA:
            add(scene, "push_012_050", [[PUSH_012_050, WAIT_BRAKE]], "push_slow",
                "慢推 0.50 m / 0.12 m/s", "Slow push 0.50 m / 0.12 m/s",
                "r0 单侧 PUSH(0°,0.12 m/s, 0.50 m)，r1 制动；慢速是否更稳。",
                "r0 single PUSH (0 deg, 0.12 m/s, 0.50 m), r1 brakes; is slower steadier?")
        if scene in ("face", "teammate"):
            add(scene, "push_pair_024_050", [[PUSH_024_050, PUSH_024_050]], "push_pair",
                "同侧双推 0.50 m / 0.24 m/s", "Same-side pair push 0.50 m / 0.24 m/s",
                "两侧同时 PUSH(0°,0.24 m/s, 0.50 m)；观察同面双推的接触点是否保持对称。",
                "Both robots PUSH (0 deg, 0.24 m/s, 0.50 m); does the same-face pair keep "
                "symmetric contact points?")

    coop_names = [name for name, _r0, _r1, _tag in COOP_SCENES]
    for scene in coop_names:
        add(scene, "pair_024_050", [[PUSH_024_050, PUSH_024_050]], "pair_mid",
            "协同中速 0.50 m / 0.24 m/s", "Cooperative medium 0.50 m / 0.24 m/s",
            "两侧同时 PUSH(0°,0.24 m/s, 0.50 m)：观察 approach 完成、双方进入 push 后的同步"
            "与接触保持。",
            "Both robots PUSH (0 deg, 0.24 m/s, 0.50 m): watch approach completion, entry into "
            "push, synchronisation and contact maintenance.")
        add(scene, "pair_012_100", [[PUSH_012_100, PUSH_012_100]], "pair_slow",
            "协同慢速 1.00 m / 0.12 m/s", "Cooperative slow 1.00 m / 0.12 m/s",
            "两侧同时 PUSH(0°,0.12 m/s, 1.00 m)：慢速长推，观察接触丢失是否更少。",
            "Both robots PUSH (0 deg, 0.12 m/s, 1.00 m): slow long push, is contact loss lower?")
        add(scene, "pair_039_050", [[PUSH_039_050, PUSH_039_050]], "pair_fast",
            "协同快推 0.50 m / 0.39 m/s", "Cooperative fast 0.50 m / 0.39 m/s",
            "两侧同时 PUSH(0°,0.39 m/s, 0.50 m)：快推下的接触保持。",
            "Both robots PUSH (0 deg, 0.39 m/s, 0.50 m): contact maintenance at the fast speed.")
        if scene in ("coop_sym_wide", "coop_uneven_gap", "coop_one_far"):
            add(scene, "pair_024_100", [[PUSH_024_100, PUSH_024_100]], "pair_long",
                "协同长推 1.00 m / 0.24 m/s", "Cooperative long 1.00 m / 0.24 m/s",
                "两侧同时 PUSH(0°,0.24 m/s, 1.00 m)：长距离协同推，观察后段接触。",
                "Both robots PUSH (0 deg, 0.24 m/s, 1.00 m): long cooperative push, late-stage "
                "contact.")
    return requests


def _dot(a, b) -> float:
    return float(a[0] * b[0] + a[1] * b[1])


def _round(value, digits: int = 4):
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None


def _geometry(row, robot_id: int) -> Optional[Dict]:
    info = row[f"r{robot_id}"]
    points = info.get("patch_points_world") or []
    tangent = info.get("tangent_world")
    normal = info.get("normal_world")
    if len(points) < 2 or not tangent or not normal:
        return None
    projections = [_dot(point, tangent) for point in points]
    low, high = min(projections), max(projections)
    offset = _dot(points[0], normal)
    centre = 0.5 * (low + high)
    margin = spec.ROBOT_RADIUS + 0.02
    target = centre if high - low < 2.0 * margin else \
        max(low + margin, min(high - margin, centre))
    robot = row["true"]["robots"][robot_id]
    projection = _dot(robot, tangent)

    def point(value):
        return (value * tangent[0] + offset * normal[0],
                value * tangent[1] + offset * normal[1])

    return {"low": low, "high": high, "centre": centre, "target": target,
            "projection": projection, "segment": (point(low), point(high)),
            "mid": point(centre), "target_point": point(target),
            "margin": margin, "tangent": tangent, "normal": normal}


def _contact_point(row, robot_id: int) -> Optional[Tuple[float, float]]:
    info = row[f"r{robot_id}"]
    local = info.get("contact_point_robot")
    if local is None or not info.get("contacting_object"):
        return None
    robot = row["true"]["robots"][robot_id]
    theta = float(robot[2])
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    return (robot[0] + local[0] * cos_a - local[1] * sin_a,
            robot[1] + local[0] * sin_a + local[1] * cos_a)


def _true_face_centre(row, robot_id: int, geometry: Optional[Dict]) -> Optional[Tuple[float, float]]:
    """Truth-only reference: the centre of the *whole* object face, not the visible segment."""
    if geometry is None:
        return None
    normal = row[f"r{robot_id}"].get("normal_world")
    if not normal:
        return None
    outward = (-float(normal[0]), -float(normal[1]))
    obj = row["true"]["object"]
    theta = float(obj[2])
    cos_a, sin_a = math.cos(theta), math.sin(theta)
    local_x = cos_a * outward[0] + sin_a * outward[1]
    local_y = -sin_a * outward[0] + cos_a * outward[1]
    half = (spec.OBJECT_WIDTH / 2.0 if abs(local_x) >= abs(local_y)
            else spec.OBJECT_HEIGHT / 2.0)
    return (float(obj[0]) + outward[0] * half, float(obj[1]) + outward[1] * half)


def _frame(row) -> Dict:
    geometry = [_geometry(row, robot) for robot in (0, 1)]
    contacts = [_contact_point(row, robot) for robot in (0, 1)]
    frame = {
        "t": _round(row["time"]), "p": int(row["period"]),
        "o": [_round(row["true"]["object"][0]), _round(row["true"]["object"][1]),
              _round(row["true"]["object"][2])],
        "r": [[_round(row["true"]["robots"][robot][0]),
               _round(row["true"]["robots"][robot][1]),
               _round(row["true"]["robots"][robot][2])] for robot in (0, 1)],
        "ct": [bool(row["r0"]["contacting_object"]), bool(row["r1"]["contacting_object"])],
        "st": [row["r0"]["stage"], row["r1"]["stage"]],
        "ex": [row["r0"]["exit_reason"], row["r1"]["exit_reason"]],
        "sk": [row["r0"]["skill"], row["r1"]["skill"]],
        "g": [_round(row["r0"].get("gap")), _round(row["r1"].get("gap"))],
        "si": [_round(row["r0"].get("signed_progress")), _round(row["r1"].get("signed_progress"))],
        "sd": [_round(row["r0"].get("progress_target_distance")),
               _round(row["r1"].get("progress_target_distance"))],
        "cr": [bool(row["r0"].get("corner_rejected")), bool(row["r1"].get("corner_rejected"))],
        "ea": [bool(row["r0"].get("edge_assigned")), bool(row["r1"].get("edge_assigned"))],
        "fs": [bool(row["r0"].get("anchor_face_switch")),
               bool(row["r1"].get("anchor_face_switch"))],
        "rd": [bool(row["r0"].get("ready")), bool(row["r1"].get("ready"))],
        "rl": [bool(row["r0"].get("release")), bool(row["r1"].get("release"))],
        "cp": [[_round(contacts[0][0]), _round(contacts[0][1])] if contacts[0] else None,
               [_round(contacts[1][0]), _round(contacts[1][1])] if contacts[1] else None],
        "pp": [[[_round(point[0]), _round(point[1])]
                for point in (row[f"r{robot}"]["patch_points_world"] or [])[:12]]
               for robot in (0, 1)],
        "seg": [[[_round(geometry[robot]["segment"][0][0]),
                  _round(geometry[robot]["segment"][0][1]),
                  _round(geometry[robot]["segment"][1][0]),
                  _round(geometry[robot]["segment"][1][1])] if geometry[robot] else None]
                for robot in (0, 1)],
        "mid": [[_round(geometry[robot]["mid"][0]), _round(geometry[robot]["mid"][1])]
                if geometry[robot] else None for robot in (0, 1)],
        "tgt": [[_round(geometry[robot]["target_point"][0]),
                 _round(geometry[robot]["target_point"][1])] if geometry[robot] else None
                for robot in (0, 1)],
        "prj": [_round(geometry[robot]["projection"]) if geometry[robot] else None
                for robot in (0, 1)],
        "fl": [[_round(row["r0"]["force"][0]), _round(row["r0"]["force"][1])],
               [_round(row["r1"]["force"][0]), _round(row["r1"]["force"][1])]],
        "rv": [[_round(row["r0"]["reference_velocity"][0]),
                _round(row["r0"]["reference_velocity"][1])],
               [_round(row["r1"]["reference_velocity"][0]),
                _round(row["r1"]["reference_velocity"][1])]],
        "tfc": [[_round(centre[0]), _round(centre[1])] if (centre := _true_face_centre(
            row, robot, geometry[robot])) else None for robot in (0, 1)],
    }
    return frame


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _pair_summary(rows: List[Dict], boundary_exits: Dict[str, str]) -> Dict:
    """Both-robot synchronisation / contact metrics for cooperative branches."""
    if not rows:
        return {}
    stages = [[row[f"r{robot}"].get("stage") for row in rows] for robot in (0, 1)]
    push = [[index for index, stage in enumerate(stages[robot]) if stage == "push"]
            for robot in (0, 1)]
    both_push = [index for index in push[0] if index in set(push[1])]
    contact = [[bool(row[f"r{robot}"].get("contacting_object")) for row in rows]
               for robot in (0, 1)]
    force_norm = [[math.hypot(*(row[f"r{robot}"].get("force") or (0.0, 0.0)))
                   for row in rows] for robot in (0, 1)]

    def fraction(indices, values):
        if not indices:
            return None
        return sum(1 for index in indices if values[index]) / len(indices)

    both_contacts = sum(1 for index in both_push if contact[0][index] and contact[1][index])
    first_both_contact = next((index for index in range(len(rows))
                               if contact[0][index] and contact[1][index]), None)
    first_contact = [next((index for index in push[robot] if contact[robot][index]), None)
                     for robot in (0, 1)]
    after_engage = []
    for robot in (0, 1):
        if first_contact[robot] is None:
            after_engage.append(None)
        else:
            engaged = [index for index in push[robot] if index >= first_contact[robot]]
            after_engage.append((sum(1 for index in engaged if contact[robot][index])
                                 / len(engaged)) if engaged else None)
    both_after = ([index for index in both_push if index >= first_both_contact]
                  if first_both_contact is not None else [])
    both_after_fraction = (sum(1 for index in both_after
                               if contact[0][index] and contact[1][index]) / len(both_after)
                           if both_after else None)
    # only a force during the push stage counts as "pushing"; approach drive force is not
    first_push_force = next((index for index in range(len(rows))
                             if ((stages[0][index] == "push" and force_norm[0][index] > 1.0)
                                 or (stages[1][index] == "push" and force_norm[1][index] > 1.0))),
                            None)
    force_before = 0.0
    if first_both_contact is None:
        force_before = None
    elif (first_push_force is not None and first_push_force < first_both_contact):
        force_before = (first_both_contact - first_push_force) * DT
    hold_forces = []
    for index, row in enumerate(rows):
        if row["r0"].get("hold"):
            hold_forces.append(force_norm[0][index])
        if row["r1"].get("hold"):
            hold_forces.append(force_norm[1][index])

    ready = [[index for index, stage in enumerate(stages[robot]) if stage == "ready"]
             for robot in (0, 1)]
    ready_both = [index for index in ready[0] if index in set(ready[1])]
    first_ready_both = ready_both[0] if ready_both else None
    released = [index for index, row in enumerate(rows)
                if row["r0"].get("release") and row["r1"].get("release")]
    first_release = released[0] if released else None
    staging_gap0 = staging_gap1 = None
    if ready[0]:
        staging_gap0 = rows[ready[0][0]]["r0"].get("gap")
    if ready[1]:
        staging_gap1 = rows[ready[1][0]]["r1"].get("gap")

    move_before = rotation_before = None
    if first_both_contact is not None and first_both_contact < len(rows):
        start = rows[0]["true"]["object"]
        stop = rows[first_both_contact]["true"]["object"]
        move_before = math.hypot(stop[0] - start[0], stop[1] - start[1])
        rotation_before = math.degrees(_wrap_angle(stop[2] - start[2]))

    return {
        "r0_contact_fraction": fraction(push[0], contact[0]),
        "r1_contact_fraction": fraction(push[1], contact[1]),
        "both_push_seconds": len(both_push) * DT,
        "both_contact_seconds": both_contacts * DT,
        "both_contact_fraction": (both_contacts / len(both_push)) if both_push else None,
        "r0_contact_after_engage_fraction": after_engage[0],
        "r1_contact_after_engage_fraction": after_engage[1],
        "both_contact_after_engage_fraction": both_after_fraction,
        "sync_delay_seconds": (abs(push[0][0] - push[1][0]) * DT
                               if push[0] and push[1] else None),
        "first_push0_seconds": push[0][0] * DT if push[0] else None,
        "first_push1_seconds": push[1][0] * DT if push[1] else None,
        "first_both_contact_seconds": first_both_contact * DT if first_both_contact is not None else None,
        "first_push_force_seconds": first_push_force * DT if first_push_force is not None else None,
        "force_before_both_contact_seconds": force_before,
        "r0_ready_seconds": len(ready[0]) * DT,
        "r1_ready_seconds": len(ready[1]) * DT,
        "both_ready_seconds": len(ready_both) * DT,
        "both_ready_time": first_ready_both * DT if first_ready_both is not None else None,
        "release_time": first_release * DT if first_release is not None else None,
        "ready_wait_seconds": ((first_release - first_ready_both) * DT
                               if first_ready_both is not None and first_release is not None
                               and first_release >= first_ready_both else None),
        "staging_gap0_m": staging_gap0,
        "staging_gap1_m": staging_gap1,
        "hold_seconds": sum(1 for row in rows
                            if row["r0"].get("hold") or row["r1"].get("hold")) * DT,
        "hold_force_max_n": max(hold_forces) if hold_forces else None,
        "hold_force_mean_n": (sum(hold_forces) / len(hold_forces)) if hold_forces else None,
        "object_move_before_both_contact_m": move_before,
        "object_rotation_before_both_contact_deg": rotation_before,
        "r0_exit": boundary_exits.get("0") or rows[-1]["r0"].get("exit_reason"),
        "r1_exit": boundary_exits.get("1") or rows[-1]["r1"].get("exit_reason"),
        "r0_signed_m": float(rows[-1]["r0"].get("signed_progress") or 0.0),
        "r1_signed_m": float(rows[-1]["r1"].get("signed_progress") or 0.0),
    }


def _summary(log: Dict, request: Dict) -> Dict:
    rows = log.get("rows", [])
    action = spec.ROBOT_ACTIONS[int(request["actions"][0][0])]
    requested = float(action.distance) if action.scale_kind == "distance" else 0.0
    empty = {"rows": 0, "requested_m": requested, "exit_reason": log.get("status"),
             "contact_fraction": None, "contact_after_engage_fraction": None,
             "pre_contact_push_seconds": None, "contact_seconds": 0.0, "push_seconds": 0.0,
             "longest_loss_seconds": 0.0, "contact_transitions": 0, "face_switches": 0,
             "corner_rejected_seconds": 0.0, "min_corner_margin_m": None,
             "first_contact_offset_m": None, "contact_shift_m": None,
             "first_contact_true_offset_m": None, "approach_start_offset_m": None, "signed_progress_m": 0.0,
             "signed_fraction": None, "push_complete": False, "contact_lost_count": 0,
             "object_displacement_m": 0.0, "object_rotation_deg": 0.0,
             "goal_progress_m": 0.0, "edge_assigned_fraction": None,
             "total_seconds": 0.0, "pair": {}}
    if not rows:
        return empty
    stages = [row["r0"].get("stage") for row in rows]
    push_rows = [index for index, stage in enumerate(stages) if stage == "push"]
    approach_rows = [index for index, stage in enumerate(stages) if stage == "approach"]
    contact = [bool(row["r0"].get("contacting_object")) for row in rows]
    contact_rows = [index for index, value in enumerate(contact) if value]
    push_contacts = sum(1 for index in push_rows if contact[index])
    contact_fraction = (push_contacts / len(push_rows)) if push_rows else None
    first_contact_push = next((index for index in push_rows if contact[index]), None)
    contact_after_engage = None
    pre_contact_push_seconds = None
    if first_contact_push is not None:
        engaged = [index for index in push_rows if index >= first_contact_push]
        contact_after_engage = (sum(1 for index in engaged if contact[index]) / len(engaged)
                                if engaged else None)
        pre_contact_push_seconds = (first_contact_push - push_rows[0]) * DT

    longest = current = 0
    transitions = 0
    previous = None
    for index in push_rows:
        if contact[index]:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        if previous is not None and contact[index] != previous:
            transitions += 1
        previous = contact[index]

    final = rows[-1]["r0"]
    signed = float(final.get("signed_progress") or 0.0)
    exit_reason = final.get("exit_reason") or log.get("status")
    face_switches = int(final.get("face_switches") or 0)
    if not face_switches:
        face_switches = sum(1 for row in rows if row["r0"].get("anchor_face_switch"))
    corner_seconds = sum(1 for row in rows if row["r0"].get("corner_rejected")) * DT
    edge_rows = [index for index in approach_rows
                 if rows[index]["r0"].get("edge_assigned")]
    edge_fraction = (len(edge_rows) / len(approach_rows)) if approach_rows else None

    first_contact = contact_rows[0] if contact_rows else None
    start_index = approach_rows[0] if approach_rows else 0
    geometry_start = _geometry(rows[start_index], 0)
    geometry_contact = _geometry(rows[first_contact], 0) if first_contact is not None else None
    approach_offset = None
    first_offset = None
    true_offset = None
    shift = None
    min_margin = None
    if geometry_start is not None:
        approach_offset = geometry_start["projection"] - geometry_start["centre"]
    if geometry_contact is not None:
        contact_point = _contact_point(rows[first_contact], 0) if first_contact is not None else None
        contact_projection = (_dot(contact_point, geometry_contact["tangent"])
                              if contact_point is not None else None)
        if contact_projection is None:
            contact_projection = geometry_contact["projection"]
        first_offset = contact_projection - geometry_contact["centre"]
        shift = contact_projection - geometry_start["projection"]
        true_centre = _true_face_centre(rows[first_contact], 0, geometry_contact)
        if true_centre is not None:
            true_offset = contact_projection - _dot(true_centre, geometry_contact["tangent"])
    margins = []
    for index in approach_rows:
        geometry = _geometry(rows[index], 0)
        if geometry is None:
            continue
        margins.append(min(geometry["projection"] - (geometry["low"] + geometry["margin"]),
                           (geometry["high"] - geometry["margin"]) - geometry["projection"]))
    if margins:
        min_margin = min(margins)

    initial_object = log.get("initial_state", {}).get("object")
    final_object = log.get("final_state", {}).get("object")
    displacement = rotation = 0.0
    if initial_object and final_object:
        displacement = math.hypot(final_object[0] - initial_object[0],
                                  final_object[1] - initial_object[1])
        rotation = math.degrees(_wrap_angle(final_object[2] - initial_object[2]))
    exit_reasons = []
    boundary_exits: Dict[str, str] = {}
    for boundary in log.get("skill_boundaries", []):
        for robot, reason in boundary.get("exit_reasons", {}).items():
            if reason:
                exit_reasons.append(reason)
                boundary_exits[str(robot)] = reason

    pair = _pair_summary(rows, boundary_exits)
    pair["is_pair"] = (spec.ROBOT_ACTIONS[int(request["actions"][0][1])].kind == "push")
    return {
        "rows": len(rows), "requested_m": requested, "exit_reason": exit_reason,
        "contact_fraction": contact_fraction, "contact_seconds": push_contacts * DT,
        "contact_after_engage_fraction": contact_after_engage,
        "pre_contact_push_seconds": pre_contact_push_seconds,
        "push_seconds": len(push_rows) * DT, "longest_loss_seconds": longest * DT,
        "contact_transitions": transitions, "face_switches": face_switches,
        "corner_rejected_seconds": corner_seconds, "min_corner_margin_m": min_margin,
        "first_contact_offset_m": first_offset, "contact_shift_m": shift,
        "first_contact_true_offset_m": true_offset,
        "approach_start_offset_m": approach_offset, "signed_progress_m": signed,
        "signed_fraction": (signed / requested) if requested > 0 else None,
        "push_complete": "push_complete" in str(exit_reason),
        "contact_lost_count": sum(1 for reason in exit_reasons if reason == "contact_lost"),
        "object_displacement_m": displacement, "object_rotation_deg": rotation,
        "goal_progress_m": float(rows[0].get("true_goal_distance", 0.0))
        - float(rows[-1].get("true_goal_distance", 0.0)),
        "edge_assigned_fraction": edge_fraction,
        "total_seconds": len(rows) * DT,
        "pair": pair,
    }


def _write_js(path: Path, variable: str, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.__DATA__ = window.__DATA__ || {};\n"
                    f"window.__DATA__[{json.dumps(variable)}] = "
                    + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n")


def _write_frames_js(path: Path, branch_id: str, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("window.__DATA__ = window.__DATA__ || {};\n"
                    "window.__DATA__.frames = window.__DATA__.frames || {};\n"
                    f"window.__DATA__.frames[{json.dumps(branch_id)}] = "
                    + json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + ";\n")


def build_html(out_dir: Path, panel: Dict, requests: List[Dict],
               summaries: Dict[str, Dict]) -> Dict:
    branches = []
    frames_dir = out_dir / "data" / "frames"
    for request in requests:
        log_path = out_dir / "logs" / f"{request['branch_id']}.json"
        if not log_path.exists():
            continue
        log = json.loads(log_path.read_text())
        frames = [_frame(row) for row in log.get("rows", [])]
        _write_frames_js(frames_dir / f"{request['branch_id']}.js",
                         request["branch_id"], {"frames": frames})
        entry = dict(request)
        entry.update({"summary": summaries.get(request["branch_id"], {}),
                      "status": log.get("status"), "frames": len(frames),
                      "physical_seconds": log.get("physical_seconds")})
        branches.append(entry)

    branch_ids = [entry["branch_id"] for entry in branches]
    manifest = {
        "panel": panel,
        "branches": branches,
        "summaries": summaries,
        "frames": {branch_id: branch_id for branch_id in branch_ids},
        "constants": {"control_dt": DT, "robot_radius": spec.ROBOT_RADIUS,
                      "object_width": spec.OBJECT_WIDTH, "object_height": spec.OBJECT_HEIGHT,
                      "lidar_range": spec.LIDAR_RANGE},
    }
    _write_js(out_dir / "data" / "manifest.js", "manifest", manifest)
    tags = "\n".join(
        f'<script src="data/frames/{branch_id}.js"></script>' for branch_id in branch_ids)
    page = TEMPLATE.replace("<!--FRAME_SCRIPTS-->", tags)
    (out_dir / "index.html").write_text(page)
    return {"branches": len(branches), "frames": sum(entry["frames"] for entry in branches)}


def write_docs(out_dir: Path, panel: Dict, requests: List[Dict],
               summaries: Dict[str, Dict]) -> None:
    import statistics
    groups: Dict[str, List[Dict]] = {}
    for request in requests:
        suffix = request["branch_id"].split("__", 1)[1]
        groups.setdefault(suffix, []).append(request)
    lines = ["# Round 035 — PUSH continuity / corner / cooperative", "",
             f"Panel: `{panel['panel_version']}` · env: `{panel['env_version']}` · "
             f"{len(panel['snapshots'])} scenes · {len(requests)} branches.", "",
             "Open `index.html` directly (all resources are local; frame scripts are static,",
             "so no server is needed).", "",
             "## Aggregate by branch kind", "",
             "| kind | n | mean contact frac | min | contact_lost | face switches | push_complete |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for suffix, items in groups.items():
        vals = [summaries[item["branch_id"]]["contact_fraction"] for item in items
                if summaries.get(item["branch_id"], {}).get("contact_fraction") is not None]
        lost = sum(summaries[item["branch_id"]].get("contact_lost_count", 0) for item in items)
        faces = sum(summaries[item["branch_id"]].get("face_switches", 0) for item in items)
        done = sum(1 for item in items if summaries[item["branch_id"]].get("push_complete"))
        lines.append(f"| {suffix} | {len(items)} | "
                     f"{statistics.mean(vals):.3f} | {min(vals):.3f} | {lost} | {faces} | {done} |")
    short = summaries.get("L1_train_0002__push_039_100")
    if short:
        short_log = out_dir / "logs" / "L1_train_0002__push_039_100.json"
        seg_len = gap = None
        if short_log.exists():
            short_rows = json.loads(short_log.read_text()).get("rows", [])
            if short_rows:
                geometry = _geometry(short_rows[-1], 0)
                seg_len = (geometry["high"] - geometry["low"]) if geometry else None
                first_contact = next((row for row in short_rows
                                      if row["r0"].get("contacting_object")), None)
                gap = first_contact["r0"].get("gap") if first_contact else None
        lines += ["", "## Short-side diagnosis: L1_train_0002", "",
                  "The old v4.5 hard guard required a 0.40 m visible segment and stopped the "
                  "0.25 m short face at gap about 0.125 m. v4.6 replaces it with a soft inset: "
                  "wide faces keep the conservative R+0.02 target, narrow faces shrink the "
                  "inset (down to a 0.1R face), and the centre remains a valid contact target.",
                  "",
                  f"Current fast branch: visible segment {seg_len:.3f} m, exit "
                  f"`{short.get('exit_reason')}`, contact fraction "
                  f"{short.get('contact_fraction')}, first-contact gap {gap}, signed progress "
                  f"{short.get('signed_progress_m'):.3f} m, object displacement "
                  f"{short.get('object_displacement_m'):.3f} m, rotation "
                  f"{short.get('object_rotation_deg'):+.1f}°."
                  if seg_len is not None and gap is not None
                  else "See the L1_train_0002 branches in the replay.",
                  ""]
    lines += ["", "## Controlled corner sweep (r1 centred; true face centre = 1.60 m)", "",
              "| scene | kind | true offset | visible offset | projection shift | corner rejected | rotation | push_complete |",
              "|---|---|---:|---:|---:|---:|---:|---|"]
    for scene in ("corner_in_004", "corner_out_002", "corner_out_006", "corner_out_014"):
        for suffix in ("push_039_100", "push_024_050", "push_012_050"):
            key = f"{scene}__{suffix}"
            summary = summaries.get(key)
            if not summary:
                continue
            lines.append(f"| {scene} | {suffix} | "
                         f"{summary['first_contact_true_offset_m']:+.3f} m | "
                         f"{summary['first_contact_offset_m']:+.3f} m | "
                         f"{summary['contact_shift_m']:+.3f} m | "
                         f"{summary['corner_rejected_seconds']:.2f} s | "
                         f"{summary['object_rotation_deg']:+.1f}° | "
                         f"{summary['push_complete']} |")
    lines += ["", "## Cooperative pair branches (empty 4.0 x 3.0 m arena, no obstacles)", "",
              "| scene | kind | r0 ct* | r1 ct* | both ct* | sync | ready wait | staging r0/r1 | push force before both | obj move before both | rotation | disp | exits |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for scene, _r0, _r1, _tag in COOP_SCENES:
        for suffix in ("pair_024_050", "pair_012_100", "pair_039_050", "pair_024_100"):
            key = f"{scene}__{suffix}"
            summary = summaries.get(key)
            if not summary or not summary.get("pair"):
                continue
            pair = summary["pair"]
            def pct(value):
                return "—" if value is None else f"{100 * value:.0f}%"
            def num(value, digits=2):
                return "—" if value is None else f"{value:.{digits}f}"
            lines.append(f"| {scene} | {suffix} | {pct(pair['r0_contact_after_engage_fraction'])} | "
                         f"{pct(pair['r1_contact_after_engage_fraction'])} | "
                         f"{pct(pair['both_contact_after_engage_fraction'])} | "
                         f"{num(pair['sync_delay_seconds'])} s | "
                         f"{num(pair['ready_wait_seconds'])} s | "
                         f"{num(pair['staging_gap0_m'], 3)}/{num(pair['staging_gap1_m'], 3)} m | "
                         f"{num(pair['force_before_both_contact_seconds'])} s | "
                         f"{num(pair['object_move_before_both_contact_m'], 3)} m | "
                         f"{summary['object_rotation_deg']:+.1f}° | "
                         f"{summary['object_displacement_m']:.2f} m | "
                         f"{pair['r0_exit']}/{pair['r1_exit']} |")
    lines += ["", "Notes: `* after first contact`. `contact fraction` over the whole push stage is "
              "lower because both robots start pushing at the 3 cm staging gap; the table's "
              "`ct*` columns measure contact after engagement. `true offset` uses the object pose "
              "(truth) and is diagnostic only; the",
              "controller sees only the visible segment. `visible offset` is relative to the",
              "visible-segment midpoint. `contact fraction` is measured only inside the PUSH",
              "stage; approach rows are excluded."]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out_dir / "README.md").write_text(
        "# Round 035 — PUSH continuity / corner contact-point evaluation\n\n"
        "Presentation round built from the frozen option testbench pipeline. It does not change\n"
        "the controller, reward or action menu. See `SUMMARY.md` for the aggregate numbers and\n"
        "`index.html` for the replay.\n\n"
        "Rebuild: `.venv/bin/python experiments/option_testbench/continuity_eval.py`\n"
        "Summaries/HTML only: `.venv/bin/python experiments/option_testbench/continuity_eval.py --skip-run`\n")
    (out_dir / "manifest.json").write_text(json.dumps(
        {"round": ROUND, "panel_version": panel["panel_version"],
         "env_version": panel["env_version"], "snapshot_count": len(panel["snapshots"]),
         "request_count": len(requests), "branch_count": len(summaries),
         "frame_count": sum(summary.get("rows", 0) for summary in summaries.values()),
         "contract_version": P.CONTRACT_VERSION, "metric_version": P.METRIC_VERSION,
         "note": "PUSH continuity / corner / cooperative presentation round"},
        indent=1) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-run", action="store_true",
                        help="reuse existing logs and only rebuild summaries / HTML")
    args = parser.parse_args()
    out_dir = ROOT / "outputs" / "option_testbench_v1" / ROUND
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, _fixtures, order = make_panel(out_dir)
    requests = make_requests(order)
    (out_dir / "requests.json").write_text(json.dumps(
        {"request_count": len(requests), "requests": requests,
         "note": "PUSH continuity / corner / cooperative presentation round"},
        indent=1, ensure_ascii=False) + "\n")
    print(f"panel {len(order)} scenes, requests {len(requests)}", flush=True)
    if not args.skip_run:
        for index, request in enumerate(requests):
            runner.run_branch(out_dir, index, out_dir, branch_wall_seconds=40.0,
                              recovery_horizon_seconds=30.0)
    summaries = {}
    for request in requests:
        path = out_dir / "logs" / f"{request['branch_id']}.json"
        if path.exists():
            summaries[request["branch_id"]] = _summary(json.loads(path.read_text()), request)
    write_docs(out_dir, panel, requests, summaries)
    info = build_html(out_dir, panel, requests, summaries)
    print(json.dumps({"index": str(out_dir / "index.html"), **info}, indent=1), flush=True)


TEMPLATE = (HERE / "continuity_template.html").read_text(encoding="utf-8")

if __name__ == "__main__":
    main()
