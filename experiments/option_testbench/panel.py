"""Round-000 panel, frozen contracts and the pre-registered request menu.

Only training instances, diagnostic trajectories or explicit development fixtures are used.
Nothing here reads validation/test results. The panel is frozen before any branch runs.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.scenes import Box, Pose  # noqa: E402
from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout, spec  # noqa: E402

HERE = Path(__file__).resolve().parent
PANEL_VERSION = "option-testbench-panel-v1.1"  # v1.1: bilingual detailed annotations only; geometry unchanged
CONTRACT_VERSION = "option-contract-v2"  # v2: calibrated PUSH/REPOSITION fulfilment (Round 003)
METRIC_VERSION = "option-metric-v2"  # v2: six calibration fixes; v1 outputs preserved

# The three pre-registered baseline options, in the frozen action-table order:
# v4 type-major menu: index = type_index * 4 + scale_index; scales are (2, 6, 12, 24) s.
#  17 = PUSH(alpha=0 deg, speed=0.16 m/s) at scale 6 s (type 4)
#  12 = REPOSITION_CCW at the 0.10 m requested distance (type 3)
#  16 = REPOSITION_CW at the 0.10 m requested distance (type 4)
#  20 = WAIT_BRAKE at scale 2 s (type 5), the explicit teammate brake/hold condition
PUSH_0_016 = 6       # type 1 (PUSH a=0, v=0.24) at 0.50 m
REPOSITION_CCW = 12  # type 3 at the 0.10 m requested distance
REPOSITION_CW = 16   # type 4 at the 0.10 m requested distance
WAIT_BRAKE = 20      # type 5 at scale 2 s
FOLLOW_6S = 25       # type 6 at scale 6 s
BASELINE_OPTIONS = {"PUSH_0_016": PUSH_0_016, "REPOSITION_CCW": REPOSITION_CCW,
                    "REPOSITION_CW": REPOSITION_CW, "WAIT_BRAKE": WAIT_BRAKE}


FOCUS_EXPLANATIONS = {
    "joint_push": {
        "zh": "共同 PUSH(0°,0.16)", "en": "Joint PUSH(0 deg, 0.16 m/s)",
        "detail_zh": "两侧同时执行 PUSH(0°,0.16 m/s)：检验联合执行是把推力叠加，还是因方向相反而抵消。",
        "detail_en": "Both robots PUSH(0 deg, 0.16 m/s) at once: does joint execution add the "
                     "pushes, or cancel them because the directions oppose?"},
    "push_plus_ccw": {
        "zh": "PUSH + 逆时针绕行(0.25 m)", "en": "PUSH + CCW reposition (0.25 m)",
        "detail_zh": "r0 执行 PUSH、r1 执行 REPOSITION_CCW(0.25 m)：一侧推动，另一侧沿可见切向绕行。",
        "detail_en": "r0 pushes while r1 repositions CCW (0.25 m) along the visible tangent."},
    "ccw_plus_push": {
        "zh": "逆时针绕行(0.25 m) + PUSH", "en": "CCW reposition (0.25 m) + PUSH",
        "detail_zh": "r0 执行 REPOSITION_CCW、r1 执行 PUSH：与 push_plus_ccw 左右互换，检验对称性。",
        "detail_en": "r0 repositions CCW while r1 pushes: the mirror of push_plus_ccw, to test "
                     "symmetry."},
    "push_plus_cw": {
        "zh": "PUSH + 顺时针绕行(0.25 m)", "en": "PUSH + CW reposition (0.25 m)",
        "detail_zh": "r0 执行 PUSH、r1 执行 REPOSITION_CW(0.25 m)：切向方向与 CCW 相反。",
        "detail_en": "r0 pushes while r1 repositions CW (0.25 m): the opposite tangential "
                     "direction from CCW."},
    "double_ccw": {
        "zh": "双侧逆时针绕行(0.25 m)", "en": "Double CCW reposition (0.25 m)",
        "detail_zh": "两侧都执行 REPOSITION_CCW(0.25 m)：检验纯边界跟随、队友间隙与间隙保持。",
        "detail_en": "Both robots reposition CCW (0.25 m): pure boundary following, teammate "
                     "clearance and gap holding."},
    "push_follow": {
        "zh": "主推 + 支撑跟随", "en": "Push + support follow",
        "detail_zh": "r0 推动，r1 在非推动侧做相对位姿跟随（不主动持续施力）。",
        "detail_en": "r0 pushes while r1 follows the relative pose on a non-push side without a "
                     "sustained push."},
    "single_push_brake": {
        "zh": "单侧 PUSH，队友持续制动", "en": "One-sided PUSH, teammate brakes/holds",
        "detail_zh": "只有 r0 执行 PUSH；r1 显式执行 WAIT_BRAKE(0.5 s) 并随后保持制动。队友仍在同一"
                     "物理世界、速度不清零，这是“另一机器人持续制动的诊断条件”，不是原始联合动作。",
        "detail_en": "Only r0 pushes; r1 explicitly runs WAIT_BRAKE(0.5 s) and then holds. The "
                     "teammate stays in the same physical world with its velocity intact: this is "
                     "the 'other robot keeps braking' diagnostic condition, not a raw joint action."},
    "recovery_2step": {
        "zh": "两步恢复（PUSH → 逆时针绕行）", "en": "Two-step recovery (PUSH then CCW)",
        "detail_zh": "恢复序列最多两次技能、累计物理时间不超过 12 s；每一步都在物理周期上检查上限。"
                     "若第一步耗时过长、剩余预算不足一个完整技能，第二步不会启动，整条分支记 "
                     "evaluation_horizon（窗口截断），而不是技能主动失败。",
        "detail_en": "A recovery sequence of at most two skills and at most 12 s of physical time; "
                     "the cap is checked on the physical clock. If step one is too slow to leave a "
                     "full skill inside the cap, step two is not started and the branch is marked "
                     "evaluation_horizon (a window cut-off), not a skill failure."},
}

def _instance(name, object_pose, robots, goal, labels):
    return layout.InstanceSpec(
        name=name, map="L0", split="dev", index=0, seed=0,
        arena=spec.ARENA, obstacles=(),
        object_pose=Pose(*object_pose),
        robot_poses=tuple(Pose(*robot) for robot in robots),
        goal_centre=tuple(goal), goal_radius=spec.GOAL_RADIUS,
        env_version=spec.ENV_VERSION), labels


def _labels(family, title_zh, title_en, problem_zh, problem_en, geometry_zh, geometry_en,
            expect_zh, expect_en, watch_zh, watch_en):
    return {"family": family, "title_zh": title_zh, "title_en": title_en,
            "problem_zh": problem_zh, "problem_en": problem_en,
            "geometry_zh": geometry_zh, "geometry_en": geometry_en,
            "expectation_zh": expect_zh, "expectation_en": expect_en,
            "watch_zh": watch_zh, "watch_en": watch_en}


def panel_definitions():
    """The eight frozen development snapshots (geometry plus bilingual annotations)."""
    half = math.pi / 2.0
    rows = [
        _instance("face", (1.60, 1.10, 0.0),
                  [(1.60, 0.545, -half), (1.60, 1.655, half)], (1.60, 1.95),
                  _labels(
                      "face",
                      "平直面中部（face）", "Flat face centre (face)",
                      "最基本的接近—接触—推动是否兑现；两侧对推时物体是否真的移动。",
                      "Does the basic approach-contact-push sequence fulfil its contract, and "
                      "does the object actually move when the two robots push from opposite sides?",
                      "物体中心 (1.60,1.10) m、朝向 0°（长边沿 x，1.00×0.25 m）；r0 在 -y 侧 "
                      "(1.60,0.545) 朝 +y，r1 在 +y 侧 (1.60,1.655) 朝 -y；两机器人到最近面的"
                      "边到边间隙都是 0.25 m；目标 (1.60,1.95)，物体到目标 0.85 m。",
                      "Object centre (1.60,1.10) m, heading 0° (long axis along x, 1.00×0.25 m); "
                      "r0 on the -y side at (1.60,0.545) facing +y, r1 on the +y side at "
                      "(1.60,1.655) facing -y; both edge-to-edge face gaps are 0.25 m; goal "
                      "(1.60,1.95), object-to-goal 0.85 m.",
                      "单侧 PUSH 应把物体推离该机器人；两侧方向相反的共同 PUSH 净推力可能接近零。"
                      "即使双方都声称 push_complete，物体也可能不动。",
                      "A one-sided PUSH should move the object away from that robot; a joint PUSH "
                      "with opposite directions may cancel to near-zero net force. Both may claim "
                      "push_complete while the object does not move.",
                      "接触建立时间、累计真实接触时长、物体位移向量、目标方向进展，以及退出字符串"
                      "与实际效果是否一致。",
                      "Contact build-up time, accumulated real contact, object displacement, "
                      "goal-direction progress, and whether the exit string matches the real effect.")),
        _instance("corner_minus", (1.60, 1.10, 0.0),
                  [(2.12, 0.545, -half), (1.08, 1.655, half)], (1.60, 1.95),
                  _labels(
                      "corner",
                      "凸角外侧微小偏移（corner_minus）", "Small offset outside the corner (corner_minus)",
                      "机器人紧贴被接触面 +x 端角点的外侧；控制器能否保持选中同一可见线段。",
                      "The robot sits just outside the +x end corner of the contacted face; can "
                      "the controller keep the same visible face segment instead of slipping off?",
                      "物体与 goal 同 face；r0 (2.12,0.545) 朝 +y，落在 -y 面 +x 角点外侧约 "
                      "0.02 m；r1 (1.08,1.655) 朝 -y 在 +y 面对称端；目标 (1.60,1.95)。",
                      "Same object and goal as face; r0 at (2.12,0.545) facing +y, about 0.02 m "
                      "outside the +x corner of the -y face; r1 at (1.08,1.655) facing -y near the "
                      "opposite end; goal (1.60,1.95).",
                      "如果选面/接触点稳定，行为应与 face 接近；微小偏移不应改变结局。",
                      "If face/contact selection is stable, behaviour should stay close to face; a "
                      "tiny offset should not flip the outcome.",
                      "选中片段点数与弧长、法向跳变、gap、接触点轨迹、退出原因，并与 corner_plus 对照。",
                      "Selected patch point count and arc, normal jumps, gap, contact-point travel, "
                      "exit reason, and the direct comparison with corner_plus.")),
        _instance("corner_plus", (1.60, 1.10, 0.0),
                  [(2.08, 0.545, -half), (1.12, 1.655, half)], (1.60, 1.95),
                  _labels(
                      "corner",
                      "凸角内侧微小偏移（corner_plus）", "Small offset inside the corner (corner_plus)",
                      "上一场景的镜像侧小偏移（约 0.04 m），其余完全不变；检验小扰动是否造成大幅切换。",
                      "Mirror-side small offset (about 0.04 m) of corner_minus with everything else "
                      "identical; does a tiny perturbation cause a large behaviour switch?",
                      "物体与 goal 同 face；r0 (2.08,0.545) 朝 +y，落在 -y 面 +x 角点内侧约 "
                      "0.02 m；r1 (1.12,1.655) 朝 -y；其余与 corner_minus 相同。",
                      "Same object and goal as face; r0 at (2.08,0.545) facing +y, about 0.02 m "
                      "inside the +x corner of the -y face; r1 at (1.12,1.655) facing -y; the rest "
                      "is identical to corner_minus.",
                      "与 corner_minus 相比，理想情况下退出层与时长应基本一致；若出现 early "
                      "contact_lost，说明选面/接触判定对微小几何敏感。",
                      "Compared with corner_minus, the exit layer and duration should ideally stay "
                      "similar; an early contact_lost would show the selection is sensitive to tiny "
                      "geometry.",
                      "是否在角点处分裂/丢失片段、是否提前 contact_lost、物理时长、物体位移，"
                      "并与 corner_minus 做同动作配对。",
                      "Whether the patch splits or is lost at the corner, early contact_lost, "
                      "physical duration, object displacement, paired with corner_minus under the "
                      "same action.")),
        _instance("edge_turn", (1.60, 1.10, 0.0),
                  [(2.25, 1.655, half), (1.00, 0.50, -half)], (1.60, 0.30),
                  _labels(
                      "edge",
                      "物体端部边沿的切向切换（edge_turn）",
                      "Tangential transition at the object end edge (edge_turn)",
                      "目标在物体下方。r0 在物体上方但起点在 +x 端部；r1 在物体下方，会把物体推向"
                      "目标反方向。r0 从端部边沿开始、预期绕该边沿做切向切换，故命名 edge_turn"
                      "（这是设计意图，不是已验证结论）。检验绕端部边沿的选面/切向切换、间隙，"
                      "以及“进度”是否对应真实位移。",
                      "The goal is below the object. r0 is above but starts at the +x end; r1 is "
                      "below and pushes the object away from the goal. r0 starts at the end edge "
                      "and is expected to turn around it tangentially, which is why the case is "
                      "named edge_turn (a design intent, not a verified outcome). The case tests "
                      "the face/tangent transition around the end edge, the gap, and whether "
                      "'progress' matches real displacement.",
                      "物体 (1.60,1.10) 朝向 0°：x∈[1.10,2.10]、y∈[0.975,1.225]；r0 (2.25,1.655) "
                      "在 +x 端外侧 0.15 m、物体上方（边到边间隙 0.28 m）；r1 (1.00,0.50) 在 −x "
                      "端外侧、物体下方（间隙 0.31 m）；目标 (1.60,0.30) 在物体下方，物体到目标 "
                      "0.80 m。",
                      "Object (1.60,1.10) heading 0°: x in [1.10,2.10], y in [0.975,1.225]; r0 "
                      "at (2.25,1.655) is 0.15 m beyond the +x end and above the object (edge gap "
                      "0.28 m); r1 at (1.00,0.50) is beyond the -x end and below the object (gap "
                      "0.31 m); goal (1.60,0.30) is below the object, 0.80 m away.",
                      "物体在上方的机器人向下推时朝目标移动（r0 侧别正确，但起点在端部）；在下方的"
                      "机器人向上推时远离目标（r1 方向相反）。因此 (4,4) 双侧 PUSH 可能互相抵消；"
                      "(4,13)（r0 推、r1 制动）已显示物体能朝目标移动约 +0.058 m。",
                      "A robot above pushes the object down toward the goal (r0 is on the right "
                      "side but starts at the end); a robot below pushes it up, away from the goal "
                      "(r1 opposes). A double PUSH (4,4) may therefore cancel; the (4,13) branch "
                      "(r0 pushes, r1 brakes) already moved the object about +0.058 m toward the "
                      "goal.",
                      "可见面/切向是否在顶面与端面之间切换（绕端部边沿）、“progress”是否对应真实"
                      "位移、目标方向进展的符号，以及 r0 是否只是停在端部。",
                      "Whether the visible face/tangent switches between the top face and the end "
                      "face (turning around the end edge), whether 'progress' matches real "
                      "displacement, the sign of goal-direction progress, and whether r0 simply "
                      "stays at the end.")),
        _instance("corridor_wide", (1.60, 0.925, 0.0),
                  [(1.60, 0.36, half), (1.60, 1.48, -half)], (1.60, 1.95),
                  _labels(
                      "corridor",
                      "宽物体—墙通道（corridor_wide）", "Wide object-wall corridor (corridor_wide)",
                      "正常的边界跟随：REPOSITION 能否在通道内保持目标间隙并沿切向推进。",
                      "Normal boundary following: can REPOSITION hold the target gap inside the "
                      "corridor and make tangential progress?",
                      "物体 (1.60,0.925) 朝向 0°，-y 面在 y=0.80 m；通道高 0.80 m；r0 "
                      "(1.60,0.36) 朝 +y 在通道内（离墙 0.18 m、离物体面 0.26 m）；r1 "
                      "(1.60,1.48) 朝 -y 在上方；目标 (1.60,1.95)。",
                      "Object (1.60,0.925) heading 0°, -y face at y=0.80 m; corridor height "
                      "0.80 m; r0 at (1.60,0.36) facing +y inside it (0.18 m to the wall, 0.26 m to "
                      "the object face); r1 at (1.60,1.48) facing -y above; goal (1.60,1.95).",
                      "间隙应能稳定保持，REPOSITION 应兑现请求距离；物体被推动时不应出现意料外接触。",
                      "The gap should stay stable and REPOSITION should fulfil the requested "
                      "distance; pushing should not create unexpected contacts.",
                      "gap 的 min/median/max、切向累计进度、意料外接触时长、墙/队友最小距离。",
                      "gap min/median/max, cumulative tangential progress, unexpected-contact time, "
                      "and minimum wall/teammate distance.")),
        _instance("corridor_tight", (1.60, 0.710, 0.0),
                  [(1.60, 0.282, half), (1.60, 1.26, -half)], (1.60, 1.75),
                  _labels(
                      "corridor",
                      "窄可通行通道（corridor_tight）", "Narrow passable corridor (corridor_tight)",
                      "机器人几何可放入但目标间隙几乎到极限；受阻、退出与恢复是预期结果，不预设一定应通过。",
                      "The robot fits but the target gap is near the geometric limit; blocking, "
                      "exit and recovery are expected outcomes, not assumed passable.",
                      "物体 (1.60,0.710) 朝向 0°，-y 面在 y=0.585 m；通道高 0.585 m；r0 "
                      "(1.60,0.282) 朝 +y（离墙约 0.052 m、离物体面约 0.123 m）；r1 "
                      "(1.60,1.26) 朝 -y；目标 (1.60,1.75)。",
                      "Object (1.60,0.710) heading 0°, -y face at y=0.585 m; corridor height "
                      "0.585 m; r0 at (1.60,0.282) facing +y (about 0.052 m to the wall and "
                      "0.123 m to the object face); r1 at (1.60,1.26) facing -y; goal (1.60,1.75).",
                      "目标间隙 0.10 m 已接近通道极限，可能出现 blocked_contact / safety_abort；"
                      "两步恢复的第一步若耗时长，第二步可能因 12 s 上限而不启动并记 evaluation_horizon。",
                      "The 0.10 m target gap is near the corridor limit; blocked_contact / "
                      "safety_abort are plausible. If the first recovery step is slow, the second "
                      "may not start under the 12 s cap and the branch is marked evaluation_horizon.",
                      "gap 下限、blocked_contact、safety_abort、恢复是否跑到第二步、以及是否"
                      "把 horizon 截断误当成技能失败。",
                      "gap floor, blocked_contact, safety_abort, whether the recovery reaches step "
                      "two, and whether a horizon cut-off is mistaken for a skill failure.")),
        _instance("teammate", (1.60, 1.10, 0.0),
                  [(1.35, 0.545, -half), (1.85, 0.545, -half)], (1.60, 1.95),
                  _labels(
                      "teammate",
                      "队友干扰：同侧近距离（teammate）",
                      "Teammate interference: same side, close together (teammate)",
                      "两机器人同在 -y 侧、相距 0.50 m；联合执行是叠加还是互相阻挡，安全退出是否触发。",
                      "Both robots are on the -y side 0.50 m apart; does joint execution add up or "
                      "block each other, and do safety exits trigger?",
                      "物体 (1.60,1.10) 朝向 0°；r0 (1.35,0.545) 与 r1 (1.85,0.545) 都朝 +y，"
                      "中心距 0.50 m（边到边间隙 0.14 m）；两机器人到 +y/-y 面间隙均 0.25 m；"
                      "目标 (1.60,1.95)。",
                      "Object (1.60,1.10) heading 0°; r0 at (1.35,0.545) and r1 at (1.85,0.545) "
                      "both facing +y, centre distance 0.50 m (edge gap 0.14 m); both face gaps "
                      "0.25 m; goal (1.60,1.95).",
                      "同侧共同 PUSH 应叠加并把物体推向目标；队友间隙与安全触发应是主要限制。",
                      "A same-side joint PUSH should add up and move the object toward the goal; "
                      "teammate clearance and safety triggers should be the main limits.",
                      "队友最近距离、safety_trigger、是否双方都保持接触、物体位移是否大于单侧 PUSH。",
                      "Teammate nearest distance, safety_trigger, whether both keep contact, and "
                      "whether displacement exceeds a one-sided PUSH.")),
    ]
    return rows


MOVING_LABELS = _labels(
    "dynamic",
    "真实动态边界：一次 PUSH 之后（moving_recovery）",
    "Real dynamic boundary after one PUSH (moving_recovery)",
    "不是初始静止态，而是真实执行一次 PUSH 后冻结的决策边界；物体与机器人都有非零速度、存在接触。",
    "Not a static start: it is the decision boundary frozen after one real PUSH, with non-zero "
    "object and robot velocities and an existing contact.",
    "由 face 出发，执行一次真实联合技能（r0 PUSH(0°,0.16)、r1 WAIT_BRAKE(0.5)）后冻结；"
    "物体从 y=1.10 移到约 y=1.312 m、仍以约 +0.114 m/s 运动，r0 在 y≈1.007 m 重新接触，"
    "共 103 个周期（5.15 s）。快照保存物理/接触/时钟/RNG 状态，恢复用精确回放而非近似克隆。",
    "Frozen from face after one real joint skill (r0 PUSH(0 deg, 0.16 m/s), r1 WAIT_BRAKE(0.5)): "
    "the object moved from y=1.10 to about y=1.312 m and still moves at about +0.114 m/s, r0 is "
    "back in contact at y about 1.007 m, across 103 periods (5.15 s). The snapshot stores the "
    "physics/contact/clock/RNG state and is restored by exact replay, not an approximate clone.",
    "惰性与接触连续性：恢复动作应从这个真实边界继续，而不是把速度清零后重新开始。",
    "Inertia and contact continuity: a recovery action should continue from this real boundary "
    "instead of zeroing the velocity and starting over.",
    "初始速度是否被保留、接触是否连续、两步恢复能否在 12 s 上限内完成第二步、以及恢复是否"
    "真的改变了物体运动。",
    "Whether the initial velocity is preserved, whether contact stays continuous, whether the "
    "two-step recovery reaches step two inside the 12 s cap, and whether recovery really changes "
    "the object motion.")


def fixture_report(instances):
    report = {}
    for instance, labels in instances:
        legality = layout.scene_legality_report([instance])
        report[instance.name] = {"labels": labels, "legal": bool(legality["ok"]),
                                 "problems": legality["problems"],
                                 "content_sha256": layout.content_hash(instance),
                                 "instance": instance.as_dict()}
    return report


def build_panel():
    instances = panel_definitions()
    report = fixture_report(instances)
    illegal = [name for name, entry in report.items() if not entry["legal"]]
    if illegal:
        raise ValueError(f"illegal development fixtures: {illegal}")
    return {instance.name: instance for instance, _ in instances}, report


def snapshot_hash(instance, branch_state=None):
    """Canonical hash of the frozen start state (geometry plus, if dynamic, motion)."""
    parts = [spec.ENV_VERSION, layout.content_hash(instance)]
    if branch_state is not None:
        parts.append(f"{branch_state['total_periods']}:{branch_state['hold_seconds']:.12f}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _state_tuple(env):
    return {"object": list(env.object_state().as_tuple()),
            "robots": [list(state.as_tuple()) for state in env.robot_states()],
            "total_periods": int(env.total_periods), "hold_seconds": float(env.hold_seconds)}


def make_moving_snapshot(instance, out_dir, prefix_action=(5, 20)):
    """Run one real joint skill and freeze the resulting boundary as a dynamic fixture."""
    world = E.RandomStartEnv(instance)
    boundary = world.joint_step(*prefix_action)
    state = world.export_branch_state()
    path = out_dir / "snapshots" / "moving_recovery.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"base_instance": instance.as_dict(), "branch_state": state}, handle)
    return {"file": str(path.relative_to(out_dir)), "base_instance": instance.name,
            "prefix_action": list(prefix_action),
            "prefix_exit": boundary["termination"],
            "prefix_seconds": boundary["periods"] * spec.CONTROL_DT,
            "state": _state_tuple(world),
            "sha256": snapshot_hash(instance, state)}


def instance_from_dict(payload):
    return layout.InstanceSpec(
        name=payload["name"], map=payload["map"], split=payload["split"],
        index=int(payload["index"]), seed=int(payload["seed"]),
        arena=tuple(payload["arena"]),
        obstacles=tuple(Box(**box) for box in payload["obstacles"]),
        object_pose=Pose(**payload["object"]),
        robot_poses=tuple(Pose(**pose) for pose in payload["robots"]),
        goal_centre=tuple(payload["goal_centre"]),
        goal_radius=float(payload["goal_radius"]),
        env_version=payload.get("env_version", spec.ENV_VERSION))


def pickle_load(path):
    with Path(path).open("rb") as handle:
        return pickle.load(handle)


def contracts():
    """Three-layer contract: controller claim, independent fulfilment, task effect."""
    return {
        "version": CONTRACT_VERSION, "env_version": spec.ENV_VERSION,
        "layers": ["controller_claim", "independent_contract", "task_effect"],
        "push": {
            "option": "PUSH", "action_contract": "keep real object contact and push along the "
            "surface normal the controller selected from its own scan",
            "legal_inputs": "robot's own lidar/patch/contact history and the public clock",
            "completion": "contact established and signed push travel along the direction "
                          "declared at execution start >= 0.80 * requested distance, before the "
                          "derived timeout",
            "contact_established": "accumulated real object-contact time >= 0.50 s",
            "claim_exits": ["push_complete"],
            "benign_exits": ["joint_window_end", "option_budget", "face_switch", "contact_lost",
                             "blocked_no_progress",
                             "patch_lost", "not_observed", "push_wall_timeout",
                             "evaluation_horizon"],
            "violation_exits": ["safety_abort", "blocked_contact", "environment_termination"],
            "window_seconds": 8.0,
            "thresholds": {"contact_established_seconds": 0.50,
                           "full_push_seconds": spec.PUSH_ACTIVE_SECONDS,
                           "progress_fraction_min": 0.80,
                           "direction_error_speed_min": 0.02,
                           "direction_error_deg_max": 45.0},
            "note": "the PUSH promise is contact pushing, not a world-direction object "
                    "translation; goal-direction progress is a separate task-effect metric",
        },
        "reposition": {
            "option": "REPOSITION", "action_contract": "move along the locally estimated "
            "visible-boundary tangent by the requested distance while holding the target gap",
            "legal_inputs": "robot's own lidar/patch/contact history and the public clock",
            "completion": "signed tangential travel along the direction declared at execution "
                          "start >= 0.80 * requested distance, before the derived timeout",
            "claim_exits": ["reposition_complete"],
            "benign_exits": ["joint_window_end", "option_budget", "face_switch",
                             "blocked_no_progress", "blocked_contact",
                             "patch_lost", "not_observed", "evaluation_horizon"],
            "violation_exits": ["safety_abort", "environment_termination"],
            "window_seconds": 8.0,
            "thresholds": {"progress_fraction_min": 0.80, "move_seconds_fraction_min": 0.80,
                           "signed_direction": True},
            "note": "going around a corner is not automatically a cross-face completion",
        },
        "unresolved": ["wall_timeout", "evaluation_horizon", "missing_log", "batch_timeout"],
        "follow": {
            "kind": "follow",
            "completion": "effective FOLLOW time >= 0.80 * requested follow duration and the "
                          "relative-pose error within tolerance for >= 0.80 of it; actual "
                          "contact must stay below the threshold",
            "claim_exits": ["follow_complete"],
            "violation_exits": ["safety_abort", "environment_termination"],
            "thresholds": {"follow_seconds_fraction_min": 0.80,
                           "within_tolerance_fraction_min": 0.80,
                           "pose_tolerance_m": spec.FOLLOW_POSE_TOLERANCE,
                           "max_contact_seconds": 0.20},
            "benign_exits": ["option_budget", "face_switch", "patch_lost",
                             "anchor_unobservable", "joint_window_end", "blocked_no_progress"],
        },
        "calibration_v2": [
            "PUSH fulfilment split into contact_established (0.50 s) and full_push (2.00 s)",
            "contact/reposition stats truncated at each robot's own skill-end period",
            "REPOSITION uses positive accumulated progress in the requested direction (no abs)",
            "teammate gap subtracts only this robot's radius (range is to the teammate surface)",
            "normal-jump count recomputed from the normal_world sequence (PUSH corner_event is 0)",
            "conditional fulfilment numerator restricted to accepted requests; rejected/unknown "
            "reported separately"],
        "coverage": {
            "unit": "frozen (snapshot, action, teammate_protocol) request",
            "request_coverage": "accepted / all pre-registered requests",
            "state_coverage": "snapshots with >=1 accepted request / panel snapshots",
            "conditional_fulfillment": "fulfilled / accepted",
            "conservative_interval": "fulfilled / accepted  ..  (fulfilled+unresolved) / accepted",
            "effective_fulfillment_coverage": "fulfilled / all pre-registered requests",
        },
    }


def requests():
    """At most three pre-registered branches per snapshot; 24 in total."""
    menu = []
    plan = {
        "face": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                 ("push_reposition_ccw", [[PUSH_0_016, REPOSITION_CCW]], "push_plus_ccw"),
                 ("reposition_ccw_push", [[REPOSITION_CCW, PUSH_0_016]], "ccw_plus_push")],
        "corner_minus": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                         ("push_reposition_cw", [[PUSH_0_016, REPOSITION_CW]], "push_plus_cw"),
                         ("reposition_ccw_ccw", [[REPOSITION_CCW, REPOSITION_CCW]], "double_ccw")],
        "corner_plus": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                        ("push_reposition_cw", [[PUSH_0_016, REPOSITION_CW]], "push_plus_cw"),
                        ("reposition_ccw_ccw", [[REPOSITION_CCW, REPOSITION_CCW]], "double_ccw")],
        "edge_turn": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                      ("reposition_ccw_push", [[REPOSITION_CCW, PUSH_0_016]], "ccw_plus_push"),
                      ("push_teammate_brake", [[4, 13]], "single_push_brake")],
        "corridor_wide": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                          ("reposition_ccw_push", [[REPOSITION_CCW, PUSH_0_016]], "ccw_plus_push"),
                          ("push_reposition_ccw", [[PUSH_0_016, REPOSITION_CCW]], "push_plus_ccw")],
        "corridor_tight": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                           ("recovery_push_then_ccw", [[PUSH_0_016, WAIT_BRAKE], [REPOSITION_CCW, WAIT_BRAKE]], "recovery_2step"),
                           ("reposition_ccw_push", [[REPOSITION_CCW, PUSH_0_016]], "ccw_plus_push")],
        "teammate": [("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                     ("push_reposition_ccw", [[PUSH_0_016, REPOSITION_CCW]], "push_plus_ccw"),
                     ("push_follow", [[PUSH_0_016, FOLLOW_6S]], "push_follow")],
        "moving_recovery": [("recovery_push_then_ccw", [[PUSH_0_016, WAIT_BRAKE], [REPOSITION_CCW, WAIT_BRAKE]], "recovery_2step"),
                            ("push_push", [[PUSH_0_016, PUSH_0_016]], "joint_push"),
                            ("reposition_ccw_push", [[REPOSITION_CCW, PUSH_0_016]], "ccw_plus_push")],
    }
    for snapshot, branches in plan.items():
        for suffix, actions, focus in branches:
            recovery = any(len(pair) > 1 for pair in actions) or len(actions) > 1
            explanation = FOCUS_EXPLANATIONS[focus]
            menu.append({"branch_id": f"{snapshot}__{suffix}", "snapshot": snapshot,
                         "actions": actions, "focus": focus,
                         "focus_zh": explanation["zh"], "focus_en": explanation["en"],
                         "focus_detail_zh": explanation["detail_zh"],
                         "focus_detail_en": explanation["detail_en"],
                         "action_labels": [[spec.ROBOT_ACTIONS[index].label for index in pair]
                                           for pair in actions],
                         "recovery": bool(recovery),
                         "teammate_protocol": ("teammate_brake_hold"
                                               if any(13 in pair for pair in actions)
                                               else "shared_clock_joint"),
                         "requested_wall_seconds": 10.0,
                         "requested_physical_seconds": 12.0 if recovery else 8.0})
    return menu


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, default=str, ensure_ascii=False))
    temporary.replace(path)


def prepare(out_dir, snapshots_dir=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = Path(snapshots_dir) if snapshots_dir else out_dir
    panel, report = build_panel()
    moving = make_moving_snapshot(panel["face"], snapshots_dir)
    frozen = {}
    for name, instance in panel.items():
        if name == "moving_recovery":
            continue
        frozen[name] = {"kind": "static", "instance": instance.as_dict(),
                        "sha256": snapshot_hash(instance)}
    frozen["moving_recovery"] = {"kind": "dynamic", "instance": moving["base_instance"],
                                  "file": moving["file"], "sha256": moving["sha256"],
                                  "labels": MOVING_LABELS}
    write_json(out_dir / "panel.json", {
        "panel_version": PANEL_VERSION, "env_version": spec.ENV_VERSION,
        "snapshot_count": 8, "fixtures": report, "snapshots": frozen,
        "moving_recovery": moving,
        "note": "development fixtures only; not read from validation/test results"})
    write_json(out_dir / "contracts" / "contracts.json", contracts())
    request_list = requests()
    assert len(request_list) <= 24, len(request_list)
    write_json(out_dir / "requests.json", {
        "request_count": len(request_list), "menu": BASELINE_OPTIONS,
        "requests": request_list,
        "note": "frozen before running; no post-hoc request removal"})
    write_json(out_dir / "manifest.json", {
        "round": "round_000", "panel_version": PANEL_VERSION,
        "contract_version": CONTRACT_VERSION, "metric_version": METRIC_VERSION,
        "env_version": spec.ENV_VERSION, "branch_wall_seconds": 10.0,
        "batch_wall_seconds": 60.0, "render_wall_seconds": 30.0,
        "test_wall_seconds": 30.0, "round_compute_seconds": 180.0,
        "request_count": len(request_list), "snapshot_count": 8,
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((ROOT / "src/coimagine/random_start").glob("*.py"))},
        "status": "prepared"})
    (out_dir / "ROUND_PLAN.md").write_text(
        "# Round 000 build plan\n\n"
        "Goal: build the bounded testbench, run the frozen baseline menu, deliver the local "
        "HTML replay, then stop for user feedback. No controller optimisation in this round.\n\n"
        "- 8 frozen development snapshots (static plus one dynamic recovery boundary).\n"
        "- <= 24 pre-registered branches, each <= 10 s wall, batch <= 60 s, render/test <= 30 s.\n"
        "- Three metric layers: controller claim, independent contract, task effect.\n"
        "- Baseline options: PUSH(0,0.16), REPOSITION_CCW(0.25), REPOSITION_CW(0.25), "
        "WAIT_BRAKE(0.5) as the teammate brake/hold condition.\n")
    return {"panel": panel, "report": report, "moving": moving, "requests": request_list}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/option_testbench_v1/round_000")
    args = parser.parse_args()
    result = prepare(args.out)
    print(json.dumps({"snapshots": list(result["panel"]),
                      "requests": len(result["requests"]),
                      "legal": all(entry["legal"] for entry in result["report"].values())},
                     indent=1))
