#!/usr/bin/env python
"""手动调参用的最小可运行脚本：完整环境构建 + 实时渲染 + 两种推动控制律。

    python scripts/tune_push_live.py                 # 实时窗口（1x 仿真时间）
    python scripts/tune_push_live.py --record out.mp4 # 无窗口，导出 MP4
    python scripts/tune_push_live.py --law compare    # V 与 X 并排跑

设计：**所有可调参数硬编码在最上面的参数区**，控制律、阶段机、表面速度估计、渲染全部写在本文件里。
只从仓库借两样东西，因为它们是"冻结的物理与传感"、自己重写会失真：

  * `coimagine.continuous2d` 的 pymunk 后端与 `SceneSpec`（真实物理）
  * `coimagine.local_contact_option.sensing.scan_surface`（360° 射线扫描与片段拟合）

除这两样之外不引用 `local_contact_option` 的 control / position / option 模块，
控制策略在这个文件里独立实现，方便你直接改。
"""

from __future__ import annotations

# =============================================================================
# 参数区：只改这里
# =============================================================================

LAW = "position"            # "velocity" | "position" | "compare"
SEED = 0

# --- 场景 -------------------------------------------------------------------
ARENA = (3.2, 2.2)
OBSTACLES = ()              # 例：((1.6, 0.33, 0.09, 0.33, 0.0), (1.6, 1.68, 0.09, 0.33, 0.0))
GOAL = (2.55, 1.10, 0.35, 0.35, 0.0)        # cx, cy, half_w, half_h, angle
OBJECT_POSE = (1.20, 1.15, -0.75)           # x, y, angle(rad)
OBJECT_SIZE = (1.0, 0.25)                   # width, height
ROBOT_POSES = ((1.79, 0.61, 2.393), (0.55, 1.69, 2.443))
ACTIVE_ROBOTS = (0,)                        # (0,), (1,) 或 (0, 1)
ALPHA_DEG = (0.0, 0.0)                      # 每台机器人的切向动作角

# --- 物理（冻结；改这里只影响本脚本，不会写回仓库）---------------------------
PHYSICS_DT = 1.0 / 120.0
CONTROL_DT = 1.0 / 20.0
ROBOT_RADIUS = 0.18
ROBOT_MASS = 4.0
OBJECT_MASS = 3.0
MAX_FORCE = 15.0
LINEAR_DAMPING = 8.0
ANGULAR_DAMPING = 0.6
FRICTION = 0.6
ELASTICITY = 0.05

# --- 传感（冻结；白名单：只给局部片段与自身状态）-----------------------------
SCAN_RANGE = 0.6
SCAN_RAYS = 72
MIN_POINTS_FOR_NORMAL = 3
GAP_TOLERANCE = 0.03        # 相邻射线的距离连续性容差
NORMAL_TAU = 0.10           # 法向一阶低通时间常数
NORMAL_JUMP_DEG = 30.0      # 两帧法向夹角超过它判定换面/角点

# --- 技能 -------------------------------------------------------------------
SPEED = 0.16                # 命令速度档：0.08 / 0.16 / 0.26
ACTIVE_DURATION = 2.0       # 有效接触推动目标时长
DELTA = 0.0                 # 位置律的推进偏置；0 表示按 DELTA_FROM_SPEED 自动取值
DELTA_FROM_SPEED = True     # True: delta = KV * SPEED / KP

# --- 速度式控制律参数（V）---------------------------------------------------
KV = 25.0                   # N s/m
A_REF_MAX = 3.0             # m/s^2，参考速度的加速度上限

# --- 位置式控制律参数（X）---------------------------------------------------
KP = 100.0                  # N/m
KD = 10.0                   # N s/m
E_MAX = 0.10                # m，位置误差上限
BETA_RAMP = 0.20            # s，虚拟推进量 0->1 爬升时间

# --- 阶段机 -----------------------------------------------------------------
APPROACH_TIMEOUT = 5.0
APPROACH_MIN_SPEED = 0.02
APPROACH_MAX_SPEED = 0.30
APPROACH_GAIN = 4.0
APPROACH_CONTACT_SPEED = 0.06
READY_HOLD = 0.10           # s 连续接触
READY_PRECONTACT_SPEED = 0.02
BOTH_READY_HOLD = 0.10      # s 双方同时 READY
CONTACT_LOSS_TOLERANCE = 0.15
PATCH_LOSS_TOLERANCE = 0.15
PUSH_WALL_LIMIT = 3.0
BRAKE_SETTLE_LIMIT = 4.0

# --- 判稳 -------------------------------------------------------------------
SETTLE_LINEAR = 0.03
SETTLE_ANGULAR = 0.08
SETTLE_SURFACE = 0.02
SETTLE_HOLD = 0.30
SURFACE_MATCH_TOLERANCE = 0.05
SURFACE_MATCH_MIN_POINTS = 3
SURFACE_FRESHNESS = 0.20

# --- 渲染 -------------------------------------------------------------------
REALTIME = True             # True: 实时窗口；--record 时自动关闭
SPEEDUP = 1.0               # 实时倍率（2.0 = 两倍速播放）
VIEW_EVERY = 1              # 每几个控制周期重画一次
TRAIL_MAX = 400             # 轨迹拖尾点数
HOLD_AT_END = 1.0           # 结束后停留秒数（不计入仿真时间）
PROGRESS = True             # 终端里每周期打印一行实时状态（任何环境都能用）

# =============================================================================
# 以下为实现，一般不需要改
# =============================================================================

import argparse
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                                   # noqa: E402

# The live window needs a GUI backend, but that has to be chosen *before* pyplot is imported.
# Recording or headless runs use Agg so they work without a display.
if "--record" in sys.argv or "--no-render" in sys.argv:
    matplotlib.use("Agg")
matplotlib.rcParams["font.size"] = 8
import matplotlib.patches as patches                                # noqa: E402
import matplotlib.pyplot as plt                                     # noqa: E402

from coimagine.continuous2d.config import Continuous2DConfig, PhysicsConfig  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv      # noqa: E402
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec      # noqa: E402
from coimagine.local_contact_option.sensing import (                # noqa: E402
    best_target_patch,
    scan_surface,
)

Vec = Tuple[float, float]


# ---------------------------------------------------------------------------
# 环境构建
# ---------------------------------------------------------------------------


def build_scene() -> SceneSpec:
    return SceneSpec(
        name="tune", category="tune", variant=0,
        arena=ARENA,
        obstacles=tuple(Box(*entry) for entry in OBSTACLES),
        goal=Box(*GOAL),
        object_pose=Pose(*OBJECT_POSE),
        robot_poses=tuple(Pose(*pose) for pose in ROBOT_POSES),
        description="manually tuned scene",
        object_width=OBJECT_SIZE[0], object_height=OBJECT_SIZE[1],
        robot_radius=ROBOT_RADIUS,
    )


def build_env() -> Continuous2DEnv:
    physics = PhysicsConfig(
        physics_dt=PHYSICS_DT, control_dt=CONTROL_DT, robot_radius=ROBOT_RADIUS,
        robot_mass=ROBOT_MASS, object_width=OBJECT_SIZE[0], object_height=OBJECT_SIZE[1],
        object_mass=OBJECT_MASS, max_force=MAX_FORCE, linear_damping=LINEAR_DAMPING,
        angular_damping=ANGULAR_DAMPING, friction=FRICTION, elasticity=ELASTICITY,
    )
    env = Continuous2DEnv(Continuous2DConfig(physics=physics))
    env.reset(build_scene(), seed=SEED)
    return env


# ---------------------------------------------------------------------------
# 感知：只给局部片段与自身状态
# ---------------------------------------------------------------------------


def sense(env: Continuous2DEnv, robot_id: int) -> Dict[str, object]:
    observation = env.observation()
    robots = tuple(state.as_tuple() for state in observation.robots)
    patches = scan_surface(
        env.scene, observation.object_pose(), observation.object_size, robots, robot_id,
        scan_range=SCAN_RANGE, scan_rays=SCAN_RAYS, timestamp=observation.time,
        min_points=MIN_POINTS_FOR_NORMAL, gap_tolerance=GAP_TOLERANCE,
    )
    patch = best_target_patch(patches)
    own = observation.robots[robot_id]
    contacting, point_robot, impulse = False, None, 0.0
    for entry in env.backend.contacts():
        if entry["pair"] != "robot_object" or int(entry["robot_id"]) != robot_id:
            continue
        contacting = True
        point = (float(entry["point"][0]), float(entry["point"][1]))
        cos, sin = math.cos(-own.angle), math.sin(-own.angle)
        dx, dy = point[0] - own.x, point[1] - own.y
        point_robot = (dx * cos - dy * sin, dx * sin + dy * cos)
        impulse = max(impulse, max(0.0, -float(entry["distance"])))
    return {
        "own": (own.x, own.y, own.angle, own.vx, own.vy, own.omega),
        "patch": patch, "contacting": contacting,
        "contact_point_robot": point_robot, "impulse": impulse,
        "time": observation.time,
    }


def unit(vector: Vec) -> Vec:
    length = math.hypot(*vector)
    return (0.0, 0.0) if length < 1e-12 else (vector[0] / length, vector[1] / length)


def clip_norm(vector: Vec, limit: float) -> Vec:
    length = math.hypot(*vector)
    if length > limit > 0.0:
        return (vector[0] * limit / length, vector[1] * limit / length)
    return vector


def to_world(point: Vec, pose: Sequence[float]) -> Vec:
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + point[0] * cos - point[1] * sin,
            pose[1] + point[0] * sin + point[1] * cos)


def patch_geometry(patch, own: Sequence[float]) -> Dict[str, object]:
    """观测片段的几何：质心 c、法向 n、切向 t、垂足 q、接触中心 p_contact、gap_signed。"""

    if patch is None:
        return {}
    points = [to_world(point, own) for point in patch.points_robot]
    c = (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    n = unit(patch.normal_world)
    t = (-n[1], n[0])
    d = (c[0] - own[0]) * n[0] + (c[1] - own[1]) * n[1]
    q = (own[0] + d * n[0], own[1] + d * n[1])
    p_contact = (q[0] - ROBOT_RADIUS * n[0], q[1] - ROBOT_RADIUS * n[1])
    # 垂足是否还在观测线段上
    low = to_world(patch.segment_robot[0], own)
    high = to_world(patch.segment_robot[1], own)
    span = math.hypot(high[0] - low[0], high[1] - low[1])
    supported = True
    if span > 1e-9:
        u = ((high[0] - low[0]) / span, (high[1] - low[1]) / span)
        proj = (q[0] - low[0]) * u[0] + (q[1] - low[1]) * u[1]
        supported = -0.01 <= proj <= span + 0.01
    return {"c": c, "n": n, "t": t, "q": q, "p_contact": p_contact,
            "d": d, "gap_signed": d - ROBOT_RADIUS, "supported": supported,
            "segment_world": (low, high)}


# ---------------------------------------------------------------------------
# 控制律（本文件内实现，可直接改）
# ---------------------------------------------------------------------------


class NormalFilter:
    def __init__(self) -> None:
        self.value: Optional[Vec] = None
        self.jump = False

    def update(self, normal: Optional[Vec], dt: float) -> Optional[Vec]:
        if normal is None:
            self.jump = False
            return self.value
        normal = unit(normal)
        if self.value is None:
            self.value, self.jump = normal, False
            return self.value
        dot = max(-1.0, min(1.0, self.value[0] * normal[0] + self.value[1] * normal[1]))
        self.jump = math.degrees(math.acos(dot)) > NORMAL_JUMP_DEG
        alpha = 1.0 - math.exp(-max(dt, 0.0) / max(NORMAL_TAU, 1e-9))
        blended = (self.value[0] + alpha * (normal[0] - self.value[0]),
                   self.value[1] + alpha * (normal[1] - self.value[1]))
        self.value = unit(blended)
        return self.value


class ReferenceRamp:
    """参考速度的加速度限幅，第一周期从零开始。"""

    def __init__(self, a_max: float) -> None:
        self.a_max = a_max
        self.value: Vec = (0.0, 0.0)

    def reset(self, value: Vec = (0.0, 0.0)) -> None:
        self.value = value

    def update(self, raw: Vec, dt: float) -> Vec:
        step = (raw[0] - self.value[0], raw[1] - self.value[1])
        bounded = clip_norm(step, self.a_max * max(dt, 1e-9))
        self.value = (self.value[0] + bounded[0], self.value[1] + bounded[1])
        return self.value


def velocity_push_force(geometry: Dict[str, object], own: Sequence[float], ramp: ReferenceRamp,
                        dt: float, alpha: float) -> Tuple[Vec, Dict[str, object]]:
    """V：参考速度 + 有界速度反馈。"""

    n = geometry.get("n")
    if not n:
        return (0.0, 0.0), {"reason": "no_surface"}
    t = geometry["t"]
    raw = (SPEED * (math.cos(alpha) * n[0] + math.sin(alpha) * t[0]),
           SPEED * (math.cos(alpha) * n[1] + math.sin(alpha) * t[1]))
    reference = ramp.update(raw, dt)
    force = clip_norm((KV * (reference[0] - own[3]), KV * (reference[1] - own[4])), MAX_FORCE)
    return force, {"reference": reference, "p_ref": None, "p_contact": None, "error": None,
                   "beta": None, "force_raw": force, "gap_signed": geometry.get("gap_signed")}


def position_push_force(geometry: Dict[str, object], own: Sequence[float], state: Dict[str, object],
                        dt: float, alpha: float, delta: float) -> Tuple[Vec, Dict[str, object]]:
    """X：接触几何位置参考 + 速度阻尼。

        p_ref = p_contact + beta * delta * direction
        e     = p_ref - p ;  F = clip(Kp * clip_norm(e, e_max) - Kd * v, F_max)
    """

    n = geometry.get("n")
    if not n:
        return (0.0, 0.0), {"reason": "no_surface"}
    t = geometry["t"]
    state["beta_time"] = state.get("beta_time", 0.0) + dt
    beta = min(1.0, state["beta_time"] / BETA_RAMP) if BETA_RAMP > 0 else 1.0
    direction = (math.cos(alpha) * n[0] + math.sin(alpha) * t[0],
                 math.cos(alpha) * n[1] + math.sin(alpha) * t[1])
    p_contact = geometry["p_contact"]
    p_ref = (p_contact[0] + beta * delta * direction[0],
             p_contact[1] + beta * delta * direction[1])
    error = (p_ref[0] - own[0], p_ref[1] - own[1])
    bounded = clip_norm(error, E_MAX)
    raw = (KP * bounded[0] - KD * own[3], KP * bounded[1] - KD * own[4])
    force = clip_norm(raw, MAX_FORCE)
    return force, {"reference": None, "p_ref": p_ref, "p_contact": p_contact, "error": error,
                   "beta": beta, "force_raw": raw, "gap_signed": geometry.get("gap_signed")}


# ---------------------------------------------------------------------------
# 表面速度估计（判稳用）
# ---------------------------------------------------------------------------


class SurfaceSpeed:
    def __init__(self) -> None:
        self.previous: Optional[Tuple[List[Vec], Sequence[float], float]] = None

    def update(self, patch, own: Sequence[float], now: float
               ) -> Tuple[Optional[float], Optional[str]]:
        if patch is None:
            return None, "no_patch"
        if now - patch.timestamp > SURFACE_FRESHNESS:
            return None, "stale_frame"
        current = [to_world(point, own) for point in patch.points_robot]
        if self.previous is None:
            self.previous = (current, own, patch.timestamp)
            return None, "no_previous_scan"
        points, pose, stamp = self.previous
        dt = patch.timestamp - stamp
        if dt <= 0.0:
            return None, "repeated_or_out_of_order"
        n = unit(patch.normal_world)
        speeds: List[float] = []
        for point in current:
            best, best_gap = None, SURFACE_MATCH_TOLERANCE
            for candidate in points:
                gap = math.hypot(point[0] - candidate[0], point[1] - candidate[1])
                if gap < best_gap:
                    best, best_gap = candidate, gap
            if best is None:
                continue
            delta = (point[0] - best[0], point[1] - best[1])
            speeds.append((delta[0] * n[0] + delta[1] * n[1]) / dt)
        if len(speeds) < SURFACE_MATCH_MIN_POINTS:
            return None, "insufficient_matches"
        self.previous = (current, own, patch.timestamp)
        return math.sqrt(sum(value * value for value in speeds) / len(speeds)), None


# ---------------------------------------------------------------------------
# 阶段机（每台机器人独立 READY；两臂共用）
# ---------------------------------------------------------------------------


class Robot:
    def __init__(self, robot_id: int, law: str, alpha: float, delta: float) -> None:
        self.robot_id = robot_id
        self.law = law
        self.alpha = alpha
        self.delta = delta
        self.stage = "approach"
        self.hold = 0.0
        self.loss = 0.0
        self.contact = False
        self.push_time = 0.0
        self.elapsed = 0.0
        self.brake_time = 0.0
        self.settle_hold = 0.0
        self.filter = NormalFilter()
        self.ramp = ReferenceRamp(A_REF_MAX)
        self.surface = SurfaceSpeed()
        self.position_state: Dict[str, object] = {}
        self.beta_time = 0.0
        self.exit: Optional[str] = None
        self.support_loss = 0.0
        self.brake_initialised = False


def run(law: str, *, render: bool, record: Optional[Path], speedup: float,
        args_progress_off: bool = False) -> Dict[str, object]:
    env = build_env()
    alpha = [math.radians(ALPHA_DEG[robot_id]) for robot_id in (0, 1)]
    delta = [0.0, 0.0]
    for robot_id in ACTIVE_ROBOTS:
        delta[robot_id] = DELTA if not DELTA_FROM_SPEED else KV * SPEED / KP
    robots = {robot_id: Robot(robot_id, law, alpha[robot_id], delta[robot_id])
              for robot_id in (0, 1)}
    both_ready = 0.0
    joint_stage = "approach"
    exit_reason = "wall_clock_end"
    trace: Dict[str, List[float]] = {key: [] for key in
                                     ("t", "force", "speed", "object_speed", "gap", "error")}
    trails = {"object": [], "robot0": [], "robot1": []}

    viewer = Viewer(render, record, speedup) if render or record else None

    for step in range(int(20.0 / CONTROL_DT)):
        now = step * CONTROL_DT
        senses = {robot_id: sense(env, robot_id) for robot_id in (0, 1)}
        geometry = {}
        for robot_id in (0, 1):
            own = senses[robot_id]["own"]
            patch = senses[robot_id]["patch"]
            normal = robots[robot_id].filter.update(
                unit(patch.normal_world) if patch else None, CONTROL_DT)
            geometry[robot_id] = patch_geometry(patch, own)
            if patch is not None and normal is not None:
                geometry[robot_id]["n"] = normal
                geometry[robot_id]["t"] = (-normal[1], normal[0])
            robot = robots[robot_id]
            robot.contact = bool(senses[robot_id]["contacting"])
            if robot.contact:
                robot.hold += CONTROL_DT
                robot.loss = 0.0
            else:
                robot.hold = 0.0
                robot.loss += CONTROL_DT
            robot.elapsed += CONTROL_DT

        # ---- 阶段机 ----
        if joint_stage in ("approach", "ready"):
            for robot_id in ACTIVE_ROBOTS:
                robot = robots[robot_id]
                if robot.stage == "approach" and robot.hold >= READY_HOLD:
                    robot.stage = "ready"
                elif robot.stage == "ready" and not robot.contact:
                    robot.stage = "approach"
                if (robot.elapsed > APPROACH_TIMEOUT
                        and not any(robots[other].stage == "push"
                                    for other in ACTIVE_ROBOTS)):
                    robot.exit = "approach_timeout"
            if any(robots[robot_id].exit for robot_id in ACTIVE_ROBOTS):
                exit_reason = next(robots[robot_id].exit for robot_id in ACTIVE_ROBOTS
                                   if robots[robot_id].exit)
                for robot_id in ACTIVE_ROBOTS:
                    robots[robot_id].stage = "brake"
                    _enter_brake(robots[robot_id], senses[robot_id]["own"])
                joint_stage = "brake"
            else:
                ready = all(robots[robot_id].stage == "ready" and robots[robot_id].hold >= READY_HOLD
                            for robot_id in ACTIVE_ROBOTS)
                both_ready = both_ready + CONTROL_DT if ready else 0.0
                if both_ready >= BOTH_READY_HOLD:
                    joint_stage = "push"
                    for robot_id in ACTIVE_ROBOTS:
                        robot = robots[robot_id]
                        robot.stage = "push"
                        robot.beta_time = 0.0
        elif joint_stage == "push":
            aborted = None
            for robot_id in ACTIVE_ROBOTS:
                robot = robots[robot_id]
                if robot.filter.jump:
                    aborted = (robot_id, "face_switch")
                    break
                if robot.law == "position" and not geometry[robot_id].get("supported", True):
                    robot.support_loss += CONTROL_DT
                else:
                    robot.support_loss = 0.0
                if robot.support_loss > PATCH_LOSS_TOLERANCE:
                    aborted = (robot_id, "patch_support_lost")
                    break
                if geometry[robot_id].get("n") is None and robot.loss > PATCH_LOSS_TOLERANCE:
                    aborted = (robot_id, "patch_lost")
                    break
                if robot.loss > CONTACT_LOSS_TOLERANCE:
                    aborted = (robot_id, "contact_lost")
                    break
                robot.push_time += CONTROL_DT
                if robot.push_time > PUSH_WALL_LIMIT:
                    aborted = (robot_id, "push_wall_timeout")
                    break
            if aborted:
                exit_reason = aborted[1]
                for robot_id in ACTIVE_ROBOTS:
                    robots[robot_id].stage = "brake"
                    _enter_brake(robots[robot_id], senses[robot_id]["own"])
                joint_stage = "brake"
            elif all(robots[robot_id].push_time >= ACTIVE_DURATION for robot_id in ACTIVE_ROBOTS):
                for robot_id in ACTIVE_ROBOTS:
                    robots[robot_id].stage = "settle_pending"
                joint_stage = "brake"
                for robot_id in ACTIVE_ROBOTS:
                    _enter_brake(robots[robot_id], senses[robot_id]["own"])
                    robots[robot_id].stage = "brake"
        elif joint_stage == "brake":
            finished = True
            for robot_id in ACTIVE_ROBOTS:
                robot = robots[robot_id]
                robot.brake_time += CONTROL_DT
                reference = robot.ramp.update((0.0, 0.0), CONTROL_DT)
                if math.hypot(*reference) > 1e-9:
                    finished = False
            if finished:
                joint_stage = "settle"
                for robot_id in ACTIVE_ROBOTS:
                    robots[robot_id].stage = "settle"
                    robots[robot_id].settle_hold = 0.0
            if any(robots[robot_id].brake_time > BRAKE_SETTLE_LIMIT for robot_id in ACTIVE_ROBOTS):
                exit_reason = "settle_timeout"
                break
        elif joint_stage == "settle":
            unobserved = False
            for robot_id in ACTIVE_ROBOTS:
                robot = robots[robot_id]
                robot.brake_time += CONTROL_DT
                own = senses[robot_id]["own"]
                rms, reason = robot.surface.update(senses[robot_id]["patch"], own, now)
                if rms is None:
                    unobserved = True
                    robot.settle_hold = 0.0
                    continue
                if (math.hypot(own[3], own[4]) < SETTLE_LINEAR
                        and abs(own[5]) < SETTLE_ANGULAR and rms < SETTLE_SURFACE):
                    robot.settle_hold += CONTROL_DT
                else:
                    robot.settle_hold = 0.0
            if all(robots[robot_id].settle_hold >= SETTLE_HOLD for robot_id in ACTIVE_ROBOTS):
                exit_reason = "settled_observed"
                break
            if any(robots[robot_id].brake_time > BRAKE_SETTLE_LIMIT for robot_id in ACTIVE_ROBOTS):
                exit_reason = "settle_unobserved" if unobserved else "settle_timeout"
                break

        # ---- 力 ----
        forces: List[Vec] = [(0.0, 0.0), (0.0, 0.0)]
        diagnostics: Dict[int, Dict[str, object]] = {}
        for robot_id in (0, 1):
            robot = robots[robot_id]
            own = senses[robot_id]["own"]
            if robot_id not in ACTIVE_ROBOTS:
                forces[robot_id] = clip_norm((KV * -own[3], KV * -own[4]), MAX_FORCE)
                diagnostics[robot_id] = {}
                continue
            if robot.stage in ("brake", "settle"):
                force = clip_norm((KV * robot.ramp.value[0] - KV * own[3],
                                   KV * robot.ramp.value[1] - KV * own[4]), MAX_FORCE)
                diagnostics[robot_id] = {}
            elif robot.stage == "push":
                if robot.law == "position":
                    force, info = position_push_force(geometry[robot_id], own,
                                                      robot.position_state, CONTROL_DT,
                                                      robot.alpha, robot.delta)
                else:
                    force, info = velocity_push_force(geometry[robot_id], own, robot.ramp,
                                                      CONTROL_DT, robot.alpha)
                diagnostics[robot_id] = info
            elif robot.stage == "ready":
                n = geometry[robot_id].get("n") or (0.0, 0.0)
                raw = (n[0] * READY_PRECONTACT_SPEED, n[1] * READY_PRECONTACT_SPEED)
                reference = robot.ramp.update(raw, CONTROL_DT)
                force = clip_norm((KV * (reference[0] - own[3]), KV * (reference[1] - own[4])),
                                  MAX_FORCE)
                diagnostics[robot_id] = {"reference": reference}
            else:  # approach
                n = geometry[robot_id].get("n")
                if n is None:
                    reference = robot.ramp.update((0.0, 0.0), CONTROL_DT)
                else:
                    gap = max(0.0, (geometry[robot_id].get("d") or 0.0) - ROBOT_RADIUS)
                    magnitude = (APPROACH_CONTACT_SPEED if gap <= 0.005
                                 else min(APPROACH_MAX_SPEED,
                                          max(APPROACH_MIN_SPEED, APPROACH_GAIN * gap)))
                    reference = robot.ramp.update((n[0] * magnitude, n[1] * magnitude), CONTROL_DT)
                force = clip_norm((KV * (reference[0] - own[3]), KV * (reference[1] - own[4])),
                                  MAX_FORCE)
                diagnostics[robot_id] = {"reference": reference}
            forces[robot_id] = force

        env.step_control((forces[0], forces[1]))
        object_state = env.backend.object_state()
        own0 = senses[0]["own"]
        info0 = diagnostics.get(0) or {}
        trace["t"].append(now)
        trace["force"].append(math.hypot(*forces[0]))
        trace["speed"].append(math.hypot(own0[3], own0[4]))
        trace["object_speed"].append(math.hypot(object_state.vx, object_state.vy))
        trace["gap"].append(info0.get("gap_signed"))
        error = info0.get("error")
        trace["error"].append(math.hypot(*error) if error else None)
        trails["object"].append((object_state.x, object_state.y))
        for robot_id in (0, 1):
            state = env.backend.robot_states()[robot_id]
            trails[f"robot{robot_id}"].append((state.x, state.y))
        for key, values in trails.items():
            if len(values) > TRAIL_MAX:
                del values[0]

        if PROGRESS and not args_progress_off:
            print(f"\r{law:8s} t={now:5.2f}s {joint_stage:8s} "
                  f"push={robots[0].push_time:4.2f}s gap={_fmt(info0.get('gap_signed'), 4):>7s} "
                  f"|e|={_fmt(_hypot(info0.get('error')), 4):>7s} "
                  f"|F|={math.hypot(*forces[0]):5.2f}N "
                  f"|v|={math.hypot(own0[3], own0[4]):5.3f} "
                  f"|v_obj|={math.hypot(object_state.vx, object_state.vy):5.3f}",
                  end="", flush=True)
        if viewer is not None and step % VIEW_EVERY == 0:
            viewer.draw(env, robots, geometry, diagnostics, forces, joint_stage, exit_reason,
                        trace, trails, now)
        if env.termination is not None:
            exit_reason = f"environment:{env.termination}"
            break

    final = env.backend.object_state()
    start = OBJECT_POSE
    result = {
        "law": law, "exit_reason": exit_reason,
        "steps": len(trace["t"]),
        "sim_seconds": len(trace["t"]) * CONTROL_DT,
        "object_displacement": math.hypot(final.x - start[0], final.y - start[1]),
        "object_rotation": abs(final.angle - start[2]),
        "terminal_object_speed": math.hypot(final.vx, final.vy),
        "terminal_object_omega": abs(final.omega),
        "max_penetration": env.backend.max_penetration(),
        "push_seconds": {robot_id: robots[robot_id].push_time for robot_id in ACTIVE_ROBOTS},
        "peak_force": max(trace["force"]) if trace["force"] else 0.0,
    }
    if viewer is not None:
        viewer.finish(exit_reason, result)
    return result


def _enter_brake(robot: Robot, own: Sequence[float]) -> None:
    if robot.brake_initialised:
        return
    robot.ramp.reset((own[3], own[4]))
    robot.brake_initialised = True
    robot.position_state = {}


# ---------------------------------------------------------------------------
# 实时渲染
# ---------------------------------------------------------------------------


class Viewer:
    def __init__(self, live: bool, record: Optional[Path], speedup: float) -> None:
        self.live = live and record is None
        self.speedup = max(0.05, speedup)
        self.widget = None
        if self.live:
            # Do NOT use plt.pause() here: it enters the Tk mainloop and waits for a timer
            # callback, which on a remote X server without a window manager can simply never
            # fire. canvas.draw() + widget.update() pumps the pending events without blocking.
            try:
                plt.ion()
                self.widget = self.figure.canvas.get_tk_widget()
            except Exception:
                self.widget = None
        self.figure, (self.axis, self.plot_axis) = plt.subplots(
            1, 2, figsize=(13.0, 5.6), dpi=100, gridspec_kw={"width_ratios": [1.35, 1.0]})
        self.writer = None
        if record is not None:
            import imageio_ffmpeg

            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
            from matplotlib.animation import FFMpegWriter

            self.writer = FFMpegWriter(fps=int(round(1.0 / CONTROL_DT)),
                                       extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p",
                                                   "-crf", "26"])
            self.writer.setup(self.figure, str(record), dpi=100)
        self.budget = CONTROL_DT * VIEW_EVERY / self.speedup
        self.last = time.perf_counter()

    def draw(self, env, robots, geometry, diagnostics, forces, stage, exit_reason,
             trace, trails, now) -> None:
        axis = self.axis
        axis.clear()
        width, height = env.scene.arena
        axis.add_patch(patches.Rectangle((0, 0), width, height, fill=False, lw=1.0, ec="black"))
        for box in env.scene.obstacles:
            axis.add_patch(patches.Rectangle(
                (box.cx - box.half_w, box.cy - box.half_h), 2 * box.half_w, 2 * box.half_h,
                angle=math.degrees(box.angle), facecolor="0.65", edgecolor="black", lw=0.6))
        goal = env.scene.goal
        axis.add_patch(patches.Rectangle((goal.cx - goal.half_w, goal.cy - goal.half_h),
                                         2 * goal.half_w, 2 * goal.half_h, fill=False,
                                         ls=":", ec="tab:green", lw=0.9))
        object_state = env.backend.object_state()
        axis.add_patch(patches.Polygon(_corners(object_state), closed=True,
                                       facecolor="#e8871a", alpha=0.85, edgecolor="black", lw=0.7))
        axis.plot([p[0] for p in trails["object"]], [p[1] for p in trails["object"]],
                  color="#e8871a", lw=0.9, alpha=0.7)
        states = env.backend.robot_states()
        for robot_id in (0, 1):
            colour = "#1668c1" if robot_id == 0 else "#17a589"
            axis.plot([p[0] for p in trails[f"robot{robot_id}"]],
                      [p[1] for p in trails[f"robot{robot_id}"]], color=colour, lw=0.8, alpha=0.6)
            axis.add_patch(patches.Circle((states[robot_id].x, states[robot_id].y), ROBOT_RADIUS,
                                          facecolor=colour, alpha=0.85, zorder=5))
            axis.annotate(f"r{robot_id}", (states[robot_id].x, states[robot_id].y), fontsize=6,
                          ha="center", va="center", color="white", zorder=6)
            item = geometry.get(robot_id) or {}
            n = item.get("n")
            if n:
                axis.arrow(states[robot_id].x, states[robot_id].y, n[0] * 0.2, n[1] * 0.2,
                           head_width=0.025, color="crimson", zorder=7, length_includes_head=True)
            q, p_contact = item.get("q"), item.get("p_contact")
            if q:
                axis.plot([q[0]], [q[1]], marker="s", ms=6, color="tab:red", zorder=10)
            if p_contact:
                axis.plot([p_contact[0]], [p_contact[1]], marker="o", ms=6.5, mfc="none",
                          mec="tab:red", mew=1.2, zorder=10)
            info = diagnostics.get(robot_id) or {}
            p_ref, error = info.get("p_ref"), info.get("error")
            if p_ref:
                axis.plot([p_ref[0]], [p_ref[1]], marker="o", ms=9, mfc="none", mec="tab:green",
                          mew=1.4, zorder=10)
                axis.plot([states[robot_id].x, p_ref[0]], [states[robot_id].y, p_ref[1]],
                          color="tab:blue", lw=1.1, alpha=0.9, zorder=8)
            reference = info.get("reference")
            if reference and math.hypot(*reference) > 1e-6:
                axis.arrow(states[robot_id].x, states[robot_id].y, reference[0], reference[1],
                           head_width=0.025, color="darkgreen", zorder=7,
                           length_includes_head=True)
            force = forces[robot_id]
            if math.hypot(*force) > 1e-9:
                axis.arrow(states[robot_id].x, states[robot_id].y,
                           force[0] / MAX_FORCE * 0.30, force[1] / MAX_FORCE * 0.30,
                           head_width=0.03, color="black", lw=1.0, zorder=8,
                           length_includes_head=True)
        axis.set_xlim(-0.06, width + 0.06)
        axis.set_ylim(-0.06, height + 0.06)
        axis.set_aspect("equal")
        axis.set_xticks([])
        axis.set_yticks([])
        info = diagnostics.get(0) or {}
        contact = {robot_id: robots[robot_id].contact for robot_id in (0, 1)}
        header = [f"law={LAW}  t={now:.2f}s  stage={stage}  exit={exit_reason}",
                  f"contact r0={int(contact[0])} r1={int(contact[1])}  "
                  f"push={robots[0].push_time:.2f}s",
                  f"gap={_fmt(info.get('gap_signed'), 4)}  |e|={_fmt(_hypot(info.get('error')), 4)}  "
                  f"beta={_fmt(info.get('beta'), 2)}",
                  f"|F|={math.hypot(*forces[0]):.2f} N  |v_obj|={math.hypot(object_state.vx, object_state.vy):.4f}"]
        axis.text(0.012, 0.985, "\n".join(header), transform=axis.transAxes, va="top", ha="left",
                  fontsize=8.5, family="DejaVu Sans Mono",
                  bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1.6))
        self._plot(trace, now)
        self.figure.tight_layout()
        if self.writer is not None:
            self.writer.grab_frame()
        elif self.live:
            started = time.perf_counter()
            self.figure.canvas.draw()
            if self.widget is not None:
                self.widget.update()
            else:
                try:
                    self.figure.canvas.flush_events()
                except Exception:
                    pass
            time.sleep(max(0.0, self.budget - (time.perf_counter() - started)))

    def _plot(self, trace, now) -> None:
        axis = self.plot_axis
        axis.clear()
        axis.plot(trace["t"], trace["force"], color="black", lw=1.1, label="|F| N")
        axis.plot(trace["t"], trace["speed"], color="tab:blue", lw=1.0, label="|v_robot| m/s")
        axis.plot(trace["t"], trace["object_speed"], color="#e8871a", lw=1.0,
                  label="|v_object| m/s")
        axis.plot(trace["t"], trace["gap"], color="tab:red", lw=0.9, label="gap_signed m")
        axis.plot(trace["t"], trace["error"], color="tab:green", lw=0.9, ls=":", label="|e| m")
        axis.axhline(MAX_FORCE, color="0.6", lw=0.7, ls=":")
        axis.axhline(0.0, color="0.6", lw=0.6)
        axis.legend(fontsize=7, loc="best")
        axis.set_xlabel("simulation time [s]")
        axis.set_title("live signals", fontsize=9)
        axis.set_xlim(0, max(2.0, now + 0.5))

    def finish(self, exit_reason: str, result: Dict[str, object]) -> None:
        if self.writer is not None:
            self.writer.finish()
            plt.close(self.figure)
            return
        if self.live:
            self.axis.set_title(f"exit={exit_reason}   {_short(result)}")
            deadline = time.perf_counter() + min(5.0, HOLD_AT_END)
            while time.perf_counter() < deadline:
                self.figure.canvas.draw()
                if self.widget is not None:
                    self.widget.update()
                time.sleep(0.02)
            plt.close(self.figure)


def _short(result: Dict[str, object]) -> str:
    return (f"disp={result['object_displacement']:.3f}m "
            f"|v|={result['terminal_object_speed']:.4f} "
            f"push={result['push_seconds'].get(0, 0.0):.2f}s")


def _corners(state) -> List[Vec]:
    cos, sin = math.cos(state.angle), math.sin(state.angle)
    return [(state.x + dx * cos - dy * sin, state.y + dx * sin + dy * cos)
            for dx, dy in ((-OBJECT_SIZE[0] / 2, -OBJECT_SIZE[1] / 2),
                           (OBJECT_SIZE[0] / 2, -OBJECT_SIZE[1] / 2),
                           (OBJECT_SIZE[0] / 2, OBJECT_SIZE[1] / 2),
                           (-OBJECT_SIZE[0] / 2, OBJECT_SIZE[1] / 2))]


def _hypot(value) -> Optional[float]:
    return None if not value else math.hypot(*value)


def _fmt(value, digits: int) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def gui_available(timeout: float = 10.0) -> bool:
    """Probe the GUI in a subprocess: a broken DISPLAY can make tkinter block forever."""

    if not sys.argv or "--record" in sys.argv or "--no-render" in sys.argv:
        return False
    code = "import tkinter; root = tkinter.Tk(); root.destroy()"
    try:
        done = subprocess.run([sys.executable, "-c", code], timeout=timeout,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return done.returncode == 0
    except Exception:
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--law", default=LAW, choices=("velocity", "position", "compare"))
    parser.add_argument("--record", default=None, help="write an mp4 instead of a live window")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--speedup", type=float, default=SPEEDUP)
    args = parser.parse_args()

    laws = ("velocity", "position") if args.law == "compare" else (args.law,)
    record = Path(args.record) if args.record else None
    render = REALTIME and not args.no_render
    if render and record is None and not gui_available():
        # No usable display: do not hang on a window nobody can see. Fall back to the same
        # frame-by-frame pipeline writing a video, and say so.
        record = ROOT / "tune_push.mp4"
        matplotlib.use("Agg")
        render = False
        print(f"!! 图形显示不可用，改为实时渲染到 {record}（同样的逐帧管线）。"
              f"纯终端观察可以直接看下面的每周期状态行。", flush=True)
    for index, law in enumerate(laws):
        path = record
        if record is not None and len(laws) > 1:
            path = record.with_name(f"{record.stem}_{law}{record.suffix}")
        result = run(law, render=render, record=path, speedup=args.speedup,
                     args_progress_off=args.no_progress)
        if PROGRESS and not args.no_progress:
            print()
        print(f"{law:9s} exit={result['exit_reason']:18s} "
              f"steps={result['steps']:3d} sim={result['sim_seconds']:.2f}s "
              f"disp={result['object_displacement']:.3f}m rot={result['object_rotation']:.3f}rad "
              f"|v|={result['terminal_object_speed']:.4f} "
              f"push={result['push_seconds']} peakF={result['peak_force']:.2f}N "
              f"pen={result['max_penetration']:.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
