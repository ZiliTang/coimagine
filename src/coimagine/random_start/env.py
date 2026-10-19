"""The random-start joint environment: one world, one clock, 225 joint skills.

Time convention (task book section 3.3) -- there is exactly one:

    O0 (initial observation) + O1 ... On (the observation after each real control period)

``n`` control periods produce ``n + 1`` frames. A skill boundary does not add a frame: the
last period of a joint skill *is* its terminal observation, and the next skill reads that same
frame as its start. Branching reuses the parent's last frame and does not insert a duplicate.

One call to :meth:`RandomStartEnv.joint_step` runs a synchronous joint skill: both robots
choose a skill, each control period computes both forces and advances **the same world once**
(``Continuous2DEnv.step_control``); a robot that finishes early is braked by the shared hold
controller while its teammate continues. The world is never frozen, teleported or reset.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import Continuous2DEnv
from . import geometry as G
from .lidar import (
    CATEGORY_NONE,
    CATEGORY_OBJECT,
    CATEGORY_STATIC,
    CATEGORY_TEAMMATE,
    LidarScan,
    SurfacePatch,
    best_patch,
    scan_lidar,
    scan_patches,
)
from .layout import InstanceSpec
from .skills import (
    HoldBrake,
    RobotKinematics,
    Skill,
    SkillObservation,
    make_skill,
)
from .spec import (
    option_timeout,
    CONTROL_DT,
    ENV_VERSION,
    FAILURE_PENALTY,
    GAMMA,
    JOINT_MAX_SECONDS,
    LIDAR_RANGE,
    MAX_FORCE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    PROGRESS_WEIGHT,
    DISTANCE_COST,
    ROBOT_ACTIONS,
    ROBOT_RADIUS,
    SUCCESS_ANGULAR,
    SUCCESS_BONUS,
    SUCCESS_DISTANCE,
    SUCCESS_HOLD_SECONDS,
    SUCCESS_LINEAR,
    TIME_BUDGET_SECONDS,
    TIME_COST,
    joint_action_index,
)

Vec = Tuple[float, float]

TERMINATION_SUCCESS = "success"
TERMINATION_BUDGET = "task_budget_exhausted"
TERMINATION_NUMERIC = "numeric_failure"
TERMINATION_PENETRATION = "penetration_limit"
TERMINATION_OUT_OF_BOUNDS = "out_of_bounds"

#: numeric anomaly classes are kept apart from ordinary task failure (section 5.1)
ANOMALY_TERMINATIONS = (TERMINATION_NUMERIC, TERMINATION_PENETRATION, TERMINATION_OUT_OF_BOUNDS)


@dataclass
class EnvConfig:
    time_budget: float = TIME_BUDGET_SECONDS
    gamma: float = GAMMA
    time_cost: float = TIME_COST
    success_bonus: float = SUCCESS_BONUS
    failure_penalty: float = FAILURE_PENALTY
    progress_weight: float = PROGRESS_WEIGHT
    distance_cost: float = DISTANCE_COST
    success_distance: float = SUCCESS_DISTANCE
    success_linear: float = SUCCESS_LINEAR
    success_angular: float = SUCCESS_ANGULAR
    success_hold: float = SUCCESS_HOLD_SECONDS
    joint_max_seconds: float = JOINT_MAX_SECONDS
    penetration_limit: float = 0.05
    # -- oriented prototype (default off: v3.1 behaviour is byte-identical) ----------------
    #: kinematic heading DOF + stable pushing side: the robot must turn its heading towards
    #: the push direction; a misaligned push is scaled by cos(error). This is a *prototype*
    #: actuation model, not a torque-coupled contact model.
    #: v4 has exactly one actuation model: an omni-directional chassis whose heading turns
    #: towards the heading its skill declares. There is no model-selection switch to mislead
    #: callers; these three values are its only parameters.
    max_omega: float = 1.5            # rad/s heading-rate limit
    align_gain: float = 3.0           # heading-rate proportional gain
    align_tolerance_deg: float = 20.0  # full force inside this heading error
    base: Continuous2DConfig = field(default_factory=Continuous2DConfig)

    def __post_init__(self):
        if self.gamma != 1.0:
            raise ValueError("time-budget-v3 uses undiscounted finite-horizon returns (gamma=1)")
        if (self.time_budget <= 0 or not math.isfinite(self.time_budget)
                or not math.isclose(self.time_budget / CONTROL_DT,
                                    round(self.time_budget / CONTROL_DT), abs_tol=1e-8)):
            raise ValueError("time_budget must be a positive integer number of control periods")
        if self.max_omega <= 0 or self.align_gain <= 0:
            raise ValueError("the v4 actuation model needs positive max_omega and align_gain")
        if not (0.0 < self.align_tolerance_deg < 90.0):
            raise ValueError("align_tolerance_deg must be in (0, 90)")

    def as_dict(self) -> Dict[str, object]:
        return {"env_version": ENV_VERSION, "time_budget": self.time_budget,
                "gamma": self.gamma, "time_cost": self.time_cost,
                "success_bonus": self.success_bonus,
                "failure_penalty": self.failure_penalty,
                 "progress_weight": self.progress_weight,
                 "distance_cost": self.distance_cost,
                "success_distance": self.success_distance,
                "success_linear": self.success_linear,
                "success_angular": self.success_angular,
                "success_hold": self.success_hold,
                "joint_max_seconds": self.joint_max_seconds,
                "penetration_limit": self.penetration_limit,
                "actuation_model": "omni_chassis_heading_target",
                "max_omega": self.max_omega,
                "align_gain": self.align_gain,
                "align_tolerance_deg": self.align_tolerance_deg,
                "physics_config_hash": self.base.config_hash()}


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _kinematics(state) -> RobotKinematics:
    return RobotKinematics(x=float(state.x), y=float(state.y), angle=float(state.angle),
                           vx=float(state.vx), vy=float(state.vy), omega=float(state.omega))


class RandomStartEnv:
    """One task instance, one living world, and a sequence of synchronous joint skills."""

    def __init__(self, instance: InstanceSpec, config: Optional[EnvConfig] = None,
                 skill_factory=None):
        self.instance = instance
        self.config = config or EnvConfig()
        # Explicit controller injection point. The default is the frozen baseline factory, so
        # every existing caller keeps the identical behaviour; a candidate controller is passed
        # here by name and is recorded in the branch log. No implicit global monkeypatch.
        self.skill_factory = skill_factory if skill_factory is not None else make_skill
        self.env = Continuous2DEnv(self.config.base)
        self.env.reset(instance.to_scene())
        self.obstacle_polygons = tuple(
            G.box_polygon(box.cx, box.cy, box.angle, box.half_w, box.half_h)
            for box in instance.obstacles)
        self.rows: List[Dict[str, object]] = []
        self.boundaries: List[Dict[str, object]] = []
        self.history: List[Dict[str, object]] = []
        self.total_periods = 0
        self.skills_used = 0
        self.world_advances = 0
        self.hold_seconds = 0.0
        self.success = False
        self.termination: Optional[str] = None
        self.discount_prefix = 1.0
        self.reward_total = 0.0
        self.reward_components: Dict[str, float] = {
            "progress": 0.0, "distance_cost": 0.0, "time_cost": 0.0,
            "success_bonus": 0.0, "failure_penalty": 0.0}
        self._episode_events: List[Dict[str, object]] = []
        # kinematic heading state of the oriented prototype; unused when oriented=False
        self.heading = [float(pose.angle) for pose in instance.robot_poses]
        self.heading_rate = [0.0, 0.0]
        # v4: there is exactly one actuation model. The robot turns towards the heading its
        # skill declares; the force is never gated, so braking, safety and avoidance stay fully
        # effective. The old diagnostic modes are gone.
        self.orient_mode = "heading_target"
        self.orient_rotate = True
        self.orient_gate = False
        self.orient_target = True

    # -- privileged truth, reward and evaluation only -----------------------
    def object_state(self):
        return self.env.backend.object_state()

    def robot_states(self):
        return self.env.backend.robot_states()

    def object_distance(self) -> float:
        state = self.object_state()
        return float(math.hypot(state.x - self.instance.goal_centre[0],
                                state.y - self.instance.goal_centre[1]))

    def success_conditions(self) -> Dict[str, object]:
        state = self.object_state()
        linear = float(math.hypot(state.vx, state.vy))
        distance = self.object_distance()
        return {
            "distance": distance, "linear": linear, "angular": abs(float(state.omega)),
            "distance_ok": bool(distance <= self.config.success_distance),
            "linear_ok": bool(linear <= self.config.success_linear),
            "angular_ok": bool(abs(float(state.omega)) <= self.config.success_angular),
        }

    # -- legal observation --------------------------------------------------
    def contacts(self) -> Dict[str, List[Tuple[str, int, float, float]]]:
        out: Dict[str, List[Tuple[str, int, float, float]]] = {"0": [], "1": []}
        for entry in self.env.backend.contacts():
            pair = str(entry["pair"])
            robot_id = entry["robot_id"]
            if robot_id is None or int(robot_id) < 0:
                continue  # object_wall and other non-robot pairs are not a robot's sensor
            key = pair.split("_")[-1] if pair.startswith("robot_") else pair
            out[str(int(robot_id))].append((key, int(robot_id),
                                            float(entry["point"][0]), float(entry["point"][1])))
        return out

    def _legal_robot_states(self) -> List[Tuple[float, ...]]:
        """Robot states as the controller may use them: the kinematic heading when oriented."""

        states = [list(state.as_tuple()) for state in self.robot_states()]
        if self.orient_rotate:
            for robot_id in (0, 1):
                states[robot_id][2] = float(self.heading[robot_id])
                states[robot_id][5] = float(self.heading_rate[robot_id])
        return [tuple(state) for state in states]

    def scan_for(self, robot_id: int, tick: int, time: float) -> LidarScan:
        states = self._legal_robot_states()
        return scan_lidar(self.instance.arena, self.obstacle_polygons,
                          (self.object_state().x, self.object_state().y, self.object_state().angle),
                          states, robot_id, tick, time)

    def observe(self, robot_id: int, tick: int, time: float,
                contacts: Optional[Dict[str, List[Tuple[str, int, float, float]]]] = None
                ) -> SkillObservation:
        """The whitelisted observation of one robot. Object truth is used only to cast rays."""

        states = self.robot_states()
        own = _kinematics(states[robot_id])
        mate = _kinematics(states[1 - robot_id])
        if self.orient_rotate:
            own = replace(own, angle=float(self.heading[robot_id]),
                          omega=float(self.heading_rate[robot_id]))
            mate = replace(mate, angle=float(self.heading[1 - robot_id]),
                           omega=float(self.heading_rate[1 - robot_id]))
        scan = self.scan_for(robot_id, tick, time)
        patches = tuple(scan_patches(scan, (own.x, own.y)))
        patch = None
        for candidate in patches:
            if candidate.valid and (patch is None or candidate.distance < patch.distance):
                patch = candidate
        contacts = self.contacts() if contacts is None else contacts
        touching_object = False
        touching_static = False
        touching_teammate = False
        contact_point: Optional[Vec] = None
        for category, _, px, py in contacts[str(robot_id)]:
            if category == "object":
                touching_object = True
            elif category == "wall":
                touching_static = True
            elif category == "robot":
                touching_teammate = True
            else:
                continue
            if category == "object" and contact_point is None:
                cos, sin = math.cos(-own.angle), math.sin(-own.angle)
                dx, dy = px - own.x, py - own.y
                contact_point = (dx * cos - dy * sin, dx * sin + dy * cos)
        static = scan.nearest(CATEGORY_STATIC)
        teammate = scan.nearest(CATEGORY_TEAMMATE)
        return SkillObservation(
            robot_id=robot_id, tick=tick, time=time, own=own, teammate=mate, scan=scan,
            patches=patches, patch=patch, contacting_object=touching_object,
            contact_point_robot=contact_point, contacting_static=touching_static,
            contacting_teammate=touching_teammate,
            goal_xy=self.instance.goal_centre, goal_radius=self.instance.goal_radius,
            arena=self.instance.arena, force_limit=MAX_FORCE,
            remaining_seconds=max(0.0, self.config.time_budget - time),
            elapsed_seconds=time,
            static_nearest=None if static is None else (static[0], static[1]),
            teammate_nearest=None if teammate is None else (teammate[0], teammate[1]))

    def boundary_observations(self) -> Tuple[SkillObservation, SkillObservation]:
        """The legal joint observation at the current instant, before any world advance."""

        contacts = self.contacts()
        now = self.total_periods * CONTROL_DT
        return (self.observe(0, self.total_periods, now, contacts),
                self.observe(1, self.total_periods, now, contacts))

    def boundary_vector(self) -> np.ndarray:
        first, second = self.boundary_observations()
        return self.observation_vector(first, second)

    # -- policy-facing vector ----------------------------------------------
    def observation_vector(self, observation_0: SkillObservation,
                           observation_1: SkillObservation) -> np.ndarray:
        """The joint legal feature vector of one control period (both robots' own views)."""

        values: List[float] = []
        for observation in (observation_0, observation_1):
            own = observation.own
            values += [own.x, own.y, math.cos(own.angle), math.sin(own.angle), own.vx, own.vy]
        own0, own1 = observation_0.own, observation_1.own
        values += [own1.x - own0.x, own1.y - own0.y,
                   math.cos(own1.angle - own0.angle), math.sin(own1.angle - own0.angle)]
        for observation in (observation_0, observation_1):
            patch = observation.patch
            values += [1.0 if (patch is not None and patch.valid) else 0.0,
                       (patch.distance / LIDAR_RANGE) if patch is not None else 0.0,
                       patch.normal_world[0] if patch is not None else 0.0,
                       patch.normal_world[1] if patch is not None else 0.0]
        values += [self.instance.goal_centre[0], self.instance.goal_centre[1],
                   self.instance.goal_radius]
        for observation in (observation_0, observation_1):
            values += [1.0 if observation.contacting_object else 0.0,
                       1.0 if observation.contacting_static else 0.0,
                       1.0 if observation.contacting_teammate else 0.0]
        elapsed = self.total_periods * CONTROL_DT
        values += [min(1.0, elapsed / self.config.time_budget),
                    0.0,  # reserved former skill-budget fraction; no action-count budget in v3
                   max(0.0, self.config.time_budget - elapsed) / self.config.time_budget]
        for observation in (observation_0, observation_1):
            scan = observation.scan
            for index in range(len(scan.valid)):
                if scan.valid[index]:
                    # v4: clip at the range so a hit at the boundary is continuous with no return
                    values.append(min(float(scan.distances[index]), LIDAR_RANGE) / LIDAR_RANGE)
                    values.append(1.0)
                    category = int(scan.category[index])
                    values += [1.0 if category == 1 else 0.0,
                               1.0 if category == 2 else 0.0,
                               1.0 if category == 3 else 0.0]
                else:
                    values += [1.0, 0.0, 0.0, 0.0, 0.0]
        return np.asarray(values, dtype=np.float32)

    def observation_fields(self) -> List[str]:
        fields: List[str] = []
        for robot_id in (0, 1):
            fields += [f"r{robot_id}_{name}" for name in
                       ("x", "y", "cos", "sin", "vx", "vy")]
        fields += ["teammate_dx", "teammate_dy", "teammate_cos", "teammate_sin"]
        for robot_id in (0, 1):
            fields += [f"r{robot_id}_patch_valid", f"r{robot_id}_patch_distance",
                       f"r{robot_id}_patch_normal_x", f"r{robot_id}_patch_normal_y"]
        fields += ["goal_x", "goal_y", "goal_radius"]
        for robot_id in (0, 1):
            fields += [f"r{robot_id}_contact_object", f"r{robot_id}_contact_static",
                       f"r{robot_id}_contact_teammate"]
        fields += ["elapsed_fraction", "reserved_zero", "remaining_fraction"]
        for robot_id in (0, 1):
            for index in range(72):
                fields += [f"r{robot_id}_lidar_distance_{index:02d}",
                           f"r{robot_id}_lidar_hit_{index:02d}",
                           f"r{robot_id}_lidar_object_{index:02d}",
                           f"r{robot_id}_lidar_static_{index:02d}",
                           f"r{robot_id}_lidar_teammate_{index:02d}"]
        return fields

    # -- one synchronous joint skill ---------------------------------------
    def joint_step(self, action_index_0: int, action_index_1: int,
                   record: bool = True) -> Dict[str, object]:
        if self.termination is not None:
            raise RuntimeError(
                f"episode already terminated with {self.termination!r}; a terminated episode "
                "may not be extended (it must not be used as a branchable parent)")
        action_0 = ROBOT_ACTIONS[action_index_0]
        action_1 = ROBOT_ACTIONS[action_index_1]
        # v4: the joint window is the option's own time budget (the longer of the two), capped
        window_seconds = min(self.config.joint_max_seconds,
                             max(option_timeout(action_0), option_timeout(action_1)))
        skills: Dict[int, Skill] = {0: self.skill_factory(action_0, MAX_FORCE),
                                    1: self.skill_factory(action_1, MAX_FORCE)}
        holds: Dict[int, HoldBrake] = {0: HoldBrake(MAX_FORCE), 1: HoldBrake(MAX_FORCE)}
        skill_ends: Dict[int, Optional[Dict[str, object]]] = {0: None, 1: None}
        skill_index = self.skills_used + 1
        start_period = self.total_periods
        start_time = start_period * CONTROL_DT
        distance_before = self.object_distance()
        distances = [distance_before]
        periods = 0
        infos: Dict[int, Dict[str, object]] = {}
        success_period: Optional[int] = None
        anomaly: Optional[str] = None
        period_rows: List[Dict[str, object]] = []

        while True:
            now = (start_period + periods) * CONTROL_DT
            contacts = self.contacts()
            observations = {robot_id: self.observe(robot_id, self.total_periods, now, contacts)
                            for robot_id in (0, 1)}
            # the policy-facing vector of this period must be built *before* the world advances
            # and before ``total_periods`` is incremented, otherwise its clock columns describe
            # the next period and a boundary frame would disagree with the skill that owns it
            observation_vector = self.observation_vector(observations[0], observations[1])
            forces: List[Vec] = [(0.0, 0.0), (0.0, 0.0)]
            for robot_id in (0, 1):
                if skills[robot_id].done:
                    force, info = holds[robot_id].step(observations[robot_id], CONTROL_DT)
                    info = {**info, "skill": f"HOLD_AFTER({skills[robot_id].action.label})"}
                else:
                    force, info = skills[robot_id].step(observations[robot_id], CONTROL_DT)
                    if skills[robot_id].done and skill_ends[robot_id] is None:
                        skill_ends[robot_id] = {
                            "period": start_period + periods, "time": now,
                            "exit_reason": skills[robot_id].exit_reason,
                            "stage": skills[robot_id].stage,
                            "speed_at_exit": skills[robot_id].speed_at_exit,
                            "residual_motion": bool(skills[robot_id].speed_at_exit > 0.03)}
                forces[robot_id] = force
                infos[robot_id] = info

            if self.orient_target:
                for robot_id in (0, 1):
                    target = infos[robot_id].get("desired_heading")
                    if target is not None:
                        desired = math.atan2(float(target[1]), float(target[0]))
                        error = _wrap_angle(desired - self.heading[robot_id])
                        omega = max(-self.config.max_omega,
                                    min(self.config.max_omega, self.config.align_gain * error))
                        self.heading_rate[robot_id] = omega
                        self.heading[robot_id] = _wrap_angle(
                            self.heading[robot_id] + omega * CONTROL_DT)
                        infos[robot_id] = {**infos[robot_id],
                                           "heading": self.heading[robot_id],
                                           "heading_rate": omega,
                                           "align_error_deg": math.degrees(error),
                                           "force_scale": 1.0,
                                           "force_gated": list(forces[robot_id]),
                                           "oriented_mode": self.orient_mode}
                    else:
                        self.heading_rate[robot_id] = 0.0
                        infos[robot_id] = {**infos[robot_id],
                                           "heading": self.heading[robot_id],
                                           "heading_rate": 0.0, "align_error_deg": None,
                                           "force_scale": 1.0,
                                           "force_gated": list(forces[robot_id]),
                                           "oriented_mode": self.orient_mode}
            self.env.step_control((forces[0], forces[1]))
            self.world_advances += 1
            self.total_periods += 1
            periods += 1
            elapsed = start_time + periods * CONTROL_DT

            conditions = self.success_conditions()
            distances.append(float(conditions["distance"]))
            if (conditions["distance_ok"] and conditions["linear_ok"]
                    and conditions["angular_ok"]):
                self.hold_seconds += CONTROL_DT
            else:
                self.hold_seconds = 0.0
            if self.hold_seconds >= self.config.success_hold:
                success_period = periods

            diagnostics = self.env.backend
            penetration = float(diagnostics.max_penetration())
            finite = bool(diagnostics.finite())
            in_bounds = bool(diagnostics.in_bounds())
            if not finite:
                anomaly = TERMINATION_NUMERIC
            elif penetration > self.config.penetration_limit:
                anomaly = TERMINATION_PENETRATION
            elif not in_bounds:
                anomaly = TERMINATION_OUT_OF_BOUNDS

            row = self._row(skill_index, start_period + periods, action_0, action_1,
                            infos, observations, forces, conditions, penetration)
            period_rows.append(row)
            if record:
                self.rows.append(row)
                self.history.append({
                    "period": self.total_periods,
                    "time": row["time"],
                    "observation": observation_vector,
                    "action_index": joint_action_index(action_index_0, action_index_1),
                    "skills": [action_0.label, action_1.label],
                    "stages": [infos[0]["stage"], infos[1]["stage"]],
                    "force": [list(forces[0]), list(forces[1])],
                })

            if success_period is not None or anomaly is not None:
                break
            if all(skills[robot_id].done for robot_id in (0, 1)):
                break
            if periods * CONTROL_DT >= window_seconds:
                break
            if elapsed >= self.config.time_budget:
                break

        # a robot whose skill was still running when the shared option window closed gets an
        # explicit exit reason; it is never left as "no reason"
        for robot_id in (0, 1):
            if skill_ends[robot_id] is None:
                final_observation = observations[robot_id]
                skill_ends[robot_id] = {
                    "period": start_period + periods,
                    "time": (start_period + periods) * CONTROL_DT,
                    "exit_reason": "joint_window_end",
                    "stage": skills[robot_id].stage,
                    "speed_at_exit": float(final_observation.own.speed),
                    "residual_motion": bool(final_observation.own.speed > 0.03)}

        # -- termination -----------------------------------------------------
        truncated = False
        if success_period is not None:
            self.success = True
            self.termination = TERMINATION_SUCCESS
        elif anomaly is not None:
            self.termination = anomaly
        elif self.total_periods * CONTROL_DT >= self.config.time_budget:
            truncated = True
            self.termination = TERMINATION_BUDGET

        terms = self._reward_terms(distances,
                                   success=success_period is not None,
                                   budget=bool(truncated))
        self.discount_prefix = terms["discount_after"]
        self.reward_total += terms["total"]
        for key in self.reward_components:
            self.reward_components[key] += terms[key]
        self.skills_used += 1

        boundary = {
            "skill_index": skill_index,
            "action_index": [int(action_index_0), int(action_index_1)],
            "action": [action_0.label, action_1.label],
            "kind": [action_0.kind, action_1.kind],
            "periods": periods,
            "start_period": start_period,
            "start_time": start_time,
            "end_time": start_time + periods * CONTROL_DT,
            "skill_ends": skill_ends,
            "joint_end_period": start_period + periods,
            "distance_before": distance_before,
            "distance_after": distances[-1],
            "discount_prefix_before": terms["discount_before"],
            "discount_prefix_after": terms["discount_after"],
            "reward": terms,
            "success": bool(success_period is not None),
            "truncated": bool(truncated),
            "anomaly": anomaly,
            "termination": self.termination,
            "goal_distance_after": self.object_distance(),
            "object_pose_after": list(self.object_state().as_tuple()[:3]),
        }
        if record:
            self.boundaries.append(boundary)
        if success_period is not None or anomaly is not None or truncated:
            self._episode_events.append({
                "kind": "termination", "reason": self.termination,
                "period": self.total_periods, "time": self.total_periods * CONTROL_DT,
                "success": self.success})
        return boundary

    # -- reward ------------------------------------------------------------
    def _reward_terms(self, distances: Sequence[float],
                       *, success: bool, budget: bool) -> Dict[str, float]:
        """Sum real per-control-period dense rewards, retaining real terminal distance.

        r_k = b*(d_k-d_(k+1)) - a*d_(k+1)*dt - c + terminal_event.
        No absorbing potential, enabling bonus, evidence bonus or per-skill reward.
        """
        if len(distances) < 2 or not all(math.isfinite(d) and d >= 0 for d in distances):
            raise ValueError("Reward requires finite non-negative initial and post-period distances")
        periods = len(distances) - 1
        progress = self.config.progress_weight * (distances[0] - distances[-1])
        distance_cost = -self.config.distance_cost * CONTROL_DT * sum(distances[1:])
        time_cost = -self.config.time_cost * periods
        event = self.config.success_bonus if success else (self.config.failure_penalty if budget else 0.0)
        return {
            "progress": float(progress),
            "distance_cost": float(distance_cost),
            "time_cost": float(time_cost),
            "success_bonus": float(event if success else 0.0),
            "failure_penalty": float(event if budget and not success else 0.0),
            "event": float(event),
            "total": float(progress + distance_cost + time_cost + event),
            "periods": int(periods),
            "discount_before": 1.0,
            "discount_after": 1.0,
            "distance_before": float(distances[0]),
            "distance_after": float(distances[-1]),
        }

    # -- rows and logs -----------------------------------------------------
    def _row(self, skill_index, period, action_0, action_1, infos, observations, forces,
             conditions, penetration) -> Dict[str, object]:
        true_states = self.robot_states()
        object_state = self.object_state()
        robot_tuples = [list(state.as_tuple()) for state in true_states]
        if self.orient_rotate:
            for robot_id in (0, 1):
                robot_tuples[robot_id][2] = float(self.heading[robot_id])
                robot_tuples[robot_id][5] = float(self.heading_rate[robot_id])
        row: Dict[str, object] = {
            "tick": int(self.env.tick),
            "period": int(period),
            "time": float(period * CONTROL_DT),
            "skill_index": int(skill_index),
            "joint_action_index": int(joint_action_index(action_0.index, action_1.index)),
            "joint_action_label": f"{action_0.label}|{action_1.label}",
            "forces": [list(forces[0]), list(forces[1])],
            "true": {
                "object": list(object_state.as_tuple()),
                "robots": robot_tuples,
            },
            "goal": list(self.instance.goal_centre),
            "goal_radius": float(self.instance.goal_radius),
            "success_conditions": conditions,
            "hold_seconds": float(self.hold_seconds),
            "penetration": float(penetration),
            "diagnostic_only": "true state, goal distance and hold timer are diagnostic truth",
            "true_goal_distance": self.object_distance(),
            "oriented": True, "oriented_mode": self.orient_mode,
        }
        for robot_id in (0, 1):
            info = infos[robot_id]
            observation = observations[robot_id]
            scan = observation.scan
            patch = observation.patch
            row[f"r{robot_id}"] = {
                "skill": info.get("skill"), "kind": info.get("kind"),
                "stage": info.get("stage"), "exit_reason": info.get("exit_reason"),
                "done": bool(info.get("done")),
                "force": list(info.get("force", (0.0, 0.0))),
                "desired_velocity": list(info.get("desired_velocity", (0.0, 0.0))),
                "reference_velocity": list(info.get("reference_velocity", (0.0, 0.0))),
                "force_norm": float(info.get("force_norm", 0.0)),
                "force_saturated": bool(info.get("force_saturated", False)),
                "progress": info.get("progress"), "progress_target": info.get("progress_target"),
                "no_progress_seconds": info.get("no_progress_seconds"),
                "gap": info.get("gap"), "gap_target": info.get("gap_target"),
                "gap_error": info.get("gap_error"),
                "normal_world": info.get("normal_world"),
                "outward_normal": info.get("outward_normal"),
                "normal_valid": info.get("normal_valid"),
                "normal_source": info.get("normal_source"),
                "tangent_world": info.get("tangent_world"),
                "tangent_direction": info.get("tangent_direction"),
                "patch_valid": info.get("patch_valid"),
                "patch_arc_rays": info.get("patch_arc_rays"),
                "patch_rms": info.get("patch_rms"),
                "patch_confidence": info.get("patch_confidence"),
                "contacting_object": bool(observation.contacting_object),
                "contact_point_robot": list(observation.contact_point_robot)
                if observation.contact_point_robot is not None else None,
                "contacting_static": bool(observation.contacting_static),
                "contacting_teammate": bool(observation.contacting_teammate),
                "obstacle_nearest_distance": info.get("obstacle_nearest_distance"),
                "obstacle_nearest_point": (list(info["obstacle_nearest_point"])
                                           if info.get("obstacle_nearest_point") else None),
                "teammate_nearest_distance": info.get("teammate_nearest_distance"),
                "avoidance_active": bool(info.get("avoidance_active", False)),
                "avoidance_direction": (list(info["avoidance_direction"])
                                        if info.get("avoidance_direction") else None),
                "push_alpha_deg": info.get("push_alpha_deg"),
                "push_speed": info.get("push_speed"),
                "effective_contact_seconds": info.get("effective_contact_seconds"),
                "corner_event": bool(info.get("corner_event", False)),
                "corner_events": info.get("corner_events"),
                "unintended_contact_seconds": info.get("unintended_contact_seconds"),
                "speed_at_exit": info.get("speed_at_exit"),
                "residual_motion": info.get("residual_motion"),
                "safety_trigger": (list(info["safety_trigger"])
                                   if info.get("safety_trigger") else None),
                "skill_time": info.get("skill_time"),
                "declared_direction": (list(info["declared_direction"])
                                       if info.get("declared_direction") else None),
                "signed_progress": info.get("signed_progress"),
                "positive_travel": info.get("positive_travel"),
                "backward_travel": info.get("backward_travel"),
                "progress_target_distance": info.get("progress_target_distance"),
                "option_timeout_seconds": info.get("option_timeout_seconds"),
                "face_switches": info.get("face_switches"),
                "heading": info.get("heading"), "heading_rate": info.get("heading_rate"),
                "desired_heading": info.get("desired_heading"),
                "align_error_deg": info.get("align_error_deg"),
                "force_scale": info.get("force_scale"),
                "force_gated": info.get("force_gated"),
                "find_direction": (list(info["find_direction"])
                                   if info.get("find_direction") else None),
                "patch_points_world": ([list(point) for point in patch.points_world]
                                       if patch is not None else []),
                # the sensor record of this period: exactly what the robot measured
                "scan": {
                    "distance": [round(value, 4) for value in scan.distances],
                    "valid": [bool(value) for value in scan.valid],
                    "category": [int(value) for value in scan.category],
                    "origin": list(scan.origin), "heading": float(scan.heading),
                },
            }
        return row

    # -- branching ---------------------------------------------------------
    def export_branch_state(self) -> Dict[str, object]:
        return {
            "env_version": ENV_VERSION,
            "snapshot": self.env.export_state(),
            "total_periods": self.total_periods,
            "skills_used": self.skills_used,
            "world_advances": self.world_advances,
            "hold_seconds": self.hold_seconds,
            "success": self.success,
            "termination": self.termination,
            "discount_prefix": self.discount_prefix,
            "reward_total": self.reward_total,
            "reward_components": dict(self.reward_components),
            "history": list(self.history),
            "rows": len(self.rows),
            "boundaries": len(self.boundaries),
            "oriented": True,
            "oriented_mode": self.orient_mode,
            "heading": list(self.heading), "heading_rate": list(self.heading_rate),
        }

    def restore_branch_state(self, state: Dict[str, object]) -> None:
        if state.get("env_version") != ENV_VERSION:
            raise ValueError("Cannot restore a branch from a different reward contract")
        if "oriented" in state and not bool(state["oriented"]):
            raise ValueError("Cannot restore a pre-v4 branch into the v4 actuation model")
        self.env.restore(state["snapshot"], exact=True)  # type: ignore[arg-type]
        self.total_periods = int(state["total_periods"])  # type: ignore[arg-type]
        self.skills_used = int(state["skills_used"])  # type: ignore[arg-type]
        self.world_advances = int(state["world_advances"])  # type: ignore[arg-type]
        self.hold_seconds = float(state["hold_seconds"])  # type: ignore[arg-type]
        self.success = bool(state["success"])
        self.termination = state["termination"]  # type: ignore[assignment]
        self.discount_prefix = float(state["discount_prefix"])  # type: ignore[arg-type]
        self.reward_total = float(state["reward_total"])  # type: ignore[arg-type]
        self.reward_components = dict(state["reward_components"])  # type: ignore[arg-type]
        self.history = list(state["history"])  # type: ignore[arg-type]
        self.rows = self.rows[: int(state["rows"])]  # type: ignore[arg-type]
        self.boundaries = self.boundaries[: int(state["boundaries"])]  # type: ignore[arg-type]
        if "heading" in state:
            self.heading = [float(value) for value in state["heading"]]  # type: ignore[arg-type]
            self.heading_rate = [float(value) for value in state["heading_rate"]]  # type: ignore[arg-type]


def recompute_episode_reward(rows: Sequence[Dict[str, object]],
                             boundaries: Sequence[Dict[str, object]],
                             initial_object_xy: Tuple[float, float],
                             config: Optional[EnvConfig] = None) -> Dict[str, float]:
    """Independent recomputation of the episode return from the raw physical log.

    It consumes the per-period log (which carries the true object position, the goal and the
    period count of each boundary) and re-derives the distances, integrated costs and the
    event terms from scratch. It never reads the reward back out of the log, so a bug in the
    environment's reward path cannot hide here.
    """

    config = config or EnvConfig()
    total = 0.0
    progress_total = 0.0
    distance_total = 0.0
    time_total = 0.0
    event_total = 0.0
    index = 0
    previous_object = (float(initial_object_xy[0]), float(initial_object_xy[1]))
    for boundary in boundaries:
        periods = int(boundary["periods"])
        success = bool(boundary["success"])
        truncated = bool(boundary["truncated"])
        progress = 0.0
        distance_cost = 0.0
        for row in rows[index:index + periods]:
            goal = row["goal"]
            object_pose = row["true"]["object"]
            before = math.hypot(previous_object[0] - goal[0], previous_object[1] - goal[1])
            after = math.hypot(object_pose[0] - goal[0], object_pose[1] - goal[1])
            progress += config.progress_weight * (before - after)
            distance_cost -= config.distance_cost * CONTROL_DT * after
            previous_object = (float(object_pose[0]), float(object_pose[1]))
        time_cost = -config.time_cost * periods
        event = config.success_bonus if success else (
            config.failure_penalty if truncated else 0.0)
        progress_total += progress
        distance_total += distance_cost
        time_total += time_cost
        event_total += event
        total += progress + distance_cost + time_cost + event
        index += periods
    return {"total": float(total), "progress": float(progress_total), "distance_cost": float(distance_total),
             "time_cost": float(time_total), "event": float(event_total),
             "discount_prefix": 1.0}
