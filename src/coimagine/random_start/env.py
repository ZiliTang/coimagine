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
from dataclasses import dataclass, field
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
    CONTROL_DT,
    FAILURE_PENALTY,
    GAMMA,
    JOINT_MAX_SECONDS,
    LIDAR_RANGE,
    MAX_FORCE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    POTENTIAL_WEIGHT,
    ROBOT_ACTIONS,
    ROBOT_RADIUS,
    SKILL_BUDGET,
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
    skill_budget: int = SKILL_BUDGET
    gamma: float = GAMMA
    time_cost: float = TIME_COST
    success_bonus: float = SUCCESS_BONUS
    failure_penalty: float = FAILURE_PENALTY
    potential_weight: float = POTENTIAL_WEIGHT
    success_distance: float = SUCCESS_DISTANCE
    success_linear: float = SUCCESS_LINEAR
    success_angular: float = SUCCESS_ANGULAR
    success_hold: float = SUCCESS_HOLD_SECONDS
    joint_max_seconds: float = JOINT_MAX_SECONDS
    penetration_limit: float = 0.05
    base: Continuous2DConfig = field(default_factory=Continuous2DConfig)

    def as_dict(self) -> Dict[str, object]:
        return {"time_budget": self.time_budget, "skill_budget": self.skill_budget,
                "gamma": self.gamma, "time_cost": self.time_cost,
                "success_bonus": self.success_bonus,
                "failure_penalty": self.failure_penalty,
                "potential_weight": self.potential_weight,
                "success_distance": self.success_distance,
                "success_linear": self.success_linear,
                "success_angular": self.success_angular,
                "success_hold": self.success_hold,
                "joint_max_seconds": self.joint_max_seconds,
                "penetration_limit": self.penetration_limit,
                "physics_config_hash": self.base.config_hash()}


def _kinematics(state) -> RobotKinematics:
    return RobotKinematics(x=float(state.x), y=float(state.y), angle=float(state.angle),
                           vx=float(state.vx), vy=float(state.vy), omega=float(state.omega))


class RandomStartEnv:
    """One task instance, one living world, and a sequence of synchronous joint skills."""

    def __init__(self, instance: InstanceSpec, config: Optional[EnvConfig] = None):
        self.instance = instance
        self.config = config or EnvConfig()
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
            "shaping": 0.0, "time_cost": 0.0, "success_bonus": 0.0, "failure_penalty": 0.0}
        self._episode_events: List[Dict[str, object]] = []

    # -- privileged truth, reward and evaluation only -----------------------
    def object_state(self):
        return self.env.backend.object_state()

    def robot_states(self):
        return self.env.backend.robot_states()

    def object_distance(self) -> float:
        state = self.object_state()
        return float(math.hypot(state.x - self.instance.goal_centre[0],
                                state.y - self.instance.goal_centre[1]))

    def potential(self) -> float:
        return -self.object_distance()

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

    def scan_for(self, robot_id: int, tick: int, time: float) -> LidarScan:
        states = [state.as_tuple() for state in self.robot_states()]
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
                   self.skills_used / max(1, self.config.skill_budget),
                   max(0.0, self.config.time_budget - elapsed) / self.config.time_budget]
        for observation in (observation_0, observation_1):
            scan = observation.scan
            values += [((scan.distances[index] / LIDAR_RANGE) if scan.valid[index] else 0.0)
                       for index in range(len(scan.valid))]
            values += [scan.category[index] / 3.0 for index in range(len(scan.valid))]
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
        fields += ["elapsed_fraction", "skills_fraction", "remaining_fraction"]
        for robot_id in (0, 1):
            fields += [f"r{robot_id}_lidar_{index:02d}" for index in range(72)]
            fields += [f"r{robot_id}_class_{index:02d}" for index in range(72)]
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
        skills: Dict[int, Skill] = {0: make_skill(action_0, MAX_FORCE),
                                    1: make_skill(action_1, MAX_FORCE)}
        holds: Dict[int, HoldBrake] = {0: HoldBrake(MAX_FORCE), 1: HoldBrake(MAX_FORCE)}
        skill_ends: Dict[int, Optional[Dict[str, object]]] = {0: None, 1: None}
        skill_index = self.skills_used + 1
        start_period = self.total_periods
        start_time = start_period * CONTROL_DT
        phi_before = self.potential()
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

            self.env.step_control((forces[0], forces[1]))
            self.world_advances += 1
            self.total_periods += 1
            periods += 1
            elapsed = start_time + periods * CONTROL_DT

            conditions = self.success_conditions()
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
            if periods * CONTROL_DT >= self.config.joint_max_seconds:
                break
            if elapsed >= self.config.time_budget:
                break

        # a robot whose skill was still running when the shared 8 s window closed gets an
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
        terminated = False
        truncated = False
        if success_period is not None:
            self.success = True
            terminated = True
            self.termination = TERMINATION_SUCCESS
        elif anomaly is not None:
            terminated = True
            self.termination = anomaly
        elif self.total_periods * CONTROL_DT >= self.config.time_budget \
                or self.skills_used + 1 >= self.config.skill_budget:
            truncated = True
            self.termination = TERMINATION_BUDGET

        phi_next = 0.0 if (terminated or truncated) else self.potential()
        terms = self._reward_terms(periods, phi_before, phi_next,
                                   success=success_period is not None,
                                   budget=bool(truncated))
        self.discount_prefix = terms["discount_after"]
        self.reward_total += terms["total"]
        for key in ("shaping", "time_cost", "success_bonus", "failure_penalty"):
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
            "phi_before": phi_before,
            "phi_next_used": phi_next,
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
    def _reward_terms(self, periods: int, phi_before: float, phi_next: float,
                      *, success: bool, budget: bool) -> Dict[str, float]:
        """The single reward implementation of section 5.2.

        ``R = gamma^n Phi_next - Phi_current - c * sum_{j<n} gamma^j + gamma^(n-1) B``
        evaluated at this joint boundary, then weighted by the episode's discount prefix so
        the same formula is valid for every boundary of a long episode.
        """

        gamma = self.config.gamma
        prefix = self.discount_prefix
        discount_to_end = gamma ** periods
        geometric = sum(gamma ** j for j in range(periods))
        shaping = prefix * self.config.potential_weight * (discount_to_end * phi_next - phi_before)
        time_cost = -prefix * self.config.time_cost * geometric
        event = 0.0
        if success:
            event = self.config.success_bonus
        elif budget:
            event = self.config.failure_penalty
        event_term = prefix * (gamma ** (periods - 1)) * event if periods > 0 else prefix * event
        return {
            "shaping": float(shaping),
            "time_cost": float(time_cost),
            "success_bonus": float(event_term if success else 0.0),
            "failure_penalty": float(event_term if budget else 0.0),
            "event": float(event_term),
            "total": float(shaping + time_cost + event_term),
            "periods": int(periods),
            "discount_before": float(prefix),
            "discount_after": float(prefix * discount_to_end),
            "phi_before": float(phi_before),
            "phi_next_used": float(phi_next),
        }

    # -- rows and logs -----------------------------------------------------
    def _row(self, skill_index, period, action_0, action_1, infos, observations, forces,
             conditions, penetration) -> Dict[str, object]:
        true_states = self.robot_states()
        object_state = self.object_state()
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
                "robots": [list(state.as_tuple()) for state in true_states],
            },
            "goal": list(self.instance.goal_centre),
            "goal_radius": float(self.instance.goal_radius),
            "success_conditions": conditions,
            "hold_seconds": float(self.hold_seconds),
            "penetration": float(penetration),
            "diagnostic_only": "true state, goal distance and hold timer are diagnostic truth",
            "true_goal_distance": self.object_distance(),
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
        }

    def restore_branch_state(self, state: Dict[str, object]) -> None:
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


def recompute_episode_reward(rows: Sequence[Dict[str, object]],
                             boundaries: Sequence[Dict[str, object]],
                             initial_object_xy: Tuple[float, float],
                             config: Optional[EnvConfig] = None) -> Dict[str, float]:
    """Independent recomputation of the episode return from the raw physical log.

    It consumes the per-period log (which carries the true object position, the goal and the
    period count of each boundary) and re-derives the potential, the discount prefix and the
    event terms from scratch. It never reads the reward back out of the log, so a bug in the
    environment's reward path cannot hide here.
    """

    config = config or EnvConfig()
    gamma = config.gamma
    total = 0.0
    prefix = 1.0
    shaping_total = 0.0
    time_total = 0.0
    event_total = 0.0
    index = 0
    previous_object = (float(initial_object_xy[0]), float(initial_object_xy[1]))
    for boundary in boundaries:
        periods = int(boundary["periods"])
        goal = rows[index]["goal"]
        phi_before = -math.hypot(previous_object[0] - goal[0], previous_object[1] - goal[1])
        last = rows[index + periods - 1]
        success = bool(boundary["success"])
        truncated = bool(boundary["truncated"])
        anomaly = boundary.get("anomaly")
        terminated = bool(success or anomaly is not None or truncated)
        if terminated:
            phi_next = 0.0
        else:
            object_pose = last["true"]["object"]
            phi_next = -math.hypot(object_pose[0] - goal[0], object_pose[1] - goal[1])
        geometric = sum(gamma ** j for j in range(periods))
        shaping = prefix * config.potential_weight * (gamma ** periods * phi_next - phi_before)
        time_cost = -prefix * config.time_cost * geometric
        event = config.success_bonus if success else (
            config.failure_penalty if truncated else 0.0)
        event_term = (prefix * (gamma ** (periods - 1)) * event if periods > 0
                      else prefix * event)
        shaping_total += shaping
        time_total += time_cost
        event_total += event_term
        total += shaping + time_cost + event_term
        prefix *= gamma ** periods
        previous_object = (float(last["true"]["object"][0]), float(last["true"]["object"][1]))
        index += periods
    return {"total": float(total), "shaping": float(shaping_total),
            "time_cost": float(time_total), "event": float(event_total),
            "discount_prefix": float(prefix)}
