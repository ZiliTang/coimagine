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
import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import Continuous2DEnv
from . import geometry as G
from .lidar_service import LidarService
from .lidar import (
    CATEGORY_NONE,
    CATEGORY_OBJECT,
    CATEGORY_STATIC,
    CATEGORY_TEAMMATE,
    LidarScan,
    SurfacePatch,
    best_patch,
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
from .peer_clearance import PeerClearancePrefix
from .anchor import AnchorTracker
from .spec import (
    BRAKE_MAX_SECONDS,
    CONTROL_DT,
    ENV_VERSION,
    FAILURE_PENALTY,
    GAMMA,
    JOINT_MAX_SECONDS,
    LIDAR_RANGE,
    MAX_FORCE,
    NO_PROGRESS_SECONDS,
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
    peer_clearance_recovery: str = "off"
    lidar_backend: str = "scalar"

    def __post_init__(self):
        if not math.isclose(self.base.physics.control_dt,CONTROL_DT,rel_tol=0.,abs_tol=1e-12):
            raise ValueError('random-start clocks currently require 20 Hz control; changing physics_dt alone is supported')
        if self.lidar_backend not in ('scalar','numpy_batch'):
            raise ValueError('unknown lidar backend')
        if self.peer_clearance_recovery not in ("off", "boundary_v1"):
            raise ValueError("unknown peer clearance recovery mode")
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
                "peer_clearance_recovery": self.peer_clearance_recovery,
                "lidar_backend": self.lidar_backend,
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


def _segment_distance(point, a, b) -> float:
    ax, ay = a
    bx, by = b
    ex, ey = bx - ax, by - ay
    denom = ex * ex + ey * ey
    t = 0.0 if denom < 1e-12 else max(0.0, min(1.0, ((point[0] - ax) * ex
                                                     + (point[1] - ay) * ey) / denom))
    return math.hypot(point[0] - (ax + t * ex), point[1] - (ay + t * ey))


def _point_polygon_distance(point, polygon) -> float:
    """Signed distance from a point to a convex polygon (negative inside); diagnostics only."""

    edges = list(zip(polygon, list(polygon[1:]) + [polygon[0]]))
    distance = min(_segment_distance(point, a, b) for a, b in edges)
    sign = None
    for a, b in edges:
        cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
        if abs(cross) < 1e-12:
            continue
        current = cross > 0
        if sign is None:
            sign = current
        elif current != sign:
            return distance
    return -distance


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _kinematics(state) -> RobotKinematics:
    return RobotKinematics(x=float(state.x), y=float(state.y), angle=float(state.angle),
                           vx=float(state.vx), vy=float(state.vy), omega=float(state.omega))


class RandomStartEnv:
    """One task instance, one living world, and a sequence of synchronous joint skills."""

    def __init__(self, instance: InstanceSpec, config: Optional[EnvConfig] = None,
                 skill_factory=None, *, action_table=None, action_version=None):
        self.instance = instance
        self.config = config or EnvConfig()
        # Explicit controller injection point. The default is the frozen baseline factory, so
        # every existing caller keeps the identical behaviour; a candidate controller is passed
        # here by name and is recorded in the branch log. No implicit global monkeypatch.
        self.skill_factory = skill_factory if skill_factory is not None else make_skill
        self.actions = tuple(ROBOT_ACTIONS if action_table is None else action_table)
        if not self.actions or any(a.index != i for i, a in enumerate(self.actions)):
            raise ValueError("action_table must have contiguous, explicit action IDs")
        self.action_version = action_version or ENV_VERSION
        self.action_codebook_hash = hashlib.sha256(json.dumps(
            [asdict(action) for action in self.actions], sort_keys=True).encode()).hexdigest()
        self._legacy_action_table = self.actions == tuple(ROBOT_ACTIONS)
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
        #: v4.1 option progress of the previous period, per robot: (signed travel, target)
        self._last_progress = {0: (0.0, 0.0), 1: (0.0, 0.0)}
        #: v4.1 anchor tracking: legal corner candidates only, with an explicit status
        self.anchor_trackers = {0: AnchorTracker(), 1: AnchorTracker()}
        # Single-entry sensor caches per robot. The key includes the requested
        # timestamp and complete ray-casting geometry, not merely the tick.
        self.lidar = LidarService()
        self._scan_cache = self.lidar.analytic_cache
        self._patch_cache = {}
        self._last_anchor = {0: None, 1: None}
        self._hold_time = {0: 0.0, 1: 0.0}
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
        obj = self.object_state()
        pose = (obj.x, obj.y, obj.angle)
        return self.lidar.analytic(self.instance.arena, self.obstacle_polygons, pose,
                                   states, robot_id, tick, time,
                                   object_polygon=self.instance.object_polygon,
                                   backend=self.config.lidar_backend)

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
        self._last_anchor[robot_id] = self.anchor_trackers[robot_id].update(
            scan, float(time), CONTROL_DT, tick=tick)
        previous = self._patch_cache.get(robot_id)
        if previous is not None and previous[0] is scan and previous[1] == (own.x, own.y):
            patches = previous[2]
        else:
            patches = tuple(scan_patches(scan, (own.x, own.y)))
            self._patch_cache[robot_id] = (scan, (own.x, own.y), patches)
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
                           observation_1: SkillObservation,
                           progress=None) -> np.ndarray:
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
        progress = progress if progress is not None else self._last_progress
        for robot_id in (0, 1):
            signed, target = progress.get(robot_id, (0.0, 0.0))
            values += [(min(1.0, signed / target) if target > 0 else 0.0),
                       min(1.0, target / 2.0)]
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
        for robot_id in (0, 1):
            fields += [f"r{robot_id}_option_progress_fraction",
                       f"r{robot_id}_option_target_fraction"]
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
    def _joint_action_index(self, first: int, second: int) -> int:
        if not (0 <= first < len(self.actions) and 0 <= second < len(self.actions)):
            raise ValueError("joint action ID outside this world's action table")
        return int(first * len(self.actions) + second)

    def joint_step(self, action_index_0: int, action_index_1: int,
                   record: bool = True) -> Dict[str, object]:
        if self.termination is not None:
            raise RuntimeError(
                f"episode already terminated with {self.termination!r}; a terminated episode "
                "may not be extended (it must not be used as a branchable parent)")
        self._joint_action_index(action_index_0, action_index_1)
        action_0 = self.actions[action_index_0]
        action_1 = self.actions[action_index_1]
        # v4: the joint window is the option's own time budget (the longer of the two), capped
        skills: Dict[int, Skill] = {0: self.skill_factory(action_0, MAX_FORCE),
                                    1: self.skill_factory(action_1, MAX_FORCE)}
        bound_followers = [i for i in (0,1) if getattr(skills[i],'completion_clock',None)=='peer_push']
        if bound_followers and (len(bound_followers)!=1 or skills[1-bound_followers[0]].kind!='push'):
            raise ValueError('push-bound FOLLOW requires a PUSH partner')
        clearance_prefix = (PeerClearancePrefix() if self.config.peer_clearance_recovery == "boundary_v1"
                            else None)
        window_seconds = min(self.config.joint_max_seconds,
                             max(skills[0].time_budget, skills[1].time_budget))
        if bound_followers:
            follower = bound_followers[0]
            pusher = 1-follower
            # PUSH keeps its original timeout. Only the follower's bounded brake
            # handoff may extend the joint window, at the original physical cost.
            window_seconds = min(self.config.joint_max_seconds,
                skills[pusher].time_budget + BRAKE_MAX_SECONDS + CONTROL_DT)
            skills[follower].bind_push(pusher, window_seconds)
        # Proportional propulsion biases use T = max(A, B) / v_ref. Actual normal
        # travel is measured independently: rigid contact can couple the velocities,
        # so this does not guarantee simultaneous exits for different distances.
        both_push = action_0.kind == action_1.kind == "push"
        requested_protocols = [getattr(skills[i], "execution_protocol", "legacy_v3") for i in (0, 1)]
        paired_reference = both_push and requested_protocols == ["proportional_normal_v1"] * 2
        if both_push and len(set(requested_protocols)) > 1:
            raise ValueError("dual PUSH skills must use the same execution protocol")
        cooperative_push = paired_reference or (both_push
                            and getattr(action_0, "coordination_key", None)
                            == getattr(action_1, "coordination_key", None))
        proportional_translation_pair = paired_reference and all(
            getattr(a, "effect_kind", None) == "translate" for a in (action_0, action_1))
        if proportional_translation_pair and len({skills[i].config.translation_completion for i in (0, 1)}) > 1:
            raise ValueError("dual translations must use the same completion measure")
        normal_reference_pair = proportional_translation_pair and all(
            getattr(skills[i], "normal_reference_mode", False) for i in (0, 1))
        normal_travel_pair = proportional_translation_pair and all(
            getattr(skills[i], "normal_travel_mode", False) for i in (0, 1))
        if normal_travel_pair and len({skills[i].config.post_push for i in (0, 1)}) > 1:
            raise ValueError("dual translations must use the same post-push protocol")
        post_follow_pair = normal_travel_pair and all(
            skills[i].config.post_push == "contact_follow" for i in (0, 1))
        if post_follow_pair and len({skills[i].config.contact_recovery for i in (0, 1)}) > 1:
            raise ValueError("dual translations must use the same contact recovery protocol")
        joint_recovery_pair = post_follow_pair and all(
            skills[i].config.contact_recovery == "joint_approach" for i in (0, 1))
        if joint_recovery_pair and any(getattr(skills[0].config, name) != getattr(skills[1].config, name)
                for name in ("recovery_enter_gap", "recovery_resume_gap", "recovery_ready_seconds",
                             "recovery_timeout", "recovery_max_attempts")):
            raise ValueError("joint recovery must share thresholds and budgets")
        recovery_active = False
        recovery_started_at = 0.0
        recovery_ready_seconds = 0.0
        recovery_attempts = 0
        post_exit = "contact_follow_then_brake" if post_follow_pair else "hold_brake"
        follow_stop_released = False
        for i in (0, 1):
            if post_follow_pair:
                skills[i].follow_enabled = True
                skills[i].recovery_enabled = joint_recovery_pair
        requested_budgets = [skills[i].time_budget for i in (0, 1)]
        released = False
        contact_released = False
        execution_events = []
        if bound_followers:
            execution_events.append({'event':'follow_bound','follower':follower,'pusher':pusher,
                                     'window_seconds':window_seconds})
        if paired_reference:
            window_seconds = min(self.config.joint_max_seconds,
                                 math.ceil(window_seconds / CONTROL_DT - 1e-9) * CONTROL_DT)
            for i in (0, 1):
                skills[i].joint_pair = True
                # Preparation/handshake shares a bounded clock; a short nominal
                # scale must not time out solely while its peer is preparing.
                skills[i].time_budget = window_seconds
        if cooperative_push:
            distances = (float(action_0.distance), float(action_1.distance))
            speeds = (float(action_0.speed), float(action_1.speed))
            reference = max(speeds[0], speeds[1], 1e-6)
            budget = max(distances) / reference
            if proportional_translation_pair:
                speeds = tuple(float(skills[i].speed_command) for i in (0, 1))
                budget = max(distances[i] / max(speeds[i], 1e-6) for i in (0, 1))
            for robot_id in (0, 1):
                if hasattr(skills[robot_id], "speed_command") and (not paired_reference or proportional_translation_pair):
                    skills[robot_id].speed_command = (distances[robot_id] / budget
                                                      if budget > 0.0 else speeds[robot_id])
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

        while True:
            now = (start_period + periods) * CONTROL_DT
            contacts = self.contacts()
            if bound_followers:
                # All peer state below was reported on PRECEDING periods. Current
                # private observations remain separate during the two step calls.
                follow_skill,push_skill=skills[follower],skills[pusher]
                follow_skill.bound_push_phase=infos.get(pusher,{}).get('stage','prepare')
                if push_skill.done and not follow_skill.done and follow_skill.bound_push_stop is None:
                    successful=push_skill.exit_reason in ('normal_travel_complete','normal_reference_complete',
                        'effect_complete','local_effect_reached','push_complete')
                    follow_skill.bound_push_stop='follow_push_complete' if successful else 'peer_push_failed'
                    execution_events.append({'event':'follow_stop','time':now,'pusher_exit':push_skill.exit_reason,
                                             'reason':follow_skill.bound_push_stop})
                elif follow_skill.done and not push_skill.done:
                    if not any(e['event']=='follow_failed' for e in execution_events):
                        push_skill.pending_exit_reason='peer_follow_failed'
                        execution_events.append({'event':'follow_failed','time':now,'reason':follow_skill.exit_reason})
            if joint_recovery_pair:
                cfg = skills[0].config
                if recovery_active:
                    failure = ("peer_recovery_failed" if any(skills[i].done for i in (0, 1)) else
                               "joint_recovery_timeout" if now - recovery_started_at >= cfg.recovery_timeout - 1e-9 else None)
                    if failure:
                        for i in (0, 1):
                            if not skills[i].done:
                                skills[i].pending_exit_reason = failure
                        execution_events.append({"event": "recovery_failed", "reason": failure, "time": now})
                        recovery_active = False
                    else:
                        recovery_ready_seconds = (recovery_ready_seconds + CONTROL_DT
                            if all(skills[i].recovery_ready for i in (0, 1)) else 0.0)
                        if recovery_ready_seconds >= cfg.recovery_ready_seconds - 1e-9:
                            for i in (0, 1):
                                skills[i].resume_recovery()
                            recovery_active = False
                            execution_events.append({"event": "recovery_resumed", "time": now,
                                                     "attempt": recovery_attempts})
                elif (all(not skills[i].done for i in (0, 1))
                        and not all(skills[i].drive_complete for i in (0, 1))
                        and any(skills[i].recovery_requested for i in (0, 1))):
                    recovery_attempts += 1
                    if recovery_attempts > cfg.recovery_max_attempts:
                        for i in (0, 1):
                            skills[i].pending_exit_reason = "joint_recovery_attempt_limit"
                        execution_events.append({"event": "recovery_failed", "reason": "attempt_limit", "time": now})
                    else:
                        requested = [skills[i].recovery_requested for i in (0, 1)]
                        for i in (0, 1):
                            skills[i].begin_recovery()
                        recovery_active = True
                        recovery_started_at = now
                        recovery_ready_seconds = 0.0
                        execution_events.append({"event": "recovery_started", "time": now,
                                                 "attempt": recovery_attempts, "requested": requested})
            # Phase flags use PRECEDING-period completion/failure reports only.
            # No current-period peer observation or future physical result is read.
            if (post_follow_pair and not recovery_active and not follow_stop_released
                    and any(skills[i].follow_started for i in (0, 1))
                    and all(skills[i].drive_complete or skills[i].done for i in (0, 1))):
                follow_stop_released = True
                for i in (0, 1):
                    skills[i].follow_stop = True
                execution_events.append({"event": "follow_stop_released", "time": now,
                                         "peer_failure": any(skills[i].done for i in (0, 1))})
            if cooperative_push:
                # v4.6: both robots stage at COOP_READY_GAP with their heading aligned to
                # the contact normal.  Only when both report ready do they leave the staging
                # clearance together; until then both apply zero force.
                for robot_id in (0, 1):
                    skills[robot_id].cooperative = True
                    skills[robot_id].hold = False
                if paired_reference and not contact_released and any(skills[i].done for i in (0, 1)):
                    for i in (0, 1):
                        if not skills[i].done:
                            skills[i].pending_exit_reason = "peer_prepare_failed"
                    if not any(e["event"] == "peer_prepare_failed" for e in execution_events):
                        execution_events.append({"event": "peer_prepare_failed", "time": now})
                elif (paired_reference and not contact_released
                      and all(skills[i].no_progress_seconds >= NO_PROGRESS_SECONDS for i in (0, 1))):
                    for i in (0, 1):
                        skills[i].pending_exit_reason = "joint_prepare_stalled"
                    execution_events.append({"event": "joint_prepare_stalled", "time": now})
                elif (not paired_reference or not released) and all(getattr(skills[robot_id], "ready", False)
                       and not skills[robot_id].done for robot_id in (0, 1)):
                    for robot_id in (0, 1):
                        skills[robot_id].release = True
                    if paired_reference:
                        released = True
                        execution_events.append({"event": "staging_released", "time": now})
                if (paired_reference and released and not contact_released
                        and all(getattr(skills[i], "contact_ready", False) and not skills[i].done for i in (0, 1))):
                    contact_released = True
                    for i in (0, 1):
                        skills[i].joint_active = True
                    execution_events.append({"event": "contact_released", "time": now})
            observations = {robot_id: self.observe(robot_id, self.total_periods, now, contacts)
                            for robot_id in (0, 1)}
            # the policy-facing vector of this period must be built *before* the world advances
            # and before ``total_periods`` is incremented, otherwise its clock columns describe
            # the next period and a boundary frame would disagree with the skill that owns it
            observation_vector = (self.observation_vector(observations[0], observations[1],
                                                          self._last_progress) if record else None)
            forces: List[Vec] = [(0.0, 0.0), (0.0, 0.0)]
            separation = clearance_prefix.step(observations, CONTROL_DT) if clearance_prefix else None
            for robot_id in (0, 1):
                if separation is not None:
                    skill = skills[robot_id]
                    skill.skill_time += CONTROL_DT
                    if skill.kind == "wait":
                        skill.progress += CONTROL_DT
                    force, info = separation[robot_id]
                    info.update(skill=skill.action.label, requested_kind=skill.kind,
                                progress_target=skill.action.distance or skill.action.duration,
                                progress_target_distance=(skill.action.distance
                                    if skill.action.scale_kind == "distance" else None),
                                skill_time=skill.skill_time)
                    if clearance_prefix.failed:
                        skill._finish("blocked_peer_clearance", observations[robot_id])
                        info.update(done=True, exit_reason="blocked_peer_clearance")
                elif skills[robot_id].done:
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
                if getattr(skills[robot_id], "normal_travel_mode", False):
                    info["completion_progress"] = skills[robot_id].normal_travel
                    info["completion_target"] = skills[robot_id].action.distance
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
            for robot_id in (0, 1):
                target = infos[robot_id].get("completion_target", infos[robot_id].get("progress_target_distance"))
                self._last_progress[robot_id] = (
                    float(infos[robot_id].get("completion_progress", infos[robot_id].get("signed_progress")) or 0.0),
                    float(target) if target else 0.0)
            _,step_diagnostics = self.env.step_control(
                (forces[0], forces[1]),
                ((self.heading[0], self.heading_rate[0]),
                 (self.heading[1], self.heading_rate[1])),minimal=True)
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

            penetration = float(step_diagnostics['penetration'])
            finite = bool(step_diagnostics['finite'])
            in_bounds = bool(step_diagnostics['in_bounds'])
            if not finite:
                anomaly = TERMINATION_NUMERIC
            elif penetration > self.config.penetration_limit:
                anomaly = TERMINATION_PENETRATION
            elif not in_bounds:
                anomaly = TERMINATION_OUT_OF_BOUNDS

            if record:
                row = self._row(skill_index, start_period + periods, action_0, action_1,
                            infos, observations, forces, conditions, penetration)
            if record and paired_reference:
                row["execution"] = {
                    "protocol": "proportional_normal_v1",
                    "phase": "recover" if recovery_active else "execute" if contact_released else "engage" if released else "prepare",
                    "normal_reference_pair": normal_reference_pair,
                    "normal_travel_pair": normal_travel_pair,
                    "active": [not skills[i].done for i in (0, 1)],
                    "normal_reference_budget": [getattr(skills[i], "normal_reference_budget", 0.) for i in (0, 1)],
                    "normal_travel": [getattr(skills[i], "normal_travel", 0.) for i in (0, 1)],
                    "drive_complete": [getattr(skills[i], "drive_complete", False) for i in (0, 1)],
                    "follow_normal_travel": [getattr(skills[i], "follow_normal_travel", 0.) for i in (0, 1)],
                    "following": [skills[i].stage == "follow" for i in (0, 1)],
                    "recovery_attempts": recovery_attempts,
                    "recovery_normal_travel": [getattr(skills[i], "recovery_normal_travel", 0.) for i in (0, 1)],
                    "exit_reason": [skills[i].exit_reason for i in (0, 1)],
                    "post_exit": post_exit,
                }
            if record:
                self.rows.append(row)
                self.history.append({
                    "period": self.total_periods,
                    "time": row["time"],
                    "observation": observation_vector,
                    "action_index": self._joint_action_index(action_index_0, action_index_1),
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
            "action_version": self.action_version,
            "action_codebook_hash": self.action_codebook_hash,
            "execution": {
                "protocol": "proportional_normal_v1" if paired_reference else "legacy_v3",
                "normal_reference_pair": normal_reference_pair,
                "normal_travel_pair": normal_travel_pair,
                "proportional_translation_pair": proportional_translation_pair,
                "requested_time_budgets": requested_budgets,
                "effective_window_seconds": window_seconds,
                "normal_speed_references": [getattr(skills[i], "speed_command", None) for i in (0, 1)],
                "normal_reference_budget": [getattr(skills[i], "normal_reference_budget", None) for i in (0, 1)],
                "normal_travel": [getattr(skills[i], "normal_travel", None) for i in (0, 1)],
                "drive_complete_times": [getattr(skills[i], "drive_complete_time", None) for i in (0, 1)],
                "follow_normal_travel": [getattr(skills[i], "follow_normal_travel", 0.) for i in (0, 1)],
                "follow_path_length": [getattr(skills[i], "follow_path_length", 0.) for i in (0, 1)],
                "contact_recovery": "joint_approach" if joint_recovery_pair else "abort",
                "recovery_attempts": recovery_attempts,
                "recovery_normal_travel": [getattr(skills[i], "recovery_normal_travel", 0.) for i in (0, 1)],
                "recovery_path_length": [getattr(skills[i], "recovery_path_length", 0.) for i in (0, 1)],
                "events": execution_events,
                "post_exit": post_exit,
                "bound_follow": ({"version":"push-bound-follow-v1","follower":follower,"pusher":pusher}
                                 if bound_followers else None),
            },
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
        boundary["peer_clearance"] = ({"version":"peer-clearance-prefix-v1",
            "entered":clearance_prefix.started,"elapsed":clearance_prefix.elapsed,
            "failed":clearance_prefix.failed,"path_m":clearance_prefix.path}
            if clearance_prefix is not None else {"version":"off","entered":False})
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
        if getattr(self.instance, 'object_polygon', None) is not None:
            local = self.instance.object_polygon
            c, sn = math.cos(object_state.angle), math.sin(object_state.angle)
            object_polygon = tuple((object_state.x + c*float(v[0]) - sn*float(v[1]),
                                    object_state.y + sn*float(v[0]) + c*float(v[1])) for v in local)
        else:
            object_polygon = G.box_polygon(object_state.x, object_state.y, object_state.angle,
                                           OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0)
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
            "joint_action_index": self._joint_action_index(action_0.index, action_1.index),
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
            "true_edge_gap": {
                str(robot_id): (_point_polygon_distance(
                    (true_states[robot_id].x, true_states[robot_id].y), object_polygon)
                    - ROBOT_RADIUS)
                for robot_id in (0, 1)},
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
                "progress_guard": info.get("progress_guard"),
                "gap": info.get("gap"), "centre_gap": info.get("centre_gap"),
                "gap_target": info.get("gap_target"),
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
                "peer_clearance": info.get("peer_clearance"),
                "requested_kind": info.get("requested_kind"),
                "declared_direction": (list(info["declared_direction"])
                                       if info.get("declared_direction") else None),
                "signed_progress": info.get("signed_progress"),
                "positive_travel": info.get("positive_travel"),
                "backward_travel": info.get("backward_travel"),
                "progress_target_distance": info.get("progress_target_distance"),
                "option_timeout_seconds": info.get("option_timeout_seconds"),
                "face_switches": info.get("face_switches"),
                "hold": info.get("hold"), "speed_command": info.get("speed_command"),
                "ready": info.get("ready"), "release": info.get("release"),
                "cooperative": info.get("cooperative"),
                "corner_rejected": info.get("corner_rejected"),
                "follow_offset": info.get("follow_offset"),
                "follow_reference_offset": info.get("follow_reference_offset"),
                "follow_pose_error": info.get("follow_pose_error"),
                "follow_time": info.get("follow_time"),
                "follow_target_gap": info.get("follow_target_gap"),
                "follow_peer_guard": info.get("follow_peer_guard"),
                "follow_binding": info.get("follow_binding"),
                "force_peak": info.get("force_peak"),
                "anchor_status": (self._last_anchor[robot_id].status
                                  if self._last_anchor[robot_id] else None),
                "anchor_candidates": ([list(point) for point in
                                       self._last_anchor[robot_id].candidates]
                                      if self._last_anchor[robot_id] else []),
                "anchor_confidence": (self._last_anchor[robot_id].confidence
                                      if self._last_anchor[robot_id] else None),
                "anchor_observation_age_s": (
                    self._last_anchor[robot_id].observation_age_s
                    if self._last_anchor[robot_id] else None),
                "anchor_legal_candidates": (
                    self._last_anchor[robot_id].legal_candidates
                    if self._last_anchor[robot_id] else None),
                "anchor_shadow_boundaries": (
                    self._last_anchor[robot_id].shadow_boundaries
                    if self._last_anchor[robot_id] else None),
                "anchor_face_switch": (
                    self._last_anchor[robot_id].face_switch
                    if self._last_anchor[robot_id] else None),
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
                "effect": info.get("effect"),
                "boundary_follow": info.get("boundary_follow"),
                "peer_detour": info.get("peer_detour"),
                "static_route_guard": info.get("static_route_guard"),
                "command_disturbance": info.get("command_disturbance"),
                "completion_progress": info.get("completion_progress"),
                "completion_target": info.get("completion_target"),
            }
        return row

    # -- branching ---------------------------------------------------------
    def export_branch_state(self) -> Dict[str, object]:
        return {
            "env_version": ENV_VERSION,
            "action_version": self.action_version,
            "peer_clearance_recovery": self.config.peer_clearance_recovery,
            "lidar_backend": self.config.lidar_backend,
            "action_labels": [action.label for action in self.actions],
            "action_codebook_hash": self.action_codebook_hash,
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
            "anchor_state": {str(robot_id): self.anchor_trackers[robot_id].state()
                             for robot_id in (0, 1)},
            "heading": list(self.heading), "heading_rate": list(self.heading_rate),
        }

    def restore_branch_state(self, state: Dict[str, object], *, allow_execution_change: bool = False) -> None:
        self.lidar.invalidate()
        self._patch_cache.clear()
        if state.get('lidar_backend','scalar') != self.config.lidar_backend and not allow_execution_change:
            raise ValueError('Cannot restore a branch with a different lidar backend')
        if (state.get("peer_clearance_recovery", "off") != self.config.peer_clearance_recovery
                and not allow_execution_change):
            raise ValueError("Cannot restore a branch with a different peer-clearance controller")
        if state.get("env_version") != ENV_VERSION:
            raise ValueError("Cannot restore a branch from a different reward contract")
        if (state.get("action_version", ENV_VERSION) != self.action_version
                or state.get("action_labels", [a.label for a in ROBOT_ACTIONS])
                != [a.label for a in self.actions]):
            raise ValueError("Cannot restore a branch with a different action codebook")
        if state.get("action_codebook_hash", self.action_codebook_hash) != self.action_codebook_hash:
            raise ValueError("Cannot restore a branch with different action parameters")
        if "action_codebook_hash" not in state and not self._legacy_action_table:
            raise ValueError("Cannot restore unversioned parameters into a custom codebook")
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
        for robot_id, tracker_state in (state.get("anchor_state") or {}).items():
            self.anchor_trackers[int(robot_id)].load_state(tracker_state)
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
