"""The three skills, their observation whitelist and their exit contract.

One skill is one of the 15 discrete actions of one robot. The unified interface is

    skill = make_skill(action, force_limit)
    force, info = skill.step(observation, dt)      # one control period
    skill.done, skill.stage, skill.exit_reason     # status

The observation whitelist (task book section 3.1) is, and only is: the robot's own
kinematics, its teammate's shared kinematics, its own scan (returns plus fitted surface
fragments), its own contact reading, the public goal region, the public arena and the public
clock. Object truth, reward, success, hidden geometry and engine snapshots have no field on
``SkillObservation`` and are never read here.

Actuation is the frozen one: a force vector at the disc centre, clamped to ``MAX_FORCE``;
there is no torque channel, so a robot's heading is set by contact alone. RE POSITION is
therefore a translation of the disc along the observed surface, not a rotation in place.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..local_contact_option.control import (
    NormalFilter,
    ReferenceVelocity,
    clip_norm,
    raw_reference,
    velocity_feedback_force,
)
from . import geometry as G
from .lidar import CATEGORY_STATIC, CATEGORY_TEAMMATE, LidarScan, SurfacePatch
from .spec import (
    A_REF_MAX,
    APPROACH_GAIN,
    REPOSITION_ALIGN_ENTRY,
    REPOSITION_GAP_DEADBAND,
    CONTACT_SEEK_SECONDS,
    APPROACH_MAX_SPEED,
    APPROACH_MIN_SPEED,
    APPROACH_TIMEOUT,
    BRAKE_MAX_SECONDS,
    BRAKE_SPEED,
    CONTACT_LOSS_TOLERANCE,
    FIND_SECONDS,
    FIND_SPEED,
    KV,
    MAX_FORCE,
    NO_PROGRESS_SECONDS,
    NORMAL_FILTER_TAU,
    NORMAL_JUMP_LIMIT_DEG,
    OBSERVATION_FAIL_TOLERANCE,
    OBSTACLE_AVOID_GAIN,
    OBSTACLE_AVOID_MARGIN,
    PUSH_ACTIVE_SECONDS,
    PUSH_WALL_SECONDS,
    REPOSITION_ALIGN_TOLERANCE,
    REPOSITION_GAP,
    REPOSITION_GAP_GAIN,
    REPOSITION_SPEED,
    REPOSITION_SPEED_MAX,
    ROBOT_RADIUS,
    SAFETY_GAP,
    SKILL_MAX_SECONDS,
    FOLLOW_GAP,
    FOLLOW_GAP_GAIN,
    FOLLOW_POSE_TOLERANCE,
    FOLLOW_TANGENT_GAIN,
    OPTION_BUDGET_EXIT,
    FACE_SWITCH_EXIT,
    option_timeout,
    RobotAction,
)

Vec = Tuple[float, float]

STAGES = ("find", "approach", "push", "align", "move", "brake", "done", "wait")

EXITS = ("push_complete", "reposition_complete", "wait_complete", "blocked_no_progress",
         "not_observed", "patch_lost", "contact_lost", "push_wall_timeout", "timeout",
         "safety_abort", "blocked_contact", "environment_termination")


@dataclass(frozen=True)
class RobotKinematics:
    x: float
    y: float
    angle: float
    vx: float
    vy: float
    omega: float

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass(frozen=True)
class SkillObservation:
    """The complete, whitelisted input set of a skill controller."""

    robot_id: int
    tick: int
    time: float                     # absolute physical time of the sample, seconds
    own: RobotKinematics
    teammate: RobotKinematics
    scan: LidarScan
    patches: Tuple[SurfacePatch, ...]
    patch: Optional[SurfacePatch]
    contacting_object: bool
    contact_point_robot: Optional[Vec]
    contacting_static: bool
    contacting_teammate: bool
    goal_xy: Vec
    goal_radius: float
    arena: Tuple[float, float]
    force_limit: float
    remaining_seconds: float
    elapsed_seconds: float
    static_nearest: Optional[Tuple[float, Vec]]
    teammate_nearest: Optional[Tuple[float, Vec]]
    skill_parameters: Dict[str, float] = field(default_factory=dict)
    stage_hint: str = ""

    def forbidden_fields(self) -> Tuple[str, ...]:
        banned = ("object", "object_pose", "object_position", "object_velocity", "object_angle",
                  "object_corners", "object_size", "face_id", "true_state", "env",
                  "environment", "backend", "space", "reward", "success", "potential",
                  "snapshot", "goal_object")
        return tuple(name for name in banned if hasattr(self, name))


def _clip_velocity(value: Vec, limit: float) -> Vec:
    return clip_norm(value, limit)


def patch_gap(patch: SurfacePatch) -> float:
    """Edge-to-edge clearance to the observed surface.

    The distance to the *nearest individual return* is used, not the perpendicular distance
    to the fitted line: the foot of that perpendicular can leave the observed segment at a
    corner, where the line distance would understate the true clearance and let the robot
    drive through the corner. Returns are surface points, so their distance *is* a clearance.
    """

    return float(patch.nearest_distance - ROBOT_RADIUS)


class Skill:
    """Base class: shared clock, whitelist bookkeeping, exit bookkeeping."""

    kind = "skill"

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        self.action = action
        self.force_limit = force_limit
        #: v4.1: the option timeout is *derived* (T_prepare + slack*D/v_ref); the requested
        #: distance is the target and this is only the exit bound when it is not reached.
        self.time_budget: Optional[float] = option_timeout(action)
        #: signed push/transfer progress along the direction declared at execution start
        self.declared_direction: Optional[Vec] = None
        self.signed_progress = 0.0
        self.positive_travel = 0.0
        self.backward_travel = 0.0
        self.face_switches = 0
        self.previous_position: Optional[Vec] = None
        self.step_vector: Vec = (0.0, 0.0)
        self.stage = "find"
        self.exit_reason: Optional[str] = None
        self.done = False
        self.progress = 0.0
        self.skill_time = 0.0
        self.no_progress_seconds = 0.0
        self.brake_time = 0.0
        self.reference = ReferenceVelocity(a_ref_max=A_REF_MAX)
        self.normal = NormalFilter(tau=NORMAL_FILTER_TAU, jump_limit_deg=NORMAL_JUMP_LIMIT_DEG)
        self.speed_at_exit = 0.0
        self.exit_time: Optional[float] = None
        self.unobserved_seconds = 0.0
        self.patch_missing_seconds = 0.0
        self.find_direction: Vec = (1.0, 0.0)
        self.find_elapsed = 0.0
        self.corner_events = 0
        self.last_tangent: Optional[Vec] = None
        self.avoidance_active = False
        self.avoidance_direction: Optional[Vec] = None
        self.last_progress_wall: float = 0.0
        self.safety_trigger: Optional[Tuple[str, float]] = None

    # -- helpers -----------------------------------------------------------
    def _note_position(self, observation: SkillObservation) -> None:
        """One per period: remember this robot's displacement since the previous period."""

        current = (observation.own.x, observation.own.y)
        self.step_vector = ((current[0] - self.previous_position[0],
                             current[1] - self.previous_position[1])
                            if self.previous_position is not None else (0.0, 0.0))
        self.previous_position = current

    def _accumulate_signed_progress(self, direction: Vec) -> float:
        """Signed projection of this period's displacement on the declared direction."""

        if self.declared_direction is None:
            self.declared_direction = direction
        norm = math.hypot(direction[0], direction[1])
        if norm < 1e-9 or self.declared_direction is None:
            return 0.0
        anchor = self.declared_direction
        anchor_norm = math.hypot(anchor[0], anchor[1])
        if anchor_norm < 1e-9:
            return 0.0
        advance = (self.step_vector[0] * anchor[0]
                   + self.step_vector[1] * anchor[1]) / anchor_norm
        self.signed_progress += advance
        if advance > 0.0:
            self.positive_travel += advance
        else:
            self.backward_travel += -advance
        return advance

    def _progress_info(self) -> Dict[str, object]:
        return {
            "declared_direction": (list(self.declared_direction)
                                   if self.declared_direction is not None else None),
            "signed_progress": self.signed_progress,
            "positive_travel": self.positive_travel,
            "backward_travel": self.backward_travel,
            "progress_target_distance": (float(self.action.distance)
                                         if getattr(self.action, "scale_kind", "")
                                         == "distance" else None),
            "option_timeout_seconds": self.time_budget,
            "face_switches": self.face_switches,
        }

    def _mark_progress(self) -> None:
        self.no_progress_seconds = 0.0

    def _finish(self, reason: str, observation: SkillObservation) -> None:
        if self.done:
            return
        self.done = True
        self.exit_reason = reason
        self.exit_time = observation.time
        self.speed_at_exit = observation.own.speed
        self.stage = "done"

    def _safety(self, observation: SkillObservation) -> Optional[str]:
        self.safety_trigger = None
        if observation.static_nearest is not None:
            distance = observation.static_nearest[0]
            if distance <= ROBOT_RADIUS + SAFETY_GAP:
                self.safety_trigger = ("static", float(distance))
                return "safety_abort"
        if observation.teammate_nearest is not None:
            # the range is measured to the teammate's *surface*, so the edge-to-edge gap is
            # ``distance - ROBOT_RADIUS`` for the teammate exactly as for a static hit
            distance = observation.teammate_nearest[0]
            if distance <= ROBOT_RADIUS + SAFETY_GAP:
                self.safety_trigger = ("teammate", float(distance))
                return "safety_abort"
        return None

    def _common_checks(self, observation: SkillObservation, dt: float) -> Optional[str]:
        if self.time_budget is not None and self.skill_time >= self.time_budget:
            return OPTION_BUDGET_EXIT
        if self.skill_time >= (self.time_budget or 0.0) + SKILL_MAX_SECONDS:
            return "timeout"
        safety = self._safety(observation)
        if safety is not None:
            return safety
        return None

    def _obstacle_avoidance(self, observation: SkillObservation, raw: Vec,
                            include_teammate: bool = False) -> Vec:
        """Bounded repulsion from the nearest *measured* static return (and the teammate).

        Only measured quantities are used: the range to the nearest return of that class and
        this robot's own pose. The avoidance direction is recorded whenever it is actually
        applied, so the rendering layer never draws a repulsion arrow the controller did not
        use.
        """

        self.avoidance_active = False
        self.avoidance_direction = None
        total = (raw[0], raw[1])
        sources = [observation.static_nearest]
        if include_teammate:
            sources.append(observation.teammate_nearest)
        for nearest in sources:
            if nearest is None:
                continue
            distance, point = nearest
            gap = distance - ROBOT_RADIUS
            if gap > OBSTACLE_AVOID_MARGIN:
                continue
            away = G.unit((observation.own.x - point[0], observation.own.y - point[1]))
            strength = OBSTACLE_AVOID_GAIN * (OBSTACLE_AVOID_MARGIN - gap)
            total = (total[0] + strength * away[0], total[1] + strength * away[1])
            self.avoidance_active = True
            if self.avoidance_direction is None:
                self.avoidance_direction = away
            else:
                self.avoidance_direction = G.unit((self.avoidance_direction[0] + away[0],
                                                   self.avoidance_direction[1] + away[1]))
        return total

    def _track_patch(self, observation: SkillObservation, dt: float) -> Optional[Vec]:
        """Update the normal filter from the fitted patch (never from truth)."""

        patch = observation.patch
        normal = self.normal.update(patch.normal_world if patch is not None else None, dt)
        if patch is None or not patch.valid:
            self.patch_missing_seconds += dt
            if normal is None:
                self.unobserved_seconds += dt
        else:
            self.patch_missing_seconds = 0.0
            self.unobserved_seconds = 0.0
        return normal if (patch is not None and patch.valid) else normal

    def _drive(self, raw: Vec, dt: float) -> Tuple[Vec, bool]:
        """Rate-limit a raw velocity reference and turn it into a bounded force."""

        reference = self.reference.update(raw, dt)
        return reference

    def _force_from_reference(self, reference: Vec, observation: SkillObservation) -> Tuple[Vec, bool]:
        kinematics = (observation.own.vx, observation.own.vy)
        force = velocity_feedback_force(reference, kinematics, KV, self.force_limit)
        norm = math.hypot(force[0], force[1])
        return force, bool(norm >= self.force_limit - 1e-9)

    # -- interface ---------------------------------------------------------
    def step(self, observation: SkillObservation, dt: float) -> Tuple[Vec, Dict[str, object]]:
        raise NotImplementedError


class HoldBrake(Skill):
    """The hold command a finished robot receives while its teammate keeps executing."""

    kind = "hold"

    def __init__(self, force_limit: float = MAX_FORCE):
        super().__init__(RobotAction(kind="wait", duration=0.0, label="HOLD"), force_limit)
        self.stage = "brake"

    def step(self, observation: SkillObservation, dt: float) -> Tuple[Vec, Dict[str, object]]:
        self.skill_time += dt
        reference = self.reference.update((0.0, 0.0), dt)
        force, saturated = self._force_from_reference(reference, observation)
        info = {
            "skill": "HOLD", "kind": "hold", "stage": "brake", "exit_reason": None,
            "done": False, "desired_velocity": (0.0, 0.0), "reference_velocity": reference,
            "force": force, "force_norm": math.hypot(*force), "force_saturated": saturated,
            "progress": 0.0, "progress_target": 0.0,
            "speed_at_exit": None, "residual_motion": False,
        }
        return force, info


class WaitBrakeSkill(Skill):
    """WAIT_BRAKE: brake with the existing law and hold; the world is never frozen."""

    kind = "wait"

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        super().__init__(action, force_limit)
        self.stage = "brake"
        self.duration = float(action.duration)

    def step(self, observation: SkillObservation, dt: float) -> Tuple[Vec, Dict[str, object]]:
        self.skill_time += dt
        self.progress += dt
        self._mark_progress()
        reference = self.reference.update((0.0, 0.0), dt)
        force, saturated = self._force_from_reference(reference, observation)
        force = self._obstacle_avoidance(observation, force)
        reason = None
        if self.progress >= self.duration:
            reason = "wait_complete"
        else:
            reason = self._safety(observation) or (
                None if self.skill_time < (self.time_budget or SKILL_MAX_SECONDS) else "timeout")
        if reason is not None:
            self._finish(reason, observation)
        info = self._info(observation, force, saturated, reference, (0.0, 0.0))
        return force, info

    def _info(self, observation, force, saturated, reference, raw) -> Dict[str, object]:
        return {
            "skill": self.action.label, "kind": self.kind, "stage": self.stage,
            "exit_reason": self.exit_reason, "done": self.done,
            "desired_velocity": raw, "reference_velocity": reference, "force": force,
            "force_norm": math.hypot(*force), "force_saturated": saturated,
            "progress": self.progress, "progress_target": self.duration,
            "desired_heading": None,
            "push_time": self.skill_time, "speed_at_exit": self.speed_at_exit,
            "residual_motion": bool(self.done and self.speed_at_exit > BRAKE_SPEED),
        }


class _VisionSkill(Skill):
    """Shared seek/approach/brake machinery of PUSH and RE POSITION."""

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        super().__init__(action, force_limit)
        self.find_started = 0.0
        self.find_index = 0
        self.find_heading0: Optional[float] = None
        self._last_gap: Optional[float] = None

    # -- FIND --------------------------------------------------------------
    def _find(self, observation: SkillObservation) -> Vec:
        """Bounded local seek: translate slowly along eight directions in turn.

        The robot's heading is not actuated (force only at the disc centre), so a search is a
        translation, not a rotation. The pattern is expressed in the world frame, uses only
        the known initial heading of this robot, and is bounded by ``FIND_SECONDS``.
        """

        if self.find_heading0 is None:
            self.find_heading0 = observation.own.angle
        self.find_elapsed += 1.0 / 20.0
        self.find_index = int(self.find_elapsed / 0.2) % 8
        angle = self.find_heading0 + self.find_index * math.pi / 4.0
        self.find_direction = (math.cos(angle), math.sin(angle))
        return (FIND_SPEED * self.find_direction[0], FIND_SPEED * self.find_direction[1])

    # -- APPROACH ----------------------------------------------------------
    def _segment_extent(self, patch: SurfacePatch):
        tangent = patch.tangent_world
        projections = [point[0] * tangent[0] + point[1] * tangent[1]
                       for point in patch.points_world]
        return min(projections), max(projections), tangent

    def _approach(self, observation: SkillObservation, patch: SurfacePatch,
                  gap_target: float) -> Vec:
        gap = patch_gap(patch)
        error = gap - gap_target
        speed = max(APPROACH_MIN_SPEED, min(APPROACH_MAX_SPEED, APPROACH_GAIN * error))
        if error <= 0.0:
            speed = -max(APPROACH_MIN_SPEED, min(APPROACH_MAX_SPEED, -APPROACH_GAIN * error))
        return (speed * patch.normal_world[0], speed * patch.normal_world[1])

    def _track_gap_progress(self, gap: float) -> None:
        if self._last_gap is None or gap < self._last_gap - 1e-3:
            self._mark_progress()
        self._last_gap = min(gap, self._last_gap) if self._last_gap is not None else gap


class PushSkill(_VisionSkill):
    """PUSH: seek the object, approach the visible surface, push along the estimated normal."""

    kind = "push"

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        super().__init__(action, force_limit)
        self.stage = "find"
        self.alpha = math.radians(float(action.alpha_deg))
        self.speed_command = float(action.speed)
        self.effective_contact = 0.0
        self.contact_seen = False
        self.contact_loss_seconds = 0.0
        self.push_wall = 0.0
        self.approach_time = 0.0
        self.corner_rejected = False
        self.reobserve_time = 0.0
        #: set by the environment while a cooperative push waits for the teammate to start
        self.hold = False

    def _tangential_reference(self, observation: SkillObservation,
                              patch: SurfacePatch) -> Vec:
        """Candidate extension point: extra reference velocity along the visible tangent.

        The default is a no-op, so the frozen baseline controller is unchanged. A candidate
        controller may override it (for example to aim the contact point inside the visible
        segment instead of at a corner). It never changes the observation or the safety gate.
        """

        return (0.0, 0.0)

    def step(self, observation: SkillObservation, dt: float) -> Tuple[Vec, Dict[str, object]]:
        self.skill_time += dt
        self.no_progress_seconds += dt
        normal = self._track_patch(observation, dt)
        patch = observation.patch if (observation.patch is not None
                                      and observation.patch.valid) else None
        raw: Vec = (0.0, 0.0)
        reason: Optional[str] = None

        if self.stage == "find":
            raw = self._find(observation)
            if patch is not None:
                self._mark_progress()
                self.stage = "approach"
            elif self.find_elapsed >= FIND_SECONDS:
                reason = "not_observed"
        elif self.stage == "approach":
            self.approach_time += dt
            if patch is None:
                if self.patch_missing_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
                elif self.unobserved_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "not_observed"
                raw = (0.0, 0.0)
            else:
                gap = patch_gap(patch)
                self._track_gap_progress(gap)
                raw = self._approach(observation, patch, 0.0)
                # v4.2/M2: never approach a corner. The contact region is the visible segment
                # inset by R; while the projection is outside it the tangential correction gets
                # priority and the radial speed is reduced, so the robot enters the face instead
                # of driving into the corner.
                low, high, tangent = self._segment_extent(patch)
                robot_proj = (observation.own.x * tangent[0]
                              + observation.own.y * tangent[1])
                margin = ROBOT_RADIUS + 0.02
                if high - low < 2.0 * margin:
                    # too short to know a face: do not push, move along the boundary to
                    # re-observe; only give up after a bounded attempt
                    self.reobserve_time += dt
                    midpoint = 0.5 * (low + high)
                    sign = 1.0 if midpoint >= robot_proj else -1.0
                    raw = (sign * REPOSITION_SPEED * tangent[0],
                           sign * REPOSITION_SPEED * tangent[1])
                    self.corner_rejected = True
                    if self.reobserve_time >= 2.0:
                        reason = "insufficient_visible_segment"
                elif not (low + margin <= robot_proj <= high - margin):
                    # only a *near-boundary* projection is treated as a corner; being far
                    # outside the interval is a normal approach along the face
                    outside = min(abs(robot_proj - (low + margin)),
                                  abs(robot_proj - (high - margin)))
                    self.corner_rejected = outside <= 0.10
                    if self.corner_rejected:
                        raw = (0.35 * raw[0], 0.35 * raw[1])
                else:
                    self.corner_rejected = False
                # PUSH starts on a sensed contact, not on a range threshold: the range is the
                # distance to the *fitted plane*, which a convex corner can make look closed
                # while the surfaces are still a few millimetres apart. The approach keeps a
                # minimum inward speed, so the last millimetres are always closed.
                if observation.contacting_object:
                    self.stage = "push"
                    self.contact_seen = True
                    self.push_wall = 0.0
                    self._mark_progress()
                elif self.approach_time >= APPROACH_TIMEOUT:
                    reason = "blocked_no_progress"
        elif self.stage == "push":
            if not self.hold:
                self.push_wall += dt
            if patch is None:
                if self.patch_missing_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
                elif normal is not None:
                    raw = raw_reference(normal, self.alpha, self.speed_command)
            else:
                raw = raw_reference(patch.normal_world if normal is None else normal,
                                    self.alpha, self.speed_command)
            if self.hold:
                # keep a light contact while waiting for the teammate: a zero-force hold loses
                # the contact (the object keeps moving) and the branch then aborts contact_lost
                direction = (patch.normal_world if patch is not None and patch.valid
                             else normal)
                if direction is not None:
                    raw = raw_reference(direction, self.alpha,
                                        min(0.05, self.speed_command))
                else:
                    raw = (0.0, 0.0)
            if observation.contacting_object and not self.hold:
                self.effective_contact += dt
                self.contact_seen = True
                self.contact_loss_seconds = 0.0
                direction = (patch.normal_world if patch is not None and patch.valid
                             else (normal if normal is not None else None))
                if direction is not None:
                    self._accumulate_signed_progress(direction)
                if self.normal.jump:
                    self.face_switches += 1
                    reason = FACE_SWITCH_EXIT
                self._mark_progress()
            else:
                self.contact_loss_seconds += dt
                if self.contact_seen and self.contact_loss_seconds >= CONTACT_LOSS_TOLERANCE:
                    reason = "contact_lost"
                elif (not self.contact_seen
                      and self.contact_loss_seconds >= CONTACT_SEEK_SECONDS):
                    reason = "blocked_no_progress"
            if (reason is None and self.action.scale_kind == "distance"
                    and self.signed_progress >= float(self.action.distance)):
                reason = "push_complete"
            # v4.1: the push stage is bounded by the *derived option timeout*, not the fixed
            # 3 s wall; otherwise a 1.00 m requested distance could never be reached.
            if reason is None and self.push_wall >= (self.time_budget or PUSH_WALL_SECONDS):
                reason = "push_wall_timeout"
        else:
            raw = (0.0, 0.0)

        if reason is None and patch is not None and self.stage in ("approach", "push"):
            correction = self._tangential_reference(observation, patch)
            raw = (raw[0] + correction[0], raw[1] + correction[1])
        if reason is None:
            reason = self._common_checks(observation, dt)
        if reason is None and self.stage != "find" and self.no_progress_seconds >= NO_PROGRESS_SECONDS:
            reason = "blocked_no_progress"

        raw = self._obstacle_avoidance(observation, raw)
        raw = _clip_velocity(raw, self.speed_command if self.stage == "push"
                             else max(APPROACH_MAX_SPEED, self.speed_command))
        reference = self._drive(raw, dt)
        force, saturated = self._force_from_reference(reference, observation)
        self.progress = self.effective_contact
        if reason is not None:
            self._finish(reason, observation)
        info = self._info(observation, force, saturated, reference, raw)
        return force, info

    def _info(self, observation, force, saturated, reference, raw) -> Dict[str, object]:
        patch = observation.patch
        return {
            "skill": self.action.label, "kind": self.kind, "stage": self.stage,
            "exit_reason": self.exit_reason, "done": self.done,
            "desired_velocity": raw, "reference_velocity": reference, "force": force,
            "force_norm": math.hypot(*force), "force_saturated": saturated,
            "normal_world": patch.normal_world if patch is not None else None,
            "normal_valid": bool(patch is not None and patch.valid),
            "desired_heading": (patch.normal_world
                                if patch is not None and patch.valid else None),
            "normal_source": "fitted_returns" if patch is not None else None,
            "tangent_world": patch.tangent_world if patch is not None else None,
            "tangent_direction": None,
            "gap": patch_gap(patch) if patch is not None else None,
            "gap_target": 0.0,
            "gap_error": patch_gap(patch) if patch is not None else None,
            "patch_valid": bool(patch is not None and patch.valid),
            "patch_arc_rays": patch.arc_rays if patch is not None else 0,
            "patch_rms": patch.rms_m if patch is not None else None,
            "patch_confidence": patch.confidence if patch is not None else None,
            "contacting_object": observation.contacting_object,
            "contact_point_robot": observation.contact_point_robot,
            "obstacle_nearest_distance": (observation.static_nearest[0]
                                          if observation.static_nearest else None),
            "obstacle_nearest_point": (observation.static_nearest[1]
                                       if observation.static_nearest else None),
            "teammate_nearest_distance": (observation.teammate_nearest[0]
                                          if observation.teammate_nearest else None),
            "avoidance_active": self.avoidance_active,
            "avoidance_direction": self.avoidance_direction,
            "push_alpha_deg": float(self.action.alpha_deg),
            "push_speed": self.speed_command,
            "effective_contact_seconds": self.effective_contact,
            "push_wall_seconds": self.push_wall,
            "corner_rejected": self.corner_rejected, "reobserve_time": self.reobserve_time,
            "hold": self.hold, "speed_command": self.speed_command,
            "edge_assigned": not self.corner_rejected,
            "progress": self.progress, "progress_target": PUSH_ACTIVE_SECONDS,
            **self._progress_info(),
            "no_progress_seconds": self.no_progress_seconds,
            "find_direction": self.find_direction if self.stage == "find" else None,
            "skill_time": self.skill_time, "corner_event": False, "corner_events": 0,
            "unintended_contact_seconds": 0.0,
            "safety_trigger": self.safety_trigger,
            "speed_at_exit": self.speed_at_exit,
            "residual_motion": bool(self.done and self.speed_at_exit > BRAKE_SPEED),
        }


class RepositionSkill(_VisionSkill):
    """RE POSITION_CW / _CCW: non-contact motion along the visible boundary.

    Sign convention (global right-handed frame, +x right, +y up, angles CCW):

    * ``n`` is the fitted surface normal *outward from the object*, i.e. from the surface
      towards the robot. It is the negated fitted approach normal, so it is still estimated
      only from returns.
    * the CCW tangent is ``rotate90_ccw(n) = (-n_y, n_x)``; the CW tangent is its negation.
      Going CW therefore means orbiting the object in the clockwise direction seen from above.
    * progress is the accumulated positive projection of this robot's own displacement on the
      tangent that was valid in that period. It is not accumulated while the object is in
      contact, while the fit is invalid, or when the displacement is backwards.
    """

    kind = "reposition"

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        super().__init__(action, force_limit)
        self.stage = "find"
        self.cw = str(action.direction) == "CW"
        self.distance_target = float(action.distance)
        self.tangent: Optional[Vec] = None
        self.tangent_direction = "CW" if self.cw else "CCW"
        self.previous_position: Optional[Vec] = None
        self.align_time = 0.0
        self.corner_event_now = False
        self.unintended_contact = 0.0
        self.gap = None
        self.move_time = 0.0

    def _tangent_from(self, patch: SurfacePatch) -> Vec:
        outward = patch.normal_outward
        ccw = G.rotate90(outward)
        return (-ccw[0], -ccw[1]) if self.cw else ccw

    def step(self, observation: SkillObservation, dt: float) -> Tuple[Vec, Dict[str, object]]:
        self.skill_time += dt
        self.no_progress_seconds += dt
        self.corner_event_now = False
        self._note_position(observation)
        normal_raw = self.normal.update(
            observation.patch.normal_world if (observation.patch is not None
                                               and observation.patch.valid) else None, dt)
        patch = observation.patch if (observation.patch is not None
                                      and observation.patch.valid) else None
        jump = bool(self.normal.jump)
        if patch is None:
            self.patch_missing_seconds += dt
            self.unobserved_seconds += dt
        else:
            self.patch_missing_seconds = 0.0
            self.unobserved_seconds = 0.0
        raw: Vec = (0.0, 0.0)
        reason: Optional[str] = None
        self.avoidance_active = False
        self.avoidance_direction = None

        if self.stage == "find":
            raw = self._find(observation)
            if patch is not None:
                self._mark_progress()
                self.stage = "align"
                self.previous_position = (observation.own.x, observation.own.y)
            elif self.find_elapsed >= FIND_SECONDS:
                reason = "not_observed"
        elif self.stage == "align":
            self.align_time += dt
            if patch is None:
                if self.unobserved_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
            else:
                outward = patch.normal_outward
                gap = patch_gap(patch)
                self.gap = gap
                error = gap - REPOSITION_GAP
                if abs(error) <= REPOSITION_ALIGN_ENTRY:
                    self.stage = "move"
                    self._mark_progress()
                    self.previous_position = (observation.own.x, observation.own.y)
                else:
                    # ``error > 0`` means the robot is further from the surface than the target
                    # gap, so the correction is *inwards*: the outward normal points from the
                    # surface towards the robot, hence the minus sign.
                    speed = max(-REPOSITION_SPEED_MAX, min(REPOSITION_SPEED_MAX,
                                                           REPOSITION_GAP_GAIN * error))
                    raw = (-speed * outward[0], -speed * outward[1])
                    self._mark_progress()
        elif self.stage == "move":
            self.move_time += dt
            if patch is None:
                if self.unobserved_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
                elif self.patch_missing_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
            else:
                candidate = self._tangent_from(patch)
                if jump or (self.last_tangent is not None
                            and candidate[0] * self.last_tangent[0]
                            + candidate[1] * self.last_tangent[1] < 0.0):
                    # a corner: keep the previous direction, count the transition, do not
                    # accumulate progress from a normal that just jumped
                    self.corner_event_now = True
                    self.corner_events += 1
                    candidate = self.last_tangent if self.last_tangent is not None else candidate
                    self.tangent = candidate
                    raw = (REPOSITION_SPEED * candidate[0], REPOSITION_SPEED * candidate[1])
                else:
                    self.last_tangent = candidate
                    self.tangent = candidate
                    gap = patch_gap(patch)
                    self.gap = gap
                    error = gap - REPOSITION_GAP
                    if abs(error) <= REPOSITION_GAP_DEADBAND:
                        error = 0.0
                    correction = REPOSITION_GAP_GAIN * error
                    raw = (REPOSITION_SPEED * candidate[0] - correction * patch.normal_outward[0],
                           REPOSITION_SPEED * candidate[1] - correction * patch.normal_outward[1])
                if observation.contacting_object:
                    # the skill intends non-contact motion; a real contact is recorded and the
                    # progress is frozen, so object motion can never be booked as repositioning
                    self.unintended_contact += dt
                    if self.unintended_contact >= 0.50:
                        reason = "blocked_contact"
                else:
                    self.unintended_contact = 0.0
                    if self.tangent is not None:
                        advance = self._accumulate_signed_progress(self.tangent)
                        if advance > 0.0 and not self.corner_event_now:
                            self.progress += advance
                            self._mark_progress()
                    if self.corner_event_now:
                        self.face_switches += 1
                        reason = FACE_SWITCH_EXIT
            if (reason is None and self.action.scale_kind == "distance"
                    and self.signed_progress >= float(self.action.distance)):
                reason = "reposition_complete"
        else:
            raw = (0.0, 0.0)

        if reason is None:
            reason = self._common_checks(observation, dt)
        if reason is None and self.stage != "find" and self.no_progress_seconds >= NO_PROGRESS_SECONDS:
            reason = "blocked_no_progress"

        raw = self._obstacle_avoidance(observation, raw, include_teammate=True)
        raw = _clip_velocity(raw, REPOSITION_SPEED_MAX if self.stage == "move"
                             else max(APPROACH_MAX_SPEED, REPOSITION_SPEED_MAX))
        reference = self._drive(raw, dt)
        force, saturated = self._force_from_reference(reference, observation)
        if reason is not None:
            self._finish(reason, observation)
        info = self._info(observation, force, saturated, reference, raw)
        return force, info

    def _info(self, observation, force, saturated, reference, raw) -> Dict[str, object]:
        patch = observation.patch
        outward = patch.normal_outward if patch is not None else None
        return {
            "skill": self.action.label, "kind": self.kind, "stage": self.stage,
            "exit_reason": self.exit_reason, "done": self.done,
            "desired_velocity": raw, "reference_velocity": reference, "force": force,
            "force_norm": math.hypot(*force), "force_saturated": saturated,
            "normal_world": patch.normal_world if patch is not None else None,
            "normal_valid": bool(patch is not None and patch.valid),
            "normal_source": "fitted_returns" if patch is not None else None,
            "outward_normal": outward,
            "tangent_world": self.tangent,
            "tangent_direction": self.tangent_direction,
            "gap": self.gap,
            "gap_target": REPOSITION_GAP,
            "gap_error": (None if self.gap is None else self.gap - REPOSITION_GAP),
            "patch_valid": bool(patch is not None and patch.valid),
            "patch_arc_rays": patch.arc_rays if patch is not None else 0,
            "patch_rms": patch.rms_m if patch is not None else None,
            "patch_confidence": patch.confidence if patch is not None else None,
            "contacting_object": observation.contacting_object,
            "contact_point_robot": observation.contact_point_robot,
            "obstacle_nearest_distance": (observation.static_nearest[0]
                                          if observation.static_nearest else None),
            "obstacle_nearest_point": (observation.static_nearest[1]
                                       if observation.static_nearest else None),
            "teammate_nearest_distance": (observation.teammate_nearest[0]
                                          if observation.teammate_nearest else None),
            "avoidance_active": self.avoidance_active,
            "avoidance_direction": self.avoidance_direction,
            "push_alpha_deg": None, "push_speed": None, "effective_contact_seconds": None,
            "push_wall_seconds": None,
            "progress": self.progress, "progress_target": self.distance_target,
            **self._progress_info(),
            "desired_heading": self.tangent,
            "no_progress_seconds": self.no_progress_seconds,
            "find_direction": self.find_direction if self.stage == "find" else None,
            "skill_time": self.skill_time,
            "corner_event": self.corner_event_now, "corner_events": self.corner_events,
            "unintended_contact_seconds": self.unintended_contact,
            "safety_trigger": self.safety_trigger,
            "speed_at_exit": self.speed_at_exit,
            "residual_motion": bool(self.done and self.speed_at_exit > BRAKE_SPEED),
        }


class SupportFollowSkill(_VisionSkill):
    """SUPPORT_FOLLOW v1: hold the relative pose to the visible surface; never push.

    The normal term is a **gap servo** (at the target clearance it commands no normal force),
    and the tangential term nulls the offset to the reference point, so the robot follows the
    object as it moves without adding sustained push. The force is additionally capped low.
    """

    kind = "follow"

    def __init__(self, action: RobotAction, force_limit: float = MAX_FORCE):
        super().__init__(action, force_limit)
        self.stage = "find"
        self.reference_offset: Optional[float] = None
        self.follow_time = 0.0
        self.last_gap: Optional[float] = None
        self.gap: Optional[float] = None
        self.centre_gap: Optional[float] = None
        self.offset: Optional[float] = None
        self.pose_error: Optional[float] = None
        self.force_peak = 0.0

    def _frame(self, observation: SkillObservation, patch: SurfacePatch):
        """The control gap is the **edge** clearance, not the centre-to-line distance.

        ``patch.distance`` is the robot-centre distance to the fitted infinite line; on a flat
        face the physical edge gap is ``nearest_distance - ROBOT_RADIUS`` (nearest_distance is
        the clearance to the surface and stays correct when the perpendicular foot leaves the
        visible segment). The two are reported separately so the offline check can compare the
        control gap with the true edge gap.
        """
        tangent = patch.tangent_world
        projections = [point[0] * tangent[0] + point[1] * tangent[1]
                       for point in patch.points_world]
        # the centroid is far more stable than the min/max midpoint when occlusion changes the
        # visible span (the midpoint drifts and creates a fictitious tangential error)
        reference = sum(projections) / max(1, len(projections))
        robot = observation.own.x * tangent[0] + observation.own.y * tangent[1]
        centre_gap = float(patch.nearest_distance)
        edge_gap = centre_gap - ROBOT_RADIUS
        self.centre_gap = centre_gap
        return (edge_gap, robot - reference, tangent, patch.normal_world)

    def step(self, observation: SkillObservation, dt: float):
        self.skill_time += dt
        self.no_progress_seconds += dt
        self._note_position(observation)
        patch = observation.patch if (observation.patch is not None
                                      and observation.patch.valid) else None
        raw: Vec = (0.0, 0.0)
        reason: Optional[str] = None
        if patch is None:
            raw = self._find(observation)
            self.patch_missing_seconds += dt
            if self.patch_missing_seconds >= OBSERVATION_FAIL_TOLERANCE:
                reason = "patch_lost"
        else:
            self.patch_missing_seconds = 0.0
            gap, offset, tangent, normal = self._frame(observation, patch)
            self.gap, self.offset = gap, offset
            radial = FOLLOW_GAP_GAIN * (gap - FOLLOW_GAP)
            if self.stage == "find":
                self.stage = "align"
            if self.stage == "align":
                raw = (radial * normal[0], radial * normal[1])
                # enter FOLLOW once the gap is workably close; the gap servo finishes the
                # approach, so a 0.02 m entry would burn most of the follow budget
                if abs(gap - FOLLOW_GAP) <= 0.08:
                    self.reference_offset = offset
                    self.stage = "follow"
            if self.stage == "follow":
                self.follow_time += dt
                error = offset - (self.reference_offset or 0.0)
                tangential = -FOLLOW_TANGENT_GAIN * error
                # radial velocity feedforward from the robot's own gap measurement: without it
                # the P term lags a moving object and the gap error grows over a long follow
                if self.last_gap is not None and dt > 0.0:
                    radial += 0.6 * (gap - self.last_gap) / dt
                self.last_gap = gap
                radial = max(-REPOSITION_SPEED_MAX, min(REPOSITION_SPEED_MAX, radial))
                raw = (radial * normal[0] + tangential * tangent[0],
                       radial * normal[1] + tangential * tangent[1])
                self.pose_error = math.hypot(gap - FOLLOW_GAP, error)
                if self.pose_error <= FOLLOW_POSE_TOLERANCE:
                    self._mark_progress()
                if self.follow_time >= (self.time_budget or 0.0):
                    reason = "follow_complete"
        if reason is None:
            reason = self._common_checks(observation, dt)
        raw = self._obstacle_avoidance(observation, raw)
        raw = _clip_velocity(raw, REPOSITION_SPEED_MAX)
        reference = self._drive(raw, dt)
        force, saturated = self._force_from_reference(reference, observation)
        force = (force[0] * 0.35, force[1] * 0.35)      # v1: bounded, no sustained push
        self.force_peak = max(self.force_peak, math.hypot(*force))
        if reason is not None:
            self._finish(reason, observation)
        info = self._info(observation, force, saturated, reference, raw)
        return force, info

    def _info(self, observation, force, saturated, reference, raw) -> Dict[str, object]:
        patch = observation.patch
        return {
            "skill": self.action.label, "kind": self.kind, "stage": self.stage,
            "exit_reason": self.exit_reason, "done": self.done,
            "desired_velocity": raw, "reference_velocity": reference, "force": force,
            "force_norm": math.hypot(*force), "force_saturated": saturated,
            "normal_world": patch.normal_world if patch is not None else None,
            "normal_valid": bool(patch is not None and patch.valid),
            "tangent_world": patch.tangent_world if patch is not None else None,
            "gap": self.gap, "centre_gap": self.centre_gap, "follow_offset": self.offset,
            "follow_reference_offset": self.reference_offset,
            "follow_pose_error": self.pose_error, "follow_time": self.follow_time,
            "force_peak": self.force_peak,
            "contacting_object": observation.contacting_object,
            "contact_point_robot": observation.contact_point_robot,
            "obstacle_nearest_distance": (observation.static_nearest[0]
                                          if observation.static_nearest else None),
            "teammate_nearest_distance": (observation.teammate_nearest[0]
                                          if observation.teammate_nearest else None),
            "avoidance_active": self.avoidance_active,
            "avoidance_direction": self.avoidance_direction,
            "patch_valid": bool(patch is not None and patch.valid),
            "safety_trigger": self.safety_trigger, "speed_at_exit": self.speed_at_exit,
            "residual_motion": bool(self.done and self.speed_at_exit > BRAKE_SPEED),
            **self._progress_info(),
        }


def make_skill(action: RobotAction, force_limit: float = MAX_FORCE) -> Skill:
    if action.kind == "push":
        return PushSkill(action, force_limit)
    if action.kind == "reposition":
        return RepositionSkill(action, force_limit)
    if action.kind == "follow":
        return SupportFollowSkill(action, force_limit)
    if action.kind == "wait":
        return WaitBrakeSkill(action, force_limit)
    raise ValueError(f"unknown skill kind {action.kind!r}")
