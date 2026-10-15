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
        if self.skill_time >= SKILL_MAX_SECONDS:
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
            reason = self._safety(observation) or (None if self.skill_time < SKILL_MAX_SECONDS
                                                   else "timeout")
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
            elif self.find_elapsed >= FIND_SECONDS or observation.elapsed_seconds >= SKILL_MAX_SECONDS:
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
            self.push_wall += dt
            if patch is None:
                if self.patch_missing_seconds >= OBSERVATION_FAIL_TOLERANCE:
                    reason = "patch_lost"
                elif normal is not None:
                    raw = raw_reference(normal, self.alpha, self.speed_command)
            else:
                raw = raw_reference(patch.normal_world if normal is None else normal,
                                    self.alpha, self.speed_command)
            if observation.contacting_object:
                self.effective_contact += dt
                self.contact_seen = True
                self.contact_loss_seconds = 0.0
                self._mark_progress()
            else:
                self.contact_loss_seconds += dt
                if self.contact_seen and self.contact_loss_seconds >= CONTACT_LOSS_TOLERANCE:
                    reason = "contact_lost"
                elif (not self.contact_seen
                      and self.contact_loss_seconds >= CONTACT_SEEK_SECONDS):
                    reason = "blocked_no_progress"
            if reason is None and self.effective_contact >= PUSH_ACTIVE_SECONDS:
                reason = "push_complete"
            if reason is None and self.push_wall >= PUSH_WALL_SECONDS:
                reason = "push_wall_timeout"
        else:
            raw = (0.0, 0.0)

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
            "progress": self.progress, "progress_target": PUSH_ACTIVE_SECONDS,
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
                    previous = self.previous_position
                    if previous is not None and self.tangent is not None:
                        step_vector = (observation.own.x - previous[0],
                                       observation.own.y - previous[1])
                        advance = (step_vector[0] * self.tangent[0]
                                   + step_vector[1] * self.tangent[1])
                        if advance > 0.0 and not self.corner_event_now:
                            self.progress += advance
                            self._mark_progress()
                self.previous_position = (observation.own.x, observation.own.y)
            if reason is None and self.progress >= self.distance_target:
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
            "no_progress_seconds": self.no_progress_seconds,
            "find_direction": self.find_direction if self.stage == "find" else None,
            "skill_time": self.skill_time,
            "corner_event": self.corner_event_now, "corner_events": self.corner_events,
            "unintended_contact_seconds": self.unintended_contact,
            "safety_trigger": self.safety_trigger,
            "speed_at_exit": self.speed_at_exit,
            "residual_motion": bool(self.done and self.speed_at_exit > BRAKE_SPEED),
        }


def make_skill(action: RobotAction, force_limit: float = MAX_FORCE) -> Skill:
    if action.kind == "push":
        return PushSkill(action, force_limit)
    if action.kind == "reposition":
        return RepositionSkill(action, force_limit)
    if action.kind == "wait":
        return WaitBrakeSkill(action, force_limit)
    raise ValueError(f"unknown skill kind {action.kind!r}")
