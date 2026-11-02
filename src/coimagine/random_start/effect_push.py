"""Short effect-directed pushes using only a robot's own observed surface history.

Opt-in controller/codebook for the existing RandomStartEnv, not a second physics model.
The rectangle dimensions below are declared controller priors, never instance truth.
Geometric availability is deliberately not a promise about hidden friction or success.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Optional, Tuple

import numpy as np

from . import spec
from .lidar import CATEGORY_STATIC, CATEGORY_TEAMMATE, SurfacePatch
from .endpoint_tracking import EndpointTracker, TrackedFace, interpolate_face, observe_endpoints
from .same_face import SameFaceReservation
from .progress_guard import ProgressGuard
from .skills import PushSkill, RepositionSkill, SupportFollowSkill, SkillObservation, make_skill, patch_gap, _wrap_angle

Vec = Tuple[float, float]
VERSION = "effect-push-v8-joint-recovery"
REPOSITION_V2_VERSION = "effect-options-v9-reposition-scales"
REPOSITION_DISTANCES_V2 = (.20,.40,.80,1.20,1.80)
DISTANCES = (0.10, 0.20, 0.30, 0.40, 0.50)
ROTATION_ANGLES = (0.10, 0.20, 0.30, 0.40, 0.50)  # radians, not metres
DIRECTION_NAMES = ("E", "NE", "N", "NW", "W", "SW", "S", "SE")
DIRECTIONS = tuple((math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)) for i in range(8))


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, x):
    return (a[0] * x, a[1] * x)


def unit(a):
    norm = math.hypot(*a)
    return mul(a, 1.0 / norm) if norm > 1e-9 else (0.0, 0.0)


def tangent(n):
    return (-n[1], n[0])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class EffectAction(spec.RobotAction):
    effect_kind: str = "translate"
    effect_direction: Vec = (0.0, 1.0)
    rotation_sign: int = 0
    rotation_angle: float = 0.0

    @property
    def coordination_key(self):
        return (self.effect_kind, self.effect_direction, self.rotation_sign)

    @property
    def effect_target(self):
        return self.distance if self.effect_kind == "translate" else self.rotation_angle


def action_table(reposition_distances=None):
    """Keep the default 66-action menu; explicit new RP scales add versioned slots."""
    out = []
    for i, direction in enumerate(DIRECTIONS):
        for j, distance in enumerate(DISTANCES):
            out.append(EffectAction(
                kind="push", speed=0.12, distance=distance, scale=distance,
                scale_kind="distance", type_index=i, scale_index=j, index=len(out),
                label=f"PUSH_{DIRECTION_NAMES[i]}[{distance:.2f}m]", effect_direction=direction))
    for sign, name in ((1, "CCW"), (-1, "CW")):
        for j, angle in enumerate(ROTATION_ANGLES):
            out.append(EffectAction(
                kind="push", speed=0.12, scale=angle, scale_kind="angle",
                type_index=8 if sign == 1 else 9, scale_index=j, index=len(out),
                label=f"ROTATE_{name}[{angle:.2f}rad]", effect_kind="rotate",
                rotation_sign=sign, rotation_angle=angle))
    for action in spec.ROBOT_ACTIONS:
        if action.kind != "push":
            if action.kind == "reposition" and reposition_distances is not None:
                if action.scale_index != 0:
                    continue
                for j,distance in enumerate(reposition_distances):
                    out.append(replace(action,index=len(out),type_index=10+action.type_index-3,
                        scale_index=j,scale=distance,distance=distance,label=f"REPOSITION_{action.direction}[{distance:.2f}m]"))
            else:
                out.append(replace(action, index=len(out), type_index=10 + action.type_index - 3))
    return tuple(out)


ACTIONS = action_table()
ACTIONS_REPOSITION_V2 = action_table(REPOSITION_DISTANCES_V2)
PUSH_BOUND_FOLLOW_VERSION = "effect-options-v10-push-bound-follow"
# Retain all 64 PUSH/RP/WAIT entries; FOLLOW has one event-bound slot, no seconds.
ACTIONS_PUSH_BOUND_FOLLOW = tuple(
    replace(a, label="SUPPORT_FOLLOW_PUSH", duration=0., scale=0., scale_kind="peer_push")
    if a.kind == "follow" else a
    for a in ACTIONS_REPOSITION_V2 if a.kind != "follow" or a.scale_index == 0)
EFFICIENT_OPTION_VERSION = "effect-options-v11-efficient-options"
# A two-second WAIT remains available, and holds after its own completion while
# an active peer executes. Longer idle is expressible as repeated decisions.
ACTIONS_EFFICIENT = tuple(replace(a,index=i) for i,a in enumerate(
    a for a in ACTIONS_PUSH_BOUND_FOLLOW if a.kind!='wait' or a.scale_index==0))
TYPE_COUNT = 14
SCALE_COUNT = 5
_PAIR_TO_ID = {(a.type_index, a.scale_index): a.index for a in ACTIONS}
WAIT_INDEX = next(a.index for a in ACTIONS if a.kind == "wait" and a.duration == 2.0)


def action_id(type_index, scale_index, *, actions=ACTIONS):
    """Table-specific ragged lookup; the versioned RP menu has five distance slots."""
    try:
        return (_PAIR_TO_ID if actions is ACTIONS else {(a.type_index,a.scale_index):a.index for a in actions})[(type_index, scale_index)]
    except KeyError:
        raise ValueError("unavailable type/scale pair") from None


@dataclass(frozen=True)
class EffectPushConfig:
    feedback: bool = True
    reposition_boundary: str = "legacy_face"
    reposition_peer_detour: str = "off"
    follow_completion: str = "legacy_window"
    # Explicit opt-in until the new controller has passed the bounded comparison.
    endpoint_tracking: bool = False
    endpoint_tau: float = 0.10
    endpoint_max_age: float = 0.25
    interpolation_factor: int = 4
    fast_approach: bool = False
    approach_tangential_speed: float = 0.20
    approach_tangential_gain: float = 3.0
    approach_brake_accel: float = 0.35
    approach_velocity_damping: float = 0.25
    approach_settle_speed: float = 0.04
    joint_execution: str = "proportional_normal_v1"
    translation_completion: str = "normal_travel"
    post_push: str = "hold_brake"  # opt-in comparison; historical exit behavior retained
    follow_speed: float = 0.16
    follow_force: float = 3.0
    follow_normal_force: float = 0.75
    follow_gap: float = 0.001
    follow_gap_speed: float = 0.03
    follow_contact_loss: float = 0.25
    follow_max_seconds: float = 8.0
    follow_max_travel: float = 0.75
    contact_recovery: str = "abort"  # opt-in joint recovery; retain v7 baseline
    recovery_enter_gap: float = 0.010
    recovery_resume_gap: float = 0.003
    recovery_ready_seconds: float = 0.10
    recovery_timeout: float = 2.0
    recovery_max_attempts: int = 3
    recovery_approach_speed: float = 0.04
    joint_engage_speed: float = 0.04
    same_face_allocation: bool = False
    same_face_clearance: float = 0.04
    same_face_target_pad: float = 0.01
    same_face_end_inset: float = 0.04
    # Public shape prior. A partial/ambiguous face is rejected rather than completed by truth.
    rectangle_size: Vec = (1.0, 0.25)
    nominal_angle_deg: float = 30.0
    face_step_deg: float = 20.0
    face_position_step: float = 0.12
    span_absolute_tolerance: float = 0.02
    min_confidence: float = 0.15
    speed: float = 0.12
    min_push_speed: float = 0.018
    target_gain: float = 1.0
    gap_gain: float = 3.0
    tangential_gain: float = 1.8
    tangential_speed: float = 0.08
    point_rate: float = 0.06
    heading_gain: float = 0.20  # contact-point metres / radian
    angular_rate_gain: float = 0.10
    #: Direct contact-point tangent feedback from fitted face-normal rotation.
    #: Positive gain maps CCW normal rotation to left (+tangent) and CW to right.
    rotation_feedback_gain: float = 0.0
    lateral_gain: float = 0.35
    heading_deadband: float = math.radians(1.0)
    lateral_deadband: float = 0.01
    cooperative_separation: float = 2 * spec.ROBOT_RADIUS + 0.12
    rotation_offset_fraction: float = 0.22
    settle_speed: float = 0.025
    settle_angular_speed: float = 0.025
    settle_seconds: float = 0.20
    prepare_seconds: float = 4.0
    translation_slack: float = 4.0
    rotation_seconds_per_rad: float = 24.0
    partial_geometry_seconds: float = 0.35
    rotation_brake_seconds: float = 0.30
    rotation_trim_speed: float = 0.035
    rotation_trim_min_speed: float = 0.008
    rotation_engage_speed: float = 0.04
    cooperative_rotation_speed: float = 0.06
    # Standalone FOLLOW only; distinct from completed PUSH contact following.
    standalone_follow_gap: float = 0.05
    standalone_follow_peer_guard: str = "off"
    mixed_contact_allocation: bool = False
    approach_normal_speed: float = 0.10
    standalone_follow_speed: float = spec.REPOSITION_SPEED_MAX
    preparation_policy: str = 'legacy'
    solo_approach_gap: float = spec.COOP_READY_GAP
    progress_guard: bool = False
    contact_tangent_damping: float = 0.

    def __post_init__(self):
        if not math.isfinite(self.contact_tangent_damping) or self.contact_tangent_damping < 0:
            raise ValueError('invalid contact tangent damping')
        if not math.isfinite(self.solo_approach_gap) or not 0 < self.solo_approach_gap <= spec.COOP_READY_GAP:
            raise ValueError('solo approach gap must be positive and no larger than the ready gap')
        if self.preparation_policy not in ('legacy','blended','hold'):
            raise ValueError('unknown preparation policy')
        if not math.isfinite(self.standalone_follow_speed) or self.standalone_follow_speed <= 0:
            raise ValueError('invalid standalone FOLLOW speed')
        if not math.isfinite(self.approach_normal_speed) or self.approach_normal_speed <= 0:
            raise ValueError('invalid approach normal speed')
        if self.mixed_contact_allocation and not self.same_face_allocation:
            raise ValueError("mixed contact allocation requires same-face allocation")
        if not math.isfinite(self.standalone_follow_gap) or self.standalone_follow_gap < 0:
            raise ValueError("invalid standalone FOLLOW gap")
        if self.standalone_follow_peer_guard not in ("off", "scan_projection_v1"):
            raise ValueError("unknown standalone FOLLOW peer guard")
        if self.follow_completion not in ("legacy_window", "maintenance_duration", "peer_push"):
            raise ValueError("unknown FOLLOW completion clock")
        if self.reposition_boundary not in ("legacy_face", "rounded_scan"):
            raise ValueError("unknown reposition boundary mode")
        if self.reposition_peer_detour not in ("off", "outer_arc"):
            raise ValueError("unknown reposition peer detour")
        if self.reposition_peer_detour != "off" and self.reposition_boundary != "rounded_scan":
            raise ValueError("peer detour requires rounded boundary")
        if self.contact_recovery not in ("abort", "joint_approach"):
            raise ValueError("unknown contact recovery protocol")
        if self.contact_recovery == "joint_approach" and self.post_push != "contact_follow":
            raise ValueError("joint recovery requires contact following")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.recovery_enter_gap, self.recovery_resume_gap, self.recovery_ready_seconds,
                self.recovery_timeout, self.recovery_approach_speed)):
            raise ValueError("recovery limits must be finite and positive")
        if self.recovery_resume_gap >= self.recovery_enter_gap:
            raise ValueError("recovery requires separated entry/resume gap thresholds")
        if type(self.recovery_max_attempts) is not int or self.recovery_max_attempts <= 0:
            raise ValueError("recovery attempts must be a positive integer")
        if self.post_push not in ("hold_brake", "contact_follow"):
            raise ValueError("unknown post-push behavior")
        if self.post_push == "contact_follow" and (self.translation_completion != "normal_travel"
                or self.joint_execution != "proportional_normal_v1"):
            raise ValueError("contact follow requires the paired measured-travel protocol")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.follow_speed, self.follow_force, self.follow_normal_force,
                self.follow_gap_speed, self.follow_contact_loss,
                self.follow_max_seconds, self.follow_max_travel)):
            raise ValueError("follow limits must be finite and positive")
        if not math.isfinite(self.follow_gap) or self.follow_gap < 0:
            raise ValueError("follow gap must be finite and nonnegative")
        if self.follow_force > spec.MAX_FORCE or self.follow_normal_force > self.follow_force:
            raise ValueError("follow force limits exceed the actuator limit")
        if self.translation_completion not in ("normal_travel", "normal_reference", "object_effect"):
            raise ValueError("unknown translation completion measure")
        SameFaceReservation(self.same_face_clearance, self.same_face_target_pad, self.same_face_end_inset)
        if self.joint_execution not in ("legacy_v3", "proportional_normal_v1"):
            raise ValueError("unknown joint execution protocol")
        if not math.isfinite(self.joint_engage_speed) or self.joint_engage_speed <= 0:
            raise ValueError("joint engagement speed must be finite and positive")
        if any(not math.isfinite(v) or v <= 0 for v in
               (self.approach_tangential_speed, self.approach_tangential_gain,
                self.approach_brake_accel, self.approach_settle_speed)):
            raise ValueError("approach speed/gain/braking limits must be finite and positive")
        if not math.isfinite(self.approach_velocity_damping) or self.approach_velocity_damping < 0:
            raise ValueError("approach velocity damping must be finite and nonnegative")
        if any(not math.isfinite(v) or v <= 0 for v in (self.endpoint_tau, self.endpoint_max_age)):
            raise ValueError("endpoint time constants must be finite and positive")
        if self.interpolation_factor < 1 or int(self.interpolation_factor) != self.interpolation_factor:
            raise ValueError("interpolation factor must be a positive integer")
        if (len(self.rectangle_size) != 2 or any(not math.isfinite(x) or x <= 0
                                               for x in self.rectangle_size)
                or abs(self.rectangle_size[0] - self.rectangle_size[1]) < 1e-6):
            raise ValueError("this prototype requires two distinct positive rectangle dimensions")
        if self.speed <= 0 or not math.isfinite(self.speed):
            raise ValueError("speed must be finite and positive")
        if not 0 < self.nominal_angle_deg < 90:
            raise ValueError("nominal direction cone must be between 0 and 90 degrees")
        if any(not math.isfinite(v) or v < 0 for v in
               (self.heading_gain, self.angular_rate_gain, self.lateral_gain, self.point_rate)):
            raise ValueError("contact feedback gains must be finite and nonnegative")


@dataclass(frozen=True)
class FaceEstimate:
    patch: SurfacePatch
    normal: Vec
    point: Vec  # estimated face midpoint, prior supported, not simulator truth
    centre: Vec
    length: float
    depth: float
    confidence: float
    centre_observable: bool = True
    evidence: str = "two_bounded_edges_plus_prior"
    tracking: Optional[TrackedFace] = None


def estimate_face(observation: SkillObservation, patch: SurfacePatch,
                  config: EffectPushConfig) -> Optional[FaceEstimate]:
    if not patch.valid or patch.confidence < config.min_confidence or len(patch.points_world) < 3:
        return None
    n = unit(patch.normal_world)
    t = tangent(n)
    projections = [dot(point, t) for point in patch.points_world]
    observed_low, observed_high = min(projections), max(projections)
    plane = sum(dot(point, n) for point in patch.points_world) / len(patch.points_world)
    if config.endpoint_tracking:
        measured = observe_endpoints(observation.scan, patch, observation.patches)
        if not any(m.observed for m in measured):
            return None
        candidates = []
        for length, depth in (config.rectangle_size, config.rectangle_size[::-1]):
            lo, hi = observed_high - length, observed_low
            for m in measured:
                if m.observed:
                    lo = max(lo, dot(m.lower, t) - m.side * length)
                    hi = min(hi, dot(m.upper, t) - m.side * length)
            if lo <= hi + .005:
                candidates.append((length, depth, lo, hi))
        if len(candidates) != 1:
            return None
        length, depth, lo, hi = candidates[0]
        point = add(mul(t, (lo + hi + length) / 2), mul(n, plane))
        return FaceEstimate(patch, n, point, add(point, mul(n, depth / 2)), length, depth,
                            min(patch.confidence, (observed_high - observed_low) / length),
                            True, "physical_endpoint_bounds_plus_prior")
    low, high = observed_low, observed_high
    low_censored = high_censored = False
    # Bracket each edge between its last return and the neighbouring ray's intersection
    # with the fitted line. Treating last-return span as the true face length makes an
    # otherwise static face appear/disappear as 5-degree lidar sampling crosses an edge.
    # Occluder/range-censored boundaries do not provide such a bracket.
    scan = observation.scan
    for point, side in ((patch.points_world[0], -1), (patch.points_world[-1], 1)):
        bearing = math.atan2(point[1] - scan.origin[1], point[0] - scan.origin[0])
        k = min(range(len(scan.ray_angles)),
                key=lambda j: abs(_wrap_angle(scan.ray_angles[j] - bearing)))
        j = (k + side) % len(scan.valid)
        ray = (math.cos(scan.ray_angles[j]), math.sin(scan.ray_angles[j]))
        denominator = dot(n, ray)
        distance = ((plane - dot(n, scan.origin)) / denominator
                    if denominator > 1e-6 else float("inf"))
        censored = (distance <= 0 or distance > spec.LIDAR_RANGE
                    or (scan.valid[j] and scan.category[j] in (CATEGORY_STATIC, CATEGORY_TEAMMATE)
                        and scan.distances[j] <= distance + 0.02))
        if censored:
            if dot(point, t) <= (observed_low + observed_high) / 2:
                low_censored = True
            else:
                high_censored = True
            continue
        projection = dot(add(scan.origin, mul(ray, distance)), t)
        low, high = min(low, projection), max(high, projection)
    span = observed_high - observed_low
    if low_censored and high_censored:
        return None
    candidates = []
    for length, depth in (config.rectangle_size, config.rectangle_size[::-1]):
        left_low = max(float("-inf") if low_censored else low, observed_high - length)
        left_high = min(observed_low, float("inf") if high_censored else high - length)
        if left_low <= left_high + config.span_absolute_tolerance:
            candidates.append((length, depth, left_low, left_high))
    if len(candidates) != 1:
        return None
    length, depth, left_low, left_high = candidates[0]
    q = (left_low + left_high + length) / 2
    point = add(mul(t, q), mul(n, plane))
    one_edge = low_censored or high_censored
    return FaceEstimate(patch, n, point, add(point, mul(n, depth / 2)), length, depth,
                        min(patch.confidence, max(0.0, span / length)) * (0.65 if one_edge else 1),
                        True, "one_bounded_edge_plus_prior" if one_edge
                        else "two_bounded_edges_plus_prior")


def available_faces(observation, action: EffectAction, config=EffectPushConfig()):
    faces = []
    for patch in observation.patches:
        face = estimate_face(observation, patch, config)
        if face is None:
            continue
        if action.effect_kind == "translate" and dot(face.normal, action.effect_direction) \
                < math.cos(math.radians(config.nominal_angle_deg)):
            continue
        faces.append(face)
    return faces


def availability(observation, action, config=EffectPushConfig()):
    """Local geometric mask, not public evidence and not an oracle for physical feasibility."""
    if not isinstance(action, EffectAction):
        if action.kind == 'follow' and ((action.scale_kind == 'peer_push') != (config.follow_completion == 'peer_push')):
            return False, 'follow_codebook_mismatch'
        return (action.kind in ("wait", "follow") or observation.patch is not None,
                "non_push")
    if available_faces(observation, action, config):
        return True, "geometry_available"
    if any(estimate_face(observation, p, config) for p in observation.patches):
        return False, "direction_unavailable"
    return False, "effect_unobservable"


def action_mask(observation, config=EffectPushConfig(), *, actions=ACTIONS):
    faces = [f for p in observation.patches
             if (f := estimate_face(observation, p, config)) is not None]
    cosine = math.cos(math.radians(config.nominal_angle_deg))
    mask = np.asarray([
        bool(faces) and (a.effect_kind == "rotate"
                        or any(dot(f.normal, a.effect_direction) >= cosine for f in faces))
        if isinstance(a, EffectAction)
        else (a.kind in ("wait", "follow") or observation.patch is not None)
        for a in actions], dtype=bool)
    for i,a in enumerate(actions):
        if a.kind == 'follow':
            mask[i] &= (a.scale_kind == 'peer_push') == (config.follow_completion == 'peer_push')
    return mask


def joint_action_mask(observations, config=EffectPushConfig(), *, actions=ACTIONS):
    """Legal joint pairs for central sampling; do not repair actions after sampling."""
    left,right=(action_mask(o,config,actions=actions) for o in observations)
    mask=left[:,None] & right[None,:]
    if config.follow_completion == 'peer_push':
        follow=np.array([a.kind=='follow' for a in actions]);push=np.array([a.kind=='push' for a in actions])
        mask &= ~(follow[:,None] & ~push[None,:])
        mask &= ~(~push[:,None] & follow[None,:])
    return mask


def action_mask_given_peer(observation, peer_action, config=EffectPushConfig(), *, actions=ACTIONS):
    """Conditional mask after the other public action is chosen."""
    mask=action_mask(observation,config,actions=actions)
    if config.follow_completion=='peer_push' and peer_action.kind!='push':
        mask &= np.array([a.kind!='follow' for a in actions])
    if config.follow_completion=='peer_push' and peer_action.kind=='follow':
        mask &= np.array([a.kind=='push' for a in actions])
    return mask


def type_scale_mask(observation, config=EffectPushConfig(), *, actions=ACTIONS):
    flat = action_mask(observation, config,actions=actions)
    mask = np.zeros((TYPE_COUNT, SCALE_COUNT), dtype=bool)
    for a in actions:
        mask[a.type_index, a.scale_index] = flat[a.index]
    return mask


class EffectPushSkill(PushSkill):
    """Declared motion intent, tracked local face, bounded contact-point outer feedback."""

    def __init__(self, action: EffectAction, force_limit=spec.MAX_FORCE,
                 config=EffectPushConfig()):
        super().__init__(action, force_limit)
        self.config = config
        self.speed_command = config.speed
        duration = (config.translation_slack * action.distance / config.speed
                    if action.effect_kind == "translate"
                    else config.rotation_seconds_per_rad * action.rotation_angle)
        self.time_budget = min(spec.JOINT_MAX_SECONDS, config.prepare_seconds + duration)
        self.face: Optional[FaceEstimate] = None
        self.initial_face: Optional[FaceEstimate] = None
        self.reference_centre: Optional[Vec] = None
        self.reference_time: Optional[float] = None
        self.initial_normal: Optional[Vec] = None
        self.last_normal: Optional[Vec] = None
        self.rotation = 0.0
        self.angular_rate = 0.0
        self.instant_angular_rate = 0.0
        self.effect_progress = 0.0
        self.effect_speed = 0.0
        self.lateral_error = 0.0
        self.heading_error = 0.0
        self.point_offset = 0.0
        self.contact_target: Optional[Vec] = None
        self.rotation_feedback = 0.0
        self.diagnostic_tangent_offset = 0.0
        self.target_before_visibility = None
        self.target_before_reservation = None
        self.target_final = None
        self.lateral_command = 0.0
        self.tangential_velocity = 0.0
        self.radial_command = 0.0
        self.first_clip = 0.0
        self.damping_term = 0.0
        self.raw_before_avoid = (0.0, 0.0)
        self.raw_after_avoid = (0.0, 0.0)
        self.raw_before_norm_cap = (0.0, 0.0)
        self.raw_after_norm_cap = (0.0, 0.0)
        self.v_ref = (0.0, 0.0)
        self.force_pre_norm_cap = (0.0, 0.0)
        self.force_post_norm_cap = (0.0, 0.0)
        self.settled_seconds = 0.0
        self.effect_status = "not_started"
        self.partial_geometry_time = 0.0
        self.role_offset: Optional[float] = None
        self.rotation_trimming = False
        self.endpoint_tracker = EndpointTracker(tau=config.endpoint_tau, max_age=config.endpoint_max_age)
        self.endpoint_status = "not_started"
        self.anchor_distance = None
        self.tangential_error = 0.0
        self.approach_speed_limit = self.config.tangential_speed
        self.approach_blend = None
        self.execution_protocol = config.joint_execution
        self.joint_pair = False
        self.joint_active = False
        self.contact_ready = False
        self.normal_reference_mode = (action.effect_kind == "translate" and config.translation_completion == "normal_reference")
        self.normal_travel_mode = (action.effect_kind == "translate" and config.translation_completion == "normal_travel")
        self.normal_reference_budget = 0.0
        self.normal_travel = 0.0
        self.normal_progress_marker = 0.0
        self.travel_previous_normal = None
        self.travel_previous_contact = False
        self.travel_previous_active = False
        self.travel_previous_time = None
        self.local_brake_seconds = 0.0
        self.pending_exit_reason = None
        self.requested_time_budget = self.time_budget
        self.previous_surface_point = None
        self.surface_normal_speed = 0.0
        self.best_preparation_error = float("inf")
        self.preparation_gap_target = None
        self.preparation_offset = None
        self.phase_progress = ProgressGuard()
        self.contact_reservation = SameFaceReservation(
            config.same_face_clearance, config.same_face_target_pad, config.same_face_end_inset)
        self.reservation_result = None
        self.contact_allocation_exit = None
        self.follow_enabled = False  # environment enables only paired measured translations
        self.drive_complete = False
        self.drive_complete_time = None
        self.drive_complete_travel = None
        self.follow_started = False
        self.follow_stop = False
        self.follow_seconds = self.follow_contact_seconds = 0.0
        self.follow_contact_loss_seconds = 0.0
        self.follow_normal_travel = self.follow_path_length = 0.0
        self.follow_increment = 0.0
        self.travel_previous_follow = False
        self.recovery_enabled = self.recovery_requested = self.recovery_ready = False
        self.recovery_resume_stage = None
        self.recovery_count = 0
        self.recovery_seconds = self.recovery_path_length = self.recovery_normal_travel = 0.0
        self.recovery_increment = 0.0
        self.travel_previous_recovery = False
        self.follow_near_seconds = 0.0

    def begin_recovery(self):
        """Executor phase signal, based only on preceding local gap reports."""
        if self.done:
            return
        self.recovery_resume_stage = self.stage
        self.stage = "recover"
        self.recovery_count += 1
        self.recovery_requested = self.recovery_ready = False

    def resume_recovery(self):
        if not self.done and self.stage == "recover":
            self.stage = self.recovery_resume_stage
            self.recovery_requested = self.recovery_ready = False

    def _measure_normal_travel(self, observation, face, dt):
        """Signed own-position travel on the PRECEDING observed face normal.

        Require contact at both endpoints of the interval. Never backfill an
        unobserved/contact-transition interval, and never use object truth.
        """
        before = self.normal_travel
        self.follow_increment = 0.0
        self.recovery_increment = 0.0
        valid_interval = (self.travel_previous_normal is not None and self.travel_previous_time is not None
                and 0 < observation.time - self.travel_previous_time <= 1.5 * dt
                and face is not None and face.patch.valid
                and dot(face.normal, self.travel_previous_normal)
                    >= math.cos(math.radians(self.config.face_step_deg)))
        if self.travel_previous_recovery:
            self.recovery_path_length += math.hypot(*self.step_vector)
            if valid_interval:
                self.recovery_increment = dot(self.step_vector, self.travel_previous_normal)
                self.recovery_normal_travel += self.recovery_increment
        elif (self.travel_previous_active and self.travel_previous_contact
                and observation.contacting_object and valid_interval):
            advance = dot(self.step_vector, self.travel_previous_normal)
            if self.travel_previous_follow:
                self.follow_normal_travel += advance
                self.follow_increment = advance
            else:
                self.normal_travel += advance
        if self.travel_previous_follow:
            self.follow_path_length += math.hypot(*self.step_vector)
        return self.normal_travel - before

    def _approach_lateral(self, error, velocity, gap, contacting):
        """Faster in free clearance; distance-limited braking near the fixed target.

        Only preparation uses this profile. In contact, retain the original slow
        tangential cap. The inherited acceleration/force limits still apply.
        """
        # Small initial corrections keep the previous controller exactly. Blend
        # continuously from 5 to 15 cm instead of switching at a hard distance.
        blend = self.approach_blend if self.approach_blend is not None else 1.0
        limit = (self.config.tangential_speed + blend *
                 (self.config.approach_tangential_speed - self.config.tangential_speed)
                 if gap >= .04 and not contacting else self.config.tangential_speed)
        stopping_limit = math.sqrt(2 * self.config.approach_brake_accel * abs(error))
        self.approach_speed_limit = min(limit, stopping_limit)
        gain = self.config.tangential_gain + blend * (self.config.approach_tangential_gain - self.config.tangential_gain)
        raw = gain * error - blend * self.config.approach_velocity_damping * velocity
        return clamp(raw, -self.approach_speed_limit, self.approach_speed_limit)

    def _observe_face(self, observation):
        faces = [face for patch in observation.patches
                 if (face := estimate_face(observation, patch, self.config)) is not None]
        if self.face is None:
            faces = available_faces(observation, self.action, self.config)
            if not faces:
                return None
            return max(faces, key=lambda f: (dot(f.normal, self.action.effect_direction)
                       if self.action.effect_kind == "translate" else f.length)
                       - 0.1 * f.patch.nearest_distance)
        compatible = [f for f in faces if abs(f.length - self.face.length) < 1e-6
                      and dot(f.normal, self.face.normal)
                      >= math.cos(math.radians(self.config.face_step_deg))
                      and math.dist(f.centre, self.face.centre) <= self.config.face_position_step]
        if compatible:
            return min(compatible, key=lambda f: math.dist(f.centre, self.face.centre))
        # Briefly maintain the association when ray quantisation/range hides an endpoint.
        # Only the normal/line displacement is observed here; tangential centre motion is
        # unknown and MUST NOT be credited as a completed translation.
        for patch in observation.patches:
            n = unit(patch.normal_world)
            if (not patch.valid or patch.confidence < self.config.min_confidence
                    or dot(n, self.face.normal) < math.cos(math.radians(self.config.face_step_deg))):
                continue
            plane = sum(dot(point, n) for point in patch.points_world) / len(patch.points_world)
            shift = plane + self.face.depth / 2 - dot(self.face.centre, n)
            if abs(shift) > self.config.face_position_step:
                continue
            centre = add(self.face.centre, mul(n, shift))
            return FaceEstimate(patch, n, sub(centre, mul(n, self.face.depth / 2)),
                                centre, self.face.length, self.face.depth, 0.3, False,
                                "history_line_only")
        return None

    def _update_effect(self, face, dt):
        if self.reference_centre is None:
            return
        if self.previous_surface_point is not None:
            rate = dot(sub(face.point, self.previous_surface_point), face.normal) / dt
            self.surface_normal_speed = .5 * self.surface_normal_speed + .5 * clamp(rate, -.20, .20)
        self.previous_surface_point = face.point
        if self.last_normal is not None:
            angle = math.atan2(cross(self.last_normal, face.normal), dot(self.last_normal, face.normal))
            self.rotation += angle
            self.instant_angular_rate = angle / dt
            self.angular_rate = 0.65 * self.angular_rate + 0.35 * angle / dt
        self.last_normal = face.normal
        progress_before = self.effect_progress
        if self.action.effect_kind == "translate":
            if face.centre_observable:
                delta = sub(face.centre, self.reference_centre)
                self.effect_progress = dot(delta, self.action.effect_direction)
                self.lateral_error = dot(delta, tangent(self.action.effect_direction))
            self.heading_error = math.atan2(cross(self.action.effect_direction, face.normal),
                                           dot(self.action.effect_direction, face.normal))
        else:
            self.effect_progress = self.action.rotation_sign * self.rotation
        self.effect_speed = 0.65 * self.effect_speed + 0.35 * (self.effect_progress - progress_before) / dt

    def _choose_contact(self, face, dt, observation=None):
        limit = max(0.0, face.length / 2 - min(0.20, face.length * 0.20))
        first_contact_target = self.contact_target is None
        if self.role_offset is None:
            self.role_offset = 0.0
            if (self.cooperative and self.action.effect_kind == "translate"
                    and observation is not None and observation.teammate_nearest is not None):
                distance, point = observation.teammate_nearest
                own = (observation.own.x, observation.own.y)
                mate = add(own, mul(unit(sub(point, own)), distance + spec.ROBOT_RADIUS))
                if dot(sub(mate, face.point), face.normal) < 0.05:
                    side = dot(sub(own, mate), tangent(face.normal))
                    if abs(side) < 1e-6:
                        side = -1 if observation.robot_id == 0 else 1
                    self.role_offset = math.copysign(min(limit, self.config.cooperative_separation / 2), side)
        if self.follow_started:
            desired = self.point_offset  # retain endpoint-relative role; no direction correction
        elif self.action.effect_kind == "rotate":
            desired = -self.action.rotation_sign * self.config.rotation_offset_fraction * face.length
        elif self.config.feedback:
            heading = math.copysign(max(0., abs(self.heading_error) - self.config.heading_deadband),
                                    self.heading_error)
            lateral = math.copysign(max(0., abs(self.lateral_error) - self.config.lateral_deadband),
                                    self.lateral_error)
            desired = (self.config.heading_gain * heading
                       + self.config.angular_rate_gain * self.angular_rate
                       + self.config.lateral_gain * lateral)
        else:
            desired = 0.0
        if self.action.effect_kind == "translate" and not self.follow_started:
            desired += self.diagnostic_tangent_offset
        desired = clamp(desired + self.role_offset, -limit, limit) - self.role_offset
        if (first_contact_target and face.tracking is not None
                and self.action.effect_kind == "rotate"):
            # The nominal lever is an initial destination, not a moving feedback
            # correction. Set it before motion; velocity/force ramps still apply.
            self.point_offset = desired
        self.point_offset += clamp(desired - self.point_offset,
                                   -self.config.point_rate * dt, self.config.point_rate * dt)
        t = tangent(face.normal)
        # Direct linear feedback on fitted face-normal rotation.  The frame is:
        # t = rotate90(normal) = robot's left; positive normal rate = CCW.
        # Thus CCW -> +t (left), CW -> -t (right), countering the object's tendency.
        self.rotation_feedback = clamp(
            self.config.rotation_feedback_gain * self.angular_rate, -limit, limit)
        offset = self.point_offset + self.role_offset + self.rotation_feedback
        if face.tracking is not None and face.tracking.anchor is not None:
            sign = 1 if face.tracking.anchor_side == 0 else -1
            self.anchor_distance = face.length / 2 + sign * offset
            target = add(face.tracking.anchor, mul(t, sign * self.anchor_distance))
        else:
            target = add(face.point, mul(t, offset))
        self.target_before_visibility = target
        if (self.config.preparation_policy == 'hold' and self.action.effect_kind == 'translate'
                and self.stage in ('approach','ready') and self.preparation_offset is not None):
            target = add(face.point,mul(t,self.preparation_offset))
        projected = [dot(p, tangent(face.normal)) for p in face.patch.points_world]
        # Confirmed endpoints use filtered physical bounds; shadow boundaries retain
        # the raw visibility cap. Never extend a target through an occluder.
        lo, hi = face.tracking.contact_limits if face.tracking else (min(projected), max(projected))
        margin = min(.02, max(0., (hi - lo) / 4))
        q = clamp(dot(target, t), lo + margin, hi - margin)
        self.contact_target = add(target, mul(t, q - dot(target, t)))
        self.target_before_reservation = self.contact_target
        if (self.config.same_face_allocation and (self.cooperative or self.config.mixed_contact_allocation)
                and self.action.effect_kind == "translate" and observation is not None):
            self.reservation_result = self.contact_reservation.project(face, observation, self.contact_target, dt,
                occupied_only=self.config.mixed_contact_allocation and not self.cooperative)
            if self.reservation_result.status == "allocated":
                self.contact_target = self.reservation_result.target
            elif self.reservation_result.status != "not_same_face":
                self.contact_allocation_exit = "same_face_" + self.reservation_result.status
        self.target_final = self.contact_target
        if face.tracking is not None and face.tracking.anchor is not None:
            self.anchor_distance = abs(dot(sub(self.contact_target, face.tracking.anchor), t))
        if self.preparation_offset is None:
            self.preparation_offset=dot(sub(self.contact_target,face.point),t)

    def _obstacle_avoidance(self, observation, raw, include_teammate=False):
        if not self.config.same_face_allocation:
            return super()._obstacle_avoidance(observation, raw, include_teammate)
        total = super()._obstacle_avoidance(observation, raw, include_teammate=False)
        if not include_teammate or observation.teammate_nearest is None:
            return total
        margin = spec.OBSTACLE_AVOID_MARGIN
        if self.reservation_result is not None and self.reservation_result.status == "allocated":
            margin = self.reservation_result.avoidance_margin
        distance, point = observation.teammate_nearest
        gap = distance - spec.ROBOT_RADIUS
        if gap <= margin:
            away = unit(sub((observation.own.x, observation.own.y), point))
            total = add(total, mul(away, spec.OBSTACLE_AVOID_GAIN * (margin - gap)))
            self.avoidance_active = True
            self.avoidance_direction = (away if self.avoidance_direction is None
                                        else unit(add(self.avoidance_direction, away)))
        return total

    def step(self, observation: SkillObservation, dt):
        self.skill_time += dt
        if self.stage != "recover":
            self.no_progress_seconds += dt
        self._note_position(observation)
        reason = self.pending_exit_reason or self._safety(observation)
        face = self._observe_face(observation)
        if face is not None and self.config.endpoint_tracking:
            tracked = self.endpoint_tracker.update(observation.scan, face.patch, observation.patches,
                                                   face.length, observation.time)
            if tracked is None:
                self.endpoint_status = "endpoint_unobservable"
                reason = reason or "effect_unobservable"
                face = replace(face, centre_observable=False)
            else:
                self.endpoint_status = tracked.status
                face = replace(face, point=tracked.midpoint,
                               centre=add(tracked.midpoint, mul(face.normal, face.depth / 2)),
                               centre_observable=tracked.status == "observed",
                               evidence="tracked_endpoint_plus_prior", tracking=tracked)
        normal_advance = self._measure_normal_travel(observation, face, dt)
        if self.normal_travel_mode and self.normal_travel > self.normal_progress_marker + .001:
            self.normal_progress_marker = self.normal_travel
            self._mark_progress()
        raw = (0.0, 0.0)
        if face is None:
            if self.face is None:
                _, unavailable = availability(observation, self.action, self.config)
                reason = reason or unavailable
            else:
                reason = reason or "effect_face_lost"
            self.effect_status = "unavailable" if self.face is None else "tracking_lost"
        else:
            self.face = face
            self.partial_geometry_time = (0.0 if face.centre_observable
                                          else self.partial_geometry_time + dt)
            if self.partial_geometry_time > self.config.partial_geometry_seconds:
                reason = reason or "effect_unobservable"
            local = replace(observation, patch=face.patch)
            gap = patch_gap(face.patch)
            if self.initial_face is None:
                self.initial_face = face
                # The effect belongs to the whole option, including preparation/contact
                # establishment. Do not reset its origin after the object already moved.
                self.reference_centre = face.centre
                self.initial_normal = self.last_normal = face.normal
                self.reference_time = observation.time
                self.heading_error = (math.atan2(cross(self.action.effect_direction, face.normal),
                                                dot(self.action.effect_direction, face.normal))
                                      if self.action.effect_kind == "translate" else 0.0)
                self.stage = "approach"
                self.effect_status = "approaching"
            self._update_effect(face, dt)
            self._choose_contact(face, dt, observation)
            reason = reason or self.contact_allocation_exit
            tangential_error = dot(sub(self.contact_target, (observation.own.x, observation.own.y)),
                                   tangent(face.normal))
            self.tangential_error = tangential_error
            if self.approach_blend is None:
                self.approach_blend = clamp((abs(tangential_error) - .05) / .10, 0., 1.)
            tangential_velocity = dot((observation.own.vx, observation.own.vy), tangent(face.normal))
            self.tangential_error = tangential_error
            self.tangential_velocity = tangential_velocity
            self.first_clip = clamp(self.config.tangential_gain * tangential_error,
                                    -self.config.tangential_speed, self.config.tangential_speed)
            lateral = self.first_clip
            self.damping_term = 0.0
            if self.stage in ('push','follow'):
                self.damping_term = self.config.contact_tangent_damping * tangential_velocity
                lateral = clamp(lateral - self.damping_term,
                                -self.config.tangential_speed, self.config.tangential_speed)
            self.lateral_command = lateral
            if (self.recovery_enabled and self.joint_active and self.stage in ("push", "follow")
                    and gap > self.config.recovery_enter_gap):
                self.recovery_requested = True
            if self.config.fast_approach and self.approach_blend > 0 and self.stage in ("approach", "ready"):
                lateral = self._approach_lateral(tangential_error, tangential_velocity,
                                                 gap, observation.contacting_object)
            aligned = abs(_wrap_angle(math.atan2(face.normal[1], face.normal[0])
                                     - observation.own.angle)) <= math.radians(spec.COOP_READY_ALIGN_DEG)
            if self.stage in ("approach", "ready"):
                # Centre/offset on the observed face before committing normal push force.
                approach_gap = spec.COOP_READY_GAP if self.cooperative else self.config.solo_approach_gap
                radial = clamp(self.config.gap_gain * (gap - approach_gap), -0.08, self.config.approach_normal_speed)
                self.preparation_gap_target = approach_gap
                settled_laterally = (not self.config.fast_approach or self.approach_blend == 0 or
                                     abs(tangential_velocity) <= self.config.approach_settle_speed)
                if abs(tangential_error) > 0.025 or not settled_laterally:
                    if face.tracking is not None:
                        # Approach a safe staging line while moving towards the
                        # fixed endpoint-relative target; keep 3 cm extra clearance
                        # until tangential alignment, rather than waiting remotely.
                        extra=.03
                        if self.config.preparation_policy in ('blended','hold'):
                            extra *= clamp(max((abs(tangential_error)-.015)/.08,
                                               (abs(tangential_velocity)-self.config.approach_settle_speed)/.15),0.,1.)
                        self.preparation_gap_target=approach_gap+extra
                        radial = clamp(self.config.gap_gain * (gap - self.preparation_gap_target),
                                       -.08, self.config.approach_normal_speed)
                    else:
                        radial = min(0.0, radial)
                close = (gap <= spec.COOP_READY_GAP + 0.005 and abs(tangential_error) <= 0.025
                         and aligned and settled_laterally)
                if self.cooperative and close:
                    self.ready = True
                    self.stage = "ready"
                    radial, lateral = 0.0, 0.0
                elif self.cooperative and not self.release:
                    self.ready = False
                    self.stage = "approach"
                if (self.stage == "ready" and self.release) or (close and not self.cooperative):
                    self.stage = "push"
                    self.effect_status = "engaging"
                    if self.joint_pair:
                        self._mark_progress()  # readiness waiting is not failed contact seeking
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                if self.joint_pair:
                    preparation_error = abs(tangential_error) + max(0., gap - spec.COOP_READY_GAP)
                    if preparation_error < self.best_preparation_error - .005:
                        self.best_preparation_error = preparation_error
                        self._mark_progress()
                else:
                    self._mark_progress()
            if self.stage == "push":
                self.contact_ready = bool(observation.contacting_object)
                if observation.contacting_object and not self.contact_seen:
                    self.effect_status = "executing"
                    self._mark_progress()
                if observation.contacting_object:
                    self.effective_contact += dt
                    self.contact_seen = True
                    self.contact_loss_seconds = 0.0
                    self._accumulate_signed_progress(self.action.effect_direction
                                                    if self.action.effect_kind == "translate" else face.normal)
                else:
                    self.contact_loss_seconds += dt
                    if (not self.recovery_enabled and self.contact_seen
                            and self.contact_loss_seconds > spec.CONTACT_LOSS_TOLERANCE):
                        reason = reason or "contact_lost"
                    elif not self.contact_seen and self.contact_loss_seconds > spec.CONTACT_SEEK_SECONDS:
                        reason = reason or "blocked_no_progress"
                remaining = self.action.effect_target - self.effect_progress
                speed = clamp(self.config.target_gain * remaining, self.config.min_push_speed,
                              self.speed_command)
                if self.rotation_trimming:
                    speed = clamp(self.config.target_gain * remaining,
                                  self.config.rotation_trim_min_speed, self.config.rotation_trim_speed)
                radial = clamp(speed + self.config.gap_gain * gap, 0.0, self.speed_command)
                if self.joint_pair and not self.joint_active:
                    # Establish contact at a common gentle, UNscaled speed. A side
                    # already touching holds, allowing its peer to catch up.
                    radial = (0.0 if observation.contacting_object else
                              clamp(self.config.joint_engage_speed + self.config.gap_gain * gap,
                                    0.0, self.config.joint_engage_speed))
                    self.effect_status = "joint_engaging"
                elif self.normal_reference_mode or self.normal_travel_mode:
                    # Actual own normal travel (or the explicitly selected legacy
                    # reference clock) determines completion, not object displacement.
                    radial = clamp(self.surface_normal_speed + self.speed_command + self.config.gap_gain * gap,
                                   0.0, self.config.speed)
                    if observation.contacting_object:
                        self.normal_reference_budget += self.speed_command * dt
                        if self.normal_reference_mode:
                            self._mark_progress()  # historical reference-budget contract
                    self.effect_status = "normal_travel_executing" if self.normal_travel_mode else "normal_reference_executing"
                if self.action.effect_kind == "rotate":
                    if not observation.contacting_object:
                        radial = min(radial, self.config.rotation_engage_speed)
                    elif self.cooperative:
                        radial = min(radial, self.config.cooperative_rotation_speed)
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                angular_coast = (max(0., self.action.rotation_sign * self.angular_rate,
                                     self.action.rotation_sign * self.instant_angular_rate)
                                 * self.config.rotation_brake_seconds
                                 if self.action.effect_kind == "rotate" else 0.)
                reached = (self.normal_travel >= self.action.distance - 1e-9 if self.normal_travel_mode else
                           self.normal_reference_budget >= self.action.distance - 1e-9
                           if self.normal_reference_mode else remaining <= 0.0 or (
                               self.contact_seen and self.action.effect_kind == "rotate"
                               and angular_coast >= remaining))
                if self.reference_centre is not None and (not self.joint_pair or self.joint_active) and reached:
                    if reason is None and not self.drive_complete:
                        self.drive_complete = True
                        self.drive_complete_time = observation.time
                        self.drive_complete_travel = self.normal_travel
                    if self.follow_enabled and reason is None:
                        self.follow_started = True
                        self.stage = "follow"
                    else:
                        self.stage = "brake"
                        self.effect_status = "settling"
                if not self.normal_travel_mode and abs(self.effect_speed) > 0.005:
                    self._mark_progress()
            if self.stage == "follow":
                self.follow_seconds += dt
                if gap <= self.config.recovery_enter_gap:
                    self.follow_near_seconds += dt
                if observation.contacting_object:
                    self.follow_contact_seconds += dt
                    self.follow_contact_loss_seconds = 0.0
                else:
                    self.follow_contact_loss_seconds += dt
                if not self.recovery_enabled and self.follow_contact_loss_seconds > self.config.follow_contact_loss:
                    reason = reason or "follow_contact_lost"
                if self.follow_seconds >= self.config.follow_max_seconds:
                    reason = reason or "follow_time_limit"
                if self.follow_path_length >= self.config.follow_max_travel:
                    reason = reason or "follow_travel_limit"
                # Follow locally observed surface motion with a small gap correction.
                # There is no active propulsion bias or nominal-direction correction.
                radial = clamp(self.surface_normal_speed + clamp(
                    self.config.gap_gain * (gap - self.config.follow_gap),
                    -self.config.follow_gap_speed, self.config.follow_gap_speed),
                    -self.config.follow_speed, self.config.follow_speed)
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                self.effect_status = "contact_following"
                if self.follow_stop:
                    self.stage = "brake"
                    self.effect_status = "joint_settling"
            if self.stage == "recover":
                self.recovery_seconds += dt
                self.recovery_ready = (gap <= self.config.recovery_resume_gap
                    and abs(dot((observation.own.vx, observation.own.vy), face.normal)) <= .06)
                radial = (0.0 if gap <= self.config.recovery_resume_gap else
                          clamp(self.config.gap_gain * (gap - self.config.follow_gap),
                                0.0, self.config.recovery_approach_speed))
                raw = add(mul(face.normal, radial), mul(tangent(face.normal), lateral))
                self.effect_status = "joint_reapproaching"
            if self.stage == "brake":
                raw = (0.0, 0.0)
                self.local_brake_seconds += dt
                settle_limit = (self.config.settle_speed if self.action.effect_kind == "translate"
                                else self.config.settle_angular_speed)
                settled_effect = self.joint_pair or self.normal_travel_mode or self.normal_reference_mode or abs(self.effect_speed) <= settle_limit
                if observation.own.speed <= self.config.settle_speed and settled_effect:
                    self.settled_seconds += dt
                else:
                    self.settled_seconds = 0.0
                if (self.settled_seconds >= self.config.settle_seconds
                        and (self.normal_travel_mode or self.normal_reference_mode or face.centre_observable
                             or self.action.effect_kind == "rotate")):
                    if self.normal_travel_mode:
                        if self.normal_travel >= self.action.distance - 1e-9:
                            reason = reason or "normal_travel_complete"
                            self.effect_status = "normal_travel_completed"
                        else:
                            # Contact recoil during braking must not turn a past
                            # threshold crossing into a false completion claim.
                            self.stage = "push"
                            self.settled_seconds = self.local_brake_seconds = 0.0
                            self.effect_status = "normal_travel_trimming"
                    elif self.normal_reference_mode:
                        reason = reason or "normal_reference_complete"
                        self.effect_status = "reference_completed"
                    elif self.effect_progress >= self.action.effect_target:
                        reason = reason or ("local_effect_reached" if self.joint_pair else "effect_complete")
                        self.effect_status = "completed"
                    else:
                        # Anticipatory braking is a control decision, not evidence that the
                        # requested effect happened. Resume a bounded slow trim if short.
                        self.stage = "push"
                        self.rotation_trimming = True
                        self.settled_seconds = 0.0
                        self.effect_status = "trimming"
                if (self.joint_pair or self.normal_travel_mode or self.normal_reference_mode) and self.local_brake_seconds >= spec.BRAKE_MAX_SECONDS:
                    reason = reason or "local_brake_timeout"
            if self.config.progress_guard and reason is None:
                phase=None;value=0.;minimum=.005
                if self.stage=='approach':
                    phase='prepare';value=-(abs(tangential_error)+max(0.,gap-spec.COOP_READY_GAP))
                elif self.stage=='push' and self.joint_pair and not self.joint_active:
                    if not observation.contacting_object:phase='engage';value=-max(0.,gap)
                elif self.stage=='push':
                    phase='translate' if self.action.effect_kind=='translate' else 'rotate'
                    value=self.normal_travel if self.normal_travel_mode else self.effect_progress
                    minimum=.005 if phase=='translate' else .01
                if self.phase_progress.update(phase,value,dt,minimum):
                    reason={'prepare':'prepare_stalled','engage':'engagement_stalled',
                            'translate':'translation_stalled','rotate':'rotation_stalled'}[phase]
            reason = reason or self._common_checks(local, dt)
            if (reason is None and self.stage == "push" and self.reference_centre is not None
                    and self.no_progress_seconds > spec.NO_PROGRESS_SECONDS):
                reason = "blocked_no_progress"
        if reason is not None:
            raw = (0.0, 0.0)
        else:
            raw = self._obstacle_avoidance(observation, raw, include_teammate=True)
        if self.stage == "follow" and reason is None:
            length = math.hypot(*raw)
            if length > self.config.follow_speed:
                raw = mul(raw, self.config.follow_speed / length)
        reference = self.reference.update(raw, dt)
        force, saturated = self._force_from_reference(reference, observation)
        if self.stage == "follow" and reason is None:
            # This caps commanded force, not unmeasured physical contact impulse.
            normal_force = dot(force, face.normal)
            force = add(force, mul(face.normal, clamp(normal_force,
                -self.config.follow_normal_force, self.config.follow_normal_force) - normal_force))
            length = math.hypot(*force)
            if length > self.config.follow_force:
                force = mul(force, self.config.follow_force / length)
            saturated = saturated or abs(normal_force) > self.config.follow_normal_force or length > self.config.follow_force
        if reason is not None:
            self._finish(reason, observation)
        local = replace(observation, patch=face.patch) if face else observation
        info = super()._info(local, force, saturated, reference, raw)
        self.progress = self.effect_progress
        info.update(progress=self.effect_progress, progress_target=self.action.effect_target)
        if self.normal_travel_mode:
            info["completion_progress"] = self.normal_travel
            info["completion_target"] = self.action.distance
        info["effect"] = {
            "version": VERSION, "kind": self.action.effect_kind,
            "direction": list(self.action.effect_direction), "rotation_sign": self.action.rotation_sign,
            "n": face.normal if face else None, "t": tangent(face.normal) if face else None,
            "heading": self.heading_error, "lateral_error": self.lateral_error,
            "omega_est": self.angular_rate, "anchor": (face.tracking.anchor if face and face.tracking else None),
            "anchor_side": (face.tracking.anchor_side if face and face.tracking else None),
            "endpoint_status": self.endpoint_status,
            "point_offset": self.point_offset, "role_offset": self.role_offset,
            "q_rot": self.rotation_feedback, "diagnostic_tangent_offset": self.diagnostic_tangent_offset,
            "target_before_visibility": self.target_before_visibility,
            "target_before_reservation": self.target_before_reservation,
            "target_final": self.target_final,
            "e_t": self.tangential_error, "v_t": self.tangential_velocity,
            "first_clip": self.first_clip, "c_v_t": self.damping_term,
            "lateral": self.lateral_command, "radial": self.radial_command,
            "raw_before_avoid": self.raw_before_avoid, "raw_after_avoid": self.raw_after_avoid,
            "raw_before_norm_cap": self.raw_before_norm_cap, "raw_after_norm_cap": self.raw_after_norm_cap,
            "v_ref": self.v_ref, "F_pre_norm_cap": self.force_pre_norm_cap,
            "F_post_norm_cap": self.force_post_norm_cap,
            "target": self.action.effect_target, "unit": "m" if self.action.effect_kind == "translate" else "rad",
            "progress": self.effect_progress, "robot_travel": self.signed_progress,
            "heading_error": self.heading_error, "lateral_error": self.lateral_error,
            "rotation_feedback": self.rotation_feedback,
            "angular_rate": self.angular_rate, "offset": self.point_offset,
            "role_offset": self.role_offset,
            "contact_target": self.contact_target, "estimated_centre": face.centre if face else None,
            "face_midpoint": face.point if face else None, "face_normal": face.normal if face else None,
            "face_points": face.patch.points_world if face else [],
            "status": self.effect_status, "feedback": self.config.feedback,
            "estimate": face.evidence if face else "unavailable", "auto_reposition": False,
            "centre_observable": bool(face and face.centre_observable),
            "geometry_age": self.partial_geometry_time,
            "observation_time": observation.time, "reference_time": self.reference_time,
            "reference_normal": self.initial_normal,
            "face_confidence": face.confidence if face else 0.0,
            "endpoint_tracking": self.config.endpoint_tracking,
            "endpoint_status": self.endpoint_status,
            "endpoint": face.tracking.as_dict() if face and face.tracking else None,
            "anchor_distance": self.anchor_distance,
            "fast_approach": self.config.fast_approach,
            "preparation_policy": self.config.preparation_policy,
            "preparation_gap_target": self.preparation_gap_target,
            "progress_guard": self.phase_progress.info if self.config.progress_guard else None,
            "approach_speed_limit": self.approach_speed_limit,
            "approach_blend": self.approach_blend,
            "joint_execution": self.execution_protocol if self.joint_pair else "single_option",
            "joint_active": self.joint_active,
            "normal_reference_budget": self.normal_reference_budget,
            "normal_reference_target": self.action.distance if self.normal_reference_mode or self.normal_travel_mode else None,
            "normal_surface_follow_speed": self.surface_normal_speed,
            "normal_travel": self.normal_travel,
            "normal_travel_target": self.action.distance if self.normal_travel_mode else None,
            "normal_travel_increment": normal_advance,
            "post_push": self.config.post_push,
            "drive_complete": self.drive_complete,
            "drive_complete_time": self.drive_complete_time,
            "drive_complete_travel": self.drive_complete_travel,
            "follow_enabled": self.follow_enabled,
            "follow_started": self.follow_started,
            "follow_stop": self.follow_stop,
            "follow_seconds": self.follow_seconds,
            "follow_contact_seconds": self.follow_contact_seconds,
            "follow_near_seconds": self.follow_near_seconds,
            "follow_normal_travel": self.follow_normal_travel,
            "follow_normal_increment": self.follow_increment,
            "follow_path_length": self.follow_path_length,
            "contact_recovery": self.config.contact_recovery,
            "recovery_enabled": self.recovery_enabled,
            "recovery_requested": self.recovery_requested,
            "recovery_ready": self.recovery_ready,
            "recovery_resume_stage": self.recovery_resume_stage,
            "recovery_count": self.recovery_count,
            "recovery_seconds": self.recovery_seconds,
            "recovery_path_length": self.recovery_path_length,
            "recovery_normal_travel": self.recovery_normal_travel,
            "recovery_normal_increment": self.recovery_increment,
            "scale_semantics": ("measured_normal_travel_m" if self.normal_travel_mode else
                                "normal_reference_m" if self.normal_reference_mode else "estimated_object_effect"),
            "same_face": asdict(self.reservation_result) if self.reservation_result else None,
            "contact_allocation_mode": ("mixed-peer-occupied-v1" if self.config.mixed_contact_allocation
                                        and not self.cooperative else "cooperative" if self.cooperative else "off"),
            "tangential_error": self.tangential_error,
            "interpolated_face_points": (interpolate_face(face.patch, self.config.interpolation_factor)
                                         if face and self.config.endpoint_tracking else []),
        }
        self.travel_previous_normal = face.normal if face is not None else None
        self.travel_previous_contact = bool(observation.contacting_object)
        self.travel_previous_active = (self.stage in ("push", "brake", "follow") and not self.done
                                       and (not self.joint_pair or self.joint_active))
        self.travel_previous_follow = self.follow_started and self.stage in ("follow", "brake") and not self.done
        self.travel_previous_recovery = self.stage == "recover" and not self.done
        self.travel_previous_time = observation.time
        return force, info


def make_effect_skill(action, force_limit=spec.MAX_FORCE, *, config=EffectPushConfig()):
    if isinstance(action, EffectAction):
        return EffectPushSkill(action, force_limit, config)
    if action.kind == "reposition":
        return RepositionSkill(action,force_limit,boundary_mode=config.reposition_boundary,
                               peer_detour=config.reposition_peer_detour,progress_guard=config.progress_guard)
    if action.kind == "follow":
        if (action.scale_kind == 'peer_push') != (config.follow_completion == 'peer_push'):
            raise ValueError("FOLLOW codebook and completion mode do not match")
        return SupportFollowSkill(action,force_limit,completion_clock=config.follow_completion,
            target_gap=config.standalone_follow_gap,peer_guard=config.standalone_follow_peer_guard,
            speed_limit=config.standalone_follow_speed)
    return make_skill(action, force_limit)
