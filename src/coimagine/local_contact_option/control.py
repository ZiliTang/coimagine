"""The new local-surface control law, and the local surface-motion estimator.

Nothing in this module can reach an object pose: every quantity comes from
:class:`~coimagine.local_contact_option.sensing.LocalContactObservation`. The law is

    v_raw = speed * (cos(alpha) * n + sin(alpha) * t)
    ||v_ref(k) - v_ref(k-1)|| <= a_ref_max * dt
    F_cmd = clip_norm(Kv * (v_ref - v_robot), F_max)

with ``n`` the filtered local surface normal (pointing from the robot towards the surface) and
``t = Rot90(n)``. There is no position integral, no direct velocity override and no
object-velocity feed-forward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .contract import (
    NORMAL_FILTER_TAU,
    NORMAL_JUMP_LIMIT,
    SURFACE_CONSISTENCY,
    SURFACE_FRESHNESS,
    SURFACE_MATCH_MIN_POINTS,
    SURFACE_MATCH_TOLERANCE,
)
from .sensing import LocalContactObservation, LocalPatch

Vec = Tuple[float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def _norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def _unit(a: Vec) -> Vec:
    length = _norm(a)
    if length < 1e-12:
        return (0.0, 0.0)
    return (a[0] / length, a[1] / length)


def clip_norm(vector: Vec, limit: float) -> Vec:
    length = _norm(vector)
    if length > limit and length > 0.0:
        scale = limit / length
        return (vector[0] * scale, vector[1] * scale)
    return vector


def rotate90(vector: Vec) -> Vec:
    return (-vector[1], vector[0])


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def angle_between(first: Vec, second: Vec) -> float:
    dot = max(-1.0, min(1.0, first[0] * second[0] + first[1] * second[1]))
    return math.acos(dot)


# ---------------------------------------------------------------------------
# normal filtering with jump detection
# ---------------------------------------------------------------------------


@dataclass
class NormalFilter:
    """First-order low-pass on the unit normal, with a face-switch test.

    Filtering the vector (rather than the angle) is what keeps a normal near +-pi from
    wrapping: the state is a unit vector, so the interpolation is the short way round.
    """

    tau: float = NORMAL_FILTER_TAU
    jump_limit_deg: float = NORMAL_JUMP_LIMIT
    value: Optional[Vec] = None
    jump: bool = False
    jump_angle: float = 0.0

    def reset(self, normal: Optional[Vec] = None) -> None:
        self.value = normal
        self.jump = False
        self.jump_angle = 0.0

    def update(self, normal: Optional[Vec], dt: float) -> Optional[Vec]:
        if normal is None:
            self.jump = False
            self.jump_angle = 0.0
            return self.value
        normal = _unit(normal)
        if self.value is None:
            self.value = normal
            self.jump = False
            self.jump_angle = 0.0
            return self.value
        delta = angle_between(self.value, normal)
        self.jump_angle = delta
        self.jump = math.degrees(delta) > self.jump_limit_deg
        alpha = 1.0 - math.exp(-max(dt, 0.0) / max(self.tau, 1e-9))
        blended = (
            self.value[0] + alpha * (normal[0] - self.value[0]),
            self.value[1] + alpha * (normal[1] - self.value[1]),
        )
        self.value = _unit(blended)
        return self.value


# ---------------------------------------------------------------------------
# reference velocity
# ---------------------------------------------------------------------------


@dataclass
class ReferenceVelocity:
    """Rate-limited reference velocity; the acceleration bound is a hard invariant.

    The reference starts explicitly at zero, so the very first control period is rate limited
    too. There is no ``None -> raw`` shortcut: that path would have let the first command jump
    straight to the cruise value, which is exactly the acceleration limit this class exists to
    enforce.
    """

    a_ref_max: float
    value: Vec = (0.0, 0.0)
    max_step_observed: float = 0.0
    initialised: bool = True

    def reset(self, value: Vec = (0.0, 0.0)) -> None:
        self.value = value
        self.max_step_observed = 0.0
        self.initialised = True

    def update(self, raw: Vec, dt: float) -> Vec:
        dt = max(dt, 1e-9)
        step = _sub(raw, self.value)
        limit = self.a_ref_max * dt
        bounded = clip_norm(step, limit)
        self.max_step_observed = max(self.max_step_observed, _norm(bounded))
        self.value = (self.value[0] + bounded[0], self.value[1] + bounded[1])
        return self.value


def raw_reference(normal: Vec, alpha: float, speed: float) -> Vec:
    tangent = rotate90(normal)
    return (
        speed * (math.cos(alpha) * normal[0] + math.sin(alpha) * tangent[0]),
        speed * (math.cos(alpha) * normal[1] + math.sin(alpha) * tangent[1]),
    )


def approach_reference(observation: LocalContactObservation, *, min_speed: float, max_speed: float,
                       gain: float, contact_speed: float) -> Optional[Vec]:
    """Reference velocity for APPROACH: along the normal, slower as the gap closes."""

    normal = observation.unit_normal_world()
    if normal is None or observation.patch is None:
        return None
    gap = max(0.0, observation.patch.distance - observation.robot_radius)
    if gap <= 0.005:
        magnitude = contact_speed
    else:
        magnitude = min(max_speed, max(min_speed, gain * gap))
    return (normal[0] * magnitude, normal[1] * magnitude)


def velocity_feedback_force(reference: Vec, velocity: Vec, kv: float, limit: float) -> Vec:
    return clip_norm((kv * (reference[0] - velocity[0]), kv * (reference[1] - velocity[1])), limit)


# ---------------------------------------------------------------------------
# control state
# ---------------------------------------------------------------------------


@dataclass
class LocalControlState:
    """Controller-internal state. It is reset between skills, never between control periods."""

    alpha: float = 0.0
    speed: float = 0.0
    force_limit: float = 15.0
    kv: float = 40.0
    a_ref_max: float = 0.5
    normal: NormalFilter = field(default_factory=NormalFilter)
    reference: Optional[ReferenceVelocity] = None
    approach_min_speed: float = 0.02
    approach_max_speed: float = 0.08
    approach_gain: float = 2.0
    approach_contact_speed: float = 0.02
    ready_precontact_speed: float = 0.01
    previous_world_points: Optional[List[Vec]] = None
    previous_pose: Optional[Tuple[float, float, float]] = None
    previous_sample_time: Optional[float] = None
    surface_speed: Optional[float] = None          # median of the per-match normal speeds (m/s)
    surface_speed_signed: Optional[float] = None   # signed mean, diagnostic only
    surface_speed_rms: Optional[float] = None      # RMS, diagnostic only
    surface_matches: int = 0
    surface_unavailable_reason: Optional[str] = None
    surface_speed_sample_time: Optional[float] = None
    monotonic_time: float = 0.0
    flags: List[str] = field(default_factory=list)
    max_step: float = 0.0

    def start(self, *, alpha: float, speed: float, force_limit: float, kv: float,
              a_ref_max: float, approach_min_speed: float = 0.02,
              approach_max_speed: float = 0.08, approach_gain: float = 2.0,
              approach_contact_speed: float = 0.02,
              ready_precontact_speed: float = 0.01) -> None:
        self.approach_min_speed = approach_min_speed
        self.approach_max_speed = approach_max_speed
        self.approach_gain = approach_gain
        self.approach_contact_speed = approach_contact_speed
        self.ready_precontact_speed = ready_precontact_speed
        self.alpha = alpha
        self.speed = speed
        self.force_limit = force_limit
        self.kv = kv
        self.a_ref_max = a_ref_max
        self.reference = ReferenceVelocity(a_ref_max=a_ref_max)
        self.normal.reset()
        self.previous_world_points = None
        self.previous_pose = None
        self.previous_sample_time = None
        self.surface_speed = None
        self.surface_speed_signed = None
        self.surface_speed_rms = None
        self.surface_matches = 0
        self.surface_unavailable_reason = None
        self.surface_speed_sample_time = None
        self.flags = []

    def note(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


def local_push_force(
    observation: LocalContactObservation,
    state: LocalControlState,
    dt: float,
    *,
    mode: str,
) -> Tuple[Vec, Dict[str, object]]:
    """One control period of the new law. Returns ``(force, diagnostics)``."""

    diagnostics: Dict[str, object] = {"mode": mode}
    patch = observation.patch
    normal = state.normal.update(observation.unit_normal_world() if patch else None, dt)
    diagnostics["normal_jump"] = state.normal.jump
    diagnostics["normal_jump_deg"] = math.degrees(state.normal.jump_angle)
    if patch is None or normal is None:
        state.note("surface_unobserved")
        state.surface_speed = None
        state.surface_speed_rms = None
        state.surface_matches = 0
        return (0.0, 0.0), {**diagnostics, "reason": "surface_unobserved"}

    if state.normal.jump:
        state.note("face_switch")

    if mode == "approach":
        raw = approach_reference(
            observation, min_speed=state.approach_min_speed,
            max_speed=state.approach_max_speed, gain=state.approach_gain,
            contact_speed=state.approach_contact_speed,
        ) or (0.0, 0.0)
    elif mode == "ready":
        magnitude = state.ready_precontact_speed
        raw = (normal[0] * magnitude, normal[1] * magnitude)
    else:
        raw = raw_reference(normal, state.alpha, state.speed)

    if state.reference is None:
        state.reference = ReferenceVelocity(a_ref_max=state.a_ref_max)
    reference = state.reference.update(raw, dt)
    state.max_step = state.reference.max_step_observed
    own = observation.own
    force = velocity_feedback_force(reference, (own.vx, own.vy), state.kv, state.force_limit)
    diagnostics.update({
        "reference": reference,
        "normal": normal,
        "reference_step": _norm(_sub(raw, reference)),
    })
    diagnostics["surface_speed"] = state.surface_speed
    diagnostics["surface_speed_signed"] = state.surface_speed_signed
    diagnostics["surface_speed_rms"] = state.surface_speed_rms
    diagnostics["surface_matches"] = state.surface_matches
    diagnostics["surface_unavailable_reason"] = state.surface_unavailable_reason
    return force, diagnostics


def brake_force(
    observation: LocalContactObservation, state: LocalControlState, dt: float
) -> Vec:
    """Brake towards a reference that is *itself* rate limited down to zero.

    The stage machine used to compute the ramp and then command ``Kv * (0 - v)``, so the ramp
    never reached the actuator. The force is now built from the rate-limited reference, which
    is the whole point of the acceleration bound.
    """

    if state.reference is None:
        state.reference = ReferenceVelocity(a_ref_max=state.a_ref_max)
    reference = state.reference.update((0.0, 0.0), dt)
    own = observation.own
    return velocity_feedback_force(reference, (own.vx, own.vy), state.kv, state.force_limit)


# ---------------------------------------------------------------------------
# local surface motion estimate
# ---------------------------------------------------------------------------


def world_points(patch: LocalPatch, pose: Tuple[float, float, float]) -> List[Vec]:
    """Map the robot-frame returns into the shared world frame using the robot's own pose."""

    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    out = []
    for px, py in patch.points_robot:
        out.append((pose[0] + px * cos - py * sin, pose[1] + px * sin + py * cos))
    return out


def update_surface_estimate(
    state: LocalControlState, patch: Optional[LocalPatch], pose: Tuple[float, float, float],
    sample_time: Optional[float], now: float, *, tolerance: float = SURFACE_MATCH_TOLERANCE,
    freshness: float = SURFACE_FRESHNESS, consistency: float = SURFACE_CONSISTENCY,
) -> Tuple[Optional[float], Optional[float], Optional[float], int, Optional[str]]:
    """Normal-direction speed of the observed surface, in m/s, from two consecutive scans.

    Returns ``(median_abs, signed_mean, rms, matches, unavailable_reason)``. The estimator
    divides the matched normal displacement by the **difference of the two scans' sample
    timestamps**, not by the control period.

    Aggregation is the **median of the per-match absolute normal speeds**, not the RMS. The RMS
    was chosen to stop a symmetric +/- motion from cancelling to zero, and it does fix that -- but
    it is dominated by mis-associations. The ray spacing on a 0.08 m patch is about 0.014 m and
    the association tolerance is 0.05 m, so when the point set changes (a ray enters or leaves
    the visible span) a point is matched to its neighbour and reports a displacement of one
    spacing, i.e. 0.014 / 0.05 s = 0.28 m/s. Measured on genuinely static states the RMS reads a
    median of 0.040 m/s against a 0.02 m/s threshold. The median of the absolute values keeps the
    anti-cancellation property and ignores the minority of mis-associated points: a really moving
    surface moves *all* of its points, so the median still sees it.

    ``rms`` and the signed mean are still returned as diagnostics.
    """

    if patch is None:
        return None, None, None, 0, "no_patch"
    if sample_time is None:
        return None, None, None, 0, "no_sample_time"
    if now - sample_time > freshness:
        return None, None, None, 0, "stale_frame"
    previous_time = state.previous_sample_time
    previous_points = state.previous_world_points
    previous_pose = state.previous_pose
    if previous_points is None or previous_pose is None or previous_time is None:
        state.previous_world_points = world_points(patch, pose)
        state.previous_pose = pose
        state.previous_sample_time = sample_time
        return None, None, None, 0, "no_previous_scan"
    dt = sample_time - previous_time
    if dt <= 0.0:
        return None, None, None, 0, "non_positive_dt" if dt < 0.0 else "repeated_timestamp"

    current = world_points(patch, pose)
    normal = patch.normal_world
    speeds: List[float] = []
    for point in current:
        best = None
        best_distance = tolerance
        for candidate in previous_points:
            distance = _norm(_sub(point, candidate))
            if distance < best_distance:
                best_distance = distance
                best = candidate
        if best is None:
            continue
        delta = _sub(point, best)
        speeds.append((delta[0] * normal[0] + delta[1] * normal[1]) / dt)
    matches = len(speeds)
    if matches < SURFACE_MATCH_MIN_POINTS:
        return None, None, None, matches, "insufficient_matches"
    absolute = sorted(abs(value) for value in speeds)
    median_abs = absolute[len(absolute) // 2] if len(absolute) % 2 else (
        0.5 * (absolute[len(absolute) // 2 - 1] + absolute[len(absolute) // 2]))
    rms = math.sqrt(sum(value * value for value in speeds) / matches)
    signed = sum(speeds) / matches
    state.previous_world_points = current
    state.previous_pose = pose
    state.previous_sample_time = sample_time
    state.surface_consistency = consistency - abs(rms - median_abs)
    return median_abs, signed, rms, matches, None
