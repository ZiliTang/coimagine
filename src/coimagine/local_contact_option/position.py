"""Arm X: the surface-position feedback push controller.

    direction = cos(alpha) * n + sin(alpha) * t
    p_ref     = p_contact + beta * delta * direction
    e         = p_ref - p
    F_raw     = Kp * clip_norm(e, e_max) - Kd * v
    F_applied = clip_norm(F_raw, F_max)

with the contact-centre reference built from the *observed* local surface only:

    d         = dot(c - p, n)          # c = observed centroid, n = filtered inward normal
    q         = p + d * n              # foot of the perpendicular
    p_contact = q - r * n              # robot centre when just touching, no penetration
    gap_signed = d - r                 # allowed to be slightly negative

The law differs from the velocity law only by the observed-gap position feedback and by the
``delta`` advance-bias semantics. At zero gap, with the reference settled, ``beta = 1`` and no
clipping, the two are algebraically identical when ``Kp * delta = Kv * s`` and ``Kd = Kv``.

Nothing in this module can reach an object pose, a true face id, an object vertex or a material
contact point. ``q`` and ``p_contact`` are the *current* local geometry, not a fixed point on
the object: this round makes no claim of material contact-point tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .sensing import LocalContactObservation, LocalPatch

Vec = Tuple[float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def _norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def _clip_norm(vector: Vec, limit: float) -> Vec:
    length = _norm(vector)
    if length > limit > 0.0:
        scale = limit / length
        return (vector[0] * scale, vector[1] * scale)
    return vector


def _rotate90(vector: Vec) -> Vec:
    return (-vector[1], vector[0])


def _to_world(point: Vec, pose: Tuple[float, float, float]) -> Vec:
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    return (pose[0] + point[0] * cos - point[1] * sin,
            pose[1] + point[0] * sin + point[1] * cos)


@dataclass
class PositionPushState:
    """Controller-internal state of the surface-position law.

    Reset rules (explicit, as the task book requires):

    * :meth:`start_push` runs once on PUSH entry: ``beta`` restarts from zero and the skill's
      ``delta``/``alpha`` are frozen for the whole push;
    * :meth:`end_push` runs when PUSH is left: the virtual advance bias is withdrawn
      (``beta`` -> 0) and the cached references are dropped, so nothing invasive is carried
      into BRAKE/SETTLE where the common brake law takes over;
    * no method here touches a physical velocity, pose, or the contact history.
    """

    beta: float = 0.0
    beta_time: float = 0.0
    active: bool = False
    p_contact: Optional[Vec] = None
    p_ref: Optional[Vec] = None
    q: Optional[Vec] = None
    gap_signed: Optional[float] = None
    signed_distance: Optional[float] = None
    error: Vec = (0.0, 0.0)
    error_bounded: Vec = (0.0, 0.0)
    force_raw: Vec = (0.0, 0.0)
    support: bool = True
    support_loss: float = 0.0
    tangential_offset: float = 0.0
    line_normal: Optional[Vec] = None
    delta: float = 0.0
    alpha: float = 0.0
    unavailable: Optional[str] = None

    def start_push(self, *, delta: float, alpha: float) -> None:
        self.line_normal = None
        self.beta = 0.0
        self.beta_time = 0.0
        self.active = True
        self.delta = float(delta)
        self.alpha = float(alpha)
        self.support_loss = 0.0
        self.unavailable = None

    def end_push(self) -> None:
        self.active = False
        self.beta = 0.0
        self.beta_time = 0.0
        self.p_ref = None
        self.p_contact = None
        self.q = None
        self.gap_signed = None
        self.signed_distance = None
        self.error = (0.0, 0.0)
        self.error_bounded = (0.0, 0.0)
        self.force_raw = (0.0, 0.0)
        self.support = True
        self.support_loss = 0.0

    def advance(self, dt: float, ramp: float) -> None:
        """Linear 0 -> 1 enable of the virtual advance over ``ramp`` seconds after entry."""

        if not self.active:
            self.beta = 0.0
            return
        self.beta_time = min(max(ramp, 1e-9), self.beta_time + dt)
        self.beta = self.beta_time / ramp if ramp > 1e-9 else 1.0


def patch_centroid_world(patch: LocalPatch, pose: Tuple[float, float, float]) -> Vec:
    """The observed patch's centroid, mapped through the robot's own pose only."""

    points = patch.points_robot
    cx = sum(point[0] for point in points) / len(points)
    cy = sum(point[1] for point in points) / len(points)
    return _to_world((cx, cy), pose)


def segment_support(q: Vec, patch: LocalPatch, pose: Tuple[float, float, float],
                    margin: float) -> Tuple[bool, float]:
    """Is the foot of the perpendicular still over the *observed* segment?

    Returns ``(supported, tangential_offset)`` where the offset is how far past the segment end
    the foot lies (negative before the start). If the foot leaves the observed span by more than
    ``margin`` the arm may not keep pushing against an extrapolated infinite plane; it reports
    ``patch_support_lost`` and the shared protocol decides what happens. No true face id, no
    object vertex and no object pose is read here.
    """

    low = _to_world(patch.segment_robot[0], pose)
    high = _to_world(patch.segment_robot[1], pose)
    direction = _sub(high, low)
    length = _norm(direction)
    if length < 1e-9:
        return False, float("inf")
    direction = (direction[0] / length, direction[1] / length)
    projection = (q[0] - low[0]) * direction[0] + (q[1] - low[1]) * direction[1]
    if projection < -margin:
        return False, projection
    if projection > length + margin:
        return False, projection - length
    return True, 0.0


def surface_reference(observation: LocalContactObservation, *, robot_radius: float,
                      margin: float) -> Optional[Dict[str, object]]:
    """Signed distance, foot of the perpendicular, contact centre and support test.

    ``gap_signed = d - r`` is allowed to go slightly negative. Clamping it with ``max(gap, 0)``
    would hide penetration from the controller, which is exactly what must not happen. The
    truth-side penetration computed by the evaluator is never fed back here.
    """

    patch = observation.patch
    filtered = observation.unit_normal_world()
    if patch is None or filtered is None:
        return None
    # The support line is a property of the *observed segment*, so it uses the segment's own
    # fitted normal. Using the low-pass filtered normal here mixes the filter's lag into the
    # geometry: during a rotating push the lag reaches ~10 degrees, which shifts the foot of
    # the perpendicular sideways by |c - p| * sin(lag) ~ 0.035 m -- comparable to the whole
    # observed span (measured 0.07-0.08 m) -- so the support test fired on the filter, not on
    # the contact leaving the visible span. The filtered normal is still the control direction.
    line_normal = patch.normal_world
    length = _norm(line_normal)
    if length < 1e-9:
        return None
    line_normal = (line_normal[0] / length, line_normal[1] / length)
    pose = (observation.own.x, observation.own.y, observation.own.angle)
    p = (pose[0], pose[1])
    c = patch_centroid_world(patch, pose)
    # d = dot(c - p, n) on the support line through the observed centroid
    d = (c[0] - p[0]) * line_normal[0] + (c[1] - p[1]) * line_normal[1]
    q = (p[0] + d * line_normal[0], p[1] + d * line_normal[1])
    p_contact = (q[0] - robot_radius * line_normal[0], q[1] - robot_radius * line_normal[1])
    supported, offset = segment_support(q, patch, pose, margin)
    return {
        "normal": filtered, "line_normal": line_normal, "centroid": c, "q": q,
        "p_contact": p_contact,
        "signed_distance": d, "gap_signed": d - robot_radius,
        "support": supported, "tangential_offset": offset,
    }


def position_push_force(
    observation: LocalContactObservation,
    position: PositionPushState,
    dt: float,
    *,
    robot_radius: float,
    margin: float,
    kp: float,
    kd: float,
    e_max: float,
    beta_ramp: float,
    force_limit: float,
) -> Vec:
    """One control period of the surface-position law (see the module docstring).

    ``beta`` ramps 0 -> 1 over ``beta_ramp`` seconds after PUSH entry. The ramp is a smooth
    enable of the *virtual advance*, not a global acceleration bound on ``p_ref``: the surface
    moves and the normal is re-estimated, so ``p_ref`` can still step. The real reference change
    and the real force change are logged, so the timing diagnostics can attribute a spike
    instead of assuming a cause.
    """

    normal = observation.unit_normal_world()
    if normal is None:
        position.unavailable = "surface_unobserved"
        return (0.0, 0.0)
    tangent = _rotate90(normal)
    direction = (
        math.cos(position.alpha) * normal[0] + math.sin(position.alpha) * tangent[0],
        math.cos(position.alpha) * normal[1] + math.sin(position.alpha) * tangent[1],
    )
    position.advance(dt, beta_ramp)
    reference = surface_reference(observation, robot_radius=robot_radius, margin=margin)
    if reference is None:
        position.unavailable = "no_patch"
        return (0.0, 0.0)
    position.unavailable = None
    p = (observation.own.x, observation.own.y)
    p_contact = reference["p_contact"]
    position.line_normal = reference["line_normal"]
    position.q = reference["q"]
    position.p_contact = p_contact
    position.signed_distance = reference["signed_distance"]
    position.gap_signed = reference["gap_signed"]
    position.support = bool(reference["support"])
    position.tangential_offset = float(reference["tangential_offset"])
    if position.support:
        position.support_loss = 0.0
    else:
        position.support_loss += dt
    bias = (position.beta * position.delta * direction[0],
            position.beta * position.delta * direction[1])
    p_ref = (p_contact[0] + bias[0], p_contact[1] + bias[1])
    position.p_ref = p_ref
    error = _sub(p_ref, p)
    position.error = error
    bounded = _clip_norm(error, e_max)
    position.error_bounded = bounded
    own = observation.own
    raw = (kp * bounded[0] - kd * own.vx, kp * bounded[1] - kd * own.vy)
    position.force_raw = raw
    return _clip_norm(raw, force_limit)


def velocity_law_force(reference: Vec, velocity: Vec, kv: float, force_limit: float) -> Vec:
    """The frozen velocity law's force, repeated here so the equivalence test can compare the
    two *implementations* rather than two copies of the same expression."""

    return _clip_norm((kv * (reference[0] - velocity[0]), kv * (reference[1] - velocity[1])),
                      force_limit)
