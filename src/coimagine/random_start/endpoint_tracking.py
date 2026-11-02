"""Causal, per-face physical endpoint tracking from one robot's own lidar.

The last return is not a corner. Endpoints are either intersections of two
observed fitted faces, bounded by adjacent rays, or explicitly censored. Public
face length may locate an unobserved end, but never turns it into an observation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Optional, Tuple

from .lidar import CATEGORY_OBJECT, CATEGORY_STATIC, CATEGORY_TEAMMATE, LidarScan, SurfacePatch
from .spec import LIDAR_RANGE

Vec = Tuple[float, float]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def mul(a, scale):
    return (a[0] * scale, a[1] * scale)


def tangent(n):
    return (-n[1], n[0])


def plane_of(patch):
    return sum(dot(p, patch.normal_world) for p in patch.points_world) / len(patch.points_world)


@dataclass(frozen=True)
class EndpointObservation:
    side: int  # 0/1: low/high along the fitted face's consistently oriented tangent
    point: Optional[Vec]
    lower: Optional[Vec]
    upper: Optional[Vec]
    kind: str
    uncertainty: float = 0.0

    @property
    def observed(self):
        return self.point is not None


def observe_endpoints(scan: LidarScan, patch: SurfacePatch, patches=()):
    """No instance geometry, raw simulator labels, or other robot's scan is used."""
    n, t, plane = patch.normal_world, tangent(patch.normal_world), plane_of(patch)
    endpoints = []
    last_points = ((patch.points_world[0], -1), (patch.points_world[-1], 1))
    ordered = sorted(last_points, key=lambda pair: dot(pair[0], t))
    for side, (point, offset) in enumerate(ordered):
        bearing = math.atan2(point[1] - scan.origin[1], point[0] - scan.origin[0])
        k = min(range(len(scan.valid)), key=lambda j: abs(math.atan2(
            math.sin(scan.ray_angles[j] - bearing), math.cos(scan.ray_angles[j] - bearing))))
        j = (k + offset) % len(scan.valid)
        # A corner split can leave a collinear edge return outside this patch.
        # Walk actual adjacent returns on this SAME line before testing its end;
        # never turn an internal fitter split into a physical endpoint.
        for _ in range(len(scan.valid) - 1):
            if not scan.valid[j] or scan.category[j] != CATEGORY_OBJECT:
                break
            direction = (math.cos(scan.ray_angles[j]), math.sin(scan.ray_angles[j]))
            hit = add(scan.origin, mul(direction, scan.distances[j]))
            if abs(dot(hit, n) - plane) >= .005:
                break
            point, k = hit, j
            j = (j + offset) % len(scan.valid)
        ray = (math.cos(scan.ray_angles[j]), math.sin(scan.ray_angles[j]))
        denom = dot(n, ray)
        distance = (plane - dot(n, scan.origin)) / denom if denom > 1e-7 else float("inf")
        kind = "silhouette_interval"
        if distance <= 0 or distance > LIDAR_RANGE:
            kind = "range_censored"
        elif (scan.valid[j] and scan.category[j] in (CATEGORY_STATIC, CATEGORY_TEAMMATE)
              and scan.distances[j] <= distance + .02):
            kind = "shadow_censored"
        elif scan.valid[j] and scan.category[j] == CATEGORY_OBJECT:
            hit = add(scan.origin, mul(ray, scan.distances[j]))
            kind = ("fragment_continues" if abs(dot(hit, n) - plane) < .005
                    else "corner_interval")
        if kind.endswith("censored") or kind == "fragment_continues":
            endpoints.append(EndpointObservation(side, None, None, None, kind))
            continue
        q0, q1 = dot(point, t), dot(add(scan.origin, mul(ray, distance)), t)
        lo, hi = min(q0, q1), max(q0, q1)
        # Analytic intersection is the sub-ray refinement when two fitted faces
        # meet. Neither interpolation nor point count supplies extra confidence.
        for other in patches:
            if other is patch or not other.valid or len(other.points_world) < 3:
                continue
            m = other.normal_world
            determinant = n[0] * m[1] - n[1] * m[0]
            if abs(determinant) < .65 or abs(dot(n, m)) > .35:
                continue
            b = plane_of(other)
            corner = ((plane * m[1] - n[1] * b) / determinant,
                      (n[0] * b - plane * m[0]) / determinant)
            q = dot(corner, t)
            other_t = tangent(m)
            other_q = [dot(p, other_t) for p in other.points_world]
            # Both observed fragments must end near the proposed common corner.
            tolerance = max(.025, 2 * math.dist(point, add(scan.origin, mul(ray, distance))))
            if (lo - .005 <= q <= hi + .005
                    and min(abs(dot(corner, other_t) - min(other_q)),
                            abs(dot(corner, other_t) - max(other_q))) <= tolerance):
                lo, hi = q - .002, q + .002
                kind = "two_face_intersection"
                break
        midpoint = add(mul(t, (lo + hi) / 2), mul(n, plane))
        endpoints.append(EndpointObservation(
            side, midpoint, add(mul(t, lo), mul(n, plane)),
            add(mul(t, hi), mul(n, plane)), kind, (hi - lo) / 2))
    return tuple(endpoints)


def interpolate_face(patch: SurfacePatch, factor=4):
    """Derived points on ONE fitted line, inside observed support, for display only.

    Does not modify SurfacePatch.arc_rays/confidence or create LidarScan returns.
    """
    if factor < 1 or int(factor) != factor:
        raise ValueError("interpolation factor must be a positive integer")
    n, t, plane = patch.normal_world, tangent(patch.normal_world), plane_of(patch)
    q = [dot(p, t) for p in patch.points_world]
    values = [a + (b - a) * j / factor for a, b in zip(q[:-1], q[1:]) for j in range(factor)]
    values.append(q[-1])
    return tuple(add(mul(t, value), mul(n, plane)) for value in values)


@dataclass
class _EndpointState:
    point: Vec
    velocity: Vec
    observed_at: float


@dataclass(frozen=True)
class TrackedFace:
    midpoint: Vec
    anchor: Optional[Vec]
    anchor_side: Optional[int]
    endpoints: tuple
    measurements: tuple
    contact_limits: Tuple[float, float]
    status: str
    age: float
    observed_count: int
    anchor_switched: bool
    endpoint_ages: tuple

    def as_dict(self):
        return {
            "status": self.status, "anchor": self.anchor, "anchor_side": self.anchor_side,
            "endpoints": self.endpoints, "observed_count": self.observed_count,
            "age": self.age, "anchor_switched": self.anchor_switched,
            "endpoint_ages": self.endpoint_ages,
            "measurements": [{"side": m.side, "point": m.point, "lower": m.lower,
                              "upper": m.upper, "kind": m.kind,
                              "uncertainty": m.uncertainty, "observed": m.observed}
                             for m in self.measurements],
        }


class EndpointTracker:
    """Per-option two-end tracker; one fresh physical endpoint suffices to anchor.

    Tangential position/velocity are filtered, the measured surface line supplies
    current normal displacement without delay. A predicted end alone never marks
    the object's full translational effect observable or complete.
    """

    def __init__(self, tau=.10, max_age=.25, max_speed=.30, residual_limit=.15):
        if any(not math.isfinite(v) or v <= 0 for v in (tau, max_age, max_speed, residual_limit)):
            raise ValueError("endpoint tracker parameters must be finite and positive")
        self.tau, self.max_age = tau, max_age
        self.max_speed, self.residual_limit = max_speed, residual_limit
        self.states = [None, None]
        self.anchor_side = None
        self.last_time = None
        self.last_normal = None
        self.last_key = None
        self.last_result = None

    def update(self, scan, patch, patches, length, time):
        key = (scan.robot_id, scan.tick, float(time))
        if key == self.last_key:
            return self.last_result
        n, t, plane = patch.normal_world, tangent(patch.normal_world), plane_of(patch)
        if (self.last_time is not None and time < self.last_time
                or self.last_normal is not None and dot(n, self.last_normal) < math.cos(math.radians(20))):
            self.states, self.anchor_side = [None, None], None
        dt = max(.001, time - self.last_time) if self.last_time is not None else .05
        alpha = dt / (self.tau + dt)
        measurements = observe_endpoints(scan, patch, patches)
        # Intersect BOTH measured endpoint intervals with the public length
        # constraint before filtering. Independently taking each interval's
        # midpoint would make a symmetric static face appear to move sideways.
        if all(m.observed for m in measurements):
            lo = max(dot(measurements[0].lower, t), dot(measurements[1].lower, t) - length)
            hi = min(dot(measurements[0].upper, t), dot(measurements[1].upper, t) - length)
            if lo <= hi + .005:
                if hi < lo:
                    lo = hi = (lo + hi) / 2
                measurements = tuple(replace(m,
                    point=add(mul(t, (lo + hi) / 2 + m.side * length), mul(n, plane)),
                    lower=add(mul(t, lo + m.side * length), mul(n, plane)),
                    upper=add(mul(t, hi + m.side * length), mul(n, plane)),
                    uncertainty=(hi - lo) / 2) for m in measurements)
            else:
                # Contradictory observed geometry must not be reconciled by truth.
                self.last_key, self.last_time, self.last_normal = key, time, n
                self.last_result = None
                return None
        observed = [m.side for m in measurements if m.observed]
        old_anchor = self.anchor_side
        for m in measurements:
            state = self.states[m.side]
            if not m.observed:
                continue
            if state is None or time - state.observed_at > self.max_age:
                self.states[m.side] = _EndpointState(m.point, (0., 0.), time)
                continue
            elapsed = time - state.observed_at
            predicted = add(state.point, mul(state.velocity, elapsed))
            predicted_q = dot(predicted, t)
            lo, hi = dot(m.lower, t), dot(m.upper, t)
            # Interval observations do not measure their midpoint. If the
            # prediction remains inside the measured bracket it needs no shift.
            measured_q = (dot(m.point, t) if m.kind == "two_face_intersection"
                          else max(lo, min(hi, predicted_q)))
            innovation = measured_q - predicted_q
            if abs(innovation) > self.residual_limit:
                # Reacquisition is explicit; don't drag a stale anchor across a jump.
                self.states[m.side] = _EndpointState(m.point, (0., 0.), time)
                continue
            gain = max(alpha, .65) if m.kind == "two_face_intersection" else alpha
            q = dot(predicted, t) + gain * innovation
            p = add(mul(t, q), mul(n, plane))
            v = dot(state.velocity, t) + .5 * gain * gain * innovation / max(elapsed, .001)
            v = max(-self.max_speed, min(self.max_speed, v))
            # Only tangential velocity is predicted. Normal motion is reobserved
            # from the current plane rather than extrapolated into empty space.
            self.states[m.side] = _EndpointState(p, mul(t, v), time)
        if self.anchor_side not in observed:
            self.anchor_side = min(observed, key=lambda i: measurements[i].uncertainty) if observed else self.anchor_side
        fresh = self.anchor_side in observed
        state = self.states[self.anchor_side] if self.anchor_side is not None else None
        age = time - state.observed_at if state else float("inf")
        if state is None or age > self.max_age:
            result = None
        else:
            anchor = add(state.point, mul(state.velocity, age))
            anchor = add(anchor, mul(n, plane - dot(anchor, n)))
            midpoint = add(anchor, mul(t, length / 2 if self.anchor_side == 0 else -length / 2))
            ends = tuple(s.point if s and time - s.observed_at <= self.max_age else None for s in self.states)
            raw = [dot(p, t) for p in patch.points_world]
            limits = [min(raw), max(raw)]
            # At an actual physical edge use its filtered estimate, not the last
            # return. At a shadow boundary retain raw visible support as a cap.
            for m in measurements:
                if m.observed and self.states[m.side] is not None:
                    limits[m.side] = dot(self.states[m.side].point, t)
            result = TrackedFace(midpoint, anchor, self.anchor_side, ends, measurements,
                                 tuple(limits), "observed" if fresh else "predicted",
                                 age, len(observed), old_anchor is not None and old_anchor != self.anchor_side,
                                 tuple(time - s.observed_at if s else None for s in self.states))
        self.last_key, self.last_time, self.last_normal = key, time, n
        self.last_result = result
        return result
