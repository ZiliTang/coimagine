"""Own-scan, order-preserving contact reservations for two pushers on one face.

No true face identity or peer private target is consumed. A visible peer on the
outward side of the locally fitted face is a conservative same-face hypothesis.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from . import spec


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def add(a, b):
    return (a[0] + b[0], a[1] + b[1])


def mul(a, b):
    return (a[0] * b, a[1] * b)


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class Reservation:
    status: str
    side: int | None = None
    interval: tuple | None = None
    preferred: tuple | None = None
    target: tuple | None = None
    peer: tuple | None = None
    target_separation: float = 0.
    avoidance_margin: float = spec.OBSTACLE_AVOID_MARGIN
    closing_speed: float = 0.
    peer_age: float = 0.


class SameFaceReservation:
    def __init__(self, clearance=.04, target_pad=.01, end_inset=.04, peer_max_age=.25):
        if any(not math.isfinite(v) or v <= 0 for v in (clearance, target_pad, end_inset, peer_max_age)):
            raise ValueError("same-face reservation parameters must be finite and positive")
        if clearance > spec.OBSTACLE_AVOID_MARGIN:
            raise ValueError("working clearance cannot exceed the conservative avoidance margin")
        self.clearance, self.target_pad = clearance, target_pad
        self.end_inset, self.peer_max_age = end_inset, peer_max_age
        self.side = None
        self.last_peer = None
        self.last_time = None
        self.last_range = None
        self.closing_speed = 0.
        self.peer_velocity = (0., 0.)

    def project(self, face, observation, preferred, dt, *, occupied_only=False):
        n, t = face.normal, (-face.normal[1], face.normal[0])
        own = (observation.own.x, observation.own.y)
        peer = None
        if observation.teammate_nearest is not None:
            distance, hit = observation.teammate_nearest
            direction = sub(hit, own)
            norm = math.hypot(*direction)
            if norm > 1e-8:
                peer = add(own, mul(direction, (distance + spec.ROBOT_RADIUS) / norm))
        if occupied_only and peer is not None:
            # A mixed-option peer need not share a target or surface. Reserve only
            # a currently observed nearby outward-side body, never an opposite face
            # or a distant robot whose tangential projection happens to overlap.
            depth = -dot(sub(peer, face.point), n)
            same_side = (spec.ROBOT_RADIUS-.03 <= depth <= spec.ROBOT_RADIUS+.35
                         and abs(dot(sub(peer, face.point), t)) <= face.length/2+spec.ROBOT_RADIUS)
            if not same_side:
                self.side = self.last_peer = self.last_time = self.last_range = None
                self.closing_speed = 0.
                return Reservation('not_same_face', target=preferred)
        if self.side is None:
            # Both centres must be on the outward side and over this face's
            # extent; adjacent/opposite faces do not enable reduced clearance.
            same_side = (peer is not None
                         and dot(sub(peer, face.point), n) <= -spec.ROBOT_RADIUS + .03
                         and abs(dot(sub(peer, face.point), t)) <= face.length / 2 + spec.ROBOT_RADIUS)
            if not same_side:
                return Reservation("not_same_face", target=preferred)
            relative = dot(sub(own, peer), t)
            self.side = -1 if relative < -1e-5 else 1 if relative > 1e-5 else (-1 if observation.robot_id == 0 else 1)
        age = 0.
        first_peer_sample = self.last_time is None
        if peer is not None:
            r = math.dist(peer, own)
            if self.last_time is not None and observation.time > self.last_time:
                elapsed = observation.time - self.last_time
                closing = (self.last_range - r) / elapsed
                self.closing_speed = .5 * self.closing_speed + .5 * clamp(closing, 0., .5)
                v = mul(sub(peer, self.last_peer), 1 / elapsed)
                speed = math.hypot(*v)
                self.peer_velocity = mul(v, min(1., .3 / max(speed, 1e-9)))
            self.last_peer, self.last_time, self.last_range = peer, observation.time, r
        elif self.last_peer is not None:
            age = observation.time - self.last_time
            if age > self.peer_max_age:
                return Reservation("peer_unobserved", side=self.side, preferred=preferred, peer_age=age)
            peer = add(self.last_peer, mul(self.peer_velocity, age))
        else:
            return Reservation("peer_unobserved", side=self.side, preferred=preferred)
        # Static clearance is smaller; rapid closure retains the old buffer.
        closing = self.closing_speed
        margin = min(spec.OBSTACLE_AVOID_MARGIN,
                     self.clearance + .1 * closing + closing * closing / (2 * spec.A_REF_MAX))
        if age > 0 or first_peer_sample:
            margin = spec.OBSTACLE_AVOID_MARGIN
        separation = 2 * spec.ROBOT_RADIUS + self.clearance + self.target_pad + .3 * age
        q = dot(face.point, t)
        lo, hi = q - face.length / 2 + self.end_inset, q + face.length / 2 - self.end_inset
        if not occupied_only and hi - lo < separation:
            return Reservation("no_room", side=self.side, preferred=preferred, peer=peer,
                               target_separation=separation, avoidance_margin=margin)
        # Both robots can estimate this separator from their own pose and peer
        # return. It avoids placing two reservation halves around inconsistent
        # independently inferred object centres.
        separator = .5 * (dot(own, t) + dot(peer, t))
        if self.side < 0:
            hi = min(hi, dot(peer, t) - separation)
            if not occupied_only:hi = min(hi, separator - separation / 2)
        else:
            lo = max(lo, dot(peer, t) + separation)
            if not occupied_only:lo = max(lo, separator + separation / 2)
        # Preserve actual visibility caps, including teammate-shadow boundaries.
        raw = [dot(p, t) for p in face.patch.points_world]
        a, b = face.tracking.contact_limits if face.tracking else (min(raw), max(raw))
        visibility_inset = min(.02, max(0., (b - a) / 4))
        lo, hi = max(lo, a + visibility_inset), min(hi, b - visibility_inset)
        if hi < lo:
            return Reservation("lane_unavailable", side=self.side, interval=(lo, hi), preferred=preferred,
                               peer=peer, target_separation=separation, avoidance_margin=margin,
                               closing_speed=closing, peer_age=age)
        target = add(preferred, mul(t, clamp(dot(preferred, t), lo, hi) - dot(preferred, t)))
        return Reservation("allocated", side=self.side, interval=(lo, hi), preferred=preferred,
                           target=target, peer=peer, target_separation=separation,
                           avoidance_margin=margin, closing_speed=closing, peer_age=age)
