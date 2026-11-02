"""Rounded local offset following from finite, own-observed surface fragments.

No object pose, rectangle dimensions, face ID, or teammate history is accepted.
A return boundary censored by shadow/range is never promoted to a physical corner.
"""
from dataclasses import dataclass
import math

from .endpoint_tracking import observe_endpoints, plane_of, dot, tangent, add, mul, sub
from .spec import ROBOT_RADIUS, REPOSITION_GAP


@dataclass(frozen=True)
class BoundaryReference:
    patch: object
    point: tuple
    outward: tuple
    distance: float
    uncertainty: float
    mode: str
    evidence: str


class PeerOuterDetour:
    """Follow the outer offset of the union of observed object and shared peer disc.

    The peer pose is already shared in SkillObservation. No peer option, force,
    hidden object geometry or occluded endpoint is needed. The inflated disc is a
    routing reference, not a relaxation of the executor's collision checks.
    """

    def __init__(self, radius=.50, blend=.04):
        self.radius = radius  # 2 * 0.18 m bodies + 0.14 m routing clearance
        self.blend = blend
        self.info = {"version": "peer-outer-v2", "active": False}

    def update(self, observation, boundary, object_error):
        own = (observation.own.x, observation.own.y)
        plane_guard = boundary is None
        if plane_guard:
            # A peer shadow may split the fitted support below our foot. Keep
            # an ALREADY ACTIVE detour with a valid currently observed plane.
            # Its infinite extension is only a conservative exclusion guard;
            # it is never a corner or a source of unseen boundary continuation.
            patch = observation.patch
            if not self.info.get("active") or patch is None or not patch.valid:
                raise ValueError("plane guard requires active detour and current observed patch")
            outward = patch.normal_outward
            plane_distance = dot(sub(own, patch.points_world[0]), outward)
            point = sub(own, mul(outward, plane_distance))
            object_error = min(object_error, plane_distance-ROBOT_RADIUS-REPOSITION_GAP)
        else:
            outward, point = boundary.outward, boundary.point
        peer = (observation.teammate.x, observation.teammate.y)
        radial = sub(own, peer)
        distance = math.hypot(*radial)
        peer_error = distance - self.radius
        # Do not route round a peer on the far side of the observed object plane.
        outside = dot(sub(peer, point), outward) >= -.05
        eligible = outside and distance > 1e-6
        active = eligible and peer_error < object_error + 5*self.blend
        weight = 0.
        error = object_error
        if active:
            # Smooth minimum of the two offset-distance fields: its gradient
            # follows the exterior of their union. A soft minimum stays below
            # BOTH distances, so smoothing adds clearance rather than cutting
            # through the overlap of the two inflated obstacles.
            difference = (peer_error-object_error)/self.blend
            weight = 1/(1+math.exp(max(-700., min(700., difference))))
            minimum = min(object_error, peer_error)
            error = minimum-self.blend*math.log1p(math.exp(-abs(difference)))
            gradient = add(mul(outward,1-weight),mul(radial,weight/distance))
            norm = math.hypot(*gradient)
            if norm > 1e-6:
                outward = mul(gradient,1/norm)
                error /= norm
            else:
                # Degenerate opposing gradients: leave the peer radially,
                # without inventing a hidden object continuation.
                outward = mul(radial,1/distance)
        self.info = {"version": "peer-outer-v2", "active": bool(active),
                     "estimate_time": observation.time,
                     "peer_centre": peer, "reference_radius": self.radius,
                     "centre_distance": distance, "reference_error": error,
                     "peer_weight": weight, "outward": outward,
                     "object_guard": "observed_plane" if plane_guard else "finite_boundary",
                     "peer_outside_observed_plane": outside}
        return outward, error


class RoundedBoundary:
    def __init__(self, tau=.10, endpoint_uncertainty=.04):
        self.tau = tau
        self.endpoint_uncertainty = endpoint_uncertainty
        self.previous_outward = None
        self.previous_face_normal = None

    def update(self, observation, dt):
        own = (observation.own.x, observation.own.y)
        candidates = []
        patches = observation.patches or (() if observation.patch is None else (observation.patch,))
        for patch in patches:
            if not patch.valid or patch.confidence < .15 or len(patch.points_world) < 3:
                continue
            n, t, plane = patch.normal_world, tangent(patch.normal_world), plane_of(patch)
            q = [dot(p, t) for p in patch.points_world]
            lo, hi = min(q), max(q)
            ends = observe_endpoints(observation.scan, patch, patches)
            trusted = [e.observed and e.uncertainty <= self.endpoint_uncertainty for e in ends]
            if trusted[0]:lo = dot(ends[0].point, t)
            if trusted[1]:hi = dot(ends[1].point, t)
            if lo > hi:continue
            along = dot(own, t)
            end = 0 if along < lo else 1 if along > hi else None
            if end is not None and not trusted[end]:
                # Occlusion/internal fitter support is not a convex corner.
                continue
            foot = max(lo, min(hi, along))
            point = add(mul(t, foot), mul(n, plane))
            radial = sub(own, point);distance = math.hypot(*radial)
            if distance < 1e-8 or dot(radial, patch.normal_outward) < -1e-5:
                continue
            outward = mul(radial, 1/distance)
            uncertainty = ends[end].uncertainty if end is not None else 0.
            candidates.append(BoundaryReference(patch, point, outward, distance, uncertainty,
                "corner_arc" if end is not None else "face_offset", ends[end].kind if end is not None else "observed_segment"))
        if not candidates:
            self.previous_outward = self.previous_face_normal = None
            return None
        best = min(candidates, key=lambda c:c.distance)
        # Near a shared corner, retain the previous observed face on numerical ties.
        compatible = [c for c in candidates if self.previous_face_normal is not None
                      and dot(c.patch.normal_world, self.previous_face_normal) > .98
                      and c.distance <= best.distance+.005]
        if compatible:best = min(compatible,key=lambda c:c.distance)
        outward = best.outward
        if self.previous_outward is not None and dot(outward,self.previous_outward) > 0:
            alpha = dt/(self.tau+dt)
            blended = add(mul(self.previous_outward,1-alpha),mul(outward,alpha))
            outward = mul(blended,1/math.hypot(*blended))
        self.previous_outward, self.previous_face_normal = outward, best.patch.normal_world
        return BoundaryReference(best.patch,best.point,outward,best.distance,best.uncertainty,best.mode,best.evidence)
