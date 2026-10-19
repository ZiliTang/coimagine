"""Anchor tracking: locate the object's face corners from the robot's own scan.

Rules that come from the round-006 occlusion measurement:

* a visible run's endpoint is a **legal corner candidate** only when the next ray is a
  non-return (``object -> none``); an endpoint that meets a teammate return is a **shadow
  boundary** and is never treated as a corner;
* one legal corner plus the observed face direction and a **deployable** face-length prior
  recovers the other end; the result is reported as a candidate, not as observed geometry;
* when the evidence is weak the tracker says so: ``uncertain`` for an inferred corner,
  ``anchor_unobservable`` when no legal corner is visible at all. It never claims to know
  hidden geometry, and ``both_corners_occluded`` stays an *offline* label;
* a material change of the tracked face raises an explicit ``face_switch`` instead of silently
  following the new surface.

Everything here consumes the robot's own scan (ray angles, distances, valid flags, categories)
and its own pose only. Object dimensions enter as a **controller prior** passed at construction,
never read from instance truth.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

Vec = Tuple[float, float]
STATUS_OBSERVABLE = "observable"
STATUS_UNCERTAIN = "uncertain"
STATUS_UNOBSERVABLE = "anchor_unobservable"


@dataclass
class RunReport:
    ray_first: int
    ray_last: int
    rays: int
    end_a_world: Vec
    end_b_world: Vec
    span_m: float
    boundary_before: int
    boundary_after: int
    before_is_occlusion: bool
    after_is_occlusion: bool
    points: List[Vec] = field(default_factory=list)


@dataclass
class AnchorEstimate:
    status: str
    candidates: List[Vec] = field(default_factory=list)
    observed_corners: List[Vec] = field(default_factory=list)
    face_direction: Optional[Vec] = None
    face_length_prior: float = 0.0
    observation_age_s: Optional[float] = None
    confidence: float = 0.0
    face_switch: bool = False
    legal_candidates: int = 0
    shadow_boundaries: int = 0
    object_rays: int = 0
    reason: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "candidates": [list(point) for point in self.candidates],
            "observed_corners": [list(point) for point in self.observed_corners],
            "face_direction": (list(self.face_direction) if self.face_direction else None),
            "face_length_prior": self.face_length_prior,
            "observation_age_s": self.observation_age_s,
            "confidence": self.confidence,
            "face_switch": self.face_switch,
            "legal_candidates": self.legal_candidates,
            "shadow_boundaries": self.shadow_boundaries,
            "object_rays": self.object_rays,
            "reason": self.reason,
        }


def object_runs(scan) -> List[RunReport]:
    """Circular runs of object returns, with each endpoint's exit category.

    The object frequently straddles ray 0, so a linear scan would split one physical face into
    two runs; runs are therefore found around the circle.
    """

    count = len(scan.valid)
    flags = [bool(scan.valid[index]) and int(scan.category[index]) == 1
             for index in range(count)]
    if not any(flags):
        return []
    if all(flags):
        runs_index = [list(range(count))]
    else:
        starts = [index for index in range(count) if flags[index] and not flags[index - 1]]
        runs_index = []
        for start_index in starts:
            run, index = [], start_index
            while flags[index]:
                run.append(index)
                index = (index + 1) % count
                if index == start_index:
                    break
            runs_index.append(run)

    def point(index):
        angle = scan.ray_angles[index]
        distance = float(scan.distances[index])
        return (scan.origin[0] + distance * math.cos(angle),
                scan.origin[1] + distance * math.sin(angle))

    reports = []
    for run in runs_index:
        first, last = run[0], run[-1]
        start_point, end_point = point(first), point(last)
        before = int(scan.category[(first - 1) % count])
        after = int(scan.category[(last + 1) % count])
        reports.append(RunReport(
            ray_first=first, ray_last=last, rays=len(run),
            end_a_world=start_point, end_b_world=end_point,
            span_m=math.dist(start_point, end_point),
            boundary_before=before, boundary_after=after,
            before_is_occlusion=before == 3, after_is_occlusion=after == 3,
            points=[point(index) for index in run]))
    return reports


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


class AnchorTracker:
    """Per-robot anchor estimate with an explicit, inspectable status."""

    def __init__(self, face_length_priors: Sequence[float] = (1.00, 0.25),
                 face_switch_deg: float = 30.0,
                 min_visible_span_m: float = 0.10) -> None:
        self.face_length_priors = tuple(float(value) for value in face_length_priors)
        self.face_switch_deg = float(face_switch_deg)
        self.min_visible_span_m = float(min_visible_span_m)
        self.prior_tolerance_m = 0.15
        self.straightness_tolerance_m = 0.02
        self.face_direction: Optional[Vec] = None
        self.last_observable_time: Optional[float] = None
        self.updates = 0
        self.last_key: Optional[Tuple[float, Optional[int]]] = None
        self.last_estimate: Optional["AnchorEstimate"] = None

    def state(self) -> Dict[str, object]:
        """Everything a branch restore needs to keep the tracker consistent."""
        return {"face_direction": (list(self.face_direction)
                                   if self.face_direction is not None else None),
                "last_observable_time": self.last_observable_time,
                "updates": self.updates,
                "last_key": (list(self.last_key) if self.last_key is not None else None)}

    def load_state(self, state: Dict[str, object]) -> None:
        direction = state.get("face_direction")
        self.face_direction = (tuple(direction) if direction is not None else None)
        self.last_observable_time = state.get("last_observable_time")
        self.updates = int(state.get("updates", 0))
        key = state.get("last_key")
        self.last_key = (float(key[0]), key[1]) if key is not None else None
        self.last_estimate = None

    def update(self, scan, time: float, dt: float, tick=None) -> AnchorEstimate:
        # one update per physical period: a second observe() at the same tick must not reset the
        # observation age or re-count a face switch
        key = (float(time), None if tick is None else int(tick))
        if self.last_key == key and self.last_estimate is not None:
            return self.last_estimate
        self.last_key = key
        self.updates += 1
        reports = object_runs(scan)
        legal: List[Vec] = []
        shadow = 0
        for report in reports:
            for end_key, occlusion_key in (("end_a_world", "before_is_occlusion"),
                                           ("end_b_world", "after_is_occlusion")):
                if report.__getattribute__(occlusion_key):
                    shadow += 1
                else:
                    legal.append(report.__getattribute__(end_key))
        span = max((report.span_m for report in reports), default=0.0)
        direction = None
        if reports:
            main = max(reports, key=lambda report: report.span_m)
            chord = (main.end_b_world[0] - main.end_a_world[0],
                     main.end_b_world[1] - main.end_a_world[1])
            norm = math.hypot(chord[0], chord[1])
            if norm > 1e-9:
                direction = (chord[0] / norm, chord[1] / norm)
        # a run that spans two faces is not a face: do not read its endpoints as corners
        if reports:
            main = max(reports, key=lambda report: report.span_m)
            if len(main.points) >= 3 and main.span_m > 1e-6:
                start, end = main.points[0], main.points[-1]
                chord = (end[0] - start[0], end[1] - start[1])
                chord_norm = math.hypot(*chord)
                if chord_norm > 1e-9:
                    for point in main.points[1:-1]:
                        cross = abs((point[0] - start[0]) * chord[1]
                                    - (point[1] - start[1]) * chord[0]) / chord_norm
                        if cross > self.straightness_tolerance_m:
                            estimate = AnchorEstimate(
                                status=STATUS_UNCERTAIN,
                                observed_corners=list(legal),
                                face_direction=None, legal_candidates=len(legal),
                                shadow_boundaries=shadow,
                                object_rays=sum(report.rays for report in reports),
                                reason="run_not_straight")
                            estimate.observation_age_s = (
                                None if self.last_observable_time is None
                                else time - self.last_observable_time)
                            self.last_estimate = estimate
                            return estimate
        face_switch = False
        if direction is not None and self.face_direction is not None:
            cosine = max(-1.0, min(1.0, direction[0] * self.face_direction[0]
                                   + direction[1] * self.face_direction[1]))
            if math.degrees(math.acos(cosine)) > self.face_switch_deg:
                face_switch = True
        if direction is not None:
            self.face_direction = direction

        estimate = AnchorEstimate(
            status=STATUS_UNOBSERVABLE, candidates=[], observed_corners=list(legal),
            face_direction=direction, legal_candidates=len(legal),
            shadow_boundaries=shadow, object_rays=sum(report.rays for report in reports),
            face_switch=face_switch)
        if span < self.min_visible_span_m or not legal:
            estimate.reason = "no_legal_corner_candidate"
            estimate.observation_age_s = (None if self.last_observable_time is None
                                          else time - self.last_observable_time)
            return estimate
        # ``object -> none`` is a *candidate* corner, never a proof; and the edge length is not
        # silently chosen by the closest prior: if no prior matches the observed span the
        # estimate stays uncertain instead of claiming a long/short edge.
        matches = [length for length in self.face_length_priors
                   if abs(length - max(span, 1e-6)) <= self.prior_tolerance_m]
        prior = (min(matches, key=lambda length: abs(length - span)) if matches else None)
        estimate.face_length_prior = prior or 0.0
        if len(legal) >= 2 and prior is not None:
            # both ends of a face are visible and legal: report them as observed candidates
            estimate.status = STATUS_OBSERVABLE
            estimate.candidates = list(legal[:2])
            estimate.confidence = 1.0
            estimate.reason = "two_legal_corners"
            self.last_observable_time = time
            estimate.observation_age_s = 0.0
            self.last_estimate = estimate
            return estimate
        # exactly one legal corner: recover the other end from the prior along the observed
        # run direction; report it as an inferred candidate, not as observed geometry
        recovered = []
        if prior is None:
            estimate.reason = "prior_mismatch"
            estimate.observation_age_s = (None if self.last_observable_time is None
                                          else time - self.last_observable_time)
            self.last_estimate = estimate
            return estimate
        for report in reports:
            for end_key, occlusion_key, other_key in (
                    ("end_a_world", "before_is_occlusion", "end_b_world"),
                    ("end_b_world", "after_is_occlusion", "end_a_world")):
                if report.__getattribute__(occlusion_key):
                    continue
                observed = report.__getattribute__(end_key)
                toward = (report.__getattribute__(other_key)[0] - observed[0],
                          report.__getattribute__(other_key)[1] - observed[1])
                norm = math.hypot(toward[0], toward[1])
                if norm < 1e-9:
                    continue
                recovered.append((observed,
                                  (observed[0] + toward[0] / norm * prior,
                                   observed[1] + toward[1] / norm * prior)))
        if recovered:
            observed, inferred = recovered[0]
            estimate.status = STATUS_UNCERTAIN
            estimate.candidates = [observed, inferred]
            # confidence decays with the observation age and the fraction inferred
            age = 0.0 if self.last_observable_time is None else time - self.last_observable_time
            estimate.confidence = max(0.0, 0.6 - 0.05 * age)
            estimate.reason = "one_legal_corner_plus_prior"
            self.last_observable_time = time
            estimate.observation_age_s = 0.0
            self.last_estimate = estimate
            return estimate
        estimate.reason = "legal_corner_without_direction"
        estimate.observation_age_s = (None if self.last_observable_time is None
                                      else time - self.last_observable_time)
        self.last_estimate = estimate
        return estimate
