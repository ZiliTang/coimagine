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
            before_is_occlusion=before == 3, after_is_occlusion=after == 3))
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
        self.face_direction: Optional[Vec] = None
        self.last_observable_time: Optional[float] = None
        self.updates = 0

    def update(self, scan, time: float, dt: float) -> AnchorEstimate:
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
        prior = min(self.face_length_priors,
                    key=lambda length: abs(length - max(span, 1e-6)))
        estimate.face_length_prior = prior
        if len(legal) >= 2:
            # both ends of a face are visible and legal: report them as observed candidates
            estimate.status = STATUS_OBSERVABLE
            estimate.candidates = list(legal[:2])
            estimate.confidence = 1.0
            estimate.reason = "two_legal_corners"
            self.last_observable_time = time
            estimate.observation_age_s = 0.0
            return estimate
        # exactly one legal corner: recover the other end from the prior along the observed
        # run direction; report it as an inferred candidate, not as observed geometry
        recovered = []
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
            return estimate
        estimate.reason = "legal_corner_without_direction"
        estimate.observation_age_s = (None if self.last_observable_time is None
                                      else time - self.last_observable_time)
        return estimate
