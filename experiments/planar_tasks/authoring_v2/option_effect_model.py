"""A room-balanced empirical model of one contact PUSH and its specified follow-up.

Coordinates at the decision boundary are public. The library stores observed joint
outcomes as development labels. Intervals are descriptive weighted quantiles, and
even a fully agreeing neighbourhood is empirical evidence, never a static proof.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import fsum, isfinite


SCHEMA = "public-push-neighbours-v1"
# A local row must agree on both flags and lie within each listed tolerance.
# These are neighbourhood widths in physical units, fixed before reading labels.
PUBLIC_SCALES = {
    "pusher_static_gap_m": 0.25,
    "peer_gap_m": 0.35,
    "peer_along_m": 0.35,
    "peer_lateral_m": 0.35,
    "target_ray_fraction": 0.10,
    "push_speed_mps": 0.15,
    "goal_along_m": 0.60,
}
PUBLIC_FLAGS = ("peer_yield", "next_face_offered")
FEATURES = (*PUBLIC_SCALES, *PUBLIC_FLAGS)
MEASUREMENTS = ("forward_m", "lateral_m", "yaw_rad", "peer_motion_m")
EVENTS = ("object_static", "endpoint_followup_available", "followup_completed")
OUTCOMES = (*MEASUREMENTS, *EVENTS)


def _number(value):
    if type(value) not in (int, float) or not isfinite(value):
        raise ValueError("expected a finite number")
    return float(value)


def _flag(value):
    if type(value) not in (bool, int, float) or value not in (0, 1):
        raise ValueError("expected an observed binary flag")
    return bool(value)


def _features(values):
    if set(values) != set(FEATURES):
        raise ValueError(f"expected only the public input fields {FEATURES}")
    if not 0 <= _number(values["target_ray_fraction"]) <= 1:
        raise ValueError("target_ray_fraction is outside [0, 1]")
    return tuple(_number(values[key]) for key in PUBLIC_SCALES) + tuple(
        _flag(values[key]) for key in PUBLIC_FLAGS
    )


def _progress(value):
    value = _number(value)
    if value <= 0:
        raise ValueError("requested contact progress must be positive")
    return value


@dataclass(frozen=True)
class _Row:
    room: str
    progress_m: float
    features: tuple
    outcome: tuple

    @classmethod
    def read(cls, row):
        room = row["room"]
        if type(room) is not str or not room:
            raise ValueError("expected a nonempty room identity for replicate weighting")
        outcome = row["outcome"]
        effects = tuple(_number(outcome[key]) for key in MEASUREMENTS)
        events = tuple(_flag(outcome[key]) for key in EVENTS)
        return cls(room, _progress(row["progress_m"]), _features(row["features"]), effects + events)

    def as_dict(self):
        return dict(room=self.room, progress_m=self.progress_m,
                    features=dict(zip(FEATURES, self.features)),
                    outcome=dict(zip(OUTCOMES, self.outcome)))


def _distance(a, b):
    n = len(PUBLIC_SCALES)
    if a[n:] != b[n:]:
        return float("inf")
    return max(abs(x - y) / scale for x, y, scale in zip(a[:n], b[:n], PUBLIC_SCALES.values()))


def _quantile(values, q):
    cumulative = 0.0
    for value, weight in sorted(values):
        cumulative += weight
        if cumulative + 1e-12 >= q:
            return value
    raise ValueError("empirical weights did not cover the requested quantile")


def _predict_from(groups, status, dose_support_rooms, local_support_rooms):
    count = len(groups)
    samples = []
    for room, rows in sorted(groups.items()):
        weight = 1 / (count * len(rows))
        for row in rows:
            samples.append(dict(room=room, weight=weight, outcome=dict(zip(OUTCOMES, row.outcome))))
    means = {}
    intervals = {}
    variances = {}
    if samples:
        for key in OUTCOMES:
            values = [(s["outcome"][key], s["weight"]) for s in samples]
            mean = fsum(value * weight for value, weight in values)
            means[key] = mean
            if key in MEASUREMENTS:
                intervals[key] = [_quantile(values, 0.1), _quantile(values, 0.9)]
                variances[key] = fsum(weight * (value - mean) ** 2 for value, weight in values)
    return dict(status=status, evidence="empirical", room_count=count,
                dose_support_rooms=dose_support_rooms, local_support_rooms=local_support_rooms, samples=samples,
                mean=means, variance=variances, interval=intervals,
                failure_rate=1 - means["followup_completed"] if samples else None)


class EffectModel:
    """Use nearby rooms for the requested dose; expose unsupported extrapolation.

    Rows from the closest prefix(es) of each room share that room's mass. Room IDs
    are never input features. Ties at the neighbour cutoff include every tied room.
    Pooled distributions use all same-dose rows with equal total mass per room.
    """

    def __init__(self, rows, min_rooms, nearest_rooms):
        if type(min_rooms) is not int or type(nearest_rooms) is not int or not 1 <= min_rooms <= nearest_rooms:
            raise ValueError("expected 1 <= min_rooms <= nearest_rooms")
        self.min_rooms = min_rooms
        self.nearest_rooms = nearest_rooms
        self._rows = tuple(sorted(rows, key=lambda r: (r.room, r.progress_m, r.features, r.outcome)))

    @classmethod
    def fit(cls, rows, *, min_rooms=3, nearest_rooms=6):
        """Fit on explicitly supplied training rows; no calibration or evaluation labels are read."""
        return cls((_Row.read(row) for row in rows), min_rooms, nearest_rooms)

    def to_dict(self):
        return dict(schema=SCHEMA, public_scales=dict(PUBLIC_SCALES), min_rooms=self.min_rooms,
                    nearest_rooms=self.nearest_rooms, rows=[r.as_dict() for r in self._rows])

    @classmethod
    def from_dict(cls, data):
        if data["schema"] != SCHEMA or data["public_scales"] != PUBLIC_SCALES:
            raise ValueError("incompatible effect model schema or neighbourhood scales")
        return cls.fit(data["rows"], min_rooms=data["min_rooms"], nearest_rooms=data["nearest_rooms"])

    def predict(self, progress_m, features):
        """Predict from the candidate and original public prefix only."""
        progress_m = _progress(progress_m)
        query = _features(features)
        groups = defaultdict(list)
        for row in self._rows:
            if row.progress_m == progress_m:
                groups[row.room].append(row)
        nearest = {}
        distances = {}
        for room, rows in groups.items():
            scored = [(_distance(query, row.features), row) for row in rows]
            best = min(d for d, _ in scored)
            if best <= 1:
                distances[room] = best
                nearest[room] = [r for d, r in scored if d <= best + 1e-12]
        if len(groups) < self.min_rooms:
            return _predict_from({}, "unknown", len(groups), len(nearest))
        if len(nearest) < self.min_rooms:
            return _predict_from(groups, "pooled", len(groups), len(nearest))
        radius = sorted(distances.values())[min(self.nearest_rooms, len(nearest)) - 1]
        local = {room: rows for room, rows in nearest.items() if distances[room] <= radius + 1e-12}
        return _predict_from(local, "local", len(groups), len(nearest))
