"""Anchor tracker: legal corners only, explicit status, never a shadow boundary as a corner."""

import math

import pytest

from coimagine.random_start import anchor


class _Scan:
    def __init__(self, origin, heading, distances, categories):
        self.origin = origin
        self.heading = heading
        self.count = len(distances)
        self.ray_angles = tuple(heading + 2.0 * math.pi * index / self.count
                                for index in range(self.count))
        self.distances = tuple(distances)
        self.valid = tuple(category != 0 for category in categories)
        self.category = tuple(categories)


def _scan_from_polyline(origin, heading, points_by_angle, count=72):
    """A tiny synthetic scan: a category per ray, distances from the given hit points."""
    distances, categories = [], []
    for index in range(count):
        angle = heading + 2.0 * math.pi * index / count
        hit = points_by_angle(angle)
        if hit is None:
            distances.append(0.0); categories.append(0)
        else:
            distance, category = hit
            distances.append(distance); categories.append(category)
    return _Scan(origin, heading, distances, categories)


def test_endpoint_next_to_a_teammate_is_not_a_corner():
    """A run that ends at a teammate return must count as a shadow boundary, not a corner."""
    origin = (0.0, 0.0)
    heading = 0.0

    def hit(angle):
        if angle < 0.5:                      # object run
            return (1.0, 1)
        if angle < 0.6:                      # teammate immediately after the run
            return (0.5, 3)
        return None

    scan = _scan_from_polyline(origin, heading, hit)
    runs = anchor.object_runs(scan)
    assert runs
    assert any(report.after_is_occlusion for report in runs)
    estimate = anchor.AnchorTracker(face_length_priors=(1.0,)).update(scan, 0.0, 0.05)
    # the legal candidate count excludes the shadow boundary
    assert estimate.shadow_boundaries >= 1


def test_no_object_returns_is_unobservable():
    scan = _scan_from_polyline((0.0, 0.0), 0.0, lambda angle: None)
    estimate = anchor.AnchorTracker().update(scan, 0.0, 0.05)
    assert estimate.status == anchor.STATUS_UNOBSERVABLE
    assert estimate.reason == "no_legal_corner_candidate"
