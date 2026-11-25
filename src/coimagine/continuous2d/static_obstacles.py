"""Observed 2-D static line segments in a shared metric frame. Create one map per scene.

Unknown space is never marked free. Queries describe observed surfaces, with no
assumption about unobserved edges of either the obstacle or the moving footprint.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Surface:
    start: tuple[float, float]
    end: tuple[float, float]
    observed_tick: int

    @property
    def length(self):
        return math.dist(self.start, self.end)

    def basis(self):
        length = self.length
        return ((self.end[0] - self.start[0]) / length,
                (self.end[1] - self.start[1]) / length)


@dataclass(frozen=True)
class DirectionalHit:
    normal_gap: float
    normal_angle: float  # relative to the requested direction, in radians
    surface: Surface


class StaticObstacleMap:
    def __init__(self):
        self._segments = {}

    def add_segment(self, start, end, tick):
        """Insert world XY endpoints. Identical segments retain their last sighting."""
        start, end = tuple(start), tuple(end)
        if any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in (start, end)):
            raise ValueError('finite planar endpoints required')
        if start == end:
            raise ValueError('segment has zero length')
        key = tuple(sorted((start, end)))
        previous = self._segments.get(key)
        if previous is None or previous.observed_tick <= tick:
            self._segments[key] = Surface(start, end, tick)

    def observe_hits(self, hits, tick, *, circular=False, min_length=.04, max_length=.85, tolerance=.008):
        """Adjacent angular samples in world XY; use None for every non-static or missing hit.

        The outer two points define a finite line, the middle observation checks
        straightness. Only wrap neighbourhoods for a full circular scan.
        """
        if len(hits) < 3:
            return
        count = len(hits) if circular else len(hits) - 2
        for i in range(count):
            a, b, c = hits[i], hits[(i + 1) % len(hits)], hits[(i + 2) % len(hits)]
            if a is None or b is None or c is None:
                continue
            length = math.dist(a, c)
            if not min_length <= length <= max_length:
                continue
            tx, ty = (c[0] - a[0]) / length, (c[1] - a[1]) / length
            residual = abs((b[0] - a[0]) * ty - (b[1] - a[1]) * tx)
            if residual <= tolerance:
                self.add_segment(a, c, tick)

    def nearest_distance(self, point):
        """Euclidean distance in metres to the closest finite observed segment."""
        distances = []
        for seg in self._segments.values():
            tx, ty = seg.basis()
            dx, dy = point[0] - seg.start[0], point[1] - seg.start[1]
            along = min(seg.length, max(0., dx * tx + dy * ty))
            distances.append(math.hypot(dx - along * tx, dy - along * ty))
        return min(distances, default=None)

    def forward_surface(self, points, direction, *, lateral_margin, min_alignment,
                        min_gap, max_gap):
        """Closest qualifying normal clearance from footprint points to an observed surface.

        lateral_margin extends the finite segment's tangential acceptance window
        around the footprint centre. The gap is along the wall normal; it does not
        predict collision-free travel distance along the requested direction.
        """
        norm = math.hypot(*direction)
        if norm == 0 or not math.isfinite(norm):
            raise ValueError('finite nonzero motion direction required')
        if not len(points):
            return None
        ux, uy = (v / norm for v in direction)
        cx, cy = (sum(p[k] for p in points) / len(points) for k in (0, 1))
        best = None
        for seg in self._segments.values():
            ax, ay = seg.start
            tx, ty = seg.basis()
            along = (cx - ax) * tx + (cy - ay) * ty
            if not -lateral_margin <= along <= seg.length + lateral_margin:
                continue
            nx, ny = -ty, tx
            if nx * (ax - cx) + ny * (ay - cy) < 0:
                nx, ny = -nx, -ny
            alignment = nx * ux + ny * uy
            if alignment <= min_alignment:
                continue
            gap = min((ax - x) * nx + (ay - y) * ny for x, y in points)
            angle = math.atan2(ux * ny - uy * nx, alignment)
            if min_gap <= gap <= max_gap and (best is None or (gap, angle) < (best.normal_gap, best.normal_angle)):
                best = DirectionalHit(gap, angle, seg)
        return best
