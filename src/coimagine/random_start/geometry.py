"""Analytic planar geometry used by the scene sampler and the range sensor.

No engine calls: the sampler, the ray caster and the legality checker must be able to run
without a physics space, so that the same code can screen a candidate (privileged truth is
allowed *here*) and cast a simulated ray later.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

Vec = Tuple[float, float]
Poly = Tuple[Vec, ...]


def rotate(vector: Vec, angle: float) -> Vec:
    cos, sin = math.cos(angle), math.sin(angle)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def rotate90(vector: Vec) -> Vec:
    """+90 degrees in the global right-handed (x, y) frame: CCW seen from above."""

    return (-vector[1], vector[0])


def unit(vector: Vec) -> Vec:
    length = math.hypot(vector[0], vector[1])
    if length < 1e-12:
        return (0.0, 0.0)
    return (vector[0] / length, vector[1] / length)


def box_polygon(cx: float, cy: float, angle: float, half_w: float, half_h: float) -> Poly:
    """Corners in a fixed order (CCW for a right-handed frame)."""

    out: List[Vec] = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        local = (sx * half_w, sy * half_h)
        offset = rotate(local, angle)
        out.append((cx + offset[0], cy + offset[1]))
    return tuple(out)


def aabb(points: Sequence[Vec]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def point_in_convex(point: Vec, poly: Poly) -> bool:
    sign = 0
    count = len(poly)
    for index in range(count):
        ax, ay = poly[index]
        bx, by = poly[(index + 1) % count]
        cross = (bx - ax) * (point[1] - ay) - (by - ay) * (point[0] - ax)
        if abs(cross) < 1e-12:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif sign != current:
            return False
    return True


def point_segment_distance(point: Vec, start: Vec, end: Vec) -> float:
    vx, vy = end[0] - start[0], end[1] - start[1]
    length_sq = vx * vx + vy * vy
    if length_sq <= 1e-18:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    t = ((point[0] - start[0]) * vx + (point[1] - start[1]) * vy) / length_sq
    t = max(0.0, min(1.0, t))
    return math.hypot(point[0] - (start[0] + t * vx), point[1] - (start[1] + t * vy))


def segments_intersect(a: Vec, b: Vec, c: Vec, d: Vec) -> bool:
    def cross(o: Vec, p: Vec, q: Vec) -> float:
        return (p[0] - o[0]) * (q[1] - o[1]) - (p[1] - o[1]) * (q[0] - o[0])

    d1, d2 = cross(c, d, a), cross(c, d, b)
    d3, d4 = cross(a, b, c), cross(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False


def segment_segment_distance(a: Vec, b: Vec, c: Vec, d: Vec) -> float:
    if segments_intersect(a, b, c, d):
        return 0.0
    return min(point_segment_distance(a, c, d), point_segment_distance(b, c, d),
               point_segment_distance(c, a, b), point_segment_distance(d, a, b))


def point_polygon_distance(point: Vec, poly: Poly) -> float:
    """0 when the point is inside or on the boundary of a convex polygon."""

    if point_in_convex(point, poly):
        return 0.0
    count = len(poly)
    return min(point_segment_distance(point, poly[i], poly[(i + 1) % count])
               for i in range(count))


def polygon_distance(first: Poly, second: Poly) -> float:
    """Smallest edge-to-edge distance between two convex polygons (0 if they overlap).

    The closest pair of two disjoint convex polygons always contains a vertex, so the vertex
    to polygon distances are sufficient.
    """

    for point in first:
        if point_in_convex(point, second):
            return 0.0
    for point in second:
        if point_in_convex(point, first):
            return 0.0
    best = min(point_polygon_distance(point, second) for point in first)
    best = min(best, min(point_polygon_distance(point, first) for point in second))
    return best


def circle_polygon_gap(centre: Vec, radius: float, poly: Poly) -> float:
    """Edge-to-edge gap between a disc and a convex polygon; negative means overlap."""

    return point_polygon_distance(centre, poly) - radius


def circle_circle_gap(first: Vec, first_radius: float, second: Vec, second_radius: float) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1]) - first_radius - second_radius


def wall_distance(point: Vec, arena: Tuple[float, float]) -> float:
    """Distance from a point to the nearest arena boundary *surface* (capsule included)."""

    width, height = arena
    from .spec import BOUNDARY_RADIUS

    inner = min(point[0], point[1], width - point[0], height - point[1])
    return inner - BOUNDARY_RADIUS


def corners_inside(poly: Poly, arena: Tuple[float, float], margin: float) -> bool:
    """Every corner at least ``margin`` away from the boundary surface."""

    from .spec import BOUNDARY_RADIUS

    inset = BOUNDARY_RADIUS + margin
    width, height = arena
    for x, y in poly:
        if x < inset or y < inset or x > width - inset or y > height - inset:
            return False
    return True


def ray_segment(origin: Vec, direction: Vec, a: Vec, b: Vec) -> Optional[float]:
    """Smallest ``t >= 0`` with ``origin + t * direction`` on the segment."""

    ex, ey = b[0] - a[0], b[1] - a[1]
    denom = direction[0] * ey - direction[1] * ex
    if abs(denom) < 1e-12:
        return None
    t = ((a[0] - origin[0]) * ey - (a[1] - origin[1]) * ex) / denom
    u = ((a[0] - origin[0]) * direction[1] - (a[1] - origin[1]) * direction[0]) / denom
    if t >= 0.0 and -1e-9 <= u <= 1.0 + 1e-9:
        return t
    return None


def ray_circle(origin: Vec, direction: Vec, centre: Vec, radius: float) -> Optional[float]:
    fx, fy = origin[0] - centre[0], origin[1] - centre[1]
    b = 2.0 * (fx * direction[0] + fy * direction[1])
    c = fx * fx + fy * fy - radius * radius
    disc = b * b - 4.0 * c
    if disc < 0.0:
        return None
    root = math.sqrt(disc)
    for t in ((-b - root) / 2.0, (-b + root) / 2.0):
        if t >= 0.0:
            return t
    return None


def polygon_edges(poly: Poly) -> List[Tuple[Vec, Vec]]:
    return [(poly[i], poly[(i + 1) % len(poly)]) for i in range(len(poly))]
