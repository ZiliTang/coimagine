"""One acquisition service, with explicit historical observation profiles.

Geometry stays inside the sensor adapter. Returned frames contain hit surfaces
and opaque hit IDs, never unseen entity poses. Analytic and Chipmunk kernels
remain separate because their rounding and tie rules are not interchangeable.

Physical results are shared only within synchronous read scopes. A physics step
must invalidate the service before/after mutation. No physical cache survives a
scope, so same-tick editor changes between calls cannot reuse stale observations.
Do not mutate the world manually inside a read scope without invalidate().
"""
from contextlib import contextmanager
from dataclasses import dataclass, replace
import math

from .lidar import LidarScan, scan_lidar
from .lidar_native import scan_batch as scan_lidar_batch
from .ray_query import nearest_hits


@dataclass(frozen=True)
class RayFrame:
    scan: LidarScan
    entity_ids: tuple
    hit_points: tuple = ()


class LidarService:
    def __init__(self):
        self.analytic_cache = {}
        self._depth = 0
        self._memo = {}

    @contextmanager
    def read_scope(self, enabled=True):
        """Share immutable scans/views during a synchronous sensor transaction."""
        if not enabled:
            yield
            return
        self._depth += 1
        try:
            yield
        finally:
            self._depth -= 1
            if not self._depth:
                self._memo.clear()

    def invalidate(self):
        self.analytic_cache.clear()
        self._memo.clear()

    def memo(self, key, build):
        if not self._depth:
            return build()
        if key not in self._memo:
            self._memo[key] = build()
        return self._memo[key]

    def analytic(self, arena, obstacles, pose, states, robot_id, tick, time,
                 *, object_polygon, backend):
        """Legacy controller profile; exact old geometry key and arithmetic."""
        key = (tick, time, pose, tuple(states), arena, obstacles, object_polygon, backend)
        previous = self.analytic_cache.get(robot_id)
        if previous is not None and previous[0] == key:
            return previous[1]
        if backend not in ('scalar', 'numpy_batch'):
            raise ValueError('unknown analytic lidar backend')
        scanner = scan_lidar_batch if backend == 'numpy_batch' else scan_lidar
        scan = scanner(arena, obstacles, pose, states, robot_id, tick, time,
                       object_polygon=object_polygon)
        self.analytic_cache[robot_id] = (key, scan)
        return scan

    def physical(self, space, robot_bodies, labels, robot_id, tick, time,
                 origin, heading, *, max_range, rays=72, native=True,
                 include_points=False, cached=True):
        """All-entity hit view. Range, heading and point protocol are explicit.

        include_points preserves Chipmunk's actual intersection coordinates for
        the historical diagnostic probe, instead of reconstructing from alpha.
        Labels are applied only to shapes actually intersected by a ray.
        """
        origin = tuple(origin)
        key = ('physical', id(space), id(labels), tuple(map(id, robot_bodies)),
               robot_id, tick, time, origin, heading, max_range, rays, native, include_points)
        def cast():
            angles = tuple(heading + j * 2 * math.pi / rays for j in range(rays))
            ends = [(origin[0] + max_range * math.cos(a),
                     origin[1] + max_range * math.sin(a)) for a in angles]
            hits = nearest_hits(space, robot_bodies[robot_id], origin, ends,
                                native=native, include_points=include_points)
            distances, valid, categories, ids, points = [], [], [], [], []
            for hit in hits:
                shape, alpha = hit[:2] if hit is not None else (None, 0.)
                label = labels.get(shape) if hit else None
                valid.append(hit is not None)
                ids.append(label)
                distances.append(float(alpha * max_range) if hit else 0.)
                categories.append(1 if label else 3 if hit and shape.body in robot_bodies else 2 if hit else 0)
                if include_points:
                    points.append(hit[2] if hit else None)
            scan = LidarScan(robot_id, tick, time, origin, heading, angles,
                             tuple(distances), tuple(valid), tuple(categories))
            return RayFrame(scan, tuple(ids), tuple(points))
        return self.memo(key, cast) if cached else cast()

    def target_view(self, frame, target):
        """Controller view: other movable entities are non-target obstacles."""
        return self.memo(('target', id(frame), target), lambda: (frame, replace(
            frame.scan, category=tuple(
                1 if label == target and label is not None else 2 if label else category
                for label, category in zip(frame.entity_ids, frame.scan.category)))))[1]
