"""Exact evidence operations batched through GEOS, with indexed hit queries."""
import math

import shapely
from shapely.geometry import LineString
from shapely import STRtree

from ..symbolic_unknown.evidence import EvidenceMap, RegionEvidence


class CachedEvidenceMap(EvidenceMap):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._eroded = None
        self._points = None
        self._hit_tree = None

    def update(self, observation):
        if observation["tick"] == self.last_tick:
            return
        # Same arrangement and cell-area rule as EvidenceMap. Keep polygon
        # extraction/area filtering in native vectorized GEOS calls rather
        # than constructing hundreds of thousands of Python geometry wrappers.
        tick = observation["tick"]
        segments = []
        for robot in observation["robots"]:
            scan = robot["scan"]
            origin = scan["origin"]
            if len(scan["distances"]) != 72:
                raise ValueError("fixed 72-ray profile required")
            for angle, distance, valid, category in zip(scan["ray_angles"], scan["distances"],
                                                       scan["valid"], scan["category"]):
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError("invalid range")
                c, s = math.cos(angle), math.sin(angle)
                length = max(0., distance-self.endpoint_erosion)
                if length > 1e-5:
                    segments.append(LineString([origin, (origin[0]+length*c, origin[1]+length*s)]))
                if valid and category == 2:
                    hit = (origin[0]+distance*c, origin[1]+distance*s)
                    self.hits[(round(hit[0],6), round(hit[1],6))] = tick
        self.scans.append(segments)
        self.scans = self.scans[-self.recent_scans:]
        if self.minimum_area is not None:
            lines = [line for scan in self.scans for line in scan]
            if not self.free.is_empty:
                lines.append(self.free.boundary)
            cells = shapely.get_parts(shapely.polygonize(shapely.get_parts(shapely.union_all(lines))))
            areas = shapely.area(cells)
            cells = cells[(areas > 1e-10) & (areas < self.minimum_area-1e-8)]
            if len(cells):
                self.ledger["certified_cells"] += len(cells)
                self.free = shapely.union_all([self.free, *cells])
        self.ticks.append(tick)
        self.last_tick = tick
        self.ledger["scans"] += 1
        self.ledger["rays"] += len(segments)
        self._eroded = None
        self._points = None
        self._hit_tree = None

    def add_initial_certificate(self, geometry, *, source):
        super().add_initial_certificate(geometry, source=source)
        self._eroded = None

    def forget_outside(self, retained_region):
        super().forget_outside(retained_region)
        self._eroded = None

    def query(self, region):
        if self._eroded is None:
            self._eroded = self.free.buffer(-self.numerical_margin)
        if self._hit_tree is None:
            self._points = shapely.points(list(self.hits)) if self.hits else []
            self._hit_tree = STRtree(self._points)
        # STRtree prunes by envelopes, then GEOS applies the original covers
        # predicate, including boundary points. No hit coordinates are dropped.
        hit_count = len(self._hit_tree.query(region, predicate="covers"))
        missing = region.difference(self._eroded)
        if hit_count:
            status = "occupied"
            if missing.area < 1e-10:
                self.ledger["sensor_conflicts"] += 1
        else:
            status = "free" if missing.is_empty else "unknown"
        return RegionEvidence(status, float(missing.area), hit_count,
                              tuple(self.ticks), self.assumptions)
