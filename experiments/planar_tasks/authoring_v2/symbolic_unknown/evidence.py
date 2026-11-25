"""Three-valued spatial evidence from public rays, never from unseen geometry.

A free ray is only a line, NOT a free angular sector. Closed arrangement cells
whose boundary consists of observed free segments can be certified empty when
their area is smaller than the declared minimum area of any connected static
obstacle. An obstacle intersecting the cell would have to cross a free boundary
or fit entirely inside the cell; both are excluded by these assumptions.

This is conditional geometry: exact static-world rays (with endpoint erosion),
correct odometry and the stated minimum-component-area prior. Without that
prior no positive-area ray cell is certified. Moving bodies are checked by the
motion layer, not assumed absent because a historical ray was free.
"""
from dataclasses import dataclass
import math

from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import polygonize, unary_union


@dataclass(frozen=True)
class RegionEvidence:
    status: str
    unknown_area_m2: float
    occupied_hits: int
    evidence_ticks: tuple
    assumptions: tuple

    @property
    def certified(self):
        return self.status == 'free'


class EvidenceMap:
    def __init__(self, *, minimum_static_component_area=None, endpoint_erosion=.003,
                 numerical_margin=.001, recent_scans=3):
        if minimum_static_component_area is not None and minimum_static_component_area <= 0:
            raise ValueError('positive minimum component area or None')
        self.minimum_area = minimum_static_component_area
        self.endpoint_erosion = endpoint_erosion
        self.numerical_margin = numerical_margin
        self.recent_scans = recent_scans
        self.free = GeometryCollection()
        self.hits = {}; self.scans = []; self.ticks = []; self.last_tick = None
        self.ledger = dict(scans=0, rays=0, certified_cells=0, sensor_conflicts=0)

    @property
    def assumptions(self):
        return ('static geometry', 'exact public ray and odometry model',
                'minimum connected static obstacle area='+str(self.minimum_area))

    def add_initial_certificate(self, geometry, *, source):
        """Explicit host initial-condition certificate, never an arbitrary fill.

        The caller must preserve its public derivation and independently audit the
        initial-condition assumption. It cannot label an unseen route this way.
        """
        if source != 'declared_initial_separation':
            raise ValueError('unrecognized initial-condition evidence')
        self.free = unary_union([self.free, geometry])

    def update(self, observation):
        tick = observation['tick']
        if tick == self.last_tick:
            return
        segments = []
        for robot in observation['robots']:
            scan = robot['scan']; origin = scan['origin']
            if len(scan['distances']) != 72:
                raise ValueError('fixed 72-ray profile required')
            for angle, distance, valid, category in zip(scan['ray_angles'], scan['distances'],
                                                       scan['valid'], scan['category']):
                if not math.isfinite(distance) or distance < 0:
                    raise ValueError('invalid range')
                c, s = math.cos(angle), math.sin(angle)
                length = max(0., distance-self.endpoint_erosion)
                if length > 1e-5:
                    segments.append(LineString([origin, (origin[0]+length*c, origin[1]+length*s)]))
                if valid and category == 2:
                    hit=(origin[0]+distance*c, origin[1]+distance*s)
                    self.hits[(round(hit[0],6),round(hit[1],6))] = tick
        self.scans.append(segments);self.scans=self.scans[-self.recent_scans:]
        if self.minimum_area is not None:
            lines=[line for scan in self.scans for line in scan]
            if not self.free.is_empty:
                lines.append(self.free.boundary)
            cells=[p for p in polygonize(unary_union(lines))
                   if 1e-10 < p.area < self.minimum_area-1e-8]
            if cells:
                self.free=unary_union([self.free,*cells])
                self.ledger['certified_cells']+=len(cells)
        self.ticks.append(tick);self.last_tick=tick
        self.ledger['scans']+=1;self.ledger['rays']+=len(segments)

    def query(self, region):
        hit_count=sum(region.covers(Point(p))for p in self.hits)
        safe=self.free.buffer(-self.numerical_margin)
        missing=region.difference(safe)
        if hit_count:
            status='occupied'
            if missing.area < 1e-10:self.ledger['sensor_conflicts']+=1
        elif missing.is_empty:
            status='free'
        else:
            status='unknown'
        return RegionEvidence(status,float(missing.area),hit_count,tuple(self.ticks),self.assumptions)

    def forget_outside(self, retained_region):
        """Memory eviction removes evidence; it never turns loss into clearance."""
        self.free=self.free.intersection(retained_region)
        self.scans=[]

    def snapshot(self):
        from shapely.geometry import mapping
        return dict(free=mapping(self.free),hits=[list(p)for p in self.hits],
                    evidence_ticks=list(self.ticks),assumptions=list(self.assumptions),ledger=dict(self.ledger))
