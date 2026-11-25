"""Explicitly built C++ public-ray kernel; coarse cells retain unknown space.

This is a NEW geometry profile, not a bit-equivalent optimization. Certificates
use a subset of closed ray polygons; all query-intersected grid cells must be
fully certified. Unknown-area summaries are conservative grid approximations.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import tempfile
import os

import numpy as np
import shapely
from shapely.geometry import GeometryCollection, box

from ..symbolic_unknown.evidence import RegionEvidence

SOURCE = Path(__file__).with_name("geometry_kernel.cpp")
CACHE = Path(__file__).resolve().parents[4] / "outputs/native/four_stage_geometry"
FLAGS = ["-std=c++17", "-O3", "-shared", "-fPIC", "-fno-fast-math", "-ffp-contract=off"]


def identity():
    return dict(profile="closed-ray-grid-cpp-passive-v2", source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                flags=FLAGS, machine=platform.machine(), system=platform.system())


def location():
    return CACHE / hashlib.sha256(json.dumps(identity(), sort_keys=True).encode()).hexdigest()


def build():
    import fcntl
    dest = location()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with (dest.parent / "build.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=dest.parent) as name:
                tmp = Path(name)
                result = subprocess.run(["g++", *FLAGS, str(SOURCE), "-o", str(tmp / "geometry.so")],
                                        text=True, capture_output=True, timeout=90)
                if result.returncode:
                    raise RuntimeError(result.stderr)
                manifest = dict(identity=identity(), binary_sha256=hashlib.sha256((tmp / "geometry.so").read_bytes()).hexdigest(),
                                compiler=subprocess.check_output(["g++", "--version"], text=True).splitlines()[0])
                (tmp / "geometry_kernel.cpp").write_bytes(SOURCE.read_bytes())
                (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                os.rename(tmp, dest)
        if not (dest / "geometry_kernel.cpp").exists():
            expected=json.loads((dest / "manifest.json").read_text())["identity"]["source_sha256"]
            if hashlib.sha256(SOURCE.read_bytes()).hexdigest()!=expected:
                raise ValueError("cannot archive a different kernel source")
            (dest / "geometry_kernel.cpp").write_bytes(SOURCE.read_bytes())
    return json.loads((dest / "manifest.json").read_text())


@lru_cache(maxsize=1)
def kernel():
    dest = location()
    manifest = json.loads((dest / "manifest.json").read_text())
    if manifest["identity"] != identity() or hashlib.sha256((dest / "geometry.so").read_bytes()).hexdigest() != manifest["binary_sha256"]:
        raise ValueError("native geometry provenance mismatch")
    lib = ctypes.CDLL(str(dest / "geometry.so"))
    ptr, integer, double = ctypes.c_void_p, ctypes.c_int, ctypes.c_double
    arr = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
    lib.cap_grid_new.argtypes = [double]*4
    lib.cap_grid_new.restype = ptr
    lib.cap_grid_delete.argtypes = [ptr]
    lib.cap_grid_delete.restype = None
    lib.cap_grid_clone.argtypes = [ptr]
    lib.cap_grid_clone.restype = ptr
    lib.cap_grid_window.argtypes = [ptr, double, double, double, double]
    lib.cap_grid_window.restype = None
    lib.cap_grid_hits.argtypes = [ptr, arr, integer]
    lib.cap_grid_hits.restype = None
    lib.cap_grid_update.argtypes = [ptr, arr]
    lib.cap_grid_update.restype = integer
    lib.cap_grid_certificate.argtypes = [ptr, arr, integer]
    lib.cap_grid_certificate.restype = integer
    lib.cap_grid_missing.argtypes = [ptr, arr, integer]
    lib.cap_grid_missing.restype = double
    lib.cap_grid_query.argtypes = [ptr, arr, integer, arr]
    lib.cap_grid_query.restype = None
    lib.cap_grid_mask.argtypes = [ptr, np.ctypeslib.ndpointer(dtype=np.uint8, flags="C_CONTIGUOUS")]
    lib.cap_grid_mask.restype = None
    lib.cap_grid_stats.argtypes = [ptr, np.ctypeslib.ndpointer(dtype=np.uint64, flags="C_CONTIGUOUS")]
    lib.cap_grid_stats.restype = None
    return lib


class NativeEvidenceMap:
    def __init__(self, *, minimum_static_component_area=.04, resolution=.025, scan_stride_ticks=16):
        if minimum_static_component_area is None or minimum_static_component_area <= 0:
            raise ValueError("declared component-area prior required")
        self.minimum_area = minimum_static_component_area
        self.resolution = resolution
        self.scan_stride_ticks = scan_stride_ticks
        self.last_kernel_tick = None
        self.numerical_margin = .001
        self.endpoint_erosion = .003
        self.ptr = None
        self.arena = None
        self.pending = []
        self.hits = {}
        self.ticks = []
        self.last_tick = None
        self._tree = None
        self._queries = {}
        self._initial_keys = set()
        self.ledger = dict(scans=0, rays=0, certified_cells=0, sensor_conflicts=0)

    def __del__(self):
        if getattr(self, "ptr", None):
            kernel().cap_grid_delete(self.ptr)
            self.ptr = None

    def __deepcopy__(self, memo):
        from copy import deepcopy
        other = type(self)(minimum_static_component_area=self.minimum_area, resolution=self.resolution,
                           scan_stride_ticks=self.scan_stride_ticks)
        memo[id(self)] = other
        for k,v in self.__dict__.items():
            if k not in ("ptr", "_tree", "_queries"):
                setattr(other, k, deepcopy(v, memo))
        if self.ptr:
            other.ptr = kernel().cap_grid_clone(self.ptr)
            if not other.ptr:
                raise MemoryError("native public grid clone")
        return other

    @property
    def assumptions(self):
        return ("static geometry", "exact public rays and odometry", "minimum static component area="+str(self.minimum_area),
                "non-target objects initially at rest and unactuated; no-contact invariant required to retain their occupied cells",
                "coarse closed-ray grid resolution="+str(self.resolution), "ray arrangement stride ticks="+str(self.scan_stride_ticks),
                "long-double clipping; area degeneracy threshold 1e-24 m2")

    def add_initial_certificate(self, geometry, *, source):
        if source != "declared_initial_separation":
            raise ValueError("unrecognized initial certificate")
        if not self.ptr:
            self.pending.append(geometry)
            return
        for p in shapely.get_parts(geometry):
            if p.geom_type != "Polygon" or p.interiors:
                continue
            # Reject non-convex certificates rather than filling their hull.
            if not p.equals(p.convex_hull):
                continue
            xy = np.ascontiguousarray(p.exterior.coords[:-1], dtype=np.float64)
            # Quantization is used ONLY to omit near-duplicate evidence, never
            # to expand a free certificate. This also bounds repeated clipping
            # of floating-point variants of a stationary object's same face.
            key=np.round(xy,5).tobytes()
            if key in self._initial_keys:
                continue
            if kernel().cap_grid_certificate(self.ptr, xy, len(xy)):
                raise RuntimeError("native certificate failed")
            self._initial_keys.add(key)
        self._queries = {}

    def update(self, o):
        if o["tick"] == self.last_tick:
            return
        if not self.ptr:
            self.arena = list(o["robots"][0]["arena"])
            self.ptr = kernel().cap_grid_new(*self.arena, self.resolution, self.minimum_area)
            if not self.ptr:
                raise ValueError("unsupported public arena/grid dimensions")
            pending, self.pending = self.pending, []
            for p in pending:
                self.add_initial_certificate(p, source="declared_initial_separation")
        rays, hits = [], []
        for robot in o["robots"]:
            s = robot["scan"]
            if len(s["distances"]) != 72:
                raise ValueError("fixed 72 rays required")
            x,y = s["origin"]
            for angle,d,valid,category,entity in zip(s["ray_angles"],s["distances"],s["valid"],s["category"],s["entity_ids"]):
                if not math.isfinite(d) or d < 0:
                    raise ValueError("invalid range")
                c, sn = math.cos(angle), math.sin(angle)
                length = max(0., d-self.endpoint_erosion)
                rays.append((x,y,x+length*c,y+length*sn))
                passive = category == 1 and entity is not None and entity != o["goal"]["target_id"]
                if valid and (category == 2 or passive):
                    hit=(x+d*c,y+d*sn)
                    hits.append(hit)
                    # Bounded cell summary; precise input remains in public
                    # observation records. Occupied cells are conservative.
                    self.hits[(math.floor(hit[0]/self.resolution),math.floor(hit[1]/self.resolution))] = o["tick"]
        if hits:
            kernel().cap_grid_hits(self.ptr,np.ascontiguousarray(hits,dtype=np.float64),len(hits))
        # Only spend arrangement work near current possible body sweeps. The
        # rest stays unknown until a later observation makes it relevant.
        points = [[r["own"]["x"], r["own"]["y"]] for r in o["robots"]]
        points += [p for e in o["entities"] if e["id"] == o["goal"]["target_id"] for p in e["points"]]
        kernel().cap_grid_window(self.ptr, min(p[0] for p in points)-.7, min(p[1] for p in points)-.7,
                                max(p[0] for p in points)+.7, max(p[1] for p in points)+.7)
        if self.last_kernel_tick is None or o["tick"]-self.last_kernel_tick >= self.scan_stride_ticks:
            if kernel().cap_grid_update(self.ptr, np.ascontiguousarray(rays,dtype=np.float64)):
                raise RuntimeError("native grid update failed")
            self.last_kernel_tick = o["tick"]
            self.ledger["kernel_updates"] = self.ledger.get("kernel_updates",0)+1
        self.last_tick = o["tick"]
        self.ticks.append(self.last_tick)
        self._tree = None
        self._queries = {}
        self.ledger["scans"] += 1
        self.ledger["rays"] += 144
        stats = np.zeros(4,dtype=np.uint64)
        kernel().cap_grid_stats(self.ptr,stats)
        self.ledger.update(certified_cells=int(stats[3]), certificates=int(stats[0]), complexity_resets=int(stats[1]))

    def query_parts(self, parts):
        if not self.ptr:
            raise ValueError("map requires public initialization")
        missing, count = 0., 0
        for p in parts:
            if p is None or p.is_empty:
                continue
            key = p.wkb
            if key not in self._queries:
                # Sixteen support halfplanes give an OUTER approximation;
                # reducing edge count cannot turn unknown space into free.
                points = np.asarray(p.convex_hull.exterior.coords[:-1],dtype=np.float64)
                angles = np.arange(16)*math.pi/8
                normals = np.column_stack((np.cos(angles),np.sin(angles)))
                support = np.max(points @ normals.T,axis=0)+self.numerical_margin+1e-10
                nxt = np.roll(normals,-1,axis=0)
                snxt = np.roll(support,-1)
                det = normals[:,0]*nxt[:,1]-normals[:,1]*nxt[:,0]
                xy = np.ascontiguousarray(np.column_stack(((support*nxt[:,1]-normals[:,1]*snxt)/det,
                                               (normals[:,0]*snxt-support*nxt[:,0])/det)),dtype=np.float64)
                result=np.zeros(2,dtype=np.float64)
                kernel().cap_grid_query(self.ptr,xy,len(xy),result)
                self._queries[key] = float(result[0]), int(result[1])
            value, hits = self._queries[key]
            count += hits
            missing += value
        status = "occupied" if count else "unknown" if missing else "free"
        return RegionEvidence(status, float(missing), count, tuple(self.ticks), self.assumptions)

    def query(self, region):
        return self.query_parts(shapely.get_parts(region))

    def mask(self):
        nx,ny = [math.ceil(v/self.resolution) for v in self.arena]
        result = np.zeros((ny,nx),dtype=np.uint8)
        kernel().cap_grid_mask(self.ptr,result)
        return result

    @property
    def free(self):
        # Offline diagnostics only; never materialized during native queries.
        ys,xs = np.nonzero(self.mask())
        h=self.resolution
        return shapely.union_all(shapely.box(xs*h,ys*h,(xs+1)*h,(ys+1)*h)) if len(xs) else GeometryCollection()


if __name__ == "__main__":
    print(json.dumps(build(),indent=2))
