"""Optional fused analytic lidar, built explicitly and loaded by source hash.

Build: python -m coimagine.random_start.lidar_native build
Select at process startup: COIMAGINE_LIDAR_KERNEL=auto|numpy|native.
Auto uses a verified existing binary, otherwise logs one NumPy fallback. Runtime
never launches a compiler. The original lidar kernels remain independent references.
"""
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

VERSION = 'analytic-fused-c-v1'
FLAGS = ('-std=c99', '-shared', '-fPIC', '-O2', '-fno-fast-math', '-ffp-contract=off')
SOURCE = Path(__file__).with_name('lidar_kernel.c')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _identity():
    return dict(version=VERSION, source_sha256=_sha(SOURCE), flags=FLAGS,
                machine=platform.machine(), system=platform.system(),
                pointer_bytes=ctypes.sizeof(ctypes.c_void_p), libc=platform.libc_ver())


def _location():
    identity = _identity()
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    root = Path(os.environ.get('COIMAGINE_LIDAR_CACHE',
                               Path.home()/'.cache'/'coimagine'/'lidar-native-v1'))
    return root/key, identity


def build():
    """Host-reviewed C only; atomic publication, separate from any author code."""
    import fcntl
    dest, identity = _location()
    dest.parent.mkdir(parents=True, exist_ok=True)
    cc = shutil.which('gcc')
    if cc is None:
        raise RuntimeError('gcc required for explicit lidar build')
    with (dest.parent/(dest.name+'.lock')).open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if dest.exists():
            _load_at(dest, identity)
            return json.loads((dest/'manifest.json').read_text())
        with tempfile.TemporaryDirectory(prefix='build-', dir=dest.parent) as temp:
            temp = Path(temp)
            command = [cc, *FLAGS, str(SOURCE), '-lm', '-o', str(temp/'lidar.so')]
            start = time.perf_counter()
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            (temp/'build.log').write_text(result.stdout+result.stderr)
            if result.returncode:
                raise RuntimeError('lidar compilation failed: '+result.stderr[-4000:])
            manifest = dict(identity=identity, compiler=cc, compiler_sha256=_sha(cc),
                            binary_sha256=_sha(temp/'lidar.so'),
                            build_seconds=time.perf_counter()-start)
            (temp/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
            # Publish a complete directory; other workers only ever see complete builds.
            os.rename(temp, dest)
    return manifest


def _load_at(dest, identity):
    manifest = json.loads((dest/'manifest.json').read_text())
    if (manifest['identity'] != json.loads(json.dumps(identity)) or
            manifest['binary_sha256'] != _sha(dest/'lidar.so')):
        raise ValueError('lidar binary/source manifest mismatch')
    library = ctypes.CDLL(str(dest/'lidar.so'))
    function = library.coimagine_lidar_scan
    # ndarray validators check dtype/contiguity; all arrays are owned host data.
    double = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    integer = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_size_t, double, ctypes.c_double, ctypes.c_double,
                         ctypes.c_double, ctypes.c_size_t, double, integer,
                         ctypes.c_size_t, double, integer, double, integer]
    function.restype = None
    function.build_manifest = manifest
    return function, manifest


@lru_cache(maxsize=1)
def kernel():
    mode = os.environ.get('COIMAGINE_LIDAR_KERNEL', 'auto')
    if mode not in ('auto', 'numpy', 'native'):
        raise ValueError('COIMAGINE_LIDAR_KERNEL must be auto, numpy or native')
    if mode == 'numpy':
        print('[lidar] analytic_kernel=numpy (explicit)', file=sys.stderr)
        return None
    dest, identity = _location()
    if not dest.exists():
        if mode == 'native':
            raise RuntimeError('native lidar is not built; run python -m coimagine.random_start.lidar_native build')
        print('[lidar] analytic_kernel=numpy reason=native_not_built', file=sys.stderr)
        return None
    function, manifest = _load_at(dest, identity)
    print('[lidar] analytic_kernel=native source='+identity['source_sha256'][:12], file=sys.stderr)
    return function


def runtime_info():
    function = kernel()
    return dict(version=VERSION, engine='numpy' if function is None else 'native',
                build=None if function is None else function.build_manifest)


def scan_batch(*args, **kwargs):
    if kernel() is None:
        from .lidar import scan_lidar_batch
        return scan_lidar_batch(*args, **kwargs)
    return scan_compiled(*args, **kwargs)


def scan_compiled(arena, obstacles, object_pose, robots, robot_id, tick, sample_time,
                  scan_range=None, scan_rays=None, object_angle_override=None,
                  object_polygon=None):
    from . import lidar as L
    function = kernel()
    if function is None:
        raise RuntimeError('scan_compiled requires a built native kernel')
    scan_range = L.LIDAR_RANGE if scan_range is None else scan_range
    scan_rays = L.LIDAR_RAYS if scan_rays is None else scan_rays
    angle = object_pose[2] if object_angle_override is None else object_angle_override
    poly = L._object_polygon_world((object_pose[0], object_pose[1], angle), object_polygon)
    robot = robots[robot_id]; origin = (robot[0], robot[1]); heading = robot[2]
    entries = L._occluder_geometry(arena, obstacles, poly, robots, robot_id)
    angles, directions = L._ray_directions(heading, scan_rays)
    segments, segment_owners, circles, circle_owners = [], [], [], []
    def segment(a, b, radius, owner):
        ex, ey = b[0]-a[0], b[1]-a[1]
        length = math.hypot(ex, ey)  # Same CPython hypot as the NumPy reference.
        divisor = max(length, 1e-12)
        segments.append((a[0], a[1], ex, ey, radius, length, ex/divisor, ey/divisor))
        segment_owners.append(owner)
    for owner, (_, _, shape) in enumerate(entries):
        if shape[0] == 'seg':
            _, a, b, radius = shape
            segment(a, b, radius, owner)
            circles.extend(((a[0], a[1], radius), (b[0], b[1], radius)))
            circle_owners.extend((owner, owner))
        elif shape[0] == 'poly':
            for a, b in L.G.polygon_edges(shape[1]):
                segment(a, b, 0., owner)
        else:
            _, centre, radius = shape
            circles.append((centre[0], centre[1], radius)); circle_owners.append(owner)
    distances = np.empty(scan_rays, dtype=np.float64)
    owners = np.empty(scan_rays, dtype=np.int64)
    function(scan_rays, directions, *origin, scan_range,
             len(segments), np.asarray(segments, dtype=np.float64).reshape(-1, 8),
             np.asarray(segment_owners, dtype=np.int64),
             len(circles), np.asarray(circles, dtype=np.float64).reshape(-1, 3),
             np.asarray(circle_owners, dtype=np.int64), distances, owners)
    valid = owners >= 0
    return L.LidarScan(robot_id=robot_id, tick=tick, sample_time=sample_time, origin=origin,
                       heading=heading, ray_angles=angles, distances=tuple(distances.tolist()),
                       valid=tuple(valid.tolist()),
                       category=tuple(entries[k][1] if v else L.CATEGORY_NONE for k,v in zip(owners,valid)),
                       raw_labels=tuple(entries[k][0] if v else None for k,v in zip(owners,valid)))


if __name__ == '__main__':
    if sys.argv[1:] != ['build']:
        raise SystemExit('usage: python -m coimagine.random_start.lidar_native build')
    print(json.dumps(build(), indent=2))
