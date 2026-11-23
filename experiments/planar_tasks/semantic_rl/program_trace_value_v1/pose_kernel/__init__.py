"""Native pose search for the pinned CAP009 in trusted experiment workers."""
import ast
import ctypes
import hashlib
import math
import os
from array import array
from pathlib import Path

from ..instrumentation import instrument, load_native

SOURCE_SHA = '48fc2ebcf6f9540f1c457a49e47e9ed7cc0cac3bb9a6a7aa17c2d083e4f5284d'
STEP = .24
DoublePtr = ctypes.POINTER(ctypes.c_double)
IntPtr = ctypes.POINTER(ctypes.c_int)


def _ptr(values, pointer):
    return ctypes.cast(values.buffer_info()[0], pointer)


class PoseKernel:
    """Caller-owned native library; no compilation or global patch on import."""
    def __init__(self, binary):
        self.library = ctypes.CDLL(str(Path(binary).resolve()))
        self.function = self.library.cap009_route
        integer, real = ctypes.c_int, ctypes.c_double
        self.function.argtypes = [integer, integer, real, real, integer, DoublePtr, IntPtr,
                                  DoublePtr, integer, DoublePtr, IntPtr, IntPtr]
        self.function.restype = integer
        if array('i').itemsize != ctypes.sizeof(integer):
            raise RuntimeError('native int layout mismatch')

    def search(self, arena, nx, ny, theta, pts, start, ga, gb):
        if not (1 <= nx <= 128 and 1 <= ny <= 128 and 0 <= start < nx*ny):
            raise ValueError('unsupported CAP009 grid or start cell')
        axes = array('d')
        directions = array('i')
        for k in range(16):
            angle = theta + k * math.pi / 16
            ux, uy = math.cos(angle), math.sin(angle)
            axes.extend((ux, uy))
            if k % 2 == 0:
                for x, y in ((ux, uy), (-ux, -uy), (-uy, ux), (uy, -ux)):
                    dd = round(math.atan2(y, x) / (math.pi/4)) % 8
                    directions.extend((round(math.cos(dd*math.pi/4)), round(math.sin(dd*math.pi/4))))
        xy = array('d', (coordinate for point in pts for coordinate in point[:2]))
        if len(xy) != 2*len(pts):
            raise ValueError('two coordinates required per public obstacle point')
        # Use CPython's hypot once per cell to retain the actual heap priorities,
        # including ties. The platform C hypot can differ in its last bit.
        heuristic = array('d', (math.hypot(i-ga, j-gb) for i in range(nx) for j in range(ny)))
        path = array('i', [0]) * (8*nx*ny)
        stats = array('i', [0]) * 5
        length = self.function(nx, ny, *arena, start, _ptr(axes, DoublePtr),
                               _ptr(directions, IntPtr), _ptr(xy, DoublePtr), len(pts),
                               _ptr(heuristic, DoublePtr), _ptr(path, IntPtr), _ptr(stats, IntPtr))
        if length < 1:
            raise RuntimeError('CAP009 native search allocation or dimension error')
        # Original Python tail follows only the parent links of the final path.
        route = path[:length]
        parents = dict(zip(route, route[1:]))
        return stats, parents


def _route_function(source, module, search=None, audit=None):
    """Retain the frozen setup, trace writes, route reconstruction and returns."""
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                    and n.name == 'pose_route')
    body = []
    for node in function.body:
        expr = ast.unparse(node)
        if search is not None:
            if isinstance(node, ast.For) and node.target.id == 'k':
                continue  # collision grids are produced inside the kernel
            if expr == 'blocked[0].discard(start)':
                continue
            if isinstance(node, ast.For) and node.target.id == 'it':
                node = ast.parse('found, best, parents = _search(arena, nx, ny, theta, pts, start, ga, gb, _tb)').body[0]
        if audit is not None and isinstance(node, ast.If) and 'found == start' in ast.unparse(node.test):
            node.body.insert(0, ast.parse('_audit([start])').body[0])
        body.append(node)
        if audit is not None and expr == 'path.reverse()':
            body.extend(ast.parse('_audit(path)').body)
    function.body = body
    env = dict(module.__dict__, _search=search, _audit=audit, _tb=None)
    tree = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(tree, '<pinned CAP009 pose route>', 'exec'), env)
    return env['pose_route']


def _replacement(source, module, kernel, schema=None, audit=None):
    if schema is not None:
        slots = {(s['function'], s['kind'], s['name']): i for i, s in enumerate(schema['slots'])}
        last_push = slots['heappush', 'assignment', 'i']
        pop_i = slots['heappop', 'assignment', 'i']
        pop_n = slots['heappop', 'assignment', 'n']
        pop_branch = slots['heappop', 'branch', 'h']
        pop_return = slots['heappop', 'return', 'reached']
    original = getattr(module, '_cap009_reference_route', module.pose_route)

    def search(arena, nx, ny, theta, pts, start, ga, gb, trace):
        stats, parents = kernel.search(arena, nx, ny, theta, pts, start, ga, gb)
        found, best, push_index, remaining, last_positive = stats
        if schema is not None and trace is not None:
            # The trace contract records the last executed scalar at each site.
            # Heap entries themselves are lists and never write a scalar slot.
            trace[last_push] = float(push_index)
            trace[pop_branch] = float(remaining != 0)
            trace[pop_return] = 1.
            if last_positive >= 0:
                trace[pop_i] = 0.
                trace[pop_n] = float(last_positive)
        return (None if found < 0 else found), best, parents

    route = _route_function(source, module, search, audit)
    def dispatch(o, g, target, goal, m, **kwargs):
        width, height = o['robots'][0]['arena']
        if not (1 <= int(width/STEP) <= 128 and 1 <= int(height/STEP) <= 128):
            return original(o, g, target, goal, m, **kwargs)
        return route(o, g, target, goal, m, **kwargs)
    return dispatch


def make_route(source, kernel, *, traced=False, audit=None):
    """Construct a scoped benchmark replacement without changing a module."""
    if hashlib.sha256(source.encode()).hexdigest() != SOURCE_SHA:
        raise ValueError('kernel requires the exact CAP009 source')
    source, schema = instrument(source) if traced else (source, None)
    module, _ = load_native(source)
    return module, _replacement(source, module, kernel, schema, audit)


def configure(module, source, *, generated=None, schema=None):
    """Select once at module startup, for the exact CAP source only."""
    from .shared import configure_shared
    digest = hashlib.sha256(source.encode()).hexdigest()
    configure_shared(module, digest, generated if generated is not None else source, schema)
    if digest != SOURCE_SHA:
        return
    mode = os.environ.get('COIMAGINE_CAP009_POSE', 'native')
    if mode not in ('native', 'reference'):
        raise ValueError('COIMAGINE_CAP009_POSE must be native or reference')
    if not hasattr(module, '_cap009_reference_route'):
        module._cap009_reference_route = module.pose_route
    if mode == 'reference':
        module.pose_route = module._cap009_reference_route
    else:
        if not hasattr(module, '_cap009_native_route'):
            from .build import default_kernel
            module._cap009_native_route = _replacement(
                generated if generated is not None else source, module, default_kernel(), schema)
        module.pose_route = module._cap009_native_route
