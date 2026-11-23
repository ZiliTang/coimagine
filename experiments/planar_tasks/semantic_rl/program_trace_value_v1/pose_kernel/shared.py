"""Specialize the two shared numeric loops, keeping the frozen policy and trace tail."""
import ast
import ctypes
from array import array
from functools import lru_cache
import math
import os

from . import DoublePtr, IntPtr, _ptr

# Pin the whole program: calls into module helpers and instrumentation are part
# of the contract, even when the top-level function bodies are identical.
PINNED = {
    'ed95db068a427234638dede2845f09fe4c130ac0d323948ffc0fc177ce9a1649',
    '9605689bba8bf4790943db4ec3b0e3134644c703261152fea1e4ade519d43575',
    '3bb1d1d714a44051e2b4e65be23b3252263b09b070991fd98d25523779a04cad',
    '0369f62a16dbffac9267dd954a656fd4cd328daf9a68b8374a55bef6e5ff364e',
    'acad4708c4e241ab8e095b3d0ad1b7fd618eb937bb4b83e3ae8672058de1756b',
    '48fc2ebcf6f9540f1c457a49e47e9ed7cc0cac3bb9a6a7aa17c2d083e4f5284d',
    '50ff6ab3197896a01d18985f52725f31b904335e25543f2762ce11d167db0f97',
}

DirectionTable = array('d', (v for k in range(8)
                             for v in (math.cos(k*math.pi/4), math.sin(k*math.pi/4))))
Hypot = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
exact_hypot = Hypot(math.hypot)
BytePtr = ctypes.POINTER(ctypes.c_ubyte)


@lru_cache(maxsize=8)
def heuristics(nx, ny):
    return array('d', (math.hypot(i,j) for i in range(nx) for j in range(ny)))


class Occupancy(bytearray):
    def __contains__(self, cell):
        return 0 <= cell < len(self) and self[cell] != 0


class SharedKernels:
    def __init__(self, lib):
        integer = ctypes.c_int
        self.clear = lib.cap_clearances
        self.clear.restype = None
        self.clear.argtypes = [DoublePtr, integer, DoublePtr, DoublePtr, DoublePtr, DoublePtr]
        self.nav = lib.cap_navigate
        self.nav.restype = integer
        self.nav.argtypes = [integer, integer, DoublePtr, DoublePtr, DoublePtr, integer,
                             DoublePtr, DoublePtr, DoublePtr, integer, DoublePtr, Hypot,
                             BytePtr, IntPtr, IntPtr]

    def clearances(self, robots, g, target, pts):
        points = array('d', (v for x,y,entity,category in pts
                             for v in (x,y,entity == target,category == 3)))
        own = array('d', (robot['own'][axis] for robot in robots for axis in ('x','y')))
        geom = array('d', g[:4])
        if len(own) != 4 or len(geom) != 4:
            raise ValueError('two robot positions and four geometry coordinates required')
        result = array('d', [0.])*24
        self.clear(_ptr(points,DoublePtr), len(pts), _ptr(geom,DoublePtr), _ptr(own,DoublePtr),
                   _ptr(DirectionTable,DoublePtr), _ptr(result,DoublePtr))
        return result[:8].tolist(), result[8:].tolist()

    def plan(self, arena, nx, ny, pos, peer, g, pts, goal):
        if not (1 <= nx <= 128 and 1 <= ny <= 128):
            raise ValueError('unsupported navigation grid')
        xy = array('d', (v for point in pts for v in (point[0],point[1])))
        geometry = array('d', g[:4])
        if len(geometry) != 4 or any(len(v)!=2 for v in (arena,pos,peer,goal)):
            raise ValueError('invalid navigation geometry')
        width, own, mate, target = (array('d',v) for v in (arena,pos,peer,goal))
        cells = nx*ny
        start = max(0,min(nx-1,int(pos[0]/.12)))*ny + max(0,min(ny-1,int(pos[1]/.12)))
        grid = Occupancy(cells)
        path = array('i',[0])*cells
        stats = array('i',[0])*4
        length = self.nav(nx,ny,_ptr(width,DoublePtr),_ptr(geometry,DoublePtr),_ptr(xy,DoublePtr),
                          len(pts),_ptr(own,DoublePtr),_ptr(mate,DoublePtr),_ptr(target,DoublePtr),
                          start,_ptr(heuristics(nx,ny),DoublePtr),exact_hypot,
                          (ctypes.c_ubyte*cells).from_buffer(grid),_ptr(path,IntPtr),_ptr(stats,IntPtr))
        if length == -2:
            raise ValueError('min() arg is an empty sequence')
        if length < 0:
            raise RuntimeError('native navigation allocation or dimension error')
        reverse = path[:length]
        parents = dict(zip(reverse,reverse[1:]))
        return grid, stats[0], (reverse[0] if length else None), parents, stats


@lru_cache(maxsize=1)
def kernels():
    from .build import default_kernel
    return SharedKernels(default_kernel().library)


def _specialize(function, module, schema):
    """Only replace the frozen loop regions. Preserve all branches and the
    original move_toward implementation, including robot-specific trace writes.
    """
    native = kernels()
    body = []
    for node in function.body:
        expr = ast.unparse(node)
        if function.name == 'clearances' and isinstance(node, ast.For):
            if node.target.id == 'k':
                body.extend(ast.parse('push, moves = _clear(o["robots"], g, target, pts)').body)
            continue
        if function.name == 'navigate':
            if expr == 'blocked = set()':
                body.extend(ast.parse('blocked, _end, _found, _parents, _stats = _plan(r["arena"], nx, ny, pos, peer, g, pts, goal)').body)
                continue
            if isinstance(node, ast.For):
                if node.target.id == 'it':
                    body.extend(ast.parse('found, parents = _found, _parents').body)
                    if schema is not None:
                        body.extend(ast.parse('_heap_trace(_tb, _stats)').body)
                continue
            if expr == 'blocked.discard(start)':
                continue
            if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'end':
                node.value = ast.Name(id='_end',ctx=ast.Load())
        body.append(node)
    function.body = body
    env = dict(module.__dict__, _clear=native.clearances, _plan=native.plan)
    if schema is not None and function.name == 'navigate':
        slots = {(s['function'],s['kind'],s['name']):i for i,s in enumerate(schema['slots'])}
        push_index = slots['heappush','assignment','i']
        pop_i = slots['heappop','assignment','i']
        pop_n = slots['heappop','assignment','n']
        pop_test = slots['heappop','branch','h']
        pop_return = slots['heappop','return','reached']
        def heap_trace(trace, stats):
            if trace is not None:
                _,last_push,remaining,last_positive = stats
                trace[push_index] = float(last_push)
                trace[pop_test] = float(remaining != 0)
                trace[pop_return] = 1.
                if last_positive >= 0:
                    trace[pop_i] = 0.
                    trace[pop_n] = float(last_positive)
        env['_heap_trace'] = heap_trace
    tree = ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    exec(compile(tree,'<pinned CAP shared kernel>','exec'),env)
    return env[function.name]


def configure_shared(module, digest, code, schema):
    if digest not in PINNED:
        return
    mode = os.environ.get('COIMAGINE_CAP_SHARED','native')
    if mode not in ('native','reference','clearances','navigate'):
        raise ValueError('COIMAGINE_CAP_SHARED must be native, reference, clearances or navigate')
    for name in ('clearances','navigate'):
        if not hasattr(module,name):
            continue
        ref = '_reference_'+name
        fast = '_native_'+name
        if not hasattr(module,ref):
            setattr(module,ref,getattr(module,name))
        original = getattr(module,ref)
        if mode not in ('native',name):
            setattr(module,name,original)
            continue
        if not hasattr(module,fast):
            definition = next(f for f in ast.parse(code).body if isinstance(f,ast.FunctionDef) and f.name==name)
            specialized = _specialize(definition,module,schema)
            if name == 'navigate':
                def navigate(r, goal, pts, peer, g, **kwargs):
                    width,height = r['arena']
                    if not (1 <= int(width/.12) <= 128 and 1 <= int(height/.12) <= 128):
                        return original(r,goal,pts,peer,g,**kwargs)
                    return specialized(r,goal,pts,peer,g,**kwargs)
                setattr(module,fast,navigate)
            else:
                setattr(module,fast,specialized)
        setattr(module,name,getattr(module,fast))
