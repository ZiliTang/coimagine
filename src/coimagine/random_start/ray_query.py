"""Nearest-hit kernel, preserving Chipmunk's all-hit traversal and ties.

No filter or body is changed, and sensor shapes remain visible. On the verified
Pymunk ABI, discard unused hit points/normals before Python object construction.
Unknown ABIs use the public reference path. This is not a new geometry engine.
"""
import pymunk

try:
    if not pymunk.version.startswith('6.9.'):
        raise ImportError('unverified native query ABI')
    from pymunk._chipmunk_cffi import ffi, lib

    @ffi.callback('void(struct cpShape *, struct cpVect, struct cpVect, double, void *)')
    def _nearest(shape, point, normal, alpha, data):
        state = ffi.from_handle(data)
        # Strict comparison matches min(..., key=alpha): keep the first tie.
        if shape not in state[0] and (state[1] == ffi.NULL or alpha < state[2]):
            state[1], state[2] = shape, alpha
except (ImportError, AttributeError):
    _nearest = None


def nearest_hits(space, own_body, origin, ends, *, native=True, include_points=False):
    """Return (shape, alpha) or None, excluding every own-body shape.

    include_points appends the exact (x, y) intersection and uses the reference
    path; otherwise native mode avoids unused point/normal allocations.
    """
    shape_filter = pymunk.ShapeFilter()
    if include_points or not native or _nearest is None:
        rows = []
        for end in ends:
            hits = (h for h in space.segment_query(origin, end, 0., shape_filter)
                    if h.shape.body is not own_body)
            hit = min(hits, key=lambda h: h.alpha, default=None)
            rows.append(None if hit is None else ((hit.shape, hit.alpha, (float(hit.point.x), float(hit.point.y)))
                                                 if include_points else (hit.shape, hit.alpha)))
        return rows
    state = [{s._shape for s in own_body.shapes}, ffi.NULL, 0.]
    handle = ffi.new_handle(state)
    rows = []
    for end in ends:
        state[1] = ffi.NULL
        lib.cpSpaceSegmentQuery(space._space, origin, end, 0., shape_filter, _nearest, handle)
        rows.append(None if state[1] == ffi.NULL else (space._get_shape(state[1]), state[2]))
    return rows
