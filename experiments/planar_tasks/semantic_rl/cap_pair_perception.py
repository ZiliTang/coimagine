"""Versioned source adapter for the authored A/B pair's fixed body-frame 72-ray sensor."""
from hashlib import sha256

DEFAULT = 'fixed72-local-v1'
LEGACY = 'fixed72-world-v0'
# Keep this spelling identical to the closed-loop-tested Cython program.
_LOCAL_CENTER = """def compact_center(points, faces, previous):
    if not points:
        return previous
    origin = (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))
    shifted = [(p[0]-origin[0], p[1]-origin[1]) for p in points]
    planes = [(n,t,d-dot(n,origin),lo-dot(t,origin),hi-dot(t,origin)) for n,t,d,lo,hi in faces]
    last = None if previous is None else [previous[0]-origin[0],previous[1]-origin[1],*previous[2:]]
    center = fit_relative(shifted,planes,last)
    return [center[0]+origin[0],center[1]+origin[1],*center[2:]]
"""


def source_for_profile(source, profile=DEFAULT):
    if profile == LEGACY:
        return source
    if profile != DEFAULT:
        raise ValueError(f"unknown A/B perception profile {profile!r}")
    signature = 'def compact_center(points,faces,previous):'
    if source.count(signature) != 1 or 'def fit_relative(' in source:
        raise ValueError("expected the frozen A/B compact_center implementation")
    return source.replace(signature, _LOCAL_CENTER + '\n' + signature.replace('compact_center', 'fit_relative'))


def fingerprint(source):
    return sha256(source.encode()).hexdigest()
