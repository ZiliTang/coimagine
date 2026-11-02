import math
import numpy as np
from experiments.option_rl_v19.synthetic_shapes import SHAPES, polygon_observation
from experiments.option_rl_v19.unknown_shape import select_simple_pair
from experiments.option_rl_v13 import controller as C


def _seg_dist(p, a, b):
    p=np.asarray(p,float); a=np.asarray(a,float); b=np.asarray(b,float)
    d=b-a; u=np.clip(float((p-a)@d)/max(1e-12,float(d@d)),0.,1.); return float(np.linalg.norm(p-(a+u*d)))


def test_synthetic_polygon_selector_pairs():
    for name, poly in SHAPES.items():
        o0=polygon_observation(name, 0, (-.45, .45), (.5, 1.2))
        o1=polygon_observation(name, 1, (.5, 1.2), (-.45,.45))
        r=select_simple_pair(o0, C.ACTIONS[2], o1, C.ACTIONS[10])
        assert r is not None, name
        q0=np.asarray(r['q0']); q1=np.asarray(r['q1'])
        assert np.linalg.norm(q0-q1) >= .30, (name, 'sep')
        # each q must lie on one polygon edge
        assert min(_seg_dist(q0, poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))) < 1e-6, name
        assert min(_seg_dist(q1, poly[i], poly[(i+1)%len(poly)]) for i in range(len(poly))) < 1e-6, name
