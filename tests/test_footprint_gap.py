import math
import pytest

from experiments.planar_tasks.relational_search.footprint_gap import FootprintHistory


def face(x, y, origin, length):
    points=[(x, y+i*length/12) for i in range(-6,7)]
    angles=[math.atan2(py-origin[1],px-origin[0]) for px,py in points]
    return {'robots':[{'scan':dict(origin=origin,valid=[False]+[True]*13+[False],
                entity_ids=[None]+['box']*13+[None],ray_angles=[0]+angles+[0],
                distances=[0]+[math.dist(p,origin) for p in points]+[0])}]}


def test_unknown_target_and_translated_observations():
    results=[]
    for shift in (0.,9.):
        shape=FootprintHistory()
        past=face(shift,shift,(shift-1.,shift),.6)
        shape.observe(past,'box')
        now=face(shift+.25,shift,(shift-.75,shift),.4)
        shape.observe(now,'box')
        bounds=shape.gap_range(.35,(1.,0.))
        assert bounds[0] < bounds[1] == .35
        results.append(bounds)
        shape.observe(now,'unseen')
        assert shape.gap_range(.35,(1.,0.)) is None
    assert results[0]==pytest.approx(results[1])
