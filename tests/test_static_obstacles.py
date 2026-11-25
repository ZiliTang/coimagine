import math

import pytest

from coimagine.continuous2d.static_obstacles import StaticObstacleMap
from experiments.planar_tasks.relational_search.static_map import update_static_map


def forward(m, points, direction):
    return m.forward_surface(points, direction, lateral_margin=0., min_alignment=.25, min_gap=0., max_gap=4.)


def test_finite_segment_and_current_footprint_in_a_shared_frame():
    m = StaticObstacleMap()
    assert m.nearest_distance((0., 0.)) is None
    m.add_segment((1., -1.), (1., 1.), 10)
    assert m.nearest_distance((2., 2.)) == pytest.approx(math.sqrt(2))
    for x in (0., .5):
        hit = forward(m, [(x, .2), (x, -.2)], (3., 0.))
        assert (hit.normal_gap, hit.normal_angle, hit.surface.observed_tick) == (1. - x, 0., 10)
    assert forward(m, [(0., 2.)], (1., 0.)) is None
    assert forward(m, [(0., 0.)], (-1., 0.)) is None


def test_public_static_categories_occlusion_and_finite_angular_support():
    m = StaticObstacleMap()
    angles = [-.1, 0., .1]
    obs = dict(tick=1, robots=[dict(scan=dict(origin=[2., 1.], ray_angles=angles,
               distances=[1/math.cos(a) for a in angles], valid=[True]*3, category=[2]*3))])
    update_static_map(m, obs)
    assert m.nearest_distance((2., 1.)) == pytest.approx(1.)
    obs['tick'] = 2
    obs['robots'][0]['scan']['category'] = [1, 1, 1]
    fresh = StaticObstacleMap()
    for layer in (m, fresh):
        update_static_map(layer, obs)
    assert fresh.nearest_distance((2., 1.)) is None
    assert forward(m, [(2.5, 1.)], (1., 0.)).surface.observed_tick == 1
    broken = StaticObstacleMap()
    broken.observe_hits([(1., 0.), None, (1., .5)], 5)
    assert broken.nearest_distance((0., 0.)) is None
