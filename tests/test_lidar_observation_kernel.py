"""Exact sensor behavior across changing scenes and nonrectangular objects."""
import math
import random

import numpy as np
import pytest

from coimagine.random_start import lidar as L
from coimagine.random_start.geometry import box_polygon
from experiments.option_rl_v19.synthetic_shapes import SHAPES


@pytest.mark.parametrize('shape', ['T', 'triangle', 'octagon'])
def test_batch_polygon_ranges_visibility_and_body_frame_match_scalar(shape):
    rng = random.Random(1029)
    obstacles = (box_polygon(1.2, 2.5, .3, .15, .4),)
    for rays in (4, 72, 288):
        for tick in range(8):
            # Repeated heading with moving geometry exercises direction reuse;
            # changing heading must still rotate the actual body-fixed rays.
            heading = .35 if tick % 2 else rng.uniform(-math.pi, math.pi)
            robots = [(rng.uniform(.2, 4.8), rng.uniform(.2, 3.8), heading, 0, 0, 0),
                      (3.8, 2.6, -.2, 0, 0, 0)]
            args = ((5., 4.), obstacles, (2., 1.7, rng.uniform(-math.pi, math.pi)),
                    robots, 0, tick, .05 * tick)
            kwargs = dict(scan_rays=rays, object_polygon=SHAPES[shape])
            a, b = L.scan_lidar(*args, **kwargs), L.scan_lidar_batch(*args, **kwargs)
            assert (a.ray_angles, a.valid, a.category, a.raw_labels, a.origin,
                    a.heading, a.tick, a.sample_time) == (
                        b.ray_angles, b.valid, b.category, b.raw_labels, b.origin,
                        b.heading, b.tick, b.sample_time)
            np.testing.assert_allclose(a.distances, b.distances, rtol=0, atol=1e-12)


def test_direction_reuse_does_not_reuse_scene_evidence():
    robots = [(1., 2., 0., 0, 0, 0), (4., 3., 0., 0, 0, 0)]
    def scan(x, tick):
        return L.scan_lidar_batch((5., 4.), (), (x, 2., 0.), robots, 0, tick, .05*tick)
    first, moved, returned = scan(2., 1), scan(3., 2), scan(2., 3)
    assert first.category[0] == moved.category[0] == L.CATEGORY_OBJECT
    assert moved.distances[0] - first.distances[0] == pytest.approx(1.)
    assert returned.distances == first.distances
    assert returned.tick == 3 and returned.sample_time == pytest.approx(.15)
