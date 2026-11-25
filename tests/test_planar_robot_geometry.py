"""Exercise the configured disc and orbit geometry on the ordinary duration runtime."""
import math

import pytest

from coimagine.random_start import spec
from experiments.planar_tasks.relational_search import pipeline_execution as execution
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig


def test_parallel_wall_orbit_uses_configured_body_and_edge_gap():
    # The initial centre is on the nominal straight-face reference in both profiles.
    compact = spec.GEOMETRY_PROFILE == 'compact20'
    radius, gap, width = (.10, .05, .39) if compact else (.18, .10, .60)
    face_x = 3. - .125
    fixture = execution.seed_fixture()
    fixture.update(scene_id='random_room_v1', movable_objects=[], object=dict(x=3., y=2.4, angle=0.),
                   room=dict(arena=[6., 4.8], obstacles=[dict(cx=face_x-width-.04, cy=2.4,
                             half_w=.04, half_h=.85, angle=0.)]),
                   robots=[dict(x=face_x-radius-gap, y=2.63, angle=0.), dict(x=4.2, y=3.3, angle=0.)])
    world = Environment(fixture, EventConfig(mode='duration'))
    backend = world.env.env.backend
    assert [body.radius for body in backend.robot_shapes] == [radius, radius]
    assert world.replay()['scene']['robot_radius'] == radius
    assert world.env.instance.to_scene().robot_radius == radius
    for _ in range(2):
        event = world.step(dict(target_id='entity-a', a0=40, a1=44, ticks=12))
        assert event['reason'] == 'duration' and event['periods'] == 12
        info = world.env.slots[0].last_info
        assert info['gap_target'] == gap and info['gap'] == pytest.approx(gap, abs=.004)
    pos = backend.robot_bodies[0].position
    assert 2.63-pos.y > .15
    assert math.isclose(face_x-pos.x-radius, gap, abs_tol=.004)
