"""Shared acquisition preserves views and never crosses world mutations."""
from dataclasses import FrozenInstanceError
import math

import pymunk
import pytest
from coimagine.random_start import lidar_service as L
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv, MovableBox
from experiments.planar_tasks.tasks import task_catalog
from experiments.planar_tasks.relational_search.physics_probe import PhysicsProbe


def world():
    env = MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance,
                              [MovableBox('b', 2., 2.85, width=.3, height=.3)])
    env.cache_public_sensors = True
    return env


def test_public_targets_and_controller_share_one_cast_per_robot(monkeypatch):
    env = world()
    original = L.nearest_hits
    calls = []
    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)
    monkeypatch.setattr(L, 'nearest_hits', counted)
    packet = env.public()
    assert len(calls) == 2
    assert set(packet['entity_legal_pairs']) == {'entity-a', 'b'}
    with env.lidar.read_scope():
        raw = env.ray_frame(0, 0, 0.)
        with env.lidar.read_scope():
            assert env.ray_frame(0, 0, 0.) is raw
            for target in ('entity-a', 'b'):
                view = env._target_scan(0, 0, 0., target)
                assert view is env._target_scan(0, 0, 0., target)
                assert view.distances is raw.scan.distances
                assert view.valid is raw.scan.valid
                assert all((c == 1) == (label == target and valid)
                           for c, label, valid in zip(view.category, raw.entity_ids, view.valid))
        assert len(calls) == 3
        with pytest.raises(FrozenInstanceError): raw.entity_ids = ()
    assert not env.lidar._memo
    assert env.public() == packet and len(calls) == 5


@pytest.mark.parametrize('changed', ['object', 'own', 'peer', 'heading', 'velocity'])
def test_same_tick_external_changes_are_visible(changed):
    env = world(); b = env.env.backend
    old = env.public()
    if changed == 'heading':
        env.heading[0] += .2
    else:
        body = b.object_body if changed == 'object' else b.robot_bodies[changed == 'peer']
        if changed == 'velocity': body.velocity = (.2, .1)
        else: body.position += (.15, .08)
        b.space.reindex_shapes_for_body(body)
    new = env.public()
    assert old != new
    # Bypass shared caches/native acquisition as an independent path.
    env.cache_public_sensors = False
    assert new == env.public()


def test_physics_step_and_target_switch_invalidate_within_scope():
    env = world(); b = env.env.backend
    with env.lidar.read_scope():
        a = env.ray_frame(0, 0, 0.)
        sensor = env._sensor(0, 0, 0., 'entity-a')
        b.apply_and_step(((15., 0.), (0., 0.)))
        # Deliberately request the same timestamp: invalidation must not rely on it.
        other = env.ray_frame(0, 0, 0.)
        assert other is not a
        assert env._sensor(0, 0, 0., 'entity-a') is not sensor
        env._commit_target('b')
        assert env.ray_frame(0, 0, 0.) is not other
        assert env.scan_for(0, 0, 0.) == env._target_scan(0, 0, 0., 'b')
    assert not env.lidar._memo


def test_scope_exception_clears_cached_results():
    env = world()
    with pytest.raises(RuntimeError):
        with env.lidar.read_scope():
            env.public()
            assert env.lidar._memo
            raise RuntimeError('interrupted consumer')
    assert env.lidar._depth == 0 and not env.lidar._memo


def test_probe_keeps_short_world_fixed_rays_and_exact_hit_points():
    env = PhysicsProbe()
    env.step_forces(((5., 1.), (3., -1.)))
    packet = env.observation(); b = env.backend
    points = {}
    for i, robot in enumerate(b.robot_bodies):
        scan = packet['robots'][i]['scan']
        origin = tuple(robot.position)
        for j in range(72):
            angle = j * 2 * math.pi / 72
            end = (origin[0] + 1.2 * math.cos(angle), origin[1] + 1.2 * math.sin(angle))
            hit = min((h for h in b.space.segment_query(origin, end, 0., pymunk.ShapeFilter())
                       if h.shape.body is not robot), key=lambda h: h.alpha, default=None)
            label = b.entity_shapes.get(hit.shape) if hit else None
            assert scan['ray_angles'][j] == angle
            assert scan['distances'][j] == (float(hit.alpha * 1.2) if hit else 0.)
            assert scan['entity_ids'][j] == label
            if label: points.setdefault(label, []).append([float(hit.point.x), float(hit.point.y)])
    assert points == {e['id']: e['points'] for e in packet['entities']}
