"""Fast observations preserve ray ties, sensor hits, self masks and step state."""
import math
import random
import pymunk
from experiments.planar_tasks.relational_search.ray_query import nearest_hits
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture, fixture, RecordedEnv


def test_native_ray_results_match_reference_including_degenerate_cases():
    rng = random.Random(71)
    space = pymunk.Space()
    own = pymunk.Body(1, 1)
    own_shapes = [pymunk.Circle(own, .1), pymunk.Circle(own, .05, (.2, 0))]
    space.add(own, *own_shapes)
    # Coincident hit distances, sensor shapes, rounded segments and polygons.
    for i in range(12):
        b = pymunk.Body(body_type=pymunk.Body.STATIC)
        b.position = (.5, 0) if i < 2 else (rng.uniform(-1, 1), rng.uniform(-1, 1))
        b.angle = rng.uniform(-math.pi, math.pi) if i >= 2 else 0
        s = pymunk.Circle(b, .12) if i < 4 else (pymunk.Poly.create_box(b, (.2, .3))
             if i % 2 else pymunk.Segment(b, (-.2, 0), (.2, 0), .03))
        s.sensor = i in (0, 5)
        space.add(b, s)
    before = [(s, s.filter, s.sensor, s.body.position, s.body.angle) for s in space.shapes]
    origins = [(0., 0.), (.38, 0.), (.5, 0.), (5., 5.)]
    origins += [(rng.uniform(-1, 1), rng.uniform(-1, 1)) for _ in range(100)]
    for origin in origins:
        ends = [(origin[0]+1.2*math.cos(j*math.pi/36),
                 origin[1]+1.2*math.sin(j*math.pi/36)) for j in range(72)]
        assert nearest_hits(space, own, origin, ends) == nearest_hits(space, own, origin, ends, native=False)
    assert before == [(s, s.filter, s.sensor, s.body.position, s.body.angle) for s in space.shapes]


def test_contact_snapshot_invalidates_at_physics_step_and_scope_exit(monkeypatch):
    f = seed_fixture(); scene, boxes = fixture(f)
    env = RecordedEnv(scene, boxes, overrides=f['overrides']); b = env.env.backend
    original = b._contacts_uncached
    calls = []
    def counted():
        calls.append(1)
        return original()
    monkeypatch.setattr(b, '_contacts_uncached', counted)
    with b.contact_snapshot(True):
        first = b.contacts()
        assert b.contacts() == first and len(calls) == 1
        # Returned records cannot poison another consumer's snapshot.
        first.append({'corrupted': True})
        assert b.contacts() == original()
        b.apply_and_step(((0., 0.), (0., 0.)))
        assert b.contacts() == original() and len(calls) == 2
    b.robot_bodies[0].position = b.object_body.position
    b.space.reindex_shapes_for_body(b.robot_bodies[0])
    assert b.contacts() == original() and len(calls) == 3


def test_full_scan_and_public_packet_match_with_native_path():
    f = seed_fixture(); scene, boxes = fixture(f)
    env = RecordedEnv(scene, boxes, overrides=f['overrides'])
    old = env.public()
    env.cache_public_sensors = True
    assert env.public() == old
    for i in (0, 1):
        env.cache_public_sensors = False
        old_scan = env._rays_uncached(i, 0, 0.)
        env.cache_public_sensors = True
        assert env._rays_uncached(i, 0, 0.) == old_scan
