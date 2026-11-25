from types import SimpleNamespace

import numpy as np

from experiments.option_rl_v19 import unknown_shape
from experiments.planar_tasks.semantic_rl import push_phases


def test_tactile_contact_requires_matching_finite_support(monkeypatch):
    face = [None, np.array([1.,0.]), np.array([0.,1.]), 2., .1, .6]
    monkeypatch.setattr(unknown_shape, 'visible_faces', lambda obs: [face])
    sensors = SimpleNamespace(contacting_object=True, own=SimpleNamespace(x=1.8,y=.3,angle=0.),
                              contact_point_robot=(.2,0.))
    env = SimpleNamespace(active_target='box',total_periods=2,_sensor=lambda *args:sensors)
    world = SimpleNamespace(env=env)
    cmd = dict(a0=0,a1=44,target_id='box',ticks=20)
    assert push_phases.contact_state(world,cmd,0,(1.,0.))=='ready'
    assert push_phases.contact_state(world,cmd,0,(-1.,0.))=='support_unobservable'
    sensors.contact_point_robot=(.2,.5)
    assert push_phases.contact_state(world,cmd,0,(1.,0.))=='contact_on_other_surface'
    sensors.contacting_object=False
    assert push_phases.push_from_contact(world,cmd,20).ticks==0


def test_push_from_contact_does_not_move_unbound_target():
    world=SimpleNamespace(env=SimpleNamespace(active_target='other'))
    out=push_phases.push_from_contact(world,dict(a0=0,a1=44,target_id='box',ticks=20),20)
    assert out.status=='target_unbound' and out.ticks==0


def test_visible_push_cone_agrees_with_menu_and_nearest_normal():
    from math import cos, radians, sin
    from experiments.option_rl_v21 import spec

    def patch(angle):
        n = (cos(radians(angle)), sin(radians(angle)))
        return SimpleNamespace(valid=True, points_world=[(0.,0.),(-n[1],n[0]),(-2*n[1],2*n[0])],
                               normal_world=n)

    obs = SimpleNamespace(skill_parameters={'local_push': True}, patches=())
    from coimagine.random_start.spec import GEOMETRY_PROFILE
    limit = 22.5 if GEOMETRY_PROFILE == 'legacy36' else 45.
    for angle, accepted in [(20., True), (limit, True), (-limit, True), (limit+.001, False),
                            (-limit-.001, False), (180., False)]:
        obs.patches = (patch(angle),)
        assert bool(spec.mask(obs)[0]) == accepted
        assert bool(unknown_shape.select_simple_point(obs, spec.ACTIONS[0])) == accepted
    obs.patches = (patch(20.), patch(-5.))
    point = unknown_shape.select_simple_point(obs, spec.ACTIONS[0])
    np.testing.assert_allclose(point['n'], obs.patches[1].normal_world)
    obs.patches = ()
    assert not spec.mask(obs)[0]


def test_rejected_push_does_not_step_and_an_explicit_orbit_does_not_push():
    from experiments.planar_tasks.relational_search.contract import NoLegalPushFace
    from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
    from experiments.planar_tasks.tasks import task_catalog
    import pytest

    env = MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance)
    menu = env.public()['entity_legal_pairs']['entity-a']
    code = next(k for k in range(0, 32, 4) if [k,44] not in menu)
    start = env.total_periods
    with pytest.raises(NoLegalPushFace) as err:
        env.step_command(dict(target_id='entity-a', a0=code, a1=44, ticks=12))
    assert err.value.robot == 0 and env.total_periods == start and not env.command_prefix
    orbit = dict(target_id='entity-a', a0=40, a1=44, ticks=12)
    event = env.step_command(orbit)
    assert event['executed'] == orbit
    assert env.total_periods-start == 12 and env.command_prefix == [orbit]
