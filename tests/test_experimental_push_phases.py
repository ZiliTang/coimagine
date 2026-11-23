from types import SimpleNamespace

import numpy as np

from experiments.planar_tasks.semantic_rl import push_phases


def test_tactile_contact_requires_matching_finite_support(monkeypatch):
    face = [None, np.array([1.,0.]), np.array([0.,1.]), 2., .1, .6]
    monkeypatch.setattr(push_phases, 'visible_faces', lambda obs: [face])
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
