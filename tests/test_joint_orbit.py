from itertools import product

import pytest

from experiments.planar_tasks.tasks import task_catalog
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
from experiments.planar_tasks.relational_search.joint_orbit import ORBITS, WAIT, YIELD, face_owner, resolve_pair
from experiments.planar_tasks.relational_search.contract import validate_action


def room():
    return MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance)


def request(a, b):
    return dict(target_id='entity-a', a0=a, a1=b, ticks=12)


def test_orbit_compatibility_and_bound_physical_execution():
    env = room()
    visible = env.public()
    menu = visible['entity_legal_pairs']['entity-a']
    assert {tuple(p) for p in menu if set(p).intersection(ORBITS)} == {
        (a,b) for a,b in product((*ORBITS, WAIT, YIELD), repeat=2) if a in ORBITS or b in ORBITS}
    with pytest.raises(ValueError):
        validate_action(request(40, 2), visible)
    toward = request(43, 40)
    e = env.step_command(toward)
    assert e['requested'] == toward and e['executed'] == request(43,YIELD)
    assert e['end_tick']-e['start_tick']==12
    assert e['handoff']['orbit_resolution']['reason']=='stagger_dual_orbit'
    assert env.offline_audit() == env.replay_commands().offline_audit()
    away = request(40,43)
    assert room().step_command(away)['executed'] == away


def test_wait_yield_and_goal_ownership_have_distinct_semantics():
    obs = room().public()
    toward = request(43, WAIT)
    assert resolve_pair(toward, obs) == (toward, None)
    # Keep a genuinely conflicting pass for either disc size. Retain the
    # observed target bearings and place the waiting side in the orbiter's path.
    from experiments.planar_tasks.relational_search.joint_orbit import _velocity, MIN_CENTRE_DISTANCE
    vx, vy = _velocity(obs['robots'][0], 43, toward['target_id'])
    speed = (vx*vx + vy*vy)**.5
    distance = MIN_CENTRE_DISTANCE + .03
    peer = obs['robots'][1]['own']
    own = obs['robots'][0]['own']
    peer.update(x=own['x'] + distance*vx/speed, y=own['y'] + distance*vy/speed)
    assert face_owner(toward, 0, obs) == 1
    yielding = request(43,YIELD)
    assert resolve_pair(yielding, obs) == (yielding, None)
    assert face_owner(yielding, 0, obs) == 0
    obs['robots'][1]['execution']['stage']='yield_blocked'
    action, reason = resolve_pair(yielding, obs)
    assert action == request(WAIT,YIELD) and reason
    assert face_owner(yielding, 0, obs) == 1
    obs['entity_legal_pairs']['entity-a'].remove([YIELD,43])
    assert face_owner(yielding, 0, obs) == 0


@pytest.mark.parametrize('mode', ['time', 'duration', 'progress'])
def test_author_request_and_executed_orbit_are_distinguishable_in_each_mode(mode):
    from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
    from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
    fixture = seed_fixture()
    fixture['object']['angle'] = 1.5707963267948966
    fixture['robots'] = [dict(x=1.25, y=y, angle=0.) for y in (1.75,2.25)]
    fixture['movable_objects'] = []
    world = Environment(fixture, EventConfig(mode=mode))
    command = request(43,40)
    event = world.step(command)
    assert event['requested'] == command
    assert event['executed'] == request(43,YIELD)
    assert event['handoff']['orbit_resolution']['reason'] == 'stagger_dual_orbit'
    if mode == 'duration':
        assert all(s['pair']==[43,YIELD] for s in event['segments'])
