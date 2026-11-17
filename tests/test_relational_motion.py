"""Target-free sensing motion, physical switching and nominal branch regression."""
from dataclasses import replace
from itertools import islice
import math

import numpy as np
import pytest
from coimagine.continuous2d.scenes import Pose, Box
from coimagine.random_start import spec as R
from experiments.planar_tasks.tasks import task_catalog
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
from experiments.planar_tasks.relational_search.contract import History, action_space, validate_action, public_observation
from experiments.planar_tasks.relational_search.motion import move_id, DIRECTIONS, SPEEDS, MOVE_ACTIONS, global_pairs
from experiments.planar_tasks.relational_search.predictor import NominalPredictor


def fixture():
    return replace(task_catalog()['planar_open_anchor'].instance, arena=(10.,8.),
        object_pose=Pose(6.6,2.,0), robot_poses=(Pose(1.,2.,0),Pose(1.,6.,0)), goal_centre=(8.,2.))


def move(direction='E', speed=.30, ticks=40, other=44):
    return dict(target_id=None,a0=move_id(direction,speed),a1=other,ticks=ticks)


def test_no_entity_required_and_motion_reveals_real_ray_hits():
    env=MultiObjectOptionEnv(fixture());before=env.offline_audit();o=env.public()
    assert o['entities']==[] and o['entity_legal_pairs']=={}
    assert len(o['global_legal_pairs'])==33**2
    assert public_observation(o)==o
    h=History();h.append(o);a=move()
    result=env.step_command(a);h.append(env.public(),a)
    assert result['handoff']['phase']=='move' and result['end_tick']==40
    assert env.robot_states()[0].x>1.3
    assert env.public()['entities'][0]['id']=='entity-a'
    assert before['entities']==env.offline_audit()['entities']
    assert before['robots'][1]==env.offline_audit()['robots'][1]
    assert env.replay_commands().offline_audit()==env.offline_audit()


@pytest.mark.parametrize('direction',DIRECTIONS)
def test_eight_world_directions_independent_of_heading(direction):
    spec=replace(fixture(),robot_poses=(Pose(4.,4.,1.1),Pose(1.,7.,0)))
    env=MultiObjectOptionEnv(spec)
    env.step_command(move(direction,ticks=20))
    delta=np.array([env.robot_states()[0].x-4.,env.robot_states()[0].y-4.])
    angle=DIRECTIONS.index(direction)*math.pi/4
    desired=np.array([math.cos(angle),math.sin(angle)])
    assert delta@desired>.1
    assert abs(np.linalg.det(np.array([delta,desired])))<1e-10


def test_speed_levels_force_acceleration_and_continuity():
    distances=[]
    for speed in SPEEDS:
        env=MultiObjectOptionEnv(fixture());env.step_command(move(speed=speed))
        distances.append(env.robot_states()[0].x-1.)
        skill=env.slots[0].skill
        old_reference=np.array(skill.reference.value)
        env.step_command(move('W',speed=speed,ticks=1))
        assert env.slots[0].skill is skill
        assert np.linalg.norm(np.array(skill.reference.value)-old_reference)<=R.A_REF_MAX*R.CONTROL_DT+1e-10
        assert math.hypot(*env.slots[0].last_info['force'])<=R.MAX_FORCE+1e-10
    assert distances==sorted(distances) and len(set(distances))==4


def test_each_robot_can_move_independently():
    env=MultiObjectOptionEnv(fixture())
    env.step_command(move('E',other=move_id('S',.20)))
    assert env.robot_states()[0].x>1.3 and env.robot_states()[1].y<5.8


def test_wall_braking_and_retreat_use_same_physics():
    env=MultiObjectOptionEnv(replace(fixture(),obstacles=(Box(2.3,2.,.1,.7),)))
    for _ in range(5):env.step_command(move(speed=.40))
    assert env.termination is None
    assert env.robot_states()[0].x<2.2-R.ROBOT_RADIUS
    assert env.slots[0].last_info['direction_unavailable']
    assert max(env.offline_audit()['penetration'].values())<1e-6
    stopped=env.robot_states()[0].x
    env.step_command(move('W'))
    assert env.robot_states()[0].x<stopped-.3


def test_all_observed_entities_are_obstacles_for_move():
    from experiments.planar_tasks.relational_search.environment import MovableBox
    spec=replace(fixture(),object_pose=Pose(8.,2.,0),goal_centre=(9.,2.))
    for target in (True,False):
        env=MultiObjectOptionEnv(replace(spec,object_pose=Pose(2.3,2.,0)) if target else spec,
            () if target else (MovableBox('b',2.3,2.,.25,1.),))
        poses=env.offline_audit()['entities']
        for _ in range(5):env.step_command(move(speed=.40))
        assert env.robot_states()[0].x<2.175-R.ROBOT_RADIUS
        assert env.slots[0].last_info['direction_unavailable']
        assert env.offline_audit()['entities']==poses


def test_move_push_brake_switching_and_recording_replay():
    env=MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance)
    env.step_command(dict(target_id='entity-a',a0=2,a1=2,ticks=20))
    result=env.step_command(move('W',ticks=1),record=True)
    assert result['end_tick']==21
    assert env.slots[0].last_info['stage']=='switching'
    env.step_command(move('W',ticks=20),record=True)
    # Return to a legal manipulation intent only; brake permits a fresh skill.
    env.step_command(dict(target_id=None,a0=44,a1=44,ticks=40))
    available=env.public()['entity_legal_pairs']['entity-a']
    a0,a1=next(p for p in available if p[0]<32 and p[1]<32)
    env.step_command(dict(target_id='entity-a',a0=a0,a1=a1,ticks=20))
    assert env.slots[0].skill.kind=='push'
    assert env.active_target=='entity-a'
    assert env.replay_commands().offline_audit()==env.offline_audit()


def test_navigation_in_lazy_menu_and_rejects_target_or_mixed_push():
    env=MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance);o=env.public()
    early=list(islice(action_space(o),8))
    assert any(a['target_id'] is None for a in early)
    assert any(a['target_id']=='entity-a' for a in early)
    for a in early:validate_action(a,o)
    for a0,a1 in global_pairs():
        validate_action(dict(target_id=None,a0=a0,a1=a1,ticks=1),o)
    for a in (dict(move(),target_id='entity-a'),dict(move(),a1=2),dict(move(),a0=999)):
        with pytest.raises(ValueError):validate_action(a,o)
    for speed in (float('nan'),True,.123,-.1):
        with pytest.raises(ValueError):move_id('E',speed)
    assert len(MOVE_ACTIONS)==32
    # All 40 durations and all independent joint velocities remain expressible.
    unseen=MultiObjectOptionEnv(fixture()).public()
    commands=list(action_space(unseen))
    assert len(commands)==40*33**2
    assert len({(a['a0'],a['a1'],a['ticks']) for a in commands})==len(commands)


def test_nominal_motion_branch_and_real_world_isolation():
    env=MultiObjectOptionEnv(task_catalog()['planar_open_anchor'].instance)
    h=History();h.append(env.public());before=env.offline_audit()
    model=NominalPredictor(h,ticks=120,calls=3,seconds=15.)
    a=move('W',ticks=20)
    first=model.predict(model.root,a)
    second=model.predict(first['handle'],a)
    repeated=model.predict(first['handle'],a)
    assert second['observation']==repeated['observation']
    assert model.ledger.used==dict(ticks=80,calls=3)
    assert second['observation']['robots'][0]['own']['x']<h.events[0]['observation']['robots'][0]['own']['x']-.2
    assert env.offline_audit()==before
    model.close()


def test_unknown_transport_geometry_is_explicit_not_hallucinated():
    env=MultiObjectOptionEnv(fixture());h=History();h.append(env.public())
    with pytest.raises(ValueError,match='insufficient observed geometry'):NominalPredictor(h)
    # Unavailable transport model must not prevent actual sensing motion.
    env.step_command(move())
    assert env.total_periods==40
