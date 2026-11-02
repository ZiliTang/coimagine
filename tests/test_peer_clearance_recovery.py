"""Own-scan separation, frozen-peer independence, real accounting and blocked pockets."""
from dataclasses import replace
from functools import partial
import math
import pytest
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import env as E,layout,spec
from coimagine.random_start.peer_clearance import PeerClearancePrefix,local_escape,public_gap
from coimagine.random_start.effect_push import ACTIONS_REPOSITION_V2 as ACTIONS,REPOSITION_V2_VERSION,make_effect_skill
from experiments.option_rl_v10.interface import CONFIG
from experiments.option_testbench.peer_clearance_eval import cases,make_world,instance_from_dict


def near_instance():
    return layout.InstanceSpec(name='near_peer',map='L0',split='dev',index=0,seed=0,arena=(4.,3.),obstacles=(),
        object_pose=Pose(2.5,2.,0.),robot_poses=(Pose(2.315,1.585,math.pi/2),Pose(2.685,1.585,math.pi/2)),
        goal_centre=(2.5,.6),goal_radius=.2)


def request(kind,direction=None,distance=None,duration=None):
    return next(a.index for a in ACTIONS if a.kind==kind and (direction is None or a.direction==direction)
                and (distance is None or a.distance==distance) and (duration is None or a.duration==duration))


@pytest.fixture(scope='module')
def recovered():
    instance=near_instance();actions=(request('reposition','CW',.4),request('wait',duration=2.))
    original=make_world(instance,False);old=original.joint_step(*actions,record=True)
    fixed=make_world(instance,True);new=fixed.joint_step(*actions,record=True)
    return original,old,fixed,new


def test_near_peer_can_finish_real_reposition_instead_of_one_tick_exit(recovered):
    _,old,world,new=recovered
    assert old['periods']==1 and all(old['skill_ends'][i]['exit_reason']=='safety_abort' for i in (0,1))
    assert new['skill_ends'][0]['exit_reason']=='reposition_complete'
    assert new['peer_clearance']['entered'] and not new['peer_clearance']['failed']
    assert old['execution']['requested_time_budgets']==new['execution']['requested_time_budgets']
    assert max(r['penetration'] for r in world.rows)==0
    assert public_gap(world.boundary_observations()[0])[0]>.06


def test_separation_motion_gets_no_nominal_travel_credit_and_reward_is_real(recovered):
    _,_,world,b=recovered
    rows=[r for r in world.rows if r['r0']['peer_clearance']]
    assert rows and rows[-1]['r0']['peer_clearance']['path_m']>0
    for row in rows:
        for i in (0,1):assert (row[f'r{i}']['signed_progress'] or 0)==0 and (row[f'r{i}']['completion_progress'] or 0)==0
    # Independently reproduce preceding-command signed boundary travel, excluding the prefix.
    pose=[(p.x,p.y) for p in world.instance.robot_poses];previous_pose=list(pose);previous=None;distance=0.
    for row in world.rows:
        info=row['r0']
        if (previous and previous['kind']=='reposition' and previous['stage']=='move' and not previous['done']
            and previous['patch_valid'] and not previous['contacting_object'] and info['patch_valid']
            and not info['contacting_object'] and previous['tangent_world']):
            t=previous['tangent_world'];distance+=sum((pose[0][j]-previous_pose[0][j])*t[j] for j in (0,1))/math.hypot(*t)
        if info['kind']=='reposition':assert info['signed_progress']==pytest.approx(distance,abs=1e-8)
        previous=info;previous_pose=pose;pose=[r[:2] for r in row['true']['robots']]
    assert distance>=.4
    audit=E.recompute_episode_reward(world.rows,[b],(2.5,2.),world.config)
    assert audit['total']==pytest.approx(b['reward']['total'])
    assert b['reward']['time_cost']==pytest.approx(-spec.TIME_COST*b['periods'])


def test_force_and_reference_acceleration_limits_are_kept(recovered):
    world=recovered[2];previous=[(0.,0.),(0.,0.)]
    for row in world.rows:
        for i in (0,1):assert math.hypot(*row['forces'][i])<=spec.MAX_FORCE+1e-8
        if row['r0']['peer_clearance']:
            for i in (0,1):
                r=row[f'r{i}']['reference_velocity'];assert math.dist(previous[i],r)<=spec.A_REF_MAX*spec.CONTROL_DT+1e-8;previous[i]=r


def test_other_scan_and_hidden_labels_do_not_change_this_robots_escape():
    obs=make_world(near_instance(),False).boundary_observations()
    private=replace(obs[1],patch=None,patches=(),scan=replace(obs[1].scan,distances=tuple(.181 for _ in obs[1].scan.distances)))
    first=PeerClearancePrefix().step(obs,.05);second=PeerClearancePrefix().step((obs[0],private),.05)
    assert first[0]==second[0]
    altered=replace(obs[0],scan=replace(obs[0].scan,raw_labels=('changed',)*len(obs[0].scan.valid)))
    assert local_escape(obs[0])==local_escape(altered)


@pytest.mark.parametrize('name,failed',[('caged',True),('one_free',False)])
def test_observed_static_pocket_stops_or_lets_only_free_side_retreat(name,failed):
    case=next(c for c in cases([]) if c['name']==name);world=make_world(instance_from_dict(case['instance']),True)
    b=world.joint_step(*case['actions'],record=True)
    assert b['peer_clearance']['failed']==failed
    assert max(r['penetration'] for r in world.rows)==0
    if failed:
        assert b['periods']==40 and all(b['skill_ends'][i]['exit_reason']=='blocked_peer_clearance' for i in (0,1))
        assert world._last_progress[0]==(0.,.4)
    else:
        released=[r for r in world.rows if (r['r0']['peer_clearance'] or {}).get('phase')=='released']
        assert released and released[0]['r0']['peer_clearance']['public_gap']>=.06
        assert b['peer_clearance']['path_m'][1]<1e-5


def test_safe_world_matches_original_forces_and_trajectory_exactly():
    case=next(c for c in cases([]) if c['name']=='healthy_rp_solo');instance=instance_from_dict(case['instance'])
    a,b=make_world(instance,False),make_world(instance,True)
    a.joint_step(*case['actions'],record=True);end=b.joint_step(*case['actions'],record=True)
    assert not end['peer_clearance']['entered']
    assert [r['true'] for r in a.rows]==[r['true'] for r in b.rows]
    assert [r['forces'] for r in a.rows]==[r['forces'] for r in b.rows]


def test_wait_clock_and_task_timeout_do_not_gain_free_time():
    actions=(request('wait',duration=2.),)*2;world=make_world(near_instance(),True);b=world.joint_step(*actions,record=True)
    assert b['peer_clearance']['entered'] and b['periods']==40
    short=E.RandomStartEnv(near_instance(),E.EnvConfig(time_budget=.5,peer_clearance_recovery='boundary_v1'),
        skill_factory=partial(make_effect_skill,config=CONFIG),action_table=ACTIONS,action_version=REPOSITION_V2_VERSION)
    e=short.joint_step(*actions,record=True)
    assert e['termination']=='task_budget_exhausted' and e['periods']==10
    assert e['reward']['failure_penalty']==spec.FAILURE_PENALTY


def test_restore_rejects_silent_controller_switch():
    a,b=make_world(near_instance(),False),make_world(near_instance(),True);state=a.export_branch_state()
    with pytest.raises(ValueError,match='different peer-clearance'):b.restore_branch_state(state)
    b.restore_branch_state(state,allow_execution_change=True)
    assert b.total_periods==a.total_periods
