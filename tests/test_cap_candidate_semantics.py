from copy import deepcopy

import numpy as np

from experiments.planar_tasks.relational_search.motion import move_id
from experiments.planar_tasks.semantic_rl.candidate_semantics import candidate_semantics


def fixture():
    angles = np.arange(72)*np.pi/36
    scan = dict(origin=[1.,2.],ray_angles=angles,distances=np.full(72,5.),valid=np.zeros(72,bool),
                category=np.zeros(72,int),entity_ids=[None]*72)
    obs = dict(tick=10,goal=dict(xy=[5.,2.],target_id='T'),
               entities=[dict(id='T',last_seen_tick=10,points=[[2,1.85],[2,2.15],[2.2,1.85],[2.2,2.15]])],
               robots=[dict(own=dict(x=1.,y=2.),scan=scan,execution={})])
    obs['robots'].append(deepcopy(obs['robots'][0]))
    scan['valid'][0] = True
    scan['category'][0] = 2
    scan['distances'][0] = 2.
    return obs


def test_push_east_hits_obstacle_and_push_west_does_not():
    obs = fixture()
    east = dict(a0=3,a1=44,ticks=60,target_id='T')
    west = dict(east,a0=19)
    a,b = (candidate_semantics(obs,v) for v in (east,west))
    assert a['sweep_static']==1 and b['sweep_static']==0
    assert a['push_goal_progress']>0>b['push_goal_progress']
    obs['entities'][0]['last_seen_tick'] -= 1
    assert candidate_semantics(obs,east)['push_box_seen']==0


def test_progress_uses_elapsed_time_and_only_matching_previous_direction():
    obs = fixture()
    old = deepcopy(obs)
    old['tick'] = 5
    old['robots'][0]['own']['x'] -= .1
    obs['robots'][0]['execution'] = dict(direction_unavailable=True)
    a = dict(a0=move_id('E'),a1=44,ticks=40,target_id=None)
    same = candidate_semantics(obs,a,old,a)
    assert np.isclose(same['repeat_robot_progress'], .4)
    assert same['repeat_blocked']==1
    opposite = candidate_semantics(obs,dict(a,a0=move_id('W')),old,a)
    assert opposite['repeat_robot_progress']==opposite['repeat_blocked']==0
    assert candidate_semantics(obs,a)['repeat_robot_progress']==0
