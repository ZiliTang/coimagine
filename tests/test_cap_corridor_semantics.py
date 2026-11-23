import numpy as np
import pytest

from experiments.planar_tasks.semantic_rl.corridor_semantics import corridor_semantics
from tools.cap_corridor_subtypes import audit_type


def observation():
    angles = np.arange(72) * (2*np.pi/72)
    scan = dict(origin=[1.,2.], ray_angles=angles, distances=np.full(72,5.),
                valid=np.zeros(72,bool), category=np.zeros(72,int), entity_ids=[None]*72)
    points = [[2.,1.85],[2.,2.15],[2.2,1.85],[2.2,2.15]]
    return dict(tick=0, goal=dict(xy=[4.,2.],target_id='T'), robots=[dict(scan=scan,arena=[8.,6.])],
                entities=[dict(id='T',points=points,last_seen_tick=0)])


@pytest.mark.parametrize('category,entity,expected',[(2,None,'static'),(1,'B','movable'),
                                                     (1,'T','clear'),(3,None,'clear')])
def test_visible_return(category,entity,expected):
    obs = observation()
    scan = obs['robots'][0]['scan']
    scan['distances'][0] = 2.
    scan['valid'][0] = True
    scan['category'][0] = category
    scan['entity_ids'][0] = entity
    res = corridor_semantics(obs)
    if expected == 'clear' and category in (1,3):
        # The return can occlude the corridor even when it is not an obstacle.
        assert not res['static_hit'] and not res['movable_hit']
    else:
        assert res['kind'] == expected


def test_unknown_when_target_stale_or_corridor_occluded():
    obs = observation()
    assert corridor_semantics(obs)['kind'] == 'clear'
    obs['robots'][0]['scan']['valid'][:] = True
    obs['robots'][0]['scan']['distances'][:] = .1
    assert corridor_semantics(obs)['kind'] == 'unknown'
    obs['tick'] = 1
    assert corridor_semantics(obs)['target_estimated'] == 0


def test_fixture_full_box_sweep():
    block = dict(cx=3.,cy=2.46,angle=0.,half_w=.15,half_h=.05)
    room = dict(object=dict(x=2.,y=2.,angle=0.),goal=dict(x=4.,y=2.),
                room=dict(obstacles=[block]),movable_objects=[])
    assert audit_type(room) == 'static'
    room['object']['angle'] = np.pi/2
    assert audit_type(room) == 'clear'
    block.update(cx=5.,cy=2.)
    assert audit_type(room) == 'clear'
