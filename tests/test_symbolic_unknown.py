import math
from copy import deepcopy

from shapely.geometry import Point,box

from experiments.planar_tasks.authoring_v2.symbolic_unknown.evidence import EvidenceMap
from experiments.planar_tasks.authoring_v2.symbolic_unknown.geometry import MotionContract,envelopes,rectangle_pose
from experiments.planar_tasks.authoring_v2.symbolic_unknown.policy import compiled,actions,schema
from experiments.planar_tasks.authoring_v2.symbolic_unknown.runtime import Controller


def observations(origins=((0.,0.),(.4,.4))):
    robots=[]
    for origin in origins:
        angles=[i*2*math.pi/72 for i in range(72)]
        robots.append(dict(scan=dict(origin=list(origin),ray_angles=angles,distances=[2.]*72,
                                    valid=[False]*72,category=[0]*72)))
    return dict(tick=0,robots=robots)


def test_no_component_prior_does_not_fill_between_rays():
    m=EvidenceMap();m.update(observations())
    assert m.free.is_empty
    assert m.query(Point(.2,.2).buffer(.02)).status=='unknown'


def test_closed_small_ray_cells_supply_conditional_positive_area_evidence():
    m=EvidenceMap(minimum_static_component_area=.04);m.update(observations())
    assert m.free.area>0
    assert m.query(Point(.2,.2).buffer(.01)).certified
    assert m.query(Point(5.,5.).buffer(.01)).status=='unknown'


def test_single_fan_has_no_closed_observed_free_boundary():
    m=EvidenceMap(minimum_static_component_area=.04);m.update(observations(origins=((0.,0.),)))
    assert m.free.is_empty


def test_memory_eviction_turns_evidence_into_unknown():
    m=EvidenceMap();m.add_initial_certificate(box(-1,-1,1,1),source='declared_initial_separation')
    assert m.query(Point(.5,.5).buffer(.01)).certified
    m.forget_outside(box(-1,-1,0,1))
    assert m.query(Point(.5,.5).buffer(.01)).status=='unknown'


def test_occupied_hit_overrides_free_certificate_and_records_conflict():
    m=EvidenceMap();m.add_initial_certificate(box(-1,-1,1,1),source='declared_initial_separation')
    m.hits[(.5,.5)]=0
    assert m.query(Point(.5,.5).buffer(.01)).status=='occupied'
    assert m.ledger['sensor_conflicts']==1


def test_empirical_rollout_success_cannot_resolve_unknown_space():
    c=Controller()
    assert not c.online_refinement(dict(certified=False),[dict(path_safe=True)]*3)['allowed']
    assert not c.online_refinement(dict(certified=True),[dict(path_safe=True),dict(error='failed branch')])['allowed']
    assert c.online_refinement(dict(certified=True),[dict(path_safe=True)])['allowed']


def values():
    z={k:(False if v['unit']=='bool'else 0.)for k,v in schema()['metrics'].items()}
    for n in actions():
        z['legal:'+n]=True;z['dynamic:'+n]=True;z['bound_valid:'+n]=True
        z['score:'+n]=101. if n.startswith('push')else 10.
    z['score:brake']=0.
    return z


def test_unknown_push_is_not_selected_and_stop_needs_its_own_certificate():
    p=compiled();z=values()
    assert p.select(z)['option']is None
    z['known:brake']=z['free:brake']=True
    assert p.select(z)['option']=='brake'
    z['known:push_1_0']=z['free:push_1_0']=True
    assert p.select(z)['option']=='push_1_0'


def test_after_real_observation_same_policy_can_enable_push():
    p=compiled();before=values();before['known:brake']=before['free:brake']=True
    after=deepcopy(before);after['known:push_1_0']=after['free:push_1_0']=True
    assert p.select(before)['option']=='brake'
    assert p.select(after)['option']=='push_1_0'
    report=p.check({k:v if type(v)is bool else[v,v]for k,v in before.items()},'brake',
        [dict(updates={k:v if type(v)is bool else[v,v]for k,v in after.items()},path_safe=True)],quality='empirical')
    assert report['status']=='conditional'


def test_motion_contract_includes_extra_stopping_distance():
    o=dict(robots=[dict(own=dict(x=0.,y=0.)),dict(own=dict(x=1.,y=0.))])
    c=MotionContract();e=envelopes(o,None,'observe',0,c,direction=0)
    assert e['robots'][0].covers(Point(.2,0))
    assert not e['robots'][0].covers(Point(-.2,0))


def test_brake_envelope_includes_nonzero_public_reference_tail():
    o=dict(robots=[dict(own=dict(x=0.,y=0.,vx=0.,vy=-.1644),execution=dict(reference_velocity=[0.,-.2])),
                   dict(own=dict(x=1.,y=0.))])
    e=envelopes(o,None,'brake',None)
    assert e['robots'][0].covers(Point(0.,-.16))
    assert not e['robots'][0].covers(Point(0.,.16))


def test_prediction_does_not_write_actual_evidence():
    c=Controller();before=c.map.snapshot()
    c.online_refinement(dict(certified=False),[dict(path_safe=True,free=True,known=True)])
    assert c.map.snapshot()==before


def test_source_policy_cannot_select_unknown_with_high_score():
    p=compiled();z=values();z['known:brake']=z['free:brake']=True
    z['score:push_0_0']=1000.
    z['free:push_0_0']=True  # an inconsistent estimate still cannot erase unknown
    assert p.select(z)['option']=='brake'


def test_hidden_obstacle_does_not_change_initial_public_decision():
    from tools.cap_unknown_safety import fixture
    from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
    from experiments.planar_tasks.authoring_v2.symbolic_general import simulation
    from experiments.planar_tasks.relational_search.contract import digest
    decisions=[];observed=[]
    for blocked in (False,True):
        world=Environment(fixture(blocked),EventConfig(mode='duration'),fast_feedback=True,record_details=False)
        o=simulation.observation(world);observed.append(digest(o));d=Controller().decide(o)
        decisions.append((d['command'],d['values'],d['certified']))
        assert d['unknown_goal']and not d['action'].startswith('push')
    assert observed[0]==observed[1]
    assert decisions[0]==decisions[1]
