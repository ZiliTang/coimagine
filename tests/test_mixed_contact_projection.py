"""Mixed PUSH reserves the nearest same-side visible unoccupied contact."""
from dataclasses import replace
import math
import pytest
from coimagine.random_start import spec
from coimagine.random_start.same_face import SameFaceReservation,dot,sub
from coimagine.random_start.effect_push import EffectPushSkill,estimate_face,make_effect_skill
from experiments.option_testbench import mixed_contact_eval as M
from experiments.option_testbench import follow_peer_eval as F


def sample():
    c=next(c for c in M.cases() if c['name']=='push_follow')
    w=F.make_world(F.instance_from_dict(c['instance']),'projected');obs=w.boundary_observations()[0]
    return c,obs,estimate_face(obs,obs.patch,M.VARIANTS['projected'])


def test_mixed_target_is_nearest_point_in_visible_unoccupied_lane():
    _,obs,face=sample();r=SameFaceReservation().project(face,obs,face.point,.05,occupied_only=True)
    assert r.status=='allocated';t=(-face.normal[1],face.normal[0]);q=dot(face.point,t)
    assert dot(r.target,t)==pytest.approx(min(max(q,r.interval[0]),r.interval[1]))
    assert dot(sub(r.target,r.peer),t)*r.side>=r.target_separation-1e-9
    assert r.target[0]<2.4 and r.preferred[0]>2.45


def test_unilateral_projection_does_not_require_equal_half_face_allocation():
    _,obs,face=sample();a,b=SameFaceReservation(),SameFaceReservation()
    uni=a.project(face,obs,face.point,.05,occupied_only=True);dual=b.project(face,obs,face.point,.05)
    assert math.dist(uni.target,face.point)<math.dist(dual.target,face.point)
    # Both remain on the original side of the peer even if preference crosses it.
    again=a.project(face,replace(obs,time=.05),uni.peer,.05,occupied_only=True)
    assert again.side==uni.side
    t=(-face.normal[1],face.normal[0]);assert dot(sub(again.target,again.peer),t)*again.side>=again.target_separation-1e-9


def test_no_remaining_visible_lane_is_explicitly_rejected():
    _,obs,face=sample();t=(-face.normal[1],face.normal[0]);q=dot(face.point,t)
    # Keep inferred face, but censor visibility to a small area occupied by peer.
    points=tuple((face.point[0]+s*t[0],face.point[1]+s*t[1]) for s in (-.01,.01))
    face=replace(face,patch=replace(face.patch,points_world=points))
    r=SameFaceReservation().project(face,obs,face.point,.05,occupied_only=True)
    assert r.status=='lane_unavailable' and r.target is None


def test_peer_leave_releases_mixed_reservation_and_tie_order_never_crosses():
    _,obs,face=sample();reserve=SameFaceReservation();assert reserve.project(face,obs,face.point,.05,occupied_only=True).status=='allocated'
    # Own range return shows a robot well away from this face.
    own=(obs.own.x,obs.own.y);peer=(3.8,2.8);v=sub(peer,own);d=math.hypot(*v);hit=(own[0]+v[0]*(d-spec.ROBOT_RADIUS)/d,own[1]+v[1]*(d-spec.ROBOT_RADIUS)/d)
    far=replace(obs,teammate_nearest=(d-spec.ROBOT_RADIUS,hit),time=.05)
    assert reserve.project(face,far,face.point,.05,occupied_only=True).status=='not_same_face'
    assert reserve.side is None


def test_controller_uses_own_peer_return_not_hidden_labels_or_other_private_data():
    c,obs,_=sample();action=F.ACTIONS[c['actions'][0]];cfg=M.VARIANTS['projected']
    changed=replace(obs,scan=replace(obs.scan,raw_labels=('hidden',)*len(obs.scan.valid)),teammate=replace(obs.teammate,x=999.,y=-999.))
    a,b=make_effect_skill(action,config=cfg),make_effect_skill(action,config=cfg)
    fa,ia=a.step(obs,.05);fb,ib=b.step(changed,.05)
    assert fa==pytest.approx(fb) and ia['effect']==ib['effect']
    assert ia['effect']['same_face']['status']=='allocated' and not a.cooperative


@pytest.fixture(scope='module')
def records():
    c=next(c for c in M.cases() if c['name']=='push_follow')
    return {v:M.run(c,v) for v in M.VARIANTS}


def test_main_case_establishes_contact_and_completes_without_changing_budget(records):
    old,new=records['baseline'],records['projected'];m=new['metric'];push=m['push_outcomes'][0]
    assert old['metric']['exits'][0]=='option_budget' and old['metric']['contact_seconds'][0]==0
    assert m['exits'][0]=='normal_travel_complete' and push['normal_travel']>=.3
    assert push['first_contact_seconds'] is not None and push['contact_seconds']>1.
    assert m['time_budgets']==old['metric']['time_budgets'] and m['min_peer_gap']>.02
    assert push['min_target_reserve_slack']>=-1e-8 and m['reward_error']<1e-8
    from coimagine.random_start.env import EnvConfig
    assert m['max_penetration']<EnvConfig().penetration_limit
    assert all(math.hypot(*f)<=spec.MAX_FORCE+1e-8 for r in new['rows'] for f in r['forces'])


def test_config_is_explicit_and_does_not_change_historical_rl_descriptor():
    from experiments.option_rl_v10.interface import metadata,validate_metadata,controller_descriptor
    data=metadata();assert 'mixed_contact_allocation' not in data['controller_config'];validate_metadata(data)
    assert controller_descriptor(M.VARIANTS['projected'])['mixed_contact_allocation']
    with pytest.raises(ValueError):replace(M.VARIANTS['projected'],same_face_allocation=False)
