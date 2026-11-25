"""Outer peer routing, blocked static corridors and preserved distance contracts."""
from dataclasses import asdict, replace
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.boundary_follow import PeerOuterDetour, RoundedBoundary
from coimagine.random_start.effect_push import EffectPushConfig, make_effect_skill, ACTIONS_REPOSITION_V2
from coimagine.random_start.env import RandomStartEnv
from experiments.option_testbench.effect_push_eval import (
    comparison_config, instance_from_dict, reposition_peer_cases, run_one)
from experiments.option_testbench.reposition_arc_audit import audit


def execute(case, enabled):
    config=asdict(comparison_config('reposition_peer_detour', enabled, {}))
    feedback=config.pop('feedback')
    return run_one(case, feedback, config, 'reposition_peer_detour')


@pytest.fixture(scope='module')
def records():
    names=['detour_rp_CCW_push_same_1.80','detour_rp_CW_push_same_1.80',
           'detour_rp_CCW_solo_1.80','detour_rp_CCW_static_gap_0.26',
           'detour_rp_CCW_static_gap_0.78','detour_rp_CCW_peer_static_blocked',
           'detour_rp_CCW_wait_same_1.80','detour_rp_CW_invisible']
    cases={c['name']:c for c in reposition_peer_cases()}
    return {name:[execute(cases[name],v) for v in (False,True)] for name in names}


@pytest.mark.parametrize('direction',['CW','CCW'])
def test_peer_blocked_push_routes_outside_without_changing_budget(records,direction):
    old,new=records[f'detour_rp_{direction}_push_same_1.80']
    assert old['metric']['exits'][0] in {'safety_abort', 'blocked_no_progress'}
    assert new['metric']['exits']==['reposition_complete','normal_travel_complete']
    assert old['case']['actions']==new['case']['actions']
    assert old['boundary']['execution']['requested_time_budgets']==new['boundary']['execution']['requested_time_budgets']
    measured=audit(new)
    assert measured['options'][0]['independent_signed_travel_m']>=1.8
    assert measured['minimum_robot_centre_distance_m']-2*spec.ROBOT_RADIUS>.06
    assert measured['physical'][0]['max_command_tangent_step_deg']<10
    assert measured['physical'][0]['max_true_body_gap_m']>spec.ROBOT_DIAMETER+.04
    assert any(r['r0'].get('peer_detour',{}).get('object_guard')=='observed_plane'
               for r in new['rows'] if r['r0'].get('peer_detour'))


@pytest.mark.parametrize('name',['detour_rp_CCW_static_gap_0.26','detour_rp_CCW_peer_static_blocked'])
def test_impossible_corridor_stops_reposition_without_physical_overlap(records,name):
    _,new=records[name]
    assert new['metric']['exits'][0]=='blocked_static_route'
    if name == 'detour_rp_CCW_peer_static_blocked' and spec.GEOMETRY_PROFILE == 'compact20':
        # The pushing peer makes physical contact. Check the orbiting disc's
        # clearance independently rather than attributing that contact to it.
        from coimagine.random_start import geometry as geom
        walls = [geom.box_polygon(**wall) for wall in new['case']['instance']['obstacles']]
        for row in new['rows']:
            truth = row['true']
            robot, peer = truth['robots']
            box = geom.box_polygon(*truth['object'][:3], spec.OBJECT_WIDTH/2, spec.OBJECT_HEIGHT/2)
            assert all(geom.circle_polygon_gap(robot[:2], spec.ROBOT_RADIUS, p) > 0 for p in [box, *walls])
            assert math.dist(robot[:2], peer[:2]) > spec.ROBOT_DIAMETER
    else:
        assert new['metric']['max_penetration']==0
    final=next(r['r0'] for r in new['rows'] if r['r0']['exit_reason']=='blocked_static_route')
    assert final['static_route_guard']['status']=='blocked'
    assert final['signed_progress']<final['progress_target']
    assert not new['rows'][0]['r0']['contacting_static']


@pytest.mark.parametrize('name',['detour_rp_CCW_solo_1.80','detour_rp_CCW_static_gap_0.78'])
def test_clear_route_preserves_old_physical_trajectory_exactly(records,name):
    old,new=records[name]
    assert old['metric']['exits'][0]==new['metric']['exits'][0]=='reposition_complete'
    assert [r['true'] for r in old['rows']]==[r['true'] for r in new['rows']]
    assert [r['forces'] for r in old['rows']]==[r['forces'] for r in new['rows']]


def test_occlusion_and_invisible_controls_remain_explicit_failures(records):
    new=records['detour_rp_CCW_wait_same_1.80'][1]
    expected = 'reposition_complete' if spec.GEOMETRY_PROFILE == 'compact20' else 'patch_lost'
    assert new['metric']['exits'][0] == expected
    assert audit(new)['minimum_robot_centre_distance_m']-2*spec.ROBOT_RADIUS>.10
    assert records['detour_rp_CW_invisible'][1]['metric']['exits'][0]=='not_observed'
    for pair in records.values():
        for record in pair:
            assert audit(record)['max_reconstruction_error_m']<1e-8
            assert all(math.hypot(*f)<=spec.MAX_FORCE+1e-9 for r in record['rows'] for f in r['forces'])


def test_obstacle_round_trip_is_not_silently_dropped():
    c=next(c for c in reposition_peer_cases() if c['name']=='detour_rp_CCW_static_gap_0.26')
    assert instance_from_dict(c['instance']).as_dict()['obstacles']==c['instance']['obstacles']


def test_detour_inputs_are_current_shared_peer_pose_and_own_observation():
    c=next(c for c in reposition_peer_cases() if c['name']=='detour_rp_CCW_push_same_1.80')
    obs=RandomStartEnv(instance_from_dict(c['instance'])).boundary_observations()[0]
    boundary=RoundedBoundary().update(obs,.05)
    assert boundary is not None
    changed=replace(obs, scan=replace(obs.scan,raw_labels=('hidden-changed',)*len(obs.scan.valid)))
    a=PeerOuterDetour();b=PeerOuterDetour()
    assert a.update(obs,boundary,.02)==b.update(changed,boundary,.02)
    assert a.info==b.info
    far=replace(obs,teammate=replace(obs.teammate,x=4.5,y=.5))
    router=PeerOuterDetour();assert router.update(far,boundary,.02)==(boundary.outward,.02)
    assert not router.info['active']


def test_factory_requires_explicit_finite_boundary_opt_in():
    a=next(a for a in ACTIONS_REPOSITION_V2 if a.kind=='reposition')
    assert make_effect_skill(a).peer_detour=='off'
    with pytest.raises(ValueError):EffectPushConfig(reposition_peer_detour='outer_arc')
    with pytest.raises(ValueError):EffectPushConfig(reposition_peer_detour='teleport')
    c=EffectPushConfig(reposition_boundary='rounded_scan',reposition_peer_detour='outer_arc')
    assert make_effect_skill(a,config=c).peer_detour=='outer_arc'
