"""Geometry-equivalent CPU batch rays and bounded efficiency candidates."""
from dataclasses import replace
import math
import random
import numpy as np
import pytest
from coimagine.random_start import env as E,spec
from coimagine.random_start.geometry import box_polygon
from coimagine.random_start.lidar import scan_lidar,scan_lidar_batch
from experiments.option_testbench import execution_efficiency_eval as X
from experiments.option_testbench.follow_peer_eval import instance_from_dict


def compare(args,**kwargs):
    a,b=scan_lidar(*args,**kwargs),scan_lidar_batch(*args,**kwargs)
    assert a.valid==b.valid and a.category==b.category and a.raw_labels==b.raw_labels
    assert a.ray_angles==b.ray_angles and a.heading==b.heading and a.origin==b.origin
    assert a.sample_time==b.sample_time and a.tick==b.tick
    np.testing.assert_allclose(a.distances,b.distances,rtol=0,atol=1e-12)
    return a,b


@pytest.mark.parametrize('rays',[4,8,72,288])
def test_batch_matches_body_fixed_ranges_categories_caps_and_occlusion(rays):
    rng=random.Random(8231)
    obstacles=(box_polygon(1.3,2.5,.3,.15,.4),box_polygon(3.5,1.,-.2,.3,.1))
    for k in range(40):
        heading=rng.uniform(-math.pi,math.pi)
        robots=[(rng.uniform(.2,4.8),rng.uniform(.2,3.8),heading,0,0,0),
                (rng.uniform(.2,4.8),rng.uniform(.2,3.8),heading+.3,0,0,0)]
        obj=(rng.uniform(1,4),rng.uniform(1,3),rng.uniform(-math.pi,math.pi))
        for side in (0,1):compare(((5.,4.),obstacles,obj,robots,side,k,.05*k),scan_rays=rays)


def test_exact_wall_axis_corner_near_tangent_and_tie_cases():
    for heading in (0.,math.pi/2,math.pi/4,math.pi-1e-12):
        for xy in ((.181,.181),(.21,2.),(2.2,1.5)):
            robots=[(*xy,heading,0,0,0),(2.8,1.5,0,0,0,0)]
            compare(((5.,4.),(),(2.5,2.,0.),robots,0,0,0.))


def test_counterfactual_orientation_override_and_range_are_preserved():
    args=((5.,4.),(),(2.5,2.,.6),[(2.2,1.5,.2,0,0,0),(2.8,1.5,0,0,0,0)],0,1,.05)
    compare(args,object_angle_override=0.,scan_range=.35)
    a,b=compare(args)
    assert len(a.distances)==len(b.distances)==72


def test_clock_guard_and_lidar_branch_configuration_are_explicit():
    base=E.EnvConfig().base
    with pytest.raises(ValueError,match='20 Hz'):E.EnvConfig(base=replace(base,physics=replace(base.physics,control_dt=.1)))
    c=next(c for c in X.cases() if c['name']=='push_follow');inst=instance_from_dict(c['instance'])
    a,b=X.make_world(inst,'eff_scalar'),X.make_world(inst,'eff_batch')
    state=a.export_branch_state();assert state['lidar_backend']=='scalar'
    with pytest.raises(ValueError,match='different lidar'):b.restore_branch_state(state)
    b.restore_branch_state(state,allow_execution_change=True)
    assert b.total_periods==0


@pytest.fixture(scope='module')
def main_runs():
    c=next(c for c in X.cases() if c['name']=='push_follow')
    return {v:X.run(c,v) for v in ('eff_scalar','eff_batch','eff_fast','eff_physics60','eff_prepare')}


def test_batch_scanner_preserves_real_forces_and_trajectory(main_runs):
    a,b=main_runs['eff_scalar'],main_runs['eff_batch']
    assert [r['true'] for r in a['rows']]==[r['true'] for r in b['rows']]
    assert [r['forces'] for r in a['rows']]==[r['forces'] for r in b['rows']]
    assert a['metric']['exits']==b['metric']['exits']


def test_fast_translation_retains_contract_quality_and_original_timeout(main_runs):
    a,b=main_runs['eff_batch']['metric'],main_runs['eff_fast']['metric']
    assert b['seconds']<a['seconds'] and b['translation_quality']['basic_effect_retained']
    assert b['time_budgets']==a['time_budgets']
    assert b['exits']==['normal_travel_complete','follow_push_complete']
    r=main_runs['eff_fast'];assert all(math.hypot(*f)<=spec.MAX_FORCE+1e-9 for row in r['rows'] for f in row['forces'])
    assert b['reward_error']<1e-8


def test_lower_physics_rate_keeps_control_period_and_cost_units(main_runs):
    c=next(c for c in X.cases() if c['name']=='push_follow');w=X.make_world(instance_from_dict(c['instance']),'eff_physics60')
    assert w.config.base.physics.control_dt==.05 and w.config.base.physics.substeps==3
    r=main_runs['eff_physics60'];b=r['boundary']
    assert r['metric']['seconds']==pytest.approx(b['periods']*.05)
    assert b['reward']['time_cost']==pytest.approx(-spec.TIME_COST*b['periods'])
    assert r['metric']['reward_error']<1e-8


def test_preparation_only_preserves_contact_speed_and_basic_translation(main_runs):
    a,b=main_runs['eff_batch']['metric'],main_runs['eff_prepare']['metric']
    assert X.PREPARE.speed==X.BASE.speed==.12
    assert X.PREPARE.point_rate==X.BASE.point_rate
    assert b['seconds']<a['seconds'] and b['translation_quality']['basic_effect_retained']
    assert b['time_budgets']==a['time_budgets']


def test_preparation_only_long_scale_still_passes_predeclared_quality_gate():
    c=next(c for c in X.cases() if c['name']=='push_0.5_follow')
    m=X.run(c,'eff_prepare')['metric']
    assert m['translation_quality']['basic_effect_retained']
    assert m['push_outcomes'][0]['normal_travel']>=.5


def test_archived_controller_descriptor_keeps_disabled_new_speed_defaults():
    from experiments.option_rl_v10.interface import metadata,validate_metadata
    meta=metadata();assert 'approach_normal_speed' not in meta['controller_config']
    assert 'standalone_follow_speed' not in meta['controller_config'];validate_metadata(meta)
