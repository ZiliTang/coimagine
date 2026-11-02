"""Standalone FOLLOW: peer anticipation, local exclusions, real clocks and gap."""
from dataclasses import replace
import math
import pytest
from coimagine.random_start import spec
from coimagine.random_start.peer_clearance import guard_follow_velocity
from coimagine.random_start.effect_push import make_effect_skill
from experiments.option_testbench.follow_peer_eval import cases,make_world,request,VARIANTS,instance_from_dict


def observation():
    case=next(c for c in cases() if c['name']=='same_face_pair')
    return make_world(instance_from_dict(case['instance']),'legacy_5cm').boundary_observations()[0]


def test_peer_closing_component_is_removed_before_two_centimetre_abort():
    obs=observation();nominal=(.18,0.)
    command,info=guard_follow_velocity(obs,nominal)
    assert info['active'] and info['feasible'] and info['public_gap']>.02
    assert command[0]<=.02000001  # 7 cm gap permits only 2 cm/s approach to a 6 cm reserve.
    assert info['commanded_gap_rate']>=info['minimum_gap_rate']-1e-9
    assert math.hypot(*command)<=.18+1e-9


def test_far_peer_does_not_change_normal_follow_command():
    obs=observation();obs=replace(obs,teammate=replace(obs.teammate,x=obs.own.x+1.))
    command,info=guard_follow_velocity(obs,(.1,.03))
    assert command==(.1,.03) and not info['active']


def test_guard_never_uses_diagnostic_hidden_labels():
    obs=observation();changed=replace(obs,scan=replace(obs.scan,raw_labels=('hidden changed',)*len(obs.scan.valid)))
    assert guard_follow_velocity(obs,(.18,0.))==guard_follow_velocity(changed,(.18,0.))


def test_observed_enclosure_brakes_when_no_safe_peer_velocity_exists():
    # Own lidar returns at body radius in all directions forbid every translation.
    obs=observation();scan=replace(obs.scan,valid=(True,)*len(obs.scan.valid),category=(2,)*len(obs.scan.valid),
        distances=(spec.ROBOT_RADIUS,)*len(obs.scan.valid))
    obs=replace(obs,scan=scan,teammate=replace(obs.teammate,x=obs.own.x+.40))
    command,info=guard_follow_velocity(obs,(.18,0.))
    assert command==(0.,0.) and info['active'] and not info['feasible']


@pytest.fixture(scope='module')
def runs():
    panel={c['name']:c for c in cases()};out={}
    for name in ('train_26735_original_next','train_26602_original_next','healthy_static'):
        for variant in ('legacy_5cm','guard_2cm'):
            from experiments.option_testbench.follow_peer_eval import run
            out[name,variant]=run(panel[name],variant)
    return out


def test_frozen_follow_no_longer_reenters_peer_safety_abort(runs):
    for name in ('train_26735_original_next',):
        old,new=runs[name,'legacy_5cm'],runs[name,'guard_2cm']
        assert any(t[0]=='teammate' for t in old['metric']['safety_triggers'])
        assert not any(t[0]=='teammate' for t in new['metric']['safety_triggers'])
        assert new['metric']['min_nominal_peer_gap']>.02
        assert new['metric']['time_budgets']==old['metric']['time_budgets']


def test_yield_time_not_credited_and_reference_force_bounds_kept(runs):
    record=runs['train_26735_original_next','guard_2cm'];previous=[0.,0.];guarded=0
    for row in record['rows']:
        for i in (0,1):
            info=row[f'r{i}'];assert math.hypot(*row['forces'][i])<=spec.MAX_FORCE+1e-8
            if info['kind']!='follow':continue
            active=bool((info.get('follow_peer_guard') or {}).get('active'))
            t=info['follow_time'];assert t-previous[i]<=spec.CONTROL_DT+1e-8
            if active and not info['follow_peer_guard'].get('maintenance_held'):
                assert t==pytest.approx(previous[i]);guarded+=1
            previous[i]=t
    assert guarded>0 and record['metric']['reward_error']<1e-8


def test_two_centimetre_gap_changes_physical_clearance_without_claiming_contact(runs):
    old,new=runs['healthy_static','legacy_5cm'],runs['healthy_static','guard_2cm']
    assert new['metric']['exits'][0]=='follow_complete'
    assert new['metric']['final_second_true_object_gaps'][0]==pytest.approx(.02,abs=.005)
    assert old['metric']['final_second_true_object_gaps'][0]==pytest.approx(.05,abs=.005)
    assert new['metric']['max_penetration']==0
    assert new['metric']['guard_seconds'][0]==0


def test_factory_preserves_legacy_defaults_and_exposes_new_follow_configuration():
    from coimagine.random_start.effect_push import ACTIONS_REPOSITION_V2
    action=ACTIONS_REPOSITION_V2[request('follow',duration=2.)]
    old,new=make_effect_skill(action),make_effect_skill(action,config=VARIANTS['guard_2cm'])
    assert old.target_gap==.05 and old.peer_guard=='off'
    assert new.target_gap==.02 and new.peer_guard=='scan_projection_v1'
    with pytest.raises(ValueError):replace(VARIANTS['guard_2cm'],standalone_follow_gap=float('nan'))


def test_new_optional_controls_do_not_invalidate_unchanged_run10_metadata():
    from experiments.option_rl_v10.interface import controller_descriptor,metadata,validate_metadata
    import hashlib,json
    old=metadata();assert 'standalone_follow_gap' not in old['controller_config']
    validate_metadata(old)
    changed=controller_descriptor(VARIANTS['guard_2cm'])
    assert changed['standalone_follow_gap']==.02
    old['controller_config']=changed
    old['controller_config_hash']=hashlib.sha256(json.dumps(changed,sort_keys=True).encode()).hexdigest()
    with pytest.raises(ValueError,match='controller_config_hash'):validate_metadata(old)
