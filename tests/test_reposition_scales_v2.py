"""Versioned larger REPOSITION scales must not reinterpret archived action IDs."""
from dataclasses import asdict
import hashlib
import json

import pytest

from coimagine.random_start import spec
from coimagine.random_start.effect_push import (
    ACTIONS,ACTIONS_REPOSITION_V2,REPOSITION_DISTANCES_V2,REPOSITION_V2_VERSION,VERSION,
    action_id,action_mask,type_scale_mask,make_effect_skill)
from coimagine.random_start.env import RandomStartEnv
from experiments.option_testbench.effect_push_eval import (
    comparison_config,hydrate_actions,instance_from_dict,replay_robot_options,reposition_scale_cases,run_one)
from experiments.option_testbench.reposition_arc_audit import local_position


def test_new_ragged_codebook_preserves_legacy_menu_and_pushes():
    assert len(spec.ROBOT_ACTIONS)==28 and len(ACTIONS)==66 and len(ACTIONS_REPOSITION_V2)==68
    assert [asdict(a) for a in ACTIONS[:50]]==[asdict(a) for a in ACTIONS_REPOSITION_V2[:50]]
    assert ACTIONS[58].kind=='wait'
    assert ACTIONS_REPOSITION_V2[58].kind=='reposition' and ACTIONS_REPOSITION_V2[58].distance==1.2
    assert [a.index for a in ACTIONS_REPOSITION_V2]==list(range(68))
    for direction in ('CW','CCW'):
        assert tuple(a.distance for a in ACTIONS_REPOSITION_V2 if a.kind=='reposition' and a.direction==direction)==REPOSITION_DISTANCES_V2
    assert len({(a.type_index,a.scale_index) for a in ACTIONS_REPOSITION_V2})==68


def test_mask_and_encoding_use_the_explicit_menu():
    c=reposition_scale_cases()[0];w=RandomStartEnv(instance_from_dict(c['instance']));obs=w.boundary_observations()[0]
    assert action_mask(obs,actions=ACTIONS_REPOSITION_V2).shape==(68,)
    mask=type_scale_mask(obs,actions=ACTIONS_REPOSITION_V2)
    assert mask[10:12,4].all() and not mask[12:,4].any()
    assert action_id(11,4,actions=ACTIONS_REPOSITION_V2)==59
    with pytest.raises(ValueError):action_id(11,4)


def test_branch_restore_rejects_a_different_menu():
    instance=instance_from_dict(reposition_scale_cases()[0]['instance'])
    new=RandomStartEnv(instance,action_table=ACTIONS_REPOSITION_V2,action_version=REPOSITION_V2_VERSION)
    old=RandomStartEnv(instance,action_table=ACTIONS,action_version=VERSION)
    with pytest.raises(ValueError,match='version|codebook'):
        old.restore_branch_state(new.export_branch_state())


def test_replay_decodes_same_numeric_id_against_its_frozen_source():
    def record(table):
        actions=[asdict(a) for a in table]
        r={'case':{'actions':[58,50]},'rows':[{'r0':{'effect':None},'r1':{'effect':None}}],
           'metric':{'status':'completed'},'boundary':{'action_codebook_hash':hashlib.sha256(json.dumps(actions,sort_keys=True).encode()).hexdigest()}}
        hydrate_actions(r,{'actions':actions});return r
    old,new=record(ACTIONS),record(ACTIONS_REPOSITION_V2)
    assert replay_robot_options(old)[0]['kind']=='wait' and replay_robot_options(old)[0]['target']==2.
    assert replay_robot_options(new)[0]['kind']=='reposition' and replay_robot_options(new)[0]['target']==1.2
    new['requested_actions'][0]={**new['requested_actions'][0],'label':'wrong'}
    with pytest.raises(ValueError,match='frozen source'):
        hydrate_actions(new,{'actions':[asdict(a) for a in ACTIONS_REPOSITION_V2]})


@pytest.mark.parametrize('direction',['CW','CCW'])
def test_long_scale_reaches_opposite_face_in_matched_solo_scene(direction):
    c=next(c for c in reposition_scale_cases() if c['name']==f'scale2_rp_{direction}_solo_1.80')
    cfg=asdict(comparison_config('reposition_scales_v2',True,{}));feedback=cfg.pop('feedback')
    r=run_one(c,feedback,cfg,'reposition_scales_v2')
    assert r['boundary']['action_version']==REPOSITION_V2_VERSION
    assert len(r['availability_mask'])==68
    assert r['metric']['option_outcomes'][0]['target']==1.8
    assert r['metric']['option_outcomes'][0]['recorded_progress']>=1.8
    assert r['metric']['exits'][0]=='reposition_complete'
    state=r['rows'][-1]['true'];x,y=local_position(state['robots'][0],state['object'])
    assert abs(x)<.5 and y>.125
