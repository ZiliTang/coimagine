import numpy as np
from experiments.planar_tasks.semantic_rl.joint_evidence import compatibility,evidence_block,WIDTH

A=dict(target_id='box',a0=3,a1=46,ticks=20)


def test_unknown_prefix_is_not_target_free_or_wrong_action():
    c=compatibility({},A)
    assert np.isnan(c[:11]).all() and c[11]==1
    assert compatibility({'target_id':None},A)[0]==0
    assert compatibility({'target_id':'box'},A)[0]==1


def test_parameter_evidence_retains_type_direction_and_scale_distinctions():
    c=compatibility(dict(A,a0=1,ticks=10),A)
    assert c[1]==1 and c[2]==1 and c[3]<0 and c[4]==0 and c[10]<0
    c=compatibility(dict(A,a0=19),A)
    assert c[1]==1 and c[2]==-1


def test_joint_binding_same_operands_no_outcome_inputs():
    state=dict(target_seen=1,blocked_robots=2,object_progress_5s=-.1)
    raw=evidence_block(state,A,A,'box',0,True,'enriched')
    joint=evidence_block(state,A,A,'box',0,True,'joint')
    assert raw.shape==joint.shape==(WIDTH,)
    np.testing.assert_array_equal(raw[:46],joint[:46])
    assert np.all(raw[46:]==0)
    assert joint[46:50].tolist()==[1,1,1,1]
    other=evidence_block(state,A,dict(A,target_id='other'),'box',0,True,'joint')
    assert other[46:50].tolist()==[0,0,0,0]
    np.testing.assert_array_equal(evidence_block(state,A,A,'box',0,True,'current'),np.zeros(WIDTH))


def test_entity_renaming_preserves_bound_evidence():
    state=dict(target_seen=1)
    x=evidence_block(state,A,A,'box',0,True,'joint')
    b=dict(A,target_id='renamed')
    np.testing.assert_array_equal(x,evidence_block(state,b,b,'renamed',0,True,'joint'))


def test_frozen_library_online_offline_arm_parity():
    from pathlib import Path
    import pytest
    from experiments.planar_tasks.authoring_v2 import random_room_coverage_v2 as V
    from experiments.planar_tasks.authoring_v2.da_joint_models import features,ModeCandidates
    from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS
    if not (V.O.PARENT/'demonstrations.json').exists():pytest.skip('frozen library unavailable')
    row=V.read(V.O.PARENT/'demonstrations.json')[0];r=V.unpack(row['result_path'])
    fs={mode:features(mode) for mode in ('current','enriched','joint')}
    try:
        hs={mode:f.new_history() for mode,f in fs.items()}
        for mode,h in hs.items():h.append(r['public_rows'][0]['observation'])
        tree=ActionTree(r['public_rows'][0]['observation']);prefix={};a=r['public_rows'][1]['action']
        for key in FIELDS:
            panels={mode:f.panel(hs[mode],tree,prefix) for mode,f in fs.items()}
            for mode,p in panels.items():
                assert p['choices']==tree.choices(prefix)
                np.testing.assert_array_equal(p['attribution_context'],panels['joint']['attribution_context'])
                masked=ModeCandidates(panels['joint']['attribution_candidates'],mode,10)[:]
                np.testing.assert_array_equal(p['attribution_candidates'],masked)
            prefix=dict(prefix,**{key:a[key]})
    finally:
        for f in fs.values():f.close()


def test_gda_forced_prefix_preserves_geometry_and_full_execution():
    import pytest
    from experiments.planar_tasks.authoring_v2 import da_joint_study as D
    from experiments.planar_tasks.semantic_rl.handoff_features import HandoffLibraryFeatures
    if not D.DONOR.exists():pytest.skip('frozen donor unavailable')
    V=D.V;model=V.I.Scorer.load(D.DONOR)
    case=V.read(D.BASE/'inputs.json')['cases'][0]
    old=V.LibraryFeatures(V.entries(),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
    fixed=HandoffLibraryFeatures(V.entries(),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
    def run(f,prefix=()):
        return V.rollout(case['fixture'],f.source,model=model,features=f,config=V.O.CFG,
            max_ticks=80,max_decisions=4,wall_seconds=60,deterministic=True,fast_feedback=True,prefix=prefix)[0]
    try:
        a=run(old);prefix=[r['action'] for r in a['public_rows'][1:3]];b=run(fixed,prefix)
        assert a['summary']['error'] is b['summary']['error'] is None
        assert V.digest(V.visible(a['public_rows']))==V.digest(V.visible(b['public_rows']))
        assert V.digest(a['replay']['frames'])==V.digest(b['replay']['frames'])
    finally:old.close();fixed.close()


def test_streamed_normalization_keeps_exact_training_updates():
    import torch
    from experiments.planar_tasks.semantic_rl.library_learning import train as old_train
    from experiments.planar_tasks.semantic_rl.joint_learning import train as new_train
    rng=np.random.default_rng(123)
    ds=dict(contexts=rng.normal(size=(2,3)),candidates=rng.normal(size=(18,4)),
            offsets=np.array([[0,2,4,6,8],[8,11,14,16,18]]),targets=np.zeros((2,4),dtype=int))
    ds['candidates'][2,1]=np.nan
    layout=dict(context_groups=['D']*3,candidate_groups=['A','A','common','common'])
    args=(ds,[0,1],layout,'GDA','test-source',None)
    a,_=old_train(*args,updates=3);b,_=new_train(*args,updates=3)
    for key,v in a.state_dict().items():assert torch.equal(v,b.state_dict()[key]),key
