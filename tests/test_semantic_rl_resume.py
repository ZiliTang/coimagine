"""A split PPO run must match uninterrupted updates, including Adam moments."""
from dataclasses import replace
from pathlib import Path
import torch
import pytest
from experiments.planar_tasks.relational_search.pipeline import Campaign,save,read
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.environment import EventConfig,Environment
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.authoring_v2.semantic_rl_parallel import Config,run,create
from experiments.planar_tasks.authoring_v2.semantic_rl_resume import schedule


def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b


def test_split_matches_uninterrupted_model_optimizer_rng_schedule(tmp_path):
    torch.set_num_threads(1);torch.manual_seed(31)
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    parent=tmp_path/'parent';c=Campaign.create(parent)
    (parent/'inputs').mkdir();(parent/'inputs/source.py').write_text(source)
    f=Features(source,fast_embedding=True,action_protocol=protocol(EventConfig(mode='duration')),
        execution_backend='trusted_inprocess',trusted_source_sha256=digest(source))
    try:
        h=f.new_history();h.append(Environment(seed_fixture()).history.events[0]['observation'])
        model=Policy([f.panel(h,ActionTree(h.events[-1]['observation']),{})],f.sha256)
        model.save(parent/'inputs/initial.pt',{})
    finally:f.close()
    cases=[dict(id=f'case{i}',family='test',split='train',fixture=seed_fixture(),fixture_sha256=digest(seed_fixture())) for i in range(3)]
    save(parent/'panels/train.json',dict(cases=cases));c.freeze()
    cfg=Config(workers=2,rounds=2,epochs=2,minibatch_size=2,max_ticks=2,max_decisions=2,episode_seconds=60,
        wall_seconds=600,temperature=1.,feature_runtime='typed_stream',execution_mode='duration',execution_backend='trusted_inprocess')
    torch.manual_seed(777);run(tmp_path/'full',parent,cfg)
    torch.manual_seed(777);run(tmp_path/'first',parent,replace(cfg,rounds=1))
    def diagnostic(*args):
        torch.rand(20)  # Callback is not allowed to perturb the training RNG.
        return dict(global_round=args[-1],success_rate=0.)
    run(tmp_path/'second',parent,replace(cfg,rounds=1),resume_from=tmp_path/'first',on_round=diagnostic)
    a,_=Policy.load(tmp_path/'full/models/final.pt',digest(source));b,_=Policy.load(tmp_path/'second/models/final.pt',digest(source))
    equal(a.state_dict(),b.state_dict())
    oa=torch.load(tmp_path/'full/models/optimizer_0001.pt',weights_only=True)
    ob=torch.load(tmp_path/'second/models/optimizer_0000.pt',weights_only=True);equal(oa,ob)
    full=read(tmp_path/'full/campaign.json');first=read(tmp_path/'first/campaign.json');second=read(tmp_path/'second/campaign.json')
    assert full['training_schedule']==first['training_schedule']+second['training_schedule']
    assert second['continuation']['episode_offset']==2 and second['continuation']['round_offset']==1
    assert second['runs'][0]['policy_version']==1
    for j in range(2):
        old=read(tmp_path/f'full/runs/a{j+2:05d}/request.json');new=read(tmp_path/f'second/runs/a{j:05d}/request.json')
        assert old['seed']==new['seed'] and old['case']==new['case']
    assert read(tmp_path/'second/metrics.json')['diagnostics'][0]['global_round']==2
    for root in ('full','first','second'):Campaign(tmp_path/root).verify_frozen()
    with pytest.raises(ValueError,match='config mismatch'):
        create(tmp_path/'bad',parent,replace(cfg,temperature=2.),resume_from=tmp_path/'first')


def test_schedule_keeps_unconsumed_shuffle_tail():
    cases=[dict(id=str(i)) for i in range(3)]
    whole=schedule(cases,12,13)
    assert whole[:5]==schedule(cases,12,5)
    assert whole[5:]==schedule(cases,12,13)[5:]
