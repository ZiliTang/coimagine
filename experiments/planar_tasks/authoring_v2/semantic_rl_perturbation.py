"""Frozen-semantics perturbation pilot: train-only PPO, sealed paired evaluation.

No author calls, task selection by outcomes, or best-checkpoint selection.
Run explicitly with --out NEW_DIRECTORY --parent FROZEN_STAGE1_CAMPAIGN.
"""
from collections import Counter
from dataclasses import asdict
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import time
import traceback

import torch
from .semantic_rl import development_fixtures
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,pack,unpack
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.learning import ppo_update
from experiments.planar_tasks.semantic_rl.runner import rollout

VERSION='frozen-semantic-perturbation-v1'
MAX_TICKS=1800
MAX_DECISIONS=300
EPISODE_SECONDS=120
TRAIN_EPISODES=24
TEMPERATURE=2.
BOUNDS={
    'small':dict(object_position=.04,object_angle=.04,robot_position=.05,robot_angle=.15,
                 obstacle_position=.08,obstacle_angle=.08),
    'medium':dict(object_position=.12,object_angle=.15,robot_position=.12,robot_angle=.30,
                  obstacle_position=.18,obstacle_angle=.20),
}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_panels():
    """Rejection uses offline initial collision geometry only, never outcome."""
    roots=development_fixtures('stage1-wall-anchor-v2');seen=set();panels={}
    for split,seed in [('train',2026111401),('test',2026111402)]:
        rng=random.Random(seed);cases=[]
        for family,base in roots.items():
            for index,level in enumerate(['small','small','medium','medium']):
                b=BOUNDS[level];rejected=[]
                for attempt in range(1000):
                    f=deepcopy(base);delta={}
                    for key,body,pos,angle in [('object',f['object'],b['object_position'],b['object_angle']),
                          *[(f'robot{i}',r,b['robot_position'],b['robot_angle']) for i,r in enumerate(f['robots'])],
                          *[(r['label'],r,b['obstacle_position'],b['obstacle_angle']) for r in f['movable_objects']]]:
                        for axis,scale in [('x',pos),('y',pos),('angle',angle)]:
                            d=rng.uniform(-scale,scale);body[axis]+=d;delta[key+'.'+axis]=d
                    try:X.fixture(f)
                    except ValueError as exc:
                        rejected.append(dict(delta=delta,reason=str(exc)));continue
                    h=digest(f)
                    if h in seen:
                        rejected.append(dict(delta=delta,reason='duplicate fixture'));continue
                    seen.add(h)
                    cases.append(dict(id=f'{split}_{family}_{index:02d}',family=family,split=split,
                        level=level,fixture=f,fixture_sha256=h,delta=delta,rejected=rejected,
                        feasibility='initial geometry valid; dynamic solvability unverified'))
                    break
                else:raise RuntimeError('geometry rejection cap; do not replace by outcome-selected cases')
        panels[split]=dict(seed=seed,bounds=BOUNDS,cases=cases)
    panels['roots']=dict(cases=[dict(id='root_'+k,family=k,split='root',level='anchor',fixture=f,
                            fixture_sha256=digest(f)) for k,f in roots.items()])
    return panels


def train_schedule(cases):
    order=[];rng=random.Random(2026111403)
    for epoch in range(2):
        ids=[c['id'] for c in cases];rng.shuffle(ids);order.extend(ids)
    assert len(order)==TRAIN_EPISODES
    return order


class PerturbationCampaign(Campaign):
    @classmethod
    def create(cls,path,parent):
        parent=Path(parent);parent_seal=Campaign(parent).verify_frozen()
        if read(parent/'campaign.json').get('fixture_profile')!='stage1-wall-anchor-v2':
            raise ValueError('wall-anchor v2 parent required')
        source=parent/'semantics/s000.py';checkpoint=parent/'semantic_fits/f000/policy.pt'
        if not source.exists() or not checkpoint.exists():raise ValueError('declared source/initial model missing')
        parent_config=read(parent/'campaign.json').get('stage1',{})
        event_config=EventConfig(**parent_config.get('event_config',{}))
        action_protocol=protocol(event_config) if event_config.mode=='duration' else None
        initial,_=Policy.load(checkpoint,digest(source.read_text()))
        if initial.action_protocol!=action_protocol:raise ValueError('parent action-scale protocol mismatch')
        panels=make_panels()
        c=super().create(path,limits=dict(execution_ticks=124200,attempts=69,resets=69,rounds=0,
                            fits=0,author_turns=0),model=dict(provider='none',name='fixed semantics PPO; no LLM'))
        (c.path/'inputs').mkdir();shutil.copyfile(source,c.path/'inputs/source.py')
        shutil.copyfile(checkpoint,c.path/'inputs/initial.pt')
        for split,panel in panels.items():save(c.path/'panels'/f'{split}.json',panel)
        schedule=train_schedule(panels['train']['cases'])
        manifest=dict(version=VERSION,parent=str(parent.resolve()),parent_seal=parent_seal,
                      source_sha256=sha(source),semantic_sha256=digest(source.read_text()),
                      initial_checkpoint_sha256=sha(checkpoint),panel_hashes={k:digest(v) for k,v in panels.items()})
        save(c.path/'inputs/manifest.json',manifest)
        save(c.path/'runtime.json',dict(python=sys.version,executable=sys.executable,torch=torch.__version__,
             git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
             git_status=subprocess.check_output(['git','status','--short'],text=True)))
        s=read(c.path/'campaign.json');s.update(profile=VERSION,runs=[],deadline=time.time()+9000,
                  training_schedule=schedule,selection='last checkpoint after fixed24episodes; no test-driven selection',
                  protocol=dict(event_config=asdict(event_config),action_protocol=action_protocol,feature_runtime=parent_config.get('feature_runtime','cached_fork'),train_temperature=TEMPERATURE,evaluation='deterministic, temperature1',
                      max_ticks=MAX_TICKS,max_decisions=MAX_DECISIONS,episode_wall_seconds=EPISODE_SECONDS,
                      optimizer='Adam',learning_rate=3e-4,ppo_epochs=4,residual_coef=.05,
                      failure_comparison_cost_seconds=180.,wall_seconds=9000,
                      fit_normalizers=False,semantic_revision=False,split_unit='initial configuration within family'))
        s['limits']['feature_batches']=72000;s['used']['feature_batches']=0
        save(c.path/'campaign.json',s)
        return cls(path)

    def verify_inputs(self):
        m=read(self.path/'inputs/manifest.json')
        if sha(self.path/'inputs/source.py')!=m['source_sha256'] or sha(self.path/'inputs/initial.pt')!=m['initial_checkpoint_sha256']:
            raise ValueError('frozen source or initialization changed')
        for name,h in m['panel_hashes'].items():
            if digest(read(self.path/'panels'/f'{name}.json'))!=h:raise ValueError('panel changed')

    def validate_case(self,case,train):
        split=case['split']
        if train and split!='train' or not train and split not in ('test','root'):
            raise ValueError('train/evaluation split violation')
        panel=read(self.path/'panels'/('roots.json' if split=='root' else split+'.json'))
        registered=next((r for r in panel['cases'] if r['id']==case['id']),None)
        if registered!=case or case['fixture_sha256']!=digest(case['fixture']):
            raise ValueError('unregistered or changed case')

    def run_case(self,case,condition,*,model=None,seed=17,train=False,checkpoint=None):
        self.validate_case(case,train)
        if condition not in ('cap','initial','rl') or (condition=='cap')!=(model is None):
            raise ValueError('condition/model mismatch')
        if train and (case['split']!='train' or condition!='rl' or model is None):
            raise ValueError('PPO collection may use train cases only')
        if not train:
            if case['split'] not in ('test','root'):raise ValueError('evaluation split')
            seal=read(self.path/'evaluation_seal.json')
            if sha(self.path/'models/final.pt')!=seal['checkpoint_sha256']:raise ValueError('evaluation model changed')
        with self.locked():
            self.verify_sources();self.verify_inputs()
            remaining=self.state['deadline']-time.time()
            if remaining<EPISODE_SECONDS:raise RuntimeError('insufficient remaining wall budget for a full admitted attempt')
            source=(self.path/'inputs/source.py').read_text()
            settings=self.state['protocol'];event_config=EventConfig(**settings.get('event_config',{}))
            features=Features(source,fast_embedding=True,runtime=settings.get('feature_runtime','legacy'),action_protocol=settings.get('action_protocol'))
            rid=f'a{len(self.state["runs"]):03d}';p=self.path/'runs'/rid;p.mkdir(parents=True)
            reserve=((5 if features.runtime=='typed_stream' else 4) if model is not None else 1)*MAX_DECISIONS+((2 if features.runtime=='typed_stream' else 1) if model is not None else 0)
            self._charge(attempts=1,resets=1,execution_ticks=MAX_TICKS,feature_batches=reserve)
            self.state['reservations'][rid]=dict(ticks=MAX_TICKS,features=reserve)
            request=dict(id=rid,case_id=case['id'],family=case['family'],split=case['split'],condition=condition,
                         train=train,seed=seed,temperature=TEMPERATURE if train else 1.,deterministic=not train,
                         checkpoint=checkpoint,fixture_sha256=case['fixture_sha256'])
            self.state['runs'].append(dict(request,status='running'));self._save();save(p/'request.json',request)
            result,batch=rollout(case['fixture'],source,model=model,features=features,config=event_config,fast_feedback=True,
                 max_ticks=MAX_TICKS,max_decisions=MAX_DECISIONS,wall_seconds=EPISODE_SECONDS,
                 seed=seed,deterministic=not train,temperature=request['temperature'],
                 on_progress=lambda v:save(p/'progress.json',dict(v,updated=time.time())))
            features.close()
            result.update(split=case['split'],case_id=case['id'],family=case['family'],condition=condition,
                          feature_batches=features.calls if model is not None or features.runtime in ('typed_batch','typed_stream') else result['summary']['decisions']+int(bool(result['summary']['error'])))
            # This is an external comparison score, not altered training reward.
            result['comparison_cost_seconds']=result['summary']['task_cost_seconds'] if result['summary']['success'] else 180.
            pack(p/'result.json.gz',result)
            if batch:pack(p/'batch.json.gz',batch)
            save(p/'summary.json',result['summary'])
            self.state['used']['execution_ticks']-=MAX_TICKS-result['summary']['ticks']
            self.state['used']['feature_batches']-=reserve-result['feature_batches']
            del self.state['reservations'][rid]
            self.state['runs'][-1].update(result['summary']);self._save()
            print(rid,case['id'],condition,result['summary']['status'],result['summary']['ticks'],flush=True)
            return self.state['runs'][-1],result,batch


def summarize(path):
    s=read(Path(path)/'campaign.json');rows=[]
    for split in ('test','root'):
        for condition in ('cap','initial','rl'):
            for family in development_fixtures('stage1-wall-anchor-v2'):
                rr=[r for r in s['runs'] if r['split']==split and r['condition']==condition and r['family']==family]
                if rr:rows.append(dict(split=split,condition=condition,family=family,n=len(rr),
                    success=sum(bool(r.get('success')) for r in rr),
                    mean_cost=sum(r['task_cost_seconds'] if r.get('success') else 180. for r in rr)/len(rr),
                    statuses=dict(Counter(r['status'] for r in rr))))
    return dict(rows=rows,used=s['used'],training_episodes=sum(r['train'] for r in s['runs']),
                scope='single training seed, paired within-family held-out initial configurations; not new task families')


def run(path,parent):
    torch.set_num_threads(1)
    c=PerturbationCampaign.create(path,parent);source=(c.path/'inputs/source.py').read_text()
    model,meta=Policy.load(c.path/'inputs/initial.pt',digest(source))
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    train={x['id']:x for x in read(c.path/'panels/train.json')['cases']}
    checkpoints=c.path/'models';checkpoints.mkdir()
    previous='inputs/initial.pt';updates=[]
    for i,cid in enumerate(read(c.path/'campaign.json')['training_schedule']):
        rec,result,batch=c.run_case(train[cid],'rl',model=model,seed=2026111500+i,train=True,checkpoint=previous)
        update=ppo_update(model,opt,batch,bootstrap=result['summary']['bootstrap']) if batch and result['summary']['eligible_for_ppo'] else dict(updates=0,skipped=True)
        previous=f'models/train_{i:03d}.pt'
        model.save(c.path/previous,dict(meta,perturbation_episode=i,train_temperature=TEMPERATURE))
        torch.save(dict(optimizer=opt.state_dict(),rng=torch.get_rng_state()),checkpoints/f'optimizer_{i:03d}.pt')
        updates.append(dict(episode=i,case_id=cid,run_id=rec['id'],checkpoint=previous,update=update))
        save(c.path/'training_progress.json',updates)
    shutil.copyfile(c.path/previous,checkpoints/'final.pt')
    c.verify_inputs()
    # No test outcomes have been generated before selecting/sealing the last model.
    save(c.path/'evaluation_seal.json',dict(at=time.time(),checkpoint_sha256=sha(checkpoints/'final.pt'),
                  source_sha256=sha(c.path/'inputs/source.py'),training_episodes=len(updates),selection='last, fixed schedule'))
    initial,_=Policy.load(c.path/'inputs/initial.pt',digest(source))
    final,_=Policy.load(checkpoints/'final.pt',digest(source))
    cases=read(c.path/'panels/test.json')['cases']+read(c.path/'panels/roots.json')['cases']
    for case in cases:
        for condition,policy,checkpoint in [('cap',None,None),('initial',initial,'inputs/initial.pt'),('rl',final,'models/final.pt')]:
            c.run_case(case,condition,model=policy,checkpoint=checkpoint)
            save(c.path/'comparison.json',summarize(c.path))
    save(c.path/'comparison.json',summarize(c.path))
    return c.freeze()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--parent',type=Path,required=True)
    args=p.parse_args()
    try:print(run(args.out,args.parent),flush=True)
    except BaseException as exc:
        # Preserve reservations on untrustworthy interruption; no silent restart.
        if args.out.exists():save(args.out/'FAILURE.json',dict(type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        raise
