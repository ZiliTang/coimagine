"""Matched offline V and Q fits for GLOBAL attention on the highest inverse dose."""
import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch

from . import cap_inverse_scale as scale
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.temporal_data import prepare
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.temporal_q import H, LiveScorer, TemporalQ

OUT = scale.OUT.parent/'cap_global_attention_20261119_01'
PARENT = scale.PARENT
SEED = 17


def predict(model, data, ids, stage):
    hist, initial, mask = data.window(ids, model.f.frame)
    row = np.arange(len(ids))
    if stage == 'V':
        caps = data.a['cap'][ids]
        actions, support, _ = model.expert_actions(hist)
        actions, support = actions[row,caps], support[row,caps]
    else:
        chosen = data.a['chosen'][ids]
        actions = torch.tensor(data.a['candidates'][ids,chosen],dtype=torch.float32)
        support = torch.tensor(data.a['sources'][ids,chosen],dtype=torch.float32)
    return model(hist, initial, mask, actions[:,None], support[:,None], expert=stage=='V')[:,0]


@torch.no_grad()
def bootstrap(model,data,ids):
    hist, initial, mask = data.window(data.next(ids), model.f.frame)
    actions, support, legal = model.expert_actions(hist)
    value = model(hist,initial,mask,actions,support,expert=True)
    best = value.masked_fill(~legal,-torch.inf).max(dim=1).values
    best = torch.where(legal.any(1), best, 0.).numpy()
    return data.a['reward'][ids]+(~data.a['done'][ids])*best


@torch.no_grad()
def measure(model,data,ids,targets,stage):
    values = np.concatenate([predict(model,data,chunk,stage).numpy() for chunk in np.array_split(ids, max(1,(len(ids)+63)//64))])
    errors = (values-targets)**2
    return dict(mse=float(errors.mean()), mean_prediction=float(values.mean()),mean_target=float(targets.mean()),n=len(ids))


def write_checkpoint(path,model,stage,updates):
    pending=path.with_suffix('.pending.pt')
    torch.save(dict(layout=model.layout, model=model.state_dict(), seed=SEED, stage=stage, updates=updates),pending)
    pending.replace(path)


def load_model(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    norms={k:state['model'][k] for k in ('mean','scale','candidate_mean','candidate_scale')}
    model=TemporalQ(state['layout'],norms)
    model.load_state_dict(state['model'])
    return model.eval()


def fit(folder,small):
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    layout=read(PARENT/'dataset.json')['layout']
    with np.load(PARENT/'normalizers.npz') as norms:
        model=TemporalQ(layout,dict(norms))
    paths=[PARENT/'arrays/train']
    if not small:
        assert (scale.OUT/'FINISHED.json').exists()
        paths.extend(scale.OUT/'parts'/str(d) for d in scale.DOSES)
    train=prepare(folder/'train', paths, model.f, 12 if small else None)
    validation=prepare(folder/'validation',[PARENT/'arrays/validation'],model.f,6 if small else None)
    if not small:
        ref=read(scale.OUT/'fit_1001/models/GA_s17/offline_complete.json')
        assert len(train)==ref['train_boundaries'] and len(validation)==ref['validation_boundaries']
    probes=[np.sort(np.random.default_rng(741).choice(len(s),min(1024,len(s)),replace=False)) for s in (train,validation)]
    result=dict(seed=SEED,history=H, train_rows=len(train),validation_rows=len(validation),
                train_episodes=int(train.a['done'].sum()),validation_episodes=int(validation.a['done'].sum()),
                train_roots=len(np.unique(train.a['root'])),validation_roots=len(np.unique(validation.a['root'])),
                parameters=sum(p.numel() for p in model.parameters()), curves={})
    start=time.monotonic()
    for stage,updates in [('V',256 if small else 16000),('Q',128 if small else 8000)]:
        rng=np.random.default_rng(SEED if stage=='V' else SEED+100000)
        if stage=='V':
            targets=train.a['y']
            valid_target=validation.a['y'][probes[1]]
        else:
            model.eval()
            # Identical one-step objective to GA-Q; next observations are real stored transitions.
            targets=np.empty(len(train),np.float32)
            for l in range(0,len(train),96):
                ix=np.arange(l,min(l+96,len(train)))
                targets[ix]=bootstrap(model,train,ix)
            valid_target=np.concatenate([bootstrap(model,validation,i) for i in np.array_split(probes[1],16)])
            assert np.isfinite(targets).all() and np.array_equal(targets[train.a['done']],train.a['reward'][train.a['done']])
            model=copy.deepcopy(model)
        optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
        points=[]
        losses=[]
        interval=32 if small else 500
        for step in range(updates+1):
            if step%interval==0 or step==updates:
                point=dict(update=step,train=measure(model,train,probes[0],targets[probes[0]],stage),
                           validation=measure(model,validation,probes[1],valid_target,stage))
                if losses:
                    point['minibatch_mse']=float(np.mean(losses))
                    losses=[]
                points.append(point)
                save(folder/'progress.json',dict(stage=stage,seconds=time.monotonic()-start,**point))
                save(folder/(stage+'_curve.json'),points)
                print(stage,step,point['train']['mse'],point['validation']['mse'],flush=True)
            if step==updates:break
            ids=rng.integers(len(train),size=64)
            optimizer.zero_grad(set_to_none=True)
            pred=predict(model,train,ids,stage)
            loss=((pred-torch.tensor(targets[ids]))**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if step==0:
                for branch in (model.experts[0], model.cross.out_proj):
                    assert branch.weight.grad.abs().max()>0
            optimizer.step()
            losses.append(loss.item())
        write_checkpoint(folder/(stage+'.pt'),model,stage,updates)
        result['curves'][stage]=points
    reloaded=load_model(folder/'Q.pt')
    with torch.no_grad():
        ids=probes[0][:32]
        torch.testing.assert_close(predict(reloaded,train,ids,'Q'),predict(model,train,ids,'Q'),rtol=0,atol=0)
    result['seconds']=time.monotonic()-start
    result['parity']=replay_parity(train,paths[0],reloaded)
    save(folder/'fit.json',result)
    return result


@torch.no_grad()
def replay_parity(data,raw_path,model):
    raw=Store(raw_path)
    live=LiveScorer(model)
    f=model.f
    error=0.
    ends=np.flatnonzero(data.a['done'])[:2]
    for j in range(int(ends[-1])+1):
        if data.a['boundary'][j]==0:live.reset()
        ids=np.array([j])
        r=data.a['raw_id'][ids]
        hist,initial,mask=data.window(ids,f.frame)
        x=torch.tensor(raw.arrays['x'][r])
        adv=f.frame+model.layout['advice_width']+2
        reconstructed=torch.cat((initial,hist[:,:,:f.frame].flatten(1),hist[:,-1,f.frame:adv]),dim=1)
        torch.testing.assert_close(reconstructed,x[:,:f.ga],rtol=0,atol=0,equal_nan=True)
        candidates=torch.tensor(data.a['candidates'][ids],dtype=torch.float32)
        sources=torch.tensor(data.a['sources'][ids],dtype=torch.float32)
        expected=model(hist,initial,mask,candidates,sources)
        actual=live.values(x,candidates,sources)
        assert np.array_equal(live.history[-1],hist[0,-1].numpy(),equal_nan=True)
        error=max(error,float((expected-actual).abs().max()))
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)
        live.executed(sources[0,data.a['chosen'][j]])
    return dict(episodes=len(ends),boundaries=int(ends[-1])+1,max_absolute_score_difference=error)


def initialize_worker(checkpoint):
    scale.family.initialize()
    fast_runtime.W.MODELS[checkpoint]=LiveScorer(load_model(checkpoint))


def evaluate_case(job):
    scorer=fast_runtime.W.MODELS[job['checkpoint']]
    scorer.reset()
    return scale.family.worker(job)


def smoke():
    folder=OUT/'smoke'
    result=read(folder/'fit.json') if (folder/'fit.json').exists() else fit(folder,True)
    checkpoint=str(folder/'Q.pt')
    initialize_worker(checkpoint)
    # Development fixtures only. Never open the 600 test fixtures in this entry.
    cases=read(scale.OUT/'anchors.json')[:3]
    rollouts=[]
    for case in cases:
        row=evaluate_case(dict(condition='attention_development_smoke',case=case,checkpoint=checkpoint,
                               epsilon=0.,directory=str(folder/'rollouts'/case['case_id'])))
        rollouts.append({k:row[k] for k in ('case_id','complete','success','decisions','ticks','stop','error')})
    save(folder/'result.json', dict(**result,rollouts=rollouts,checkpoint_sha256=sha(checkpoint)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['smoke','full'])
    if parser.parse_args().stage=='smoke':
        smoke()
    else:
        from .cap_temporal_compare import full
        full()
