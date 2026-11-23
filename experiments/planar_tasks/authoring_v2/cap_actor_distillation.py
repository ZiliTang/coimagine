"""Direct GA actor distillation from a priority teacher or observed successful action sets."""
import argparse
import csv
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from . import cap_retention_supervised as previous
from .cap_trace_value_long import sha, phase
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime, full_worker
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel

PARENT = previous.PARENT
ARCHIVE = previous.BASE/'cap_retention_supervised_20261119_01'
CAPS = previous.CAPS
SEEDS = (17, 29)
ARMS = ('single', 'multi')
WIDTH = previous.WIDTH
STEPS = 6000


class Actor(QModel):
    def __init__(self):
        with np.load(PARENT/'normalizers.npz') as norms:
            super().__init__(dict(width=WIDTH, trace_width=0), norms['mean'][:WIDTH],
                             norms['scale'][:WIDTH], 'GA', norms['candidate_mean'], norms['candidate_scale'])
        self.net[-1] = nn.Identity()

    def values(self, x, candidates, sources):
        return super().values(x[:, :WIDTH], candidates, sources)


def history_keys(path):
    with gzip.open(path, 'rt') as stream:
        record = json.load(stream)
    return [row['history_hash'] for row in record['public_rows'][:-1]]


def set_loss(logits, valid, target):
    return (logits.masked_fill(~valid, -torch.inf).logsumexp(1)
            - logits.masked_fill(~target, -torch.inf).logsumexp(1))


class Prefixes:
    def __init__(self, split_name, bundle, index, split, ranking, out):
        store = Store(PARENT/'arrays'/split_name)
        a = store.arrays
        starts = np.flatnonzero(a['boundary']==0)
        lengths = np.diff(np.r_[starts,len(store)])
        names = split['train']+split['validation']
        tasks = []
        successes = []
        for start, length in zip(starts, lengths):
            root, cap = int(a['root'][start]), int(a['cap'][start])
            record = index[names[root], CAPS[cap]]
            assert record['decisions']==length and record['success']==a['y'][start]
            if record['success']:
                successes.append((int(start),int(length),root,cap))
                tasks.append(bundle/record['path'])
        nodes = {}
        first, roots, counts, on_teacher, initial, keys = [], [], [], [], [], []
        teachers = {}
        for _,_,root,cap in successes:
            teachers[root] = min(teachers.get(root,cap),cap,key=ranking.index)
        with ProcessPoolExecutor(8) as pool:
            for (start,length,root,cap), histories in zip(successes,pool.map(history_keys,tasks,chunksize=8),strict=True):
                assert len(histories)==length
                for boundary,key in enumerate(histories):
                    row = start+boundary
                    action = int(a['chosen'][row])
                    assert a['valid'][row,action] and a['sources'][row,action,cap]
                    node = nodes.setdefault((root,key),len(first))
                    if node == len(first):
                        first.append(row)
                        roots.append(root)
                        keys.append(key)
                        counts.append(np.zeros(7,np.int16))
                        on_teacher.append(-1)
                        initial.append(boundary==0)
                    else:
                        assert initial[node]==(boundary==0)
                        for k in ('x','candidates','sources','valid'):
                            assert np.array_equal(a[k][row],a[k][first[node]],equal_nan=True), (names[root],key,k)
                    counts[node][action] += 1
                    if cap == teachers[root]:
                        assert on_teacher[node]==-1
                        on_teacher[node]=action
        self.counts = np.stack(counts)
        self.target = self.counts>0
        self.single = np.array(on_teacher)
        self.roots = np.array(roots)
        self.initial = np.array(initial)
        self.features = [np.asarray(a['x'][np.asarray(first[i:i+256]),:WIDTH]).copy()
                         for i in range(0,len(first),256)]
        self.features = np.concatenate(self.features)
        self.candidates=a['candidates'][first]
        self.sources=a['sources'][first]
        self.valid=a['valid'][first]
        assert np.all(~self.target | self.valid)
        ids=np.arange(len(first))
        self.by_root = {}
        for arm, eligible in [('single',self.single>=0),('multi',self.target.any(1))]:
            buckets=[ids[eligible & (self.roots==r)] for r in sorted(teachers)]
            self.by_root[arm]=(np.concatenate(buckets),np.cumsum([0]+[len(b) for b in buckets]))
        np.savez_compressed(out/f'{split_name}_prefixes.npz',row=first,root=self.roots,history_hash=keys,
                            successful_votes=self.counts,priority_action=self.single,initial=self.initial)
        save(out/f'{split_name}_inventory.json',dict(roots=len(teachers),success_episodes=len(successes),
             successful_boundaries=int(self.counts.sum()),unique_success_prefixes=len(first),
             priority_boundaries=int((self.single>=0).sum()),multi_action_prefixes=int((self.target.sum(1)>1).sum()),
             multi_action_initial=int(((self.target.sum(1)>1)&self.initial).sum()),
             teachers={names[r]:CAPS[c] for r,c in teachers.items()}))

    def tensors(self, ids):
        return [torch.from_numpy(v[ids]) for v in (self.features,self.candidates,self.sources,self.valid)]


def fit(out, train, arm, seed):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    model=Actor()
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
    nodes, offsets=train.by_root[arm]
    losses=[]
    draws=np.zeros(len(offsets)-1,int)
    for step in range(1,STEPS+1):
        r=rng.integers(len(draws),size=64)
        u=rng.random(64)
        ids=nodes[offsets[r]+(u*(offsets[r+1]-offsets[r])).astype(int)]
        np.add.at(draws,r,1)
        x,c,s,valid=train.tensors(ids)
        if arm=='multi':
            target=torch.from_numpy(train.target[ids])
        else:
            target=nn.functional.one_hot(torch.from_numpy(train.single[ids]),7).bool()
        loss=set_loss(model.values(x,c,s),valid,target).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step%500==0:
            phase(out,'fit',arm=arm,seed=seed,step=step,loss=float(np.mean(losses[-500:])))
    path=out/f'{arm}_s{seed}.pt'
    torch.save(model.state_dict(),path)
    save(path.with_suffix('.json'),dict(root_draws=draws.tolist(),mean_loss_by_100=np.mean(np.reshape(losses,(-1,100)),1).tolist()))
    return model


@torch.inference_mode()
def assess(model, data):
    chunks=[]
    for start in range(0,len(data.roots),256):
        x,c,s,valid=data.tensors(slice(start,start+256))
        chunks.append(model.values(x,c,s).masked_fill(~valid,-torch.inf).log_softmax(1).double().numpy())
    logp=np.concatenate(chunks)
    prob=np.exp(logp)
    choice=prob.argmax(1)
    rows=np.arange(len(choice))
    pset=np.sum(prob*data.target,axis=1)
    chosen_teacher=data.single.clip(min=0)
    is_teacher=data.single>=0
    multi=data.target.sum(1)>1
    def group(mask,target_prob,hit,loss):
        def mean(value):
            return float(np.mean([value[mask & (data.roots==r)].mean() for r in np.unique(data.roots[mask])]))
        return dict(prefixes=int(mask.sum()),roots=int(len(np.unique(data.roots[mask]))),
                    probability=mean(target_prob),top1=mean(hit),nll=mean(loss))
    sets={}
    for name,mask in [('all',rows>=0),('disagree',data.valid.sum(1)>1),('several',multi),
                      ('several_initial',multi&data.initial),('single_known',~multi)]:
        sets[name]=group(mask,pset,data.target[rows,choice],-np.log(pset.clip(min=1e-300)))
    teachers={}
    for name,mask in [('all',is_teacher),('disagree',is_teacher & (data.valid.sum(1)>1))]:
        teachers[name]=group(mask,prob[rows,chosen_teacher],choice==chosen_teacher,-logp[rows,chosen_teacher])
    return dict(priority=teachers,success_set=sets)


def evaluation_job(job):
    path=job['checkpoint']
    if path not in full_worker.MODELS:
        model=Actor()
        model.load_state_dict(torch.load(path,weights_only=True))
        full_worker.MODELS.clear()
        full_worker.MODELS[path]=model.eval()
    try:
        result=full_worker.run(job)
        if not result['complete']:
            raise RuntimeError(result)
        return result
    finally:
        if fast_runtime.ACTIVE is not None:
            fast_runtime.ACTIVE.close_reads()
            fast_runtime.ACTIVE=None


def evaluate(out, panel, ref):
    checkpoints=read(out/'FROZEN.json')
    assert all(sha(out/name)==digest for name,digest in checkpoints.items())
    jobs=[]
    for case in panel:
        for seed in SEEDS:
            for arm in ARMS:
                name=f'{arm}_s{seed}'
                jobs.append(dict(case=case,condition=name,checkpoint=str(out/(name+'.pt')),
                                 directory=str(out/'evaluation'/name/case['case_id'])))
    # Keep model loads amortized but interleave complete scene blocks across conditions.
    jobs.sort(key=lambda j:j['condition'])
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=fast_runtime.initialize,
                             initargs=(ref['bundle'],ref['priority_bundle'])) as pool:
        futures=[pool.submit(evaluation_job,job) for job in jobs]
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n%25==0:
                phase(out,'evaluation',completed=n,total=len(jobs))


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'source.py')
    panel=read(ARCHIVE/'panel.json')['cases']
    ref=read(PARENT/'inputs.json')
    bundle=Path(ref['bundle'])
    split=read(PARENT/'split.json')
    assert len(set(split['train']))==608 and len(set(split['validation']))==152
    assert set(split['train']).isdisjoint(split['validation'])
    dev={r['case_id']:r['content_sha256'] for r in read(bundle/'used_scene_identities.json') if r['role']=='development760'}
    assert {dev[k] for k in split['train']}.isdisjoint(dev[k] for k in split['validation'])
    assert len({r['case_id'] for r in panel})==len({r['content_sha256'] for r in panel})==200
    assert set(dev).isdisjoint(r['case_id'] for r in panel) and set(dev.values()).isdisjoint(r['content_sha256'] for r in panel)
    rows=read(bundle/'index.json')
    index={(r['case_id'],r['cap_id']):r for r in rows}
    totals=[sum(index[r,c]['success'] for r in split['train']) for c in CAPS]
    ranking=sorted(range(7),key=lambda i:(-totals[i],CAPS[i]))
    save(out/'protocol.json',dict(seeds=SEEDS,arms=ARMS,updates=STEPS,batch=64,adam_lr=0.0003,grad_clip=1,
         model='GA candidate scorer with two 64-unit ReLU layers and a linear logit; initialized from scratch',
         sample='same uniform successful training root draws; uniform eligible unique prefixes within root, per arm',
         labels='priority whole successful trajectory vs union of actual successful requested actions at exact same root and history hash',
         primary='multi minus single, greedy closed-loop retention in archived seven-CAP union, paired by scene across two fixed seeds',
         panel='existing viewed 200-scene retention panel; 1800 ticks/300 decisions',panel_sha256=sha(ARCHIVE/'panel.json'),
         source_sha256=sha(Path(__file__)),normalizers_sha256=sha(PARENT/'normalizers.npz'),split_sha256=sha(PARENT/'split.json'),
         archived_reference_sha256=sha(ARCHIVE/'per_scene.csv'),cap_training_success=dict(zip(CAPS,map(int,totals))),ranking=[CAPS[i] for i in ranking]))
    torch.set_num_threads(4)
    train=Prefixes('train',bundle,index,split,ranking,out)
    val=Prefixes('validation',bundle,index,split,ranking,out)
    perf={}
    for seed in SEEDS:
        for arm in ARMS:
            key=f'{arm}_s{seed}'
            model=fit(out,train,arm,seed)
            perf[key]={k:assess(model,d) for k,d in [('train',train),('validation',val)]}
            save(out/'fit_metrics.json',perf)
        assert read(out/f'single_s{seed}.json')['root_draws']==read(out/f'multi_s{seed}.json')['root_draws']
    del train,val
    save(out/'FROZEN.json',{p.name:sha(p) for p in sorted(out.glob('*.pt'))})
    evaluate(out,panel,ref)
    phase(out,'rollouts_complete',completed=800)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.out.resolve())
