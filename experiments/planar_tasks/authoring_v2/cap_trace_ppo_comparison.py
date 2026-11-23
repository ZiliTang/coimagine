"""Matched trace ablation of the complete Q-initialized PPO training pipeline."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import multiprocessing as mp
import shutil

import numpy as np
import torch

from . import cap_trace_value_ppo as ppo
from .cap_trace_value_long import phase, sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.relational_search.contract import digest

BASE = Path('outputs/semantic_rl').resolve()
PARENT = BASE/'cap_trace_value_full_20261118_02'
REFERENCE = BASE/'cap_trace_value_ppo_20261118_01'
SHUFFLE = BASE/'cap_trace_value_shuffle500_20261118_01'
SEALED = BASE/'cap_trace_value_long_20261118_01/panels/confirmation.json'
SEEDS = (17,29,43)
ARMS = ('GA','GLOBAL')


class MatchedPolicy(ppo.Policy):
    def forward(self,x,candidates,sources,valid):
        if self.actor.arm == 'GA':
            width = self.actor.layout['trace_width']
            x = x.clone()
            x[:, -width:] = torch.nan
        return super().forward(x,candidates,sources,valid)


def init_worker(bundle,priority):
    # Original rollout creates models through this class in the worker process.
    ppo.Policy = MatchedPolicy
    ppo.fast_runtime.initialize(bundle,priority)


def origin(arm,seed):
    return PARENT/'models'/f'{arm}_s{seed}'/'online_r4.pt'


def mask_check(model,seed):
    replay=PARENT/'replay'/f'GA_s{seed}'/'r4'
    arrays=[torch.from_numpy(np.load(replay/(k+'.npy'),mmap_mode='r')[::103].copy())
            for k in ('x','candidates','sources','valid')]
    x,c,s,v=arrays
    changed=x.clone()
    width=model.actor.layout['trace_width']
    changed[:,-width:]=123.
    changed[:,-width::2]=torch.nan
    # A nonconstant critic head is necessary to detect potential leakage at initialization.
    with torch.no_grad():
        weight=model.critic[-1].weight.clone()
        model.critic[-1].weight.fill_(.03125)
        old,first=model(x,c,s,v)
        new,second=model(changed,c,s,v)
        assert torch.equal(old.logits,new.logits) and torch.equal(first,second)
        model.critic[-1].weight.copy_(weight)
        greedy=model.actor.values(x,c,s).masked_fill(~v,-torch.inf).argmax(1)
        assert torch.equal(greedy,old.logits.argmax(1))
    return len(x)


def prepare(out):
    out.mkdir(parents=True,exist_ok=False)
    old=read(REFERENCE/'inputs.json')
    ref=read(SHUFFLE/'inputs.json')
    assert read(REFERENCE/'FINISHED.json')['complete']
    assert all(sha(k)==v for k,v in old['source_hashes'].items())
    assert sha(SEALED)==ref['sealed_confirmation_sha256']
    assert sha(REFERENCE/'schedule.json')==old['schedule_sha256']==sha(SHUFFLE/'schedule.json')
    cp_hash=read(PARENT/'ALL_MODELS_FROZEN.json')['files']
    parents={str(origin(a,s)):sha(origin(a,s)) for a in ARMS for s in SEEDS}
    assert all(h==cp_hash[str(Path(f).relative_to(PARENT))] for f,h in parents.items())
    assert parents[str(origin('GLOBAL',17))]==old['checkpoint_sha256']
    panels=read(PARENT/'panels/manifest.json')['files']
    assert all(sha(PARENT/'panels'/f)==h for f,h in panels.items())
    assert all(len(read(REFERENCE/'training'/f'r{r}'/'results.json'))==50 for r in range(5,41))
    schedule=read(REFERENCE/'schedule.json')
    for seed in SEEDS:
        save(out/f'schedule_s{seed}.json',[dict(round=b['round'],cases=[
            dict(case_id=c['case_id'],action_seed=c['action_seed']+(seed-17)*10000)
            for c in b['cases']]) for b in schedule])
    reserved=read(SEALED)['cases']
    known=read(Path(ref['bundle'])/'used_scene_identities.json')
    for name in panels:
        known+=read(PARENT/'panels'/name)['cases']
    assert len(reserved)==len({c['content_sha256'] for c in reserved})==len({c['case_id'] for c in reserved})==200
    for key in ('case_id','content_sha256'):
        assert {c[key] for c in reserved}.isdisjoint(c[key] for c in known)
    for c in reserved:
        assert digest({k:v for k,v in c['fixture'].items() if k!='scene_id'})==c['content_sha256']
    extra=[Path(__file__),Path('tools/cap_trace_ppo_comparison_report.py')]
    for f in extra: shutil.copy(f,out/f.name)
    save(out/'protocol.json',dict(seeds=SEEDS,arms=ARMS,parents=parents,ppo=old,
        sealed_sha=sha(SEALED),sources={str(f):sha(f) for f in extra},
        reuse={str(REFERENCE/f):sha(REFERENCE/f) for f in ('initial.pt','actor_critic_r40.pt','FINISHED.json')},
        primary='GLOBAL minus GA at fixed PPO round40, mean paired success difference across seeds',
        secondary='each arm post minus pre; difference in improvements',
        uncertainty='10000 paired scene bootstrap within object count; seed pairs fixed; show every seed',
        training='same 36x50 schedule, paired action seeds; each arm own same-seed r4 checkpoint; frozen GLOBAL17 reused',
        scope='trace available to entire training pipeline; actor and critic both blinded for GA',
        confirmation_executions=2400,additional_training=9000))
    return ref


def train_arm(out,pool,cases,arm,seed):
    dest=out/f'{arm}_s{seed}'
    dest.mkdir()
    torch.manual_seed(seed)
    model=MatchedPolicy(origin(arm,seed))
    rng=np.random.default_rng(seed)
    initial=dest/'initial.pt'
    torch.save(model.state_dict(),initial)
    if arm=='GA':
        save(dest/'mask_check.json',dict(matched=mask_check(model,seed)))
    optim=torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),dict(params=model.critic.parameters(),lr=3e-4)])
    checkpoint=initial
    history=[]
    for block in read(out/f'schedule_s{seed}.json'):
        r=block['round']
        directory=dest/'training'/f'r{r}'
        phase(out,'collect',arm=arm,seed=seed,round=r)
        rows=ppo.collect(pool,[(cases[item['case_id']],directory/item['case_id'],origin(arm,seed),
            checkpoint,item['action_seed'],True) for item in block['cases']],directory)
        metrics=ppo.fit(model,optim,rows,rng)
        checkpoint=dest/f'actor_critic_r{r}.pt'
        torch.save(model.state_dict(),checkpoint)
        history.append(dict(round=r,success=sum(row['success'] for row in rows),count=len(rows),**metrics))
        save(dest/'history.json',history)
    return initial,checkpoint


def run(out):
    from tools.cap_trace_ppo_comparison_report import report
    ref=prepare(out)
    torch.set_num_threads(4)
    cases={c['case_id']:c for c in read(PARENT/'panels/online_training.json')['cases']}
    checkpoints={}
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=init_worker,
                             initargs=(ref['bundle'],ref['priority_bundle'])) as pool:
        for seed in SEEDS:
            for arm in ARMS:
                if (arm,seed)==('GLOBAL',17):
                    files=(REFERENCE/'initial.pt',REFERENCE/'actor_critic_r40.pt')
                else:
                    files=train_arm(out,pool,cases,arm,seed)
                checkpoints[arm,seed]=files
        frozen={str(f):sha(f) for files in checkpoints.values() for f in files}
        assert len(frozen)==12
        proto=read(out/'protocol.json')
        assert all(sha(f)==h for f,h in {**proto['reuse'],**proto['parents'],**proto['sources'],**proto['ppo']['source_hashes']}.items())
        save(out/'FROZEN.json',frozen)
        assert sha(SEALED)==proto['sealed_sha']
        panel=read(SEALED)['cases']
        for (arm,seed),files in checkpoints.items():
            for stage,cp in zip(('pre','post'),files):
                assert sha(cp)==frozen[str(cp)]
                phase(out,'confirmation',arm=arm,seed=seed,stage_name=stage)
                dest=out/'confirmation'/f'{arm}_s{seed}_{stage}'
                ppo.collect(pool,[(c,dest/c['case_id'],origin(arm,seed),cp,17,False) for c in panel],dest)
    report(out,panel)
    phase(out,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out.resolve())
