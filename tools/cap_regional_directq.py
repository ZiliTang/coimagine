"""Offline regional GA-Q from verified public decision histories and frozen proposals."""
import csv
import gzip
import hashlib
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_five as study
from tools.cap_regional_evaluate import paired
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl.trusted_native import owned_copy
from experiments.planar_tasks.semantic_rl.memory_flow import current_event
from experiments.planar_tasks.semantic_rl.relational_transfer_features import RelativeGeometry, action_encoding
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime, full_data, full_learner
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import flatten
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.instrumentation import load_native
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import load_checkpoint

OUT=study.OUT/'directq'
DOCS=study.ROOT/'docs/research/key_findings/2026-11-19_regional_directq'
REG=study.OUT/'regional'
CKPT=OUT/'models/GA_s17/offline.pt'
read,save=study.read,study.save
MODULES=[]
MODEL=None


def progress(stage, **details):
    save(OUT/'progress.json',dict(stage=stage,time=time.time(),**details))


class Proposals:
    def __init__(self):
        self.geometry=RelativeGeometry(8)
        self.memory=[mod.initial_memory() for mod in MODULES]

    def panel(self,history):
        event=current_event(history)
        obs=event['observation']
        self.geometry.advance(history)
        event=dict(event,executed_ticks=None if history.previous_tick is None else obs['tick']-history.previous_tick)
        pool=[];sources=[];advice=[];proposals=[]
        for i,mod in enumerate(MODULES):
            self.memory[i]=mod.update_memory(self.memory[i],owned_copy(event))
            context=mod.prepare_context(owned_copy(self.memory[i]),owned_copy(obs))
            action=context['proposals'][0] if context['proposals'] else None
            proposals.append(action)
            encoded=np.full(86,np.nan,dtype=np.float32)
            encoded[-1]=0
            if action is not None:
                try:validate_action(action,obs)
                except (ValueError,KeyError,TypeError):pass
                else:
                    encoded[:69]=action_encoding(action,self.geometry.goal_id,self.geometry.theta)
                    encoded[69:85]=self.geometry.candidate_geometry(action)
                    encoded[-1]=1
                    if action not in pool:
                        pool.append(action);sources.append([])
                    sources[pool.index(action)].append(i)
            advice.append(encoded)
        if not pool:
            action=dict(target_id=None,a0=44,a1=44,ticks=1)
            validate_action(action,obs)
            pool=[action];sources=[[]]
        packet=dict(g=self.geometry.vector,advice=np.stack(advice),pool=pool,sources=sources,
                    trace=[],budget=[(1800-obs['tick'])/1800,(300-obs['decision'])/300])
        return flatten(packet),*full_data.candidate_arrays(packet,self),pool,proposals


def episode(job):
    cid,fixture,sid,own,base,mode=job
    directory=OUT/('shards' if mode=='replay' else 'evaluation')/(sid if mode=='replay' else mode)
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(cid+'.json')
    if path.exists():return read(path)
    if mode=='replay':
        reference=read(base/(cid+'.json'))
        with gzip.open(base/(cid+'.json.gz'),'rt') as f: original=json.load(f)
    env=fast_runtime.ScopedEnvironment(fixture,study.CFG,fast_feedback=True,record_details=False,
                                       history_factory=fast_runtime.AuditedCachedHistory)
    h=env.history; lib=Proposals(); trace=[]
    values={k:[] for k in ('x','candidates','sources','valid','chosen','duration')}
    try:
        for i in range(300):
            x,c,s,valid,pool,proposals=lib.panel(h)
            if mode=='replay':
                action=original[i]['a']
                if proposals[own]!=action or action not in pool:raise ValueError(f'{sid}/{cid}/{i} action mismatch')
                chosen=pool.index(action)
            else:
                with torch.inference_mode():
                    scores=MODEL.values(torch.from_numpy(x[None]),torch.from_numpy(c[None]),torch.from_numpy(s[None]))[0].numpy()
                chosen=int(np.argmax(np.where(valid,scores,-np.inf)))
                if not np.isfinite(scores[chosen]):raise ValueError('non-finite action score')
                action=pool[chosen]
                trace.append(dict(chosen=chosen,sources=s[chosen].nonzero()[0].tolist(),action=action,score=float(scores[chosen])))
            ev=env.step(action,remaining_ticks=1800-env.env.total_periods)
            if mode=='replay':
                if original[i]['tick']!=ev['end_tick']:raise ValueError(f'{sid}/{cid}/{i} tick mismatch')
                for k,v in zip(values,(x,c,s,valid,chosen,ev['periods'])):values[k].append(v)
            if env.env.termination is not None or env.env.total_periods>=1800:break
        n=i+1;success=bool(env.env.success)
        row=dict(case_id=cid,cap_id=sid,boundaries=n,success=success,ticks=env.env.total_periods,
                 termination=env.env.termination,history=h.sha256)
        if mode=='replay':
            assert reference['complete'] and (n,success,h.sha256)==(reference['decisions'],reference['success'],reference['history']),row
            arrays={k:np.asarray(v) for k,v in values.items()}
            arrays.update(y=np.full(n,float(success),np.float32),cap=np.full(n,own,np.int64),
                          root=np.full(n,int(cid.split('_')[-2])*100+int(cid.split('_')[-1]),np.int64),
                          boundary=np.arange(n,dtype=np.int32),reward=np.zeros(n,np.float32),done=np.zeros(n,np.bool_))
            arrays['reward'][-1]=float(success);arrays['done'][-1]=True
            temp=path.with_suffix('.pending.npz')
            np.savez_compressed(temp,**arrays)
            temp.replace(path.with_suffix('.npz'))
            row.update(path=str(path.with_suffix('.npz')),layout=dict(width=len(x),geometry_width=len(lib.geometry.vector),
                       trace_width=0,trace_widths=[0]*len(MODULES),cap_count=len(MODULES),advice_width=86*len(MODULES)))
        else:
            with gzip.open(path.with_suffix('.json.gz'),'wt',compresslevel=1) as f:json.dump(trace,f)
        save(path,row);return row
    finally:
        env.close_reads();fast_runtime.ACTIVE=None


def batch(jobs,workers,stage):
    done=[];began=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(episode,j) for j in jobs]
        try:
            for f in as_completed(futures):
                done.append(f.result())
                if len(done)%5==0 or len(done)==len(jobs):
                    progress(stage,completed=len(done),total=len(jobs),seconds=time.monotonic()-began)
        except BaseException:
            for f in futures:f.cancel()
            raise
    return sorted(done,key=lambda r:(r['case_id'],r['cap_id']))


def report(panels,sweeps,best):
    DOCS.mkdir(parents=True,exist_ok=True)
    metrics={}; rows=[]
    for panel,fixtures in panels.items():
        records=[]
        for cid,fixture in sorted(fixtures.items()):
            if panel=='dev':
                wins={rev['sid']:int(cid in rev['wins']) for rev in sweeps}
            else:
                wins={r['sid']:int(read(study.OUT/'evaluation'/('regional_'+r['sid'])/(cid+'.json'))['success']) for r in sweeps}
            q=read(OUT/'evaluation'/panel/(cid+'.json'))
            records.append(dict(panel=panel,case_id=cid,objects=1+len(fixture['movable_objects']),**wins,
                                best=wins[best],union=max(wins.values()),directq=int(q['success'])))
        rows.extend(records)
        vec=lambda k:np.array([r[k] for r in records],dtype=int)
        diff=paired(vec('best'),vec('directq'),vec('objects'))
        metrics[panel]=dict(**diff,union=int(vec('union').sum()),union_retained=int((vec('union')*vec('directq')).sum()),
                            outside_union=int(((1-vec('union'))*vec('directq')).sum()))
    with (DOCS/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    outputs=dict(metrics=metrics,fit=read(CKPT.parent/'offline_complete.json'),protocol=read(OUT/'protocol.json'),
                 model_sha256=hashlib.sha256(CKPT.read_bytes()).hexdigest())
    save(DOCS/'results.json',outputs)
    lines=['# 区域精化库的离线 directQ','',
           '按开发集选定单 CAP '+best+'。共享 64×64 GA-Q 在五个成员的开发轨迹上学习，seed17。每个决策边界只在当时的合法提案中选择一个动作，同动作合并提案来源。','',
           '| 场景 | 单 CAP | directQ | 救回 / 丢失 | 净变化（百分点，配对 95% 区间） | 库并集 | 并集内保留 / 并集外新增 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name,label in [('dev','开发（160）'),('fresh','独立新房间（200）'),('near','开发房间扰动（160）')]:
        m=metrics[name]
        lines.append(f"| {label} | {m['first']} | {m['second']} | {m['rescued']} / {m['lost']} | {m['pp']:+.1f} [{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] | {m['union']} | {m['union_retained']} / {m['outside_union']} |")
    lines+=['','同一批 160 个房间的 800 条成功及失败开发轨迹逐例重放，核对每步提案、结束 tick 及最终公开历史哈希。V 按决策边界均匀抽样，回归输出该轨迹的终局成败。Q 回归真实后继最高合法专家 V，终局用实际成功标签，折扣为 1。V 与 Q 各训练 16,000 和 8,000 步，batch 64，Adam 学习率 0.0003。归一化只用开发数据。两个留出面板都在训练前冻结，固定更新步数。区间按物体数分层、同场景配对重抽 10,000 次，条件于这一套作者程序与一个训练种子。开发成绩属于样本内。','',
            '[逐场景结果](per_scene.csv) · [参数、训练规模及统计](results.json)']
    (DOCS/'README.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'FINISHED.json',outputs)
    progress('finished',completed=len(rows),total=len(rows))


def main():
    global MODULES,MODEL
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    study_state=read(REG/'study.json');sweeps=study_state['sweeps']
    best=max(sweeps,key=lambda r:len(r['wins']))['sid']
    frozen=read(study.OUT/'frozen.json')
    config=dict(seed=17,cap_order=study_state['final'],sources=frozen['policies'],best=best,train_roots=160,
                train_episodes=800,v_updates=16000,q_updates=8000,batch=64,learning_rate=.0003,
                reward='terminal_success',continuation='max_legal_frozen_V',gamma=1,trace=False,
                panels=dict(dev=160,fresh=200,near=160),held_sha256=frozen['panel'])
    if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==config
    else:save(OUT/'protocol.json',config)
    if (OUT/'FINISHED.json').exists():return
    progress('prepare',completed=0,total=800)
    for sid in config['cap_order']:
        p=REG/'semantics'/(sid+'.py')
        assert hashlib.sha256(p.read_bytes()).hexdigest()==config['sources']['regional_'+sid]
        MODULES.append(load_native(p.read_text())[0])
    dev=read(study.WS/'regional/development_fixtures.json')
    assert study.digest(dev)==read(study.OUT/'protocol.json')['dev_sha']
    jobs=[(cid,f,r['sid'],i,REG/'logs'/r['key'],'replay') for cid,f in sorted(dev.items()) for i,r in enumerate(sweeps)]
    shards=batch(jobs,10,'replay')
    progress('assemble',completed=len(shards),total=800)
    if not (OUT/'arrays/train/manifest.json').exists():full_data.assemble(OUT/'arrays/train',shards)
    save(OUT/'dataset.json',dict(layout=shards[0]['layout'],train_episodes=len(shards)))
    progress('fit')
    full_learner.train_offline(OUT,'GA',17,evaluate_validation=False)
    MODEL=load_checkpoint(CKPT)[0]
    assert hashlib.sha256((study.OUT/'held.json').read_bytes()).hexdigest()==config['held_sha256']
    held=read(study.OUT/'held.json')
    panels={'dev':dev,**{p:{cid:f for cid,f in held.items() if cid.startswith(p+'_')} for p in ('fresh','near')}}
    for panel,fixtures in panels.items():
        jobs=[(cid,f,'directq',None,None,panel) for cid,f in sorted(fixtures.items())]
        batch(jobs,14,'evaluate_'+panel)
    progress('report')
    report(panels,sweeps,best)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        progress('error',error=repr(exc));raise
