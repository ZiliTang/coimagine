"""Matched supervised representation arms, isolated from historical drivers."""
import argparse
from pathlib import Path
import time
import numpy as np
import torch
from . import da_joint_study as D
from . import random_room_coverage_v2 as V
from experiments.planar_tasks.semantic_rl.joint_evidence import JointEvidenceFeatures,WIDTH,MODES
from experiments.planar_tasks.semantic_rl.joint_learning import train

BASE=D.BASE;HOST=D.HOST;CACHE={};SEEDS=(17,29);UPDATES=28000


def features(mode='joint'):
    return JointEvidenceFeatures(V.entries(),V.O.protocol(V.O.CFG),mode)


def extract_worker(job):
    torch.set_num_threads(1);p=Path(job['path']);p.mkdir(parents=True,exist_ok=True)
    if (p/'complete.json').exists():return V.read(p/'complete.json')
    if V.sha(job['result_path'])!=job['sha256']:raise ValueError('source trajectory changed')
    if 'extract' not in CACHE:CACHE['extract']=features()
    f=CACHE['extract'];result=V.unpack(job['result_path'])
    if not result['summary']['success']:raise ValueError('failed positive trajectory')
    contexts=[];xs=[];offsets=[];targets=[];meta=[];cursor=0;started=time.monotonic()
    for tr in V.O.extracted_rows(result,f):
        if tr['boundary']<job.get('start_boundary',0):continue
        panels=tr['panels'];contexts.append(panels[0]['attribution_context']);off=[cursor]
        for panel in panels:
            xs.append(panel['attribution_candidates']);cursor+=len(panel['choices']);off.append(cursor)
        offsets.append(off);targets.append(tr['indices']);meta.append(dict(boundary=tr['boundary'],history_hash=tr['history_hash']))
    if not contexts:raise ValueError('empty supervised suffix')
    for k,a in dict(contexts=np.array(contexts,dtype='float32'),candidates=np.concatenate(xs).astype('float32'),
                    offsets=np.array(offsets),targets=np.array(targets)).items():np.save(p/(k+'.npy'),a)
    V.save(p/'metadata.json',meta);V.save(p/'layout.json',f.layout)
    row=dict(task=job['task'],path=str(p),teacher=job['teacher'],result_path=job['result_path'],
        result_sha256=job['sha256'],start_boundary=job.get('start_boundary',0),boundaries=len(contexts),
        candidates=cursor,source_sha=f.sha256,action_protocol=f.action_protocol,
        context_width=len(contexts[0]),candidate_width=xs[0].shape[1],wall_seconds=time.monotonic()-started)
    V.save(p/'complete.json',row);return row


def extract_base():
    out=BASE/'data_base'
    if (out/'extraction.json').exists():return
    old=V.read(V.BASE/'data_expanded/extraction.json')['rows']
    initial={r['task']:r for r in V.read(V.O.PARENT/'demonstrations.json')}
    jobs=[]
    for r in old:
        info=r if 'result_path' in r else initial[r['task']]
        path=info['result_path']
        jobs.append(dict(task=r['task'],teacher=info['teacher'],result_path=path,
            sha256=V.sha(path),path=str(out/'shards'/r['task'])))
    V.save(out/'jobs.json',jobs);V.HOST=HOST
    rows=V.bounded(jobs,extract_worker,workers=4,stage='extract_joint_operands')
    V.combine(out,sorted(rows,key=lambda r:r['task']))


def extract_repair():
    out=BASE/'data_repair'
    if (out/'extraction.json').exists():return
    states=[s for s in V.read(BASE/'selection.json')['states'] if s['split']=='repair_train']
    rows=V.read(BASE/'branches/result.json')['rows'];jobs=[];selection=[]
    for s in states:
        wins=[r for r in rows if r['job']['case_id']==s['case_id'] and r['success']]
        if not wins:continue
        best=min(wins,key=lambda r:(r['ticks'],r['job']['kind']))
        # Every successful continuation is demonstrated as a full suffix, not
        # falsely asserted to be a good one-step action under another receiver.
        jobs.append(dict(task=s['case_id']+'__repair',teacher=best['job']['condition'],
            result_path=best['result_path'],sha256=V.sha(best['result_path']),start_boundary=s['cut'],
            path=str(out/'shards'/s['case_id'])))
        selection.append(dict(case_id=s['case_id'],kind=best['job']['kind'],ticks=best['ticks'],
            successes=[r['job']['kind'] for r in wins],supervision='fastest successful coherent suffix; no failed-action negatives'))
    V.save(out/'selection.json',selection);V.save(out/'jobs.json',jobs);V.HOST=HOST
    new=V.bounded(jobs,extract_worker,workers=4,stage='extract_repair_suffixes') if jobs else []
    V.combine(out,V.read(BASE/'data_base/extraction.json')['rows']+sorted(new,key=lambda r:r['task']))


class ModeCandidates:
    """One immutable joint cache; arm-specific masks applied per minibatch."""
    def __init__(self,array,mode,members):
        self.array=array;self.shape=array.shape;self.mode=mode;self.start=self.shape[1]-members*WIDTH
    def __getitem__(self,key):
        x=np.array(self.array[key],copy=True)
        if self.mode=='current':x[...,self.start:]=0
        elif self.mode=='enriched':
            for begin in range(self.start,self.shape[1],WIDTH):x[...,begin+46:begin+WIDTH]=0
        return x


def dataset(tier,mode):
    p=BASE/('data_'+tier)
    ds={k:np.load(p/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')}
    layout=V.read(p/'layout.json');layout['evidence_mode']=mode
    ds['candidates']=ModeCandidates(ds['candidates'],mode,len(layout['members']))
    f=features(mode);sha=f.sha256;f.close()
    return ds,layout,sha,V.read(p/'extraction.json')


def fit_worker(job):
    tier,mode,seed=job['tier'],job['mode'],job['seed'];name=f'{tier}_{mode}_s{seed}'
    p=BASE/'fit'/name;p.mkdir(parents=True,exist_ok=True)
    if (p/'complete.json').exists():return V.read(p/'complete.json')
    ds,layout,sha,ex=dataset(tier,mode);ids=list(range(len(ds['contexts'])));curve=[]
    def progress(r):
        curve.append(r);V.save(p/'progress.json',r)
        if r['update']%2000==0:V.save(p/'curve.json',curve);print('FIT',name,r,flush=True)
    model,curve=train(ds,ids,layout,'GDA',sha,ex['action_protocol'],updates=UPDATES,
        seed=seed,wall_seconds=7200,progress=progress)
    acc=V.I.accuracy(model,ds,ids);V.save(p/'accuracy.json',acc);V.save(p/'curve.json',curve)
    parameters=sum(p.numel() for p in model.parameters())
    meta=dict(**job,updates=UPDATES,demonstrations=ex['demonstrations'],parameters=parameters,
        data_sha=V.sha(BASE/('data_'+tier)/'extraction.json'))
    model.save(p/'model.pt',meta);loaded=V.I.Scorer.load(p/'model.pt')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    result=dict(name=name,**meta,accuracy=acc);V.save(p/'complete.json',result);return result


def fit(tier):
    V.HOST=HOST
    jobs=[dict(tier=tier,mode=mode,seed=seed) for seed in SEEDS for mode in MODES]
    rows=V.bounded(jobs,fit_worker,workers=3,stage='fit_'+tier)
    V.save(BASE/'fit'/f'{tier}_result.json',dict(rows=rows))


def names():
    return [f'{tier}_{mode}_s{seed}' for tier in ('base','repair') for seed in SEEDS for mode in MODES]


def model_runtime(name):
    if CACHE.get('name')==name:return CACHE['feature'],CACHE['model']
    if 'feature' in CACHE:CACHE['feature'].close()
    mode=name.split('_')[1];model=V.I.Scorer.load(BASE/'fit'/name/'model.pt');f=features(mode)
    CACHE.update(name=name,feature=f,model=model)
    return f,model


def eval_worker(job):
    D.runtime=model_runtime
    return D.worker(job)


def evaluate():
    # All fixed versions are sealed before ANY fresh outcome can be generated.
    hashes={n:V.sha(BASE/'fit'/n/'model.pt') for n in names()}
    seal=BASE/'fit/seal.json'
    if seal.exists() and V.read(seal)!=hashes:raise ValueError('evaluation models changed')
    V.save(seal,hashes)
    inputs=V.read(BASE/'inputs.json');states=V.read(BASE/'selection.json')['states']
    case_map={r['id']:r for r in inputs['cases']}
    regression=[dict(r,split='development_regression') for n in range(2,6)
                for r in [x for x in inputs['cases'] if x['movable_count']==n][:6]]
    jobs=[]
    for name in names():
        for c in inputs['fresh']+regression:
            jobs.append(dict(condition=name,case_id=c['id'],fixture=c['fixture'],split=c['split'],
                evaluation='full',path=str(BASE/'evaluation/runs'/name/'full'/c['id'])))
        for s in states:
            jobs.append(dict(condition=name,case_id=s['case_id'],fixture=case_map[s['case_id']]['fixture'],
                split=s['split'],evaluation='handoff',cut=s['cut'],reference=s['reference'],
                reference_sha256=s['reference_sha256'],path=str(BASE/'evaluation/runs'/name/'handoff'/s['case_id'])))
    V.save(BASE/'evaluation/jobs.json',jobs);V.HOST=HOST
    rows=V.bounded(jobs,eval_worker,workers=16,stage='fixed_model_evaluation')
    V.save(BASE/'evaluation/result.json',dict(rows=rows))
    issues=[r for r in rows if r['stop']=='wall_budget' or r.get('error')]
    if issues:raise RuntimeError(f'evaluation infrastructure issues: {len(issues)}')
    return rows


def summarize():
    rows=V.read(BASE/'evaluation/result.json')['rows'];branches=V.read(BASE/'branches/result.json')['rows']
    states=V.read(BASE/'selection.json')['states'];fits={n:V.read(BASE/'fit'/n/'complete.json') for n in names()}
    metrics={}
    for name in names():
        metrics[name]={}
        for split in ('joint_fresh','development_regression','repair_train','handoff_holdout'):
            rr=[r for r in rows if r['job']['condition']==name and r['job']['split']==split]
            metrics[name][split]=dict(success=sum(r['success'] for r in rr),n=len(rr),
                successful_cases=[r['job']['case_id'] for r in rr if r['success']])
    diagnosis=[]
    for state in states:
        rr=[r for r in branches if r['job']['case_id']==state['case_id']]
        identity=next(r for r in rr if r['job']['kind']=='identity')
        actions=[r for r in rr if r['job']['kind'].startswith('action')]
        caps=[r for r in rr if r['job']['kind'].startswith('cap')]
        diagnosis.append(dict(case_id=state['case_id'],split=state['split'],cut=state['cut'],
            original_success=identity['success'],action_rescue=not identity['success'] and any(r['success'] for r in actions),
            cap_rescue=not identity['success'] and any(r['success'] for r in caps),
            variants={r['job']['kind']:dict(success=r['success'],ticks=r['ticks']) for r in rr}))
    paired={};rng=np.random.default_rng(1781)
    for tier in ('base','repair'):
        for alternative in ('enriched','joint'):
            ids=sorted(r['id'] for r in V.read(BASE/'inputs.json')['fresh'])
            diffs=[]
            for case in ids:
                values=[]
                for seed in SEEDS:
                    a=case in metrics[f'{tier}_{alternative}_s{seed}']['joint_fresh']['successful_cases']
                    b=case in metrics[f'{tier}_current_s{seed}']['joint_fresh']['successful_cases']
                    values.append(float(a)-float(b))
                diffs.append(np.mean(values))
            d=np.asarray(diffs);boot=np.mean(d[rng.integers(0,len(d),(5000,len(d)))],axis=1)
            paired[f'{tier}_{alternative}-current']=dict(mean_difference=float(d.mean()),
                paired_case_bootstrap95=np.quantile(boot,[.025,.975]).tolist(),
                note='seeds averaged within each case before case-resampling; conditional on these2 fits')
    donors=V.read(BASE/'donors/result.json')['rows']
    result=dict(status='complete',metrics=metrics,diagnosis=diagnosis,fits=fits,paired=paired,
        physical_episodes=len(rows)+len(branches)+len(donors),
        physical_ticks=sum(r['ticks'] for r in rows+branches+donors),
        invalid_first_attempt_reserved_episodes=16,invalid_first_attempt_tick_upper_bound=28800,
        scope='candidate-parameter/condition binding, no new predicates or world model',
        repair_selection=V.read(BASE/'data_repair/selection.json'))
    V.save(HOST/'RESULT.json',result)
    lines=['# D/A联合表示与真实接管：实验结果','',
        '固定10 CAP、完整合法动作、原270示范与duration协议。三表示×两数据×两训练seed；结果未用于选择checkpoint。',
        '', '| 数据 | 表示 | seed | 新80例 | 开发回归 | 训练接管 | 留出接管 | 示范拟合 |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in names():
        tier,mode,seed=name.split('_');m=metrics[name];a=fits[name]['accuracy']
        counts=['%d/%d'%(m[k]['success'],m[k]['n']) for k in ('joint_fresh','development_regression','repair_train','handoff_holdout')]
        lines.append('| '+' | '.join([tier,mode,seed,*counts,f"{a['exact_actions']}/{a['boundaries']}"])+' |')
    lines += ['', '## 解释边界','',
        '- enriched与joint共享原始操作数；joint额外提供32个条件×建议兼容性乘积/专家。current对应新增槽全零。三者参数总数相同，活跃输入及归纳偏置不同。',
        '- 修复训练取每训练根最快成功完整后缀；保留多解诊断，不把未测试/失败动作作为负样本。本轮尚未加入显式多正例排序损失。',
        '- 接管分叉使用确定性真实前缀重放，核对公开观测及物理帧；每状态identity核对全程一致。',
        '- 初次分叉发现原LibraryFeatures强制前缀未维护几何历史。首次attempt作废保留，上界16执行/28800 tick单独计。修复仅用于本轮适配器，旧产物不变。',
        '- 两训练seed共享场景；置信区间按场景配对，并不覆盖全部训练随机性。80例首次用于固定最终模型，开发回归不是泛化证据。',
        '', f'完整逐场景数据：`{HOST}/RESULT.json`。']
    path=V.ROOT/'docs/agent_exchange/reviews/DA_Joint_Representation_20261117.md'
    path.write_text('\n'.join(lines)+'\n')
    V.save(HOST/'progress.json',dict(stage='complete',episodes=result['physical_episodes']))


def preflight():
    """Frozen operands match cached original features; reject accidental ABI drift."""
    original=V.read(V.BASE/'data_expanded/extraction.json')['rows']
    extracted=V.read(BASE/'data_base/extraction.json')['rows']
    old={r['task']:r for r in original};checks=[]
    for row in extracted:
        p=Path(row['path']);q=Path(old[row['task']]['path'])
        for key in ('contexts','offsets','targets'):
            np.testing.assert_array_equal(np.load(p/(key+'.npy'),mmap_mode='r'),np.load(q/(key+'.npy'),mmap_mode='r'))
        np.testing.assert_array_equal(np.load(p/'candidates.npy',mmap_mode='r')[:,:251],np.load(q/'candidates.npy',mmap_mode='r'))
        checks.append(row['task'])
    V.save(BASE/'data_base/parity.json',dict(original_channels_exact=True,trajectories=len(checks)))


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract_base','extract_repair','fit_base','fit_repair','evaluate','summarize','preflight'])
    a=p.parse_args();D.verify_inputs()
    if a.stage.startswith('fit_'):fit(a.stage[4:])
    else:globals()[a.stage]()


if __name__=='__main__':main()
