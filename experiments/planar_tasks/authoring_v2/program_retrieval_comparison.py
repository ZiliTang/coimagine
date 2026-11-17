"""R0/Rt vs frozen GDA on the already inspected100-case diagnostic panel.

Uses exactly the270 full demonstrations used by expanded_GDA. No new teachers,
training, tuning, or access to the running scaling study's200 fresh test cases.
"""
import collections
import json
from pathlib import Path
import time
import traceback

import numpy as np

from . import random_room_coverage_v2 as V
from .program_retrieval_baselines import execute
from experiments.planar_tasks.semantic_rl.program_retrieval import GeometryEncoder, RetrievalIndex, sha
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory, VERSION as STREAM_VERSION
from experiments.planar_tasks.relational_search.contract import History

BASE=V.ROOT/'outputs/semantic_rl/program_retrieval_comparison_20261117_01'
HOST=V.ROOT/'outputs/semantic_rl_authors/program_retrieval_comparison_20261117_01'
PARENT=BASE/'parent'
DATA=V.BASE/'data_expanded'
CONDITIONS={'R0':'initial','Rt':'boundary'}


def make(path):
    if (path/'campaign.json').exists():
        c=V.Campaign(path);c.verify_sources();return c
    return V.Campaign.create(path,fixture=V.read(V.PARENT/'fresh_test.json')['cases'][0]['fixture'],
        sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)],group='program-retrieval-comparison-v1',
        limits=dict(attempts=200,resets=200,execution_ticks=360000,fits=0,rounds=0,author_turns=0))


def demo_manifest():
    old=V.read(V.O.PARENT/'demonstrations.json')
    new=V.read(DATA/'selected.json')
    demos=[dict(task=d['task'],teacher=d['teacher'],result_path=d['result_path'],sha256=d['sha256']) for d in old+new]
    extraction=V.read(DATA/'extraction.json')
    assert len(demos)==extraction['demonstrations']==270
    assert [d['task'] for d in demos]==[r['task'] for r in extraction['rows']]
    assert len({d['task'] for d in demos})==270
    return demos


def prepare():
    if V.frozen(PARENT):return
    refs=[DATA,V.O.PARENT,V.BASE/'evaluation']
    seals={str(p):V.Campaign(p).verify_frozen() for p in refs}
    demos=demo_manifest()
    cases=V.read(V.PARENT/'fresh_test.json')['cases']
    assert len(cases)==100 and not ({d['task'] for d in demos}&{c['id'] for c in cases})
    c=make(PARENT)
    V.save(PARENT/'demonstrations.json',demos)
    V.save(PARENT/'cases.json',dict(cases=cases,note='already inspected100-case fixed diagnostic; not new blind test'))
    V.save(PARENT/'protocol.json',dict(version='program-retrieval-comparison-v1',inputs=seals,
        demonstrations=270,boundaries=9292,conditions=CONDITIONS,k=1,representation='snapshot',
        R0='one initial point per selected successful demonstration; only witnessed teacher gets positive label; others unknown',
        Rt='all9292 action-before boundaries from same270 demonstrations; no terminal states',
        provenance='no extra outcome tables; identical positive trajectories to expanded_GDA',
        workers=4,max_ticks=1800,max_decisions=300,rollout_wall_seconds=1800,
        max_episodes=200,max_total_ticks=360000,whole_study_wall_seconds=10800,
        untouched_fresh_test='random_room_gda_scaling_20261117_03 fresh200 excluded',
        interpretation='snapshot retrieval vs historical/program-informed GDA: performance comparison, not isolated representation/architecture attribution',
        latency='concurrent GDA collection/training; trace timings descriptive, not controlled speed benchmark'))
    V.finish(c)


def build_indexes():
    p=BASE/'indexes'
    if V.frozen(p):return
    c=make(p);entries=V.entries();names=[e['name'] for e in entries]
    sources={e['name']:V.digest(e['source']) for e in entries}
    demos=V.read(PARENT/'demonstrations.json')
    vectors=[];labels=[];records=[];initial=[];initial_labels=[];initial_records=[]
    started=time.monotonic()
    for j,d in enumerate(demos):
        path=Path(d['result_path'])
        if sha(path)!=d['sha256']:raise ValueError('demonstration file changed')
        result=V.unpack(path)
        assert result['summary']['success']
        assert result['source_sha256']==sources[d['teacher']]
        assert result['action_protocol']==V.O.protocol(V.O.CFG)
        h=StreamHistory();verifier=StreamHistory() if result.get('history_protocol')==STREAM_VERSION else History()
        encoder=GeometryEncoder('snapshot')
        y=np.full(len(names),np.nan);y[names.index(d['teacher'])]=1.
        for i,row in enumerate(result['public_rows'][:-1]):
            h.append(row['observation'],row['action']);verifier.append(row['observation'],row['action'])
            if verifier.sha256!=row['history_hash']:raise ValueError('corrupt source history')
            x=encoder.advance(h)
            record=dict(task=d['task'],cap=d['teacher'],boundary=i,history_hash=h.sha256,result_path=str(path))
            if i==0:
                initial.append(x);initial_labels.append(y);initial_records.append(record)
            vectors.append(x);labels.append(y);records.append(record)
        if (j+1)%10==0:
            V.update(stage='build_same_demo_indexes',completed=j+1,total=270)
            print('INDEX',j+1,'/270',flush=True)
    assert len(vectors)==V.read(DATA/'extraction.json')['boundaries']==9292
    metadata=dict(sources=sources,action_protocol=V.O.protocol(V.O.CFG),case_ids=[d['task'] for d in demos],
        data_seal=sha(DATA/'manifest.json'),representation='snapshot',
        selection='exact same270 positive trajectories as expanded_GDA; complete boundary coverage, no extra outcomes',
        evidence_limit='successful own continuation does not establish arbitrary takeover success')
    for mode,x,y,r,evidence in [('initial',initial,initial_labels,initial_records,'initial_outcomes'),
                               ('boundary',vectors,labels,records,'successful_prefix')]:
        index=RetrievalIndex(np.array(x),np.array(y),names,dict(metadata,records=r),evidence=evidence,k=1)
        index.save(p/mode)
    V.save(p/'BUILD.json',dict(trajectories=270,initial_points=len(initial),boundary_points=len(vectors),
        elapsed_seconds=time.monotonic()-started,new_ticks=0,new_llm_calls=0))
    V.finish(c)


def worker(job):
    path=Path(job['out'])
    if (path/'summary.json').exists():return V.read(path/'summary.json')
    if path.exists():raise ValueError('incomplete previous rollout; explicit recovery required')
    row=execute(BASE/'indexes'/job['mode'],job['mode'],job['case'],path,wall_seconds=1800)
    row.update(condition=job['condition'],split='fixed_diagnostic',movable_count=job['case']['movable_count'])
    V.save(path/'summary.json',row)
    return row


def evaluate():
    p=BASE/'evaluation'
    if V.frozen(p):return
    c=make(p);cases=V.read(PARENT/'cases.json')['cases']
    jobs=[dict(condition=name,mode=mode,case=case,out=str(p/'runs'/name/case['id'])) for name,mode in CONDITIONS.items() for case in cases]
    V.save(p/'jobs.json',jobs)
    rows=V.bounded(jobs,worker,workers=4,stage='fixed_retrieval_diagnostic')
    V.save(p/'result.json',dict(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows)))
    infra=[r for r in rows if r['stop']=='wall_budget' or r.get('error') and r['error'].get('message')!='CAP first recommendation missing/illegal']
    V.save(p/'execution_audit.json',dict(infrastructure_errors=infra))
    if infra:raise RuntimeError('infrastructure failure; preserve and inspect')
    assert len(rows)==200 and sum(r['ticks'] for r in rows)<=360000
    with c.locked():
        c.state['used'].update(attempts=len(rows),resets=len(rows),execution_ticks=sum(r['ticks'] for r in rows));c._save()
    V.finish(c)


def summarize():
    rr=V.read(BASE/'evaluation/result.json')['rows']
    old=V.read(V.HOST_ORIGINAL/'RESULT.json')['rows']
    old=[r for r in old if r['split']=='fresh_test']
    teachers=[r for r in old if r['condition'].startswith('CAP-')]
    reference=[r for r in old if r['condition'] in ('CAP-RR-003','expanded_GDA','repaired_GDA')]
    rows=rr+reference;names=list(CONDITIONS)+['CAP-RR-003','expanded_GDA','repaired_GDA']
    wins={n:{r['case_id'] for r in rows if r['condition']==n and r['success']} for n in names}
    teacher_union={r['case_id'] for r in teachers if r['success']}
    summary={};paired={}
    rng=np.random.default_rng(2026111761)
    ids=sorted(r['case_id'] for r in old if r['condition']=='expanded_GDA')
    for n in names:
        sub=[r for r in rows if r['condition']==n]
        assert len(sub)==len({r['case_id'] for r in sub})==100
        summary[n]=dict(success=len(wins[n]),n=100,retained_teacher=len(wins[n]&teacher_union),
            outside_teacher=len(wins[n]-teacher_union),by_movable={str(i):sum(r['success'] for r in sub if r['movable_count']==i) for i in range(2,6)},
            switches=sum(r.get('routing',{}).get('switches',0) for r in sub) if n in CONDITIONS else None,
            elapsed_wall_seconds=sum(r['wall_seconds'] for r in sub))
        if n in CONDITIONS:
            a,b=wins[n],wins['expanded_GDA']
            delta=np.array([int(i in a)-int(i in b) for i in ids])
            boot=np.mean(delta[rng.integers(0,100,size=(10000,100))],axis=1)
            paired[n]=dict(both=sorted(a&b),retrieval_only=sorted(a-b),GDA_only=sorted(b-a),neither=sorted(set(ids)-(a|b)),
                success_rate_delta=float(delta.mean()),paired_bootstrap_95_percentile=np.quantile(boot,[.025,.975]).tolist())
    result=dict(status='complete',summary=summary,paired=paired,teacher_union=len(teacher_union),
        new_episodes=200,new_ticks=sum(r['ticks'] for r in rr),no_training=True,no_llm=True,
        limitation='previously inspected diagnostic100, no distance tuning; snapshot and GDA inputs differ; single frozen GDA training seed',
        seals={str(p):V.Campaign(p).verify_frozen() for p in (PARENT,BASE/'indexes',BASE/'evaluation')})
    V.save(HOST/'RESULT.json',result)
    lines=['# 同270示范：检索基线与GDA的固定100例横向诊断','',
        '这是已查看的上一轮100例，不是新盲测。R0/Rt同270条完整示范，k=1不调参；Rt用全部9292边界。无新CAP/训练。','',
        '| 方法 | 成功/100 | 与GDA共同成功 | 本方法独有 | GDA独有 |','|---|---:|---:|---:|---:|']
    for n in names:
        p=paired.get(n)
        lines.append(f"| {n} | {summary[n]['success']} | "+(' | '.join(str(len(p[k])) for k in ['both','retrieval_only','GDA_only']) if p else '— | — | —')+' |')
    lines+=['',f'教师从初态可解并集{len(teacher_union)}/100，不是动态切换的理论上界。',
        '快照检索与包含历史、动作建议和描述语义的GDA并非等表示消融；此轮只比较可部署方法的结果，不单独归因架构。并行负载下耗时只作描述，不作为严格性能基准。',
        f'原始结果：`{HOST}/RESULT.json`。']
    (V.ROOT/'docs/agent_exchange/reviews/CAP_Retrieval_Comparison_20261117.md').write_text('\n'.join(lines)+'\n')
    V.save(HOST/'FINISHED.json',dict(status='complete',episodes=200,ticks=result['new_ticks']))
    V.update(stage='complete',completed=200,total=200)


def main():
    V.HOST_ORIGINAL=V.HOST;V.HOST=HOST
    prepare();build_indexes();evaluate();summarize()


if __name__=='__main__':
    try:main()
    except BaseException:
        V.save(HOST/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));V.update(stage='failed',status='failed');raise
