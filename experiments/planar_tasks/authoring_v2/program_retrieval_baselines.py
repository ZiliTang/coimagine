"""Build frozen R0/Rt indexes and execute bounded routing baselines.

No automatic benchmark or LLM invocation. Build requires an explicit completed
development collection; final/validation splits are rejected.
"""
import argparse
import collections
import json
from pathlib import Path
import time

import numpy as np

from . import random_room_coverage_v2 as V
from experiments.planar_tasks.semantic_rl.program_retrieval import GeometryEncoder, RetrievalIndex, RoutedFeatures, sha
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory


def select_roots(rows,max_cases=None):
    roots=collections.defaultdict(set)
    for r in rows:
        if r['split'] not in ('dev','newdev','development'):
            raise ValueError('index input must contain development only; no validation/test')
        roots[r['movable_count']].add(r['case_id'])
    queues=[sorted(roots[k]) for k in sorted(roots)]
    ordered=[q[i] for i in range(max(map(len,queues),default=0)) for q in queues if i<len(q)]
    return ordered if max_cases is None else ordered[:max_cases]


def sample_boundaries(n,limit):
    if n<=0 or limit<1:
        raise ValueError('positive decision count/sample cap required')
    return np.unique(np.linspace(0,n-1,min(n,limit),dtype=int)).tolist()


def build(rows_path,out,*,representation='snapshot',k=1,max_cases=None,points_per_trajectory=16,allowed_cases=None):
    rows_path=Path(rows_path);out=Path(out)
    if out.exists():
        raise ValueError('new immutable output directory required')
    V.Campaign(rows_path.parent).verify_frozen()
    rows=V.read(rows_path)['rows']
    ids=select_roots(rows,max_cases)
    if allowed_cases is not None:
        ids=[i for i in ids if i in allowed_cases]
    rows=[r for r in rows if r['case_id'] in ids]
    entries=V.entries();names=[e['name'] for e in entries]
    if any(r['condition'] not in names for r in rows):
        raise ValueError('only frozen CAP outcome rows allowed')
    if len({(r['case_id'],r['condition']) for r in rows})!=len(rows):
        raise ValueError('duplicate outcome labels')
    roots={};root_hash={};labels={i:np.full(len(names),np.nan) for i in ids}
    vectors=[];prefix_labels=[];prefix_records=[];inputs={}
    started=time.monotonic()
    for row in sorted(rows,key=lambda r:(r['case_id'],r['condition'])):
        path=Path(row['result_path']);result=V.unpack(path);inputs[str(path)]=sha(path)
        if result['action_protocol']!=V.O.protocol(V.O.CFG):
            raise ValueError('scale protocol changed')
        if result['summary']['success']!=row['success']:
            raise ValueError('outcome disagreement')
        source=next(e for e in entries if e['name']==row['condition'])['source']
        if result['source_sha256']!=V.digest(source):
            raise ValueError('teacher source mismatch')
        case=row['case_id'];member=names.index(row['condition'])
        # Runtime infrastructure failures are unknown; explicit policy-invalid
        # advice counts as failure as in the frozen CAP evaluation protocol.
        error=row.get('error')
        infra=row['stop']=='wall_budget' or error and error.get('message')!='CAP first recommendation missing/illegal'
        if not infra:
            labels[case][member]=int(row['success'])
        h=StreamHistory();enc=GeometryEncoder(representation)
        decisions=len(result['decisions'])
        selected=set(sample_boundaries(decisions,points_per_trajectory)) if row['success'] else set()
        for i,public in enumerate(result['public_rows']):
            if i>=max(selected,default=0)+1:
                break
            h.append(public['observation'],public['action'])
            if h.sha256!=public['history_hash']:
                raise ValueError('public history identity mismatch')
            x=enc.advance(h)
            if i==0:
                if case in root_hash and root_hash[case]!=h.sha256:
                    raise ValueError('CAPs did not see identical initial observation')
                root_hash[case]=h.sha256;roots[case]=x
            if i in selected:
                y=np.full(len(names),np.nan);y[member]=1.
                vectors.append(x);prefix_labels.append(y)
                prefix_records.append(dict(case_id=case,cap=row['condition'],boundary=i,history_hash=h.sha256,
                    result_path=str(path),evidence='successful own continuation, not cross-CAP handoff label'))
    if not vectors or len(roots)!=len(ids):
        raise ValueError('empty/incomplete development evidence')
    metadata=dict(sources={e['name']:V.digest(e['source']) for e in entries},action_protocol=V.O.protocol(V.O.CFG),
        representation=representation,geometry='shared G entity/scan encoder; snapshot retains last-seen cache and episode frame',
        rows_path=str(rows_path),rows_sha256=sha(rows_path),case_ids=ids,
        selection='all CAP outcomes at selected roots; up to fixed16 evenly spaced successful boundaries per trajectory',
        points_per_trajectory=points_per_trajectory,unknown_labels='NaN, never negative',
        purpose='development index; coverage/label budget must be matched before formal architecture claims')
    r0=RetrievalIndex(np.array([roots[i] for i in ids]),np.array([labels[i] for i in ids]),names,
        dict(metadata,records=[dict(case_id=i,history_hash=root_hash[i]) for i in ids]),
        evidence='initial_outcomes',representation=representation,k=k)
    rt=RetrievalIndex(np.array(vectors),np.array(prefix_labels),names,dict(metadata,records=prefix_records),
        evidence='successful_prefix',representation=representation,k=k)
    out.mkdir(parents=True)
    r0.save(out/'initial');rt.save(out/'boundary')
    V.save(out/'BUILD.json',dict(status='complete',development_cases=len(ids),CAP_rollouts=len(rows),
        initial_rows=len(r0.vectors),boundary_rows=len(rt.vectors),elapsed_seconds=time.monotonic()-started,
        inputs=inputs,new_simulation_ticks=0,new_llm_calls=0))
    return V.read(out/'BUILD.json')


def execute(index_path,mode,case,out,*,max_ticks=1800,max_decisions=300,wall_seconds=300):
    out=Path(out)
    if out.exists():
        raise ValueError('new rollout directory required')
    index=RetrievalIndex.load(index_path)
    expected='initial_outcomes' if mode=='initial' else 'successful_prefix'
    if index.evidence!=expected:
        raise ValueError('index/mode mismatch')
    f=RoutedFeatures(V.entries(),index,V.O.protocol(V.O.CFG),mode=mode)
    out.mkdir(parents=True)
    try:
        result,_=V.rollout(case['fixture'],f.source,features=f,model=None,config=V.O.CFG,
            max_ticks=max_ticks,max_decisions=max_decisions,wall_seconds=wall_seconds,
            seed=17,deterministic=True,fast_feedback=True)
        result['routing_trace']=f.trace
        result['routing_summary']=dict(mode=mode,index=index.identity,representation=index.representation,
            switches=sum(t['switched'] for t in f.trace),selection_counts=dict(collections.Counter(t['selected'] for t in f.trace)),
            index_case_overlap=case['id'] in index.metadata['case_ids'],
            prepare_seconds=sum(t['prepare_seconds'] for t in f.trace),
            retrieval_seconds=sum(t['retrieval_seconds'] for t in f.trace))
        V.pack(out/'result.json.gz',result,compresslevel=1)
        summary=dict(case_id=case['id'],**result['summary'],routing=result['routing_summary'])
        V.save(out/'summary.json',summary)
        V.save(out/'provenance.json',dict(index_path=str(index_path),index_sha256=sha(Path(index_path)/'index.json'),
            fixture_sha256=V.digest(case['fixture']),source_sha256=f.sha256,
            implementations={str(p):sha(p) for p in [Path(__file__),Path(__file__).parents[1]/'semantic_rl/program_retrieval.py']},
            limits=dict(max_ticks=max_ticks,max_decisions=max_decisions,wall_seconds=wall_seconds),
            purpose='engineering smoke unless separately frozen benchmark protocol'))
        return summary
    finally:
        f.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('build');b.add_argument('--rows',type=Path,required=True);b.add_argument('--out',type=Path,required=True)
    b.add_argument('--representation',choices=('snapshot','history'),default='snapshot')
    b.add_argument('--k',type=int,default=1);b.add_argument('--max-cases',type=int)
    b.add_argument('--points-per-trajectory',type=int,default=16)
    r=sub.add_parser('run');r.add_argument('--index',type=Path,required=True);r.add_argument('--mode',choices=('initial','boundary'),required=True)
    r.add_argument('--cases',type=Path,required=True);r.add_argument('--case-id',required=True);r.add_argument('--out',type=Path,required=True)
    r.add_argument('--max-ticks',type=int,default=1800);r.add_argument('--max-decisions',type=int,default=300)
    r.add_argument('--wall-seconds',type=float,default=300)
    a=p.parse_args()
    if a.command=='build':
        result=build(a.rows,a.out,representation=a.representation,k=a.k,max_cases=a.max_cases,points_per_trajectory=a.points_per_trajectory)
        result={k:v for k,v in result.items() if k!='inputs'}
    else:
        cases=V.read(a.cases)['cases'];case=next(c for c in cases if c['id']==a.case_id)
        result=execute(a.index,a.mode,case,a.out,max_ticks=a.max_ticks,max_decisions=a.max_decisions,wall_seconds=a.wall_seconds)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
