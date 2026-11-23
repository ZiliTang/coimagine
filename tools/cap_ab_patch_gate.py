"""Inspect the first distinct CAP command and test small public one-time gates."""
from collections import Counter
from pathlib import Path
import json
import sys

from tools import cap_authored_handoff as exp
from tools import cap_authored_handoff_graph as graph

OUT=exp.OUT.parent/'cap_ab_patch_gate_20261123_01'
DOC=exp.DOC


def trace(fixture, sequence):
    policies=[exp.s.phase.FrozenCAP(m) for m in (exp.AM,exp.BM)]
    first=None
    counts=Counter()
    post=[]
    with exp.s.phase.world(fixture) as (w,_):
        def update(h):
            for p in policies:p.update(h)
        update(w.history)
        while not exp.s.ended(w):
            obs=exp.s.observe(w)
            arm=0
            if first is None:
                a,b=[p.propose(obs) for p in policies]
                if a!=b:
                    mem=policies[1].memory
                    rec=mem['recovery']['proposal']
                    if rec is None:
                        assert a['ticks']>b['ticks'] and all(a[k]==b[k] for k in ('a0','a1','target_id'))
                    c=mem['c']
                    import math
                    gap=math.dist(c[:2],obs['goal']['xy']) if c is not None else None
                    first=dict(tick=obs['tick'],h=w.history.sha256,a=a,b=b,recovery=rec is not None,
                               gap=gap,touch=any(r['contacting_object'] for r in obs['robots']),
                               engaged=any(r['execution'].get('stage')=='push' and r['contacting_object'] for r in obs['robots']),
                               same_pair=all(a[k]==b[k] for k in ('a0','a1','target_id')),
                               b_stage=b['target_id'] is None,b_shorter=b['ticks']<a['ticks'])
                    arm=int(sequence[0]=='B')
                cmd=(a,b)[arm]
            else:
                arm=int(sequence[-1]=='B')
                cmd=policies[arm].propose(obs)
            if first and sequence=='inspect':return first,None
            exp.s.tick_action(w,cmd,exp.g.prep.prior.EXECUTOR,update,counts)
            if first and len(post)<5:
                post.append(dict(tick=w.env.total_periods,cmd=cmd,
                     gap=exp.g.prep.goal_gap(w,w.env.primary_id,obs['goal']['xy'])))
        result=exp.s.output(w,counts)
        result['post']=post
        return first,result


def worker(job):
    fold,k,f=job
    path=OUT/fold/(k+'.json')
    if path.exists():return
    src=exp.read(exp.OUT/fold/(k+'.json'))
    a,b=[src[m]['result'] for m in ('A','B')]
    first,end=trace(f,'inspect')
    if not first:assert a['history']==b['history']==end['history']
    if fold=='train' and first:
        assert first['h'] in src['A']['origin_hashes'] and first['h'] in src['B']['origin_hashes']
    suffix={}
    if fold=='test' and a['success']!=b['success']:
        for seq in ('AA','BB','AB','BA'):
            init,r=trace(f,seq)
            assert first==init
            if seq in ('AA','BB'):assert r['history']==src[seq[0]]['result']['history']
            suffix[seq]=dict(success=r['success'],ticks=r['ticks'],post=r['post'])
    exp.save(path,dict(room=k,a=a['success'],b=b['success'],first=first,suffix=suffix))


def conditions(s):
    return dict(always=True,recovery=s['recovery'],same_pair=s['same_pair'],touch=s['touch'],
                engaged=s['engaged'],b_stage=s['b_stage'],shorter=s['b_shorter'],
                near=s['gap'] is not None and s['gap']<=.5,
                mid=s['gap'] is not None and s['gap']<=.89,
                late=s['tick']>=900)


def choice(row,rule):
    if not row['first']:return row['a']
    flag,rev=rule
    return row['b'] if conditions(row['first'])[flag]!=rev else row['a']


def rows(fold):
    return [exp.read(p) for p in sorted((OUT/fold).glob('*.json'))]


def main():
    exp.init(Path(exp.read(exp.OUT/'member.json')['path']).read_text(), profile=exp.perception.LEGACY)
    dev=exp.dev()
    exp.pool(worker,[('train',p.stem,dev[p.stem]) for p in (exp.OUT/'train').glob('*.json')])
    train=rows('train')
    assert len(train)==176
    cs=conditions(next(r['first'] for r in train if r['first']))
    rules=[(flag,rev) for flag in cs for rev in (False,True)]
    scores=[sum(choice(r,rule) for r in train) for rule in rules]
    rule=rules[scores.index(max(scores))]
    exp.save(OUT/'frozen.json',dict(rule=rule,scores=[[k,v] for k,v in zip(rules,scores)]))
    fresh=graph.fresh_panel()
    exp.pool(worker,[('test',k,f) for k,f in fresh.items()])
    print('finished',rule,flush=True)


if __name__=='__main__':main()
