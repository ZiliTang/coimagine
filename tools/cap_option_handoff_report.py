"""Select one development handoff, then evaluate the three links on untouched rooms."""
import argparse
import hashlib
from itertools import product

import numpy as np

from tools import cap_option_handoff as exp

FLAGS = ['touch','close','progress','clear']
CONDITIONS = FLAGS+[f'node{i}' for i in range(6)]


def has(state, name):
    if state is None:
        return False
    if name == 'any':
        return True
    if name.startswith('node'):
        return state['node'] == int(name[4:])
    return name in state['flags']


def cases(fold):
    paths = sorted((exp.OUT/fold).glob('*.json'))
    panel = exp.read(exp.OUT/(fold+'_panel.json'))
    assert len(panel)==len(paths)==200
    data = [exp.read(p) for p in paths]
    assert all(r['identity']==panel[r['key']] for r in data)
    return data


def action_effect(arm):
    return int(arm['edges'][1]['gain'] >= .02)


def result(arms, key):
    return arms[key]['result']['success']


def rows_for(data, x, y):
    rows = []
    for record in data:
        arms = record['arms']
        root = arms['AA']['root']
        eligible = has(root,x) and not has(root,y)
        a_mid, b_mid = arms['AA']['mid'],arms['BA']['mid']
        a_hit,b_hit = (eligible and has(mid,y) for mid in (a_mid,b_mid))
        switch_a = bool(a_hit and a_mid['distinct'])
        switch_b = bool(b_hit and b_mid['distinct'])
        choice = 'AB' if switch_a else 'AA'
        # Replace only the first preparation in eligible rooms. Off the rule keep A.
        b_prep_choice = ('BB' if switch_b else 'BA') if eligible else 'AA'
        row = dict(room=record['key'],prefix=bool(root),eligible=int(eligible),
                   a_hit=int(a_hit),b_hit=int(b_hit),switch=int(switch_a),
                   switch_b=int(switch_b),a_gain=0,b_gain=0,b_after_b_gain=0,
                   chosen=result(arms,choice),b_prepared=result(arms,b_prep_choice),
                   single_b=record['single_b']['success'],
                   **{k:result(arms,k) for k in exp.ARMS})
        if switch_a:
            row.update(a_gain=action_effect(arms['AA']),b_gain=action_effect(arms['AB']))
        if switch_b:
            row['b_after_b_gain']=action_effect(arms['BB'])
        rows.append(row)
    return rows


def summarize(rows, intervals=False):
    a = {k:np.asarray([r[k] for r in rows],int) for k in rows[0] if k!='room'}
    n = len(rows)
    def counts(k):
        return int(a[k].sum())
    def diff(u,v,mask=None):
        keep = a[mask].astype(bool) if mask else np.ones(n,bool)
        delta=(a[u]-a[v])[keep]
        out=dict(n=len(delta),first=int(a[u][keep].sum()),second=int(a[v][keep].sum()),
                 plus=int((delta>0).sum()),minus=int((delta<0).sum()),
                 delta=float(delta.mean()) if len(delta) else None)
        if intervals and len(delta):
            rng=np.random.default_rng(4103)
            samples=delta[rng.integers(len(delta),size=(4000,len(delta)))].mean(1)
            out['ci']=np.quantile(samples,[.025,.975]).tolist()
        return out
    return dict(n=n,prefix=counts('prefix'),eligible=counts('eligible'),active=counts('switch'),
                formation=diff('a_hit','b_hit','eligible'),local=diff('b_gain','a_gain','switch'),
                terminal=diff('chosen','AA'),vs_bprep=diff('chosen','b_prepared'),
                vs_b=diff('chosen','single_b'),paired_at_y=diff('AB','AA','switch'),
                b_progress_by_origin={'A':[counts('b_gain'),counts('switch')],
                                      'B':[counts('b_after_b_gain'),counts('switch_b')]},
                both_failed=int(((a['AA']==0)&(a['single_b']==0)).sum()),
                both_rescued=int(((a['AA']==0)&(a['single_b']==0)&(a['chosen']==1)).sum()),
                unconditional={k:counts(k) for k in exp.ARMS},
                hindsight_four=int(np.maximum.reduce([a[k] for k in exp.ARMS]).sum()))


def freeze():
    data = cases('dev')
    prev = exp.read(exp.previous.OUT/'panels.json')
    # Both natural controls must reproduce the frozen library runs.
    for r in data:
        for sid, run in zip(prev['pair'],(r['arms']['AA']['result'],r['single_b'])):
            ref=exp.read(exp.s.OUT/'singles/development'/exp.g.prep.prior.EXECUTOR/sid/(r['key']+'.json'))
            assert all(run[k]==ref[k] for k in ('success','ticks','decisions','history'))
    candidates=[]
    for x,y in product(['any']+CONDITIONS,CONDITIONS):
        sm=summarize(rows_for(data,x,y))
        if sm['eligible'] < 24 or sm['active'] < 12:
            continue
        d=[sm[k]['delta'] for k in ('formation','local','terminal')]
        candidates.append(dict(x=x,y=y,positive=sum(v>0 for v in d),dev=sm))
    candidates.sort(key=lambda r:(-r['positive'],-r['dev']['terminal']['delta'],
                                  -r['dev']['local']['delta'],-r['dev']['formation']['delta'],r['x'],r['y']))
    winner=candidates[0] if candidates else None
    f=dict(primary=bool(winner and winner['positive']==3),winner=winner,
           candidate_count=len(candidates),passing=sum(r['positive']==3 for r in candidates),
           pair=prev['pair'],seed=exp.SEED,
           spec_sha256=hashlib.sha256((exp.DOC/'plan.md').read_bytes()).hexdigest())
    exp.save(exp.DOC/'dev_candidates.json',candidates)
    path=exp.OUT/'frozen.json'
    if path.exists():
        assert exp.read(path)==f
    else:
        exp.save(path,f)
    print(f,flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('phase',choices=['freeze','report'])
    args=ap.parse_args()
    if args.phase=='freeze':
        freeze()
    else:
        from tools.cap_option_handoff_publish import publish
        publish()
