"""Paired real option successors with frozen Q/V and common continuation policies."""
import gzip
import hashlib
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from . import cap_inverse_scale as scale
from . import cap_policy_switch as switch
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v, load_checkpoint
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS

OUT = scale.OUT.parent/'cap_successor_value_20261119_01'
SEED = 2026111947
DEPTHS = (0, 5, 12, 25)
Q = switch.MODEL/'offline.pt'
V = switch.MODEL/'V.pt'
NETWORKS = {}


def freeze():
    previous = read(switch.OUT/'protocol.json')
    assert sha(Q) == previous['q_sha256'] and sha(V) == previous['v_sha256']
    used = set(read(scale.OUT/'forbidden.json'))
    for p in [*scale.OUT.glob('batches/*_plan.json'),
              *(scale.OUT.parent/'cap_semantic_disagreement_20261119_01').glob('*_panel.json')]:
        used.update(r['content_sha256'] for r in read(p))
    cases = []
    for r in scale.family.rooms.generate(SEED)['cases']:
        key = scale.family.content(r['fixture'])
        assert key not in used
        used.add(key)
        cases.append(dict(case_id='succ_dev_'+r['id'], fixture=r['fixture'], content_sha256=key,
                          movable_count=r['movable_count']))
    cases.sort(key=lambda c: (c['movable_count'], hashlib.sha256(c['case_id'].encode()).hexdigest()))
    for i,c in enumerate(cases):
        c['min_boundary'] = DEPTHS[i % len(DEPTHS)]
    assert len(cases) == len({c['case_id'] for c in cases}) == 200
    panel = OUT/'panel.json'
    if panel.exists():
        assert read(panel) == cases
    else:
        save(panel,cases)
    spec = dict(panel_sha256=sha(panel), seed=SEED, q=sha(Q), v=sha(V), training_seed=17,
                primary='GA-Q continuation on both branches. Pair top two distinct legal Q proposals at first eligible boundary at/after assigned depth. Direct Q vs max legal frozen V at actual option end.',
                control='Continue with CAP009 on both branches, evaluate V009. Report pure pairs only, requiring legal CAP009 in every subsequent decision of both branches.',
                terminal='Immediate true terminal or exhausted budget uses observed binary reward (gamma=1), also report pairs with both successors nonterminal.',
                ties='Stable pool order for Q. Equal successor scores retain the original Q choice; disclose tie counts.',
                exclusions='Report all scenes ending without an eligible fork, plus incomplete runs separately. One fork per scene.',
                uncertainty='20000 paired scene bootstrap draws; exact two-sided binomial on discordant policy selections; single checkpoint.',
                caps=CAPS, depths=list(DEPTHS), driver_sha256=sha(__file__))
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json') == spec
    else:
        save(OUT/'protocol.json',spec)
    return cases


class Branch:
    def __init__(self,depth,rank,continuation):
        self.depth=depth
        self.rank=rank
        self.continuation=continuation
        self.index=-1
        self.fork=None
        self.successor=None
        self.unsupported=0

    def values(self,x,candidates,sources):
        self.index+=1
        scores=NETWORKS['q'].values(x,candidates,sources)
        allowed=sources[0].any(dim=1)
        supported=torch.where(allowed)[0].tolist()
        if self.fork is None and self.index>=self.depth and len(supported)>=2:
            order=sorted(supported,key=lambda i:-float(scores[0,i]))
            slots=order[:2]
            self.fork=dict(index=self.index,slots=slots,q=[float(scores[0,i]) for i in slots])
            slot=slots[self.rank]
        elif self.fork is not None:
            if self.index==self.fork['index']+1:
                legal=sources[0].any(dim=0)
                g=NETWORKS['v'].layout['geometry_width']
                assert torch.equal(legal,x[0,g:g+7*86].reshape(7,86)[:,-1]==1)
                values=all_v(NETWORKS['v'],x)[0]
                assert torch.isfinite(values).all()
                self.successor=dict(legal=legal.tolist(),v=values.tolist(),
                                    max_v=float(values[legal].max()) if legal.any() else 0.,
                                    v009=float(values[0]))
            if self.continuation=='009':
                options=torch.where(sources[0,:,0].bool())[0].tolist()
                if not options:
                    self.unsupported+=1
                    return scores
                slot=options[0]
            else:
                return scores
        else:
            return scores
        forced=torch.zeros_like(scores)
        forced[0,slot]=1.
        return forced


def initialize():
    scale.family.initialize()
    for kind,path in [('q',Q),('v',V)]:
        network,meta=load_checkpoint(path)
        assert meta['kind']==kind.upper() and meta['seed']==17 and meta['arm']=='GA'
        NETWORKS[kind]=network


def execute(case,rank,continuation):
    entry=Branch(case['min_boundary'],rank,continuation)
    fast_runtime.W.MODELS['branch']=entry
    with TemporaryDirectory(dir=OUT) as directory:
        try:
            s=fast_runtime.W.run(dict(case=case,condition='successor_value',checkpoint='branch',
                                      epsilon=0.,directory=directory))
        finally:
            if fast_runtime.ACTIVE is not None:
                fast_runtime.ACTIVE.close_reads()
                fast_runtime.ACTIVE=None
        assert s['complete'] and s['error'] is None,s
        with gzip.open(Path(directory)/'result.json.gz','rt') as stream:
            replay=json.load(stream)
    result=dict(success=int(s['success']),ticks=s['ticks'],stop=s['stop'],termination=s['termination'],
                decisions=s['decisions'],fallback=entry.unsupported,final_hash=replay['public_rows'][-1]['history_hash'])
    if entry.fork is None:
        return result
    i=entry.fork['index']
    scores=entry.fork['q']
    assert all(np.isfinite(scores)) and scores[0]>=scores[1]
    slots=entry.fork['slots']
    decision=replay['decisions'][i]
    assert decision['chosen']==slots[rank]
    candidates=[decision['candidates'][slot] for slot in slots]
    assert candidates[0]!=candidates[1]
    terminal=i+1==s['decisions']
    assert (entry.successor is None)==terminal
    if terminal:
        successor=dict(max_v=result['success'],v009=result['success'],v=None,legal=None)
    else:
        successor=entry.successor
    result.update(fork=dict(index=i,tick=decision['event']['start_tick'],q=scores,
                            hash=replay['public_rows'][i]['history_hash'],candidates=candidates,
                            sources=[decision['sources'][slot] for slot in slots]),
                  successor=dict(**successor,terminal=terminal,periods=decision['event']['periods'],
                                 hash=replay['public_rows'][i+1]['history_hash']))
    return result


def paired(case):
    cid=case['case_id']; path=OUT/'cases'/(cid+'.json')
    if path.exists():
        return read(path)
    top=execute(case,0,'ga')
    row=dict(case_id=cid,movable_count=case['movable_count'],min_boundary=case['min_boundary'],
             eligible='fork' in top,branches={'ga':[top]})
    if row['eligible']:
        row['branches']['ga'].append(execute(case,1,'ga'))
        row['branches']['009']=[execute(case,i,'009') for i in (0,1)]
        for i in (0,1):
            ga=row['branches']['ga'][i]; cap=row['branches']['009'][i]
            assert ga['fork']==cap['fork']==top['fork']
            assert ga['successor']==cap['successor']
        assert row['branches']['ga'][0]['successor']['hash']!=row['branches']['ga'][1]['successor']['hash']
    save(path,row)
    return row


def main():
    cases=freeze()
    with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        futures=[pool.submit(paired,c) for c in cases]
        for i,f in enumerate(as_completed(futures),1):
            f.result()
            save(OUT/'progress.json',dict(scenes=i,total=len(cases)))
    from .cap_successor_value_report import report
    report()


if __name__=='__main__':
    main()
