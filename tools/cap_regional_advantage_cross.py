"""Isolate the first differing physical action from the choice of its continuation."""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch

from tools import cap_regional_advantage as ref

OUT=ref.OUT/'cross'
study=ref.study


@torch.inference_mode()
def cross(f, landmark, first, suffix):
    env=ref.ppo.environment(f)
    lib=ref.cpp.Proposals()
    try:
        for step in range(300):
            panel=lib.panel(env.history)
            x,c,s,_,pool,raw=panel
            if step==landmark['step']:
                assert env.history.sha256==landmark['prefix']
                i,j=ref.choices(panel)[:2]
                assert i!=j
                proposals=[pool[i],pool[j]]
                a=proposals[first]
            elif step<landmark['step'] or suffix==0:
                logits=ref.ID(torch.from_numpy(x[None]))[0]
                a=raw[int(logits.argmax())] if logits.isfinite().any() else pool[0]
            else:
                scores=ref.ACTION.values(*(torch.from_numpy(arr[None]) for arr in (x,c,s)))[0,:len(pool)]
                a=pool[int(scores.argmax())]
            env.step(a,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        assert step>=landmark['step']
        return dict(success=int(env.env.success),ticks=env.env.total_periods,history=env.history.sha256,proposals=proposals)
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def case(parent):
    cid=parent['parent']
    path=OUT/(cid+'.json')
    if path.exists():
        return study.read(path)
    r=study.read(ref.OUT/'cases'/f'{cid}_0.json')
    row=dict(parent=cid,objects=parent['objects'],eligible='divergence' in r['stages'],
             id_id=r['baseline']['success'],action_action=r['labels']['initial']['success'])
    if row['eligible']:
        landmark=r['stages']['divergence']
        outputs=[cross(parent['fixtures'][0],landmark,*pair) for pair in [(1,0),(0,1)]]
        assert outputs[0]['proposals']==outputs[1]['proposals']
        a,b=outputs[0]['proposals']
        row['changed_fields']=','.join(k for k in sorted(a.keys()|b.keys()) if a.get(k)!=b.get(k))
        row.update(prefix=landmark['prefix'],step=landmark['step'],action_id=outputs[0]['success'],
                   id_action=outputs[1]['success'],crosses=outputs)
    study.save(path,row)
    return row


def main():
    ref.init()
    parents=study.read(ref.OUT/'panel.json')
    assert study.ext.old.digest(parents)==study.read(ref.OUT/'protocol.json')['panel']
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        rows=list(pool.map(case,parents))
    result=dict(n=len(rows),eligible=sum(r['eligible'] for r in rows),source=study.sha(__file__))
    for suffix in ('id','action'):
        pairs=[(r['id_'+suffix],r['action_'+suffix]) for r in rows if r['eligible']]
        result[suffix]=dict(alt_only=pairs.count((0,1)),ref_only=pairs.count((1,0)),
                            both=pairs.count((1,1)),neither=pairs.count((0,0)))
    pair=[(r['action_id']-r['id_id'],r['action_action']-r['id_action']) for r in rows if r['eligible']]
    result['reverse']=sum(a*b<0 for a,b in pair)
    result['tie_change']=sum((a==0)!=(b==0) for a,b in pair)
    docs=study.DOCS.parent/'2026-11-20_regional_advantage'
    docs.mkdir(parents=True,exist_ok=True)
    with (docs/'first_action.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['parent','objects','eligible','step','prefix','changed_fields','id_id','action_id','id_action','action_action'],extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    study.save(docs/'first_action.json',result)


if __name__=='__main__':
    main()
