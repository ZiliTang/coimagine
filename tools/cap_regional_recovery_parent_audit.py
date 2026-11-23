"""Compare the authored member with its unmodified parent at identical actor prefixes."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch

from tools import cap_regional_recovery_member as exp
from tools import cap_regional_cycle_source as source

study=exp.study
OUT=exp.OUT


def check(job):
    panel,row,origin,baseline,candidate=job
    parent_id=exp.ref.cpp.SID_ORDER.index('n_s002')
    trace,parent=source.replay(row['fixture'],entry=origin['step'],lock=parent_id)
    assert parent['complete'] and trace[origin['step']]['prefix']==origin['prefix']
    if candidate is None:
        candidate,_,actions=exp.run(row,'suffix',origin)
    else:
        actions=candidate.get('actions')
        candidate=candidate['new']
    split=None
    if actions is not None:
        for i,(a,b) in enumerate(zip(trace,actions)):
            if tuple(a['action'])!=exp.cycles.identity(b):
                split=dict(step=i,prefix=a['prefix'],parent=a['action'],new=exp.cycles.identity(b))
                assert i>=origin['step']
                break
        if split is None:
            assert parent['history']==candidate['hash']
    return dict(cid=row['cid'],objects=row['objects'],panel=panel,step=origin['step'],prefix=origin['prefix'],
                old=baseline['success'],parent=parent['success'],new=candidate['success'],first_change=split)


def duplicate_control(row):
    control,_,_=exp.run(row,'mix')
    assert control['distinct_from_pool']==0
    tested=study.read(OUT/'evaluation'/(row['cid']+'.json'))
    return dict(parent=row['cid'],objects=row['objects'],control=control,recovery=tested['new']['success'],
                duplicate=control['success'],original=tested['old']['success'])


def main():
    exp.init()
    jobs=[]
    for row in study.read(OUT/'train.json'):
        r=study.read(OUT/'development'/(row['cid']+'.json'))
        origin=dict(step=r['fork_step'],prefix=r['fork_prefix'])
        jobs.append(('train_cycle' if r['triggered'] else 'train_ordinary',row,origin,r['old'],r))
    for row in study.read(OUT/'test.json'):
        r=study.read(OUT/'evaluation'/(row['cid']+'.json'))
        if r['branch']:
            jobs.append(('test_cycle',row,r['old']['hit'],r['old'],dict(new=r['branch'])))
    select={'cm6820491_dev_3_19','fresh_dev_3_17'}
    for group,cid,n,f in study.read(exp.cycles.OUT/'panels.json'):
        if cid in select:
            r=study.read(exp.cycles.OUT/'rooms'/group/(cid+'.json'))['original']
            jobs.append(('author_example',dict(cid=cid,objects=n,fixture=f),r['windows'][0],r,None))
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        data=list(pool.map(check,jobs))
    metrics={}
    for panel in ['train_cycle','train_ordinary','test_cycle']:
        subset=[dict(r,parent_id=r['parent'],parent=r['cid']) for r in data if r['panel']==panel]
        metrics[panel]=study.compare(subset,'parent_id','new')
    # Keep the already-fitted margin head and all rollout settings, substituting only
    # the original parent for the eleventh proposal, including its routing inputs.
    exp.NATIVE=study.dq.load_native(study.ext.sourcepath('n_s002').read_text())[0]
    exp.HEAD=exp.margin_model()
    exp.HEAD.load_state_dict(torch.load(OUT/'head.pt',weights_only=True))
    exp.HEAD.eval()
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        control=list(pool.map(duplicate_control,study.read(OUT/'test.json')))
    metrics['fresh_duplicate_vs_recovery']=study.compare(control,'duplicate','recovery')
    metrics['fresh_original_vs_duplicate']=study.compare(control,'original','duplicate')
    study.save(OUT/'parent_audit.json',dict(comparisons=metrics,rooms=data,duplicate_control=control))
    print(metrics,flush=True)


if __name__=='__main__':
    main()
