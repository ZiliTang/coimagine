"""Run the frozen 6.2 library and the existing GA directQ fit on identical new rooms."""
import csv
import time
import traceback

import numpy as np
import torch

from tools import cap_general_specialists as campaign
from tools import cap_general_specialists_families as fam
from tools import cap_regional_directq as dq
from tools.cap_regional_evaluate import paired

OUT=campaign.OUT


def stage(name,**fields):
    campaign.save(OUT/'post_progress.json',dict(stage=name,**fields))


def frozen_runs(sid,fixtures):
    path=OUT/'runs'/sid
    missing={k:f for k,f in fixtures.items() if not (path/(k+'.json')).is_file()}
    if missing:
        campaign.base.run_batch((campaign.CAM/'semantics'/(sid+'.py')).read_text(),missing,path,16)
    rows={cid:campaign.read(path/(cid+'.json')) for cid in fixtures}
    if not all(r['complete'] and not r['error'] for r in rows.values()):
        raise RuntimeError('incomplete frozen evaluation of '+sid)
    return rows


def evaluate(s):
    if (OUT/'FINISHED.json').exists():return
    assert s['stage']=='frozen' and campaign.base.host.RoomCampaign(campaign.CAM).verify_frozen()==s['seal']
    protocol=campaign.read(OUT/'protocol.json')
    refs=campaign.read(campaign.CAM/'knowledge/option_source_manifest.json')
    assert campaign.base.digest(refs)==protocol['reference_sha']
    assert all(campaign.sha(campaign.WORK/'option_source'/p)==h for p,h in refs.items())
    raw=campaign.read(OUT/'development.json')
    extra=campaign.read(OUT/'extra.json')
    held=campaign.read(OUT/'held.json')
    assert campaign.base.digest(raw)==protocol['pool'] and campaign.base.digest(held)==protocol['held']
    dev=raw|extra
    assert len(dev)==len(raw)+len(extra)
    sids=[r['sid'] for r in s['rounds']]
    assert 1<=len(sids)==len(set(sids))<=5
    single=max(s['rounds'],key=lambda r:len(r['wins']))['sid']
    torch.set_num_threads(1)
    dq.OUT=OUT/'directq'
    checkpoint=dq.OUT/'models/GA_s17/offline.pt'
    jobs=[]
    for i,sid in enumerate(sids):
        stage('开发轨迹补齐',sid=sid,member=i+1,total=len(sids))
        frozen_runs(sid,dev)
        source=(campaign.CAM/'semantics'/(sid+'.py')).read_text()
        dq.MODULES.append(dq.load_native(source)[0])
        jobs.extend((k,f,sid,i,OUT/'runs'/sid,'replay') for k,f in sorted(dev.items()))
    stage('真实轨迹重放与动作校验',episodes=len(jobs))
    shards=dq.batch(jobs,12,'replay')
    if not (dq.OUT/'arrays/train/manifest.json').exists():
        dq.full_data.assemble(dq.OUT/'arrays/train',shards)
    campaign.save(dq.OUT/'dataset.json',dict(layout=shards[0]['layout'],train_episodes=len(shards)))
    stage('拟合 GA directQ',v_updates=16000,q_updates=8000,seed=17)
    dq.full_learner.train_offline(dq.OUT,'GA',17,evaluate_validation=False)
    dq.MODEL=dq.load_checkpoint(checkpoint)[0]
    evaluated={}
    for i,sid in enumerate(sids):
        stage('新房间单成员闭环',sid=sid,member=i+1,total=len(sids))
        evaluated[sid]=frozen_runs(sid,held)
    stage('新房间混合闭环',rooms=len(held))
    dq.batch([(k,f,'mix',None,None,'fresh') for k,f in sorted(held.items())],12,'evaluate_fresh')
    cases=[]
    for cid,f in sorted(held.items()):
        scores={sid:int(evaluated[sid][cid]['success']) for sid in sids}
        mix=campaign.read(dq.OUT/'evaluation/fresh'/(cid+'.json'))
        cases.append(dict(room=cid,objects=1+len(f['movable_objects']),**scores,union=max(scores.values()),
            mix=int(mix['success']),general_motif=fam.describe(OUT/'runs'/sids[0],cid)['label']))
    with (OUT/'held_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(cases[0]));writer.writeheader();writer.writerows(cases)
    def values(key):return np.array([r[key] for r in cases])
    result=dict(general_vs_mix=paired(values(sids[0]),values('mix'),values('objects')),
        best_dev_sid=single,best_vs_mix=paired(values(single),values('mix'),values('objects')),
        union=int(values('union').sum()),members={sid:int(values(sid).sum()) for sid in sids},
        rooms=len(cases),train_rooms=len(dev),train_episodes=len(shards),boundaries=sum(r['boundaries'] for r in shards),
        sha256=campaign.sha(checkpoint),scope='single author/library, single fit seed, fresh paired natural rooms')
    campaign.save(OUT/'FINISHED.json',result)
    stage('完成')


if __name__=='__main__':
    try:
        while True:
            state=campaign.read(campaign.STATE)
            if state['stage']=='incomplete':break
            if state['stage']=='frozen':
                evaluate(state);break
            time.sleep(5)
    except Exception:
        stage('中断',error=traceback.format_exc())
        raise
