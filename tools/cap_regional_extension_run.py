"""Supervise the regional continuation, seal five submissions, fit GA-Q and compare policies."""
import csv
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_directq as dq
from tools import cap_regional_extension as ext
from tools.cap_regional_evaluate import paired,sha

OUT=ext.OUT
DOCS=ext.old.ROOT/'docs/research/key_findings/2026-11-19_regional_extension'
read,save=ext.read,ext.save


def stage(name,**details):
    save(OUT/'pipeline.json',dict(stage=name,time=time.time(),**details))


def freeze():
    cfg=read(OUT/'protocol.json');state=read(OUT/'author_state.json')
    assert len(state['sweeps'])==5 and not state['active']
    used=read(OUT/'budget.json');assert used['tokens']<=ext.MAX_TOKENS
    for r in state['sweeps']:
        assert r['accepted']<=used['deadline'] and r['hash']==sha(ext.sourcepath(r['sid']))
        rows=[read(f) for f in (OUT/'sweeps'/r['sid']).glob('*.json')]
        assert len(rows)==215 and all(z['complete'] for z in rows)
        assert sorted(z['id'] for z in rows if z['success'])==r['wins']
    new=ext.old.host.RoomCampaign(ext.CAM)
    if read(ext.CAM/'campaign.json')['status']!='frozen':new.freeze()
    seals=dict(new=new.verify_frozen(),old=ext.old.host.RoomCampaign(ext.old.OUT/'regional').verify_frozen())
    sids=cfg['old_sids']+[r['sid'] for r in state['sweeps']]
    sources={sid:sha(ext.sourcepath(sid)) for sid in sids}
    assert len(set(sources.values()))==len(sids)==10
    frozen=dict(sids=sids,sources=sources,seals=seals,held=cfg['held_sha'],development=cfg['dev_sha'])
    if (OUT/'frozen.json').exists():assert read(OUT/'frozen.json')==frozen
    else:save(OUT/'frozen.json',frozen)
    return sids


def init_models(sids):
    dq.MODULES=[]
    for sid in sids:
        path=ext.sourcepath(sid)
        assert sha(path)==read(OUT/'frozen.json')['sources'][sid]
        dq.MODULES.append(dq.load_native(path.read_text())[0])


def compare(rows,first,second):
    a=np.array([r[first] for r in rows]);b=np.array([r[second] for r in rows])
    parents=sorted({r['parent'] for r in rows});mp={k:i for i,k in enumerate(parents)}
    # Each parent contributes exactly two neighbors in the perturbation panel.
    ix=np.array([mp[r['parent']] for r in rows])
    delta=b-a
    sums=np.bincount(ix,weights=delta);counts=np.bincount(ix);assert np.all(counts==counts[0])
    obj={r['parent']:r['objects'] for r in rows}
    stratum=np.array([obj[k] for k in parents])
    rng=np.random.default_rng(17);draw=np.zeros(10000)
    for k in np.unique(stratum):
        vals=sums[stratum==k];draw+=vals[rng.integers(len(vals),size=(10000,len(vals)))].sum(1)
    return dict(n=len(rows),families=len(parents),first=int(a.sum()),second=int(b.sum()),rescued=int(sum(delta>0)),
                lost=int(sum(delta<0)),pp=float(100*delta.mean()),ci=(100*np.quantile(draw,[.025,.975])/len(rows)).tolist())


def results(panels,sids,best):
    DOCS.mkdir(parents=True,exist_ok=True)
    cfg=read(OUT/'protocol.json');baseline=read(OUT/'baseline.json');sweeps=read(OUT/'author_state.json')['sweeps']
    wins={**baseline,**{r['sid']:r['wins'] for r in sweeps}}
    rows=[];metrics={};old_best=max(ext.SIDES,key=lambda s:sum(k.startswith('dev_') for k in wins[s]))
    for panel,fixtures in panels.items():
        group=[]
        for cid,f in sorted(fixtures.items()):
            if panel in ('dev','aug'):cap={s:int(cid in wins[s]) for s in sids}
            else:cap={s:int(read(OUT/'evaluation'/s/(cid+'.json'))['success']) for s in sids}
            if panel=='near':parent=cid.removeprefix('near_').rsplit('_',1)[0]
            else:parent=cfg['families'].get(cid,cid)
            q={run:int(read(OUT/run/'evaluation'/panel/(cid+'.json'))['success']) for run in ('directq','oldq')}
            group.append(dict(panel=panel,case_id=cid,parent=parent,objects=1+len(f['movable_objects']),**cap,**q,
                         best=cap[best],old_best=cap[old_best],old_union=max(cap[s] for s in ext.SIDES),union=max(cap.values())))
        rows.extend(group)
        pairs=[('old_union','union'),('old_best','best'),('best','directq'),('oldq','directq')]
        # Development augmentation has unequal family sizes. Its 55 raw counts are descriptive.
        metrics[panel]={key:sum(r[key] for r in group) for key in ('best','old_best','old_union','union','directq','oldq')}
        metrics[panel].update(n=len(group),union_retained=sum(r['union']*r['directq'] for r in group),
            outside=sum((1-r['union'])*r['directq'] for r in group))
        if panel!='aug':metrics[panel]['paired']={a+'_to_'+b:compare(group,a,b) for a,b in pairs}
    with (DOCS/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    family=[]
    for parent in cfg['parents']:
        base=next(r for r in rows if r['case_id']==parent and r['panel']=='dev')
        neighbors=[r for r in rows if r['panel']=='near' and r['parent']==parent]
        train_near=[r for r in rows if r['panel']=='aug' and r['parent']==parent]
        assert len(neighbors)==2 and train_near
        family.append(dict(parent=parent,parent_repaired=base['union'],old_near=sum(r['old_union'] for r in neighbors),
            new_near=sum(r['union'] for r in neighbors),directq_near=sum(r['directq'] for r in neighbors),
            train_neighbors=len(train_near),old_train_fraction=np.mean([r['old_union'] for r in train_near]),
            new_train_fraction=np.mean([r['union'] for r in train_near]),
            same_new_member_parent_and_both_neighbors=int(any(base[s] and all(r[s] for r in neighbors) for s in sids[5:]))))
    with (DOCS/'families.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=family[0]);w.writeheader();w.writerows(family)
    all_results=dict(metrics=metrics,old_best=old_best,best=best,fit=read(OUT/'directq/models/GA_s17/offline_complete.json'),
        checkpoint=sha(OUT/'directq/models/GA_s17/offline.pt'),author=read(OUT/'budget.json'),rounds=sweeps,
        original_repaired=sum(r['parent_repaired'] for r in family),
        same_member_robust=sum(r['same_new_member_parent_and_both_neighbors'] for r in family),
        augmented_family_equal=dict(old=float(np.mean([r['old_train_fraction'] for r in family])),
                                   new=float(np.mean([r['new_train_fraction'] for r in family]))))
    save(DOCS/'results.json',all_results)
    lines=['# 区域 CAP 第六至十轮与 directQ','','原五成员库继续开发。困难开发池由 45 个原未解父房间和 55 个新扰动组成。自然留出为新冻结的 200 个独立房间，邻域留出为每个原未解父房间各两个新扰动。','',
        '| 面板 | 场景数 | 原最佳单 CAP | 新库开发选定单 CAP | 五成员并集 | 十成员并集 | 原 directQ | 新 directQ |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for p,m in metrics.items():
        lines.append('| '+' | '.join(map(str,[p,*[m[k] for k in ('n','old_best','best','old_union','union','oldq','directq')]]))+' |')
    u=all_results['author']
    lines+=['',f"原 45 个失败父房间修复 {all_results['original_repaired']} 个。由同一个新增成员解决父房间及其两个未见扰动的有 {all_results['same_member_robust']} 族。",'',
        f"原最佳 {old_best}，扩展库的最佳单程序 {best}。两者均在原 160 个开发场景按成功数选择。V 与 Q 继续采用 seed17、16,000 和 8,000 次更新，输入无 trace。新 directQ 使用 215 个开发初态的全部十成员轨迹，旧模型保留原来的训练数据，因此 Q 的差值包含支持库和开发数据同时改变的影响。",'',
        f"作者开发上限为五轮合计 30 分钟和 40,000,000 个输入加输出 token（含缓存读取）。最后计量为 {u['tokens']:,} token，作者窗口 {u['seconds']:.0f} 秒。后台预标注、冻结重放、拟合和评测单独计时。",'',
        '[逐场景配对](per_scene.csv) · [按父房间统计](families.csv) · [完整统计及配对区间](results.json)']
    (DOCS/'README.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'FINISHED.json',dict(complete=True,report=sha(DOCS/'results.json')))
    stage('finished')


def evaluate_experts():
    frozen=read(OUT/'frozen.json');held=read(OUT/'held.json')
    assert ext.old.digest(held)==frozen['held']
    for sid in frozen['sids']:
        assert sha(ext.sourcepath(sid))==frozen['sources'][sid]
        save(OUT/'cap_progress.json',dict(active=sid))
        rr=ext.old.run_batch(ext.sourcepath(sid).read_text(),held,OUT/'evaluation'/sid,12)
        assert len(rr)==len(held) and all(r['complete'] for r in rr)
    save(OUT/'cap_progress.json',dict(complete=True))


def run():
    stage('author')
    while not (OUT/'author_state.json').exists():time.sleep(2)
    while True:
        state=read(OUT/'author_state.json');budget=ext.usage()
        if (OUT/'author_closed.json').exists():
            budget=read(OUT/'author_closed.json')
        save(OUT/'budget.json',budget)
        if len(state['sweeps'])==5:break
        if budget['seconds_left']<=0 or budget['tokens']>=ext.MAX_TOKENS-300000:
            if state['active'] and state['active']['kind']=='sweep' and len(state['sweeps'])==4:
                time.sleep(2);continue
            raise RuntimeError('author limit before five completed submissions')
        time.sleep(2)
    sids=freeze();stage('frozen')
    cfg=read(OUT/'protocol.json');dev=read(OUT/ext.DEV);held=read(OUT/'held.json')
    assert ext.old.digest(held)==cfg['held_sha'] and ext.old.digest(dev)==cfg['dev_sha']
    state=read(OUT/'author_state.json')
    wins={**read(OUT/'baseline.json'),**{r['sid']:r['wins'] for r in state['sweeps']}}
    best=max(sids,key=lambda s:sum(k.startswith('dev_') for k in wins[s]))
    panels={p:{k:v for k,v in ({**dev,**held}).items() if k.startswith(p+'_')} for p in ('dev','aug','fresh','near')}
    torch.set_num_threads(1)
    dq.OUT=OUT/'directq';dq.OUT.mkdir(exist_ok=True)
    init_models(sids)
    jobs=[(cid,f,sid,i,ext.refs(sid,cid),'replay') for cid,f in sorted(dev.items()) for i,sid in enumerate(sids)]
    stage('directq_replay')
    shards=dq.batch(jobs,18,'replay')
    if not (dq.OUT/'arrays/train/manifest.json').exists():dq.full_data.assemble(dq.OUT/'arrays/train',shards)
    save(dq.OUT/'dataset.json',dict(layout=shards[0]['layout'],train_episodes=len(shards)))
    stage('directq_fit')
    # Physics evaluations and the single-threaded fit share only frozen, read-only inputs.
    child=subprocess.Popen([sys.executable,'-u','-m',__name__ if __name__!='__main__' else 'tools.cap_regional_extension_run','experts'])
    dq.full_learner.train_offline(dq.OUT,'GA',17,evaluate_validation=False)
    for mode,source_sids,ckpt in [('directq',sids,OUT/'directq/models/GA_s17/offline.pt'),
                                   ('oldq',ext.SIDES,ext.old.OUT/'directq/models/GA_s17/offline.pt')]:
        init_models(source_sids);dq.OUT=OUT/mode;dq.OUT.mkdir(exist_ok=True)
        dq.MODEL=dq.load_checkpoint(ckpt)[0]
        for panel,fixtures in panels.items():
            stage(mode+'_evaluation',panel=panel)
            dq.batch([(k,v,mode,None,None,panel) for k,v in sorted(fixtures.items())],12,'evaluate_'+panel)
    stage('cap_evaluation')
    if child.wait()!=0:raise RuntimeError('incomplete frozen expert evaluation; see run.log')
    results(panels,sids,best)


if __name__=='__main__':
    try:
        if sys.argv[1:]==['experts']:evaluate_experts()
        else:run()
    except BaseException as exc:
        stage('error',error=repr(exc));raise
