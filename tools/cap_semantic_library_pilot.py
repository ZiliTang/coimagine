"""Matched author workspaces and capped random-room CAP probes for the semantic library pilot."""
import argparse
import ast
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path

from tools import cap_general_specialists as previous
from tools import cap_regional_five as base

OUT=previous.OUT.parent/'cap_semantic_library_20261123_01'
WORK=previous.WORK.parent/'semantic_library_20261123_01'
ARMS=('usual','specialized')
FIELDS=('observed','static_hit','movable_hit','target_estimated','swept_visibility','observed_clear','unknown','visible_other_movables','goal_wall_distance_m')
OPS={'eq':lambda a,b:a==b,'ge':lambda a,b:a>=b,'le':lambda a,b:a<=b}
read,save=base.read,base.save


def matches(rule, row):
    return all(OPS[op](row[k],v) for k,op,v in rule['all'])


def prepare():
    dev=read(previous.OUT/'development.json')
    assert len(dev)==160 and base.digest(dev)==read(previous.OUT/'protocol.json')['pool']
    source=previous.CAM/'semantics/s025.py'
    assert hashlib.sha256(source.read_bytes()).digest()==hashlib.sha256((previous.WORK/'member_r1.py').read_bytes()).digest()
    archive=base.ROOT/'docs/research/key_findings/2026-11-22_corridor_subtypes/rooms.csv'
    start={}
    with archive.open() as f:
        for r in csv.DictReader(f):
            if r['room'] not in dev:continue
            obs={k:r[k] if k=='observed' else float(r[k]) for k in FIELDS}
            y=read(previous.OUT/'runs/s025'/(r['room']+'.json'))
            assert y['complete'] and not y['error'] and r['s025']==str(int(y['success']))
            start[r['room']]=dict(**obs,base_success=int(y['success']))
    assert len(start)==160
    OUT.mkdir(exist_ok=True)
    save(OUT/'development.json',dev)
    save(OUT/'initial.json',start)
    # Two identical nonadaptive 32-room probe panels, stratified by observed corridor.
    rng=random.Random(23117)
    panel=[]
    for group,n in [('clear',13),('movable',9),('static',5),('both',2),('unknown',3)]:
        panel.extend(rng.sample(sorted(k for k,r in start.items() if r['observed']==group),n))
    save(OUT/'probe.json',sorted(panel))
    for arm in ARMS:
        path=WORK/arm
        path.mkdir(exist_ok=True,parents=True)
        shutil.copyfile(source,path/'basis.py')
        shutil.copytree(previous.WORK/'guide',path/'guide',dirs_exist_ok=True)
        shutil.copytree(previous.WORK/'option_source',path/'option_source',dirs_exist_ok=True)
        shutil.copyfile(previous.WORK/'initial_public.json',path/'initial_public.json')
        shutil.copyfile(base.ROOT/'experiments/planar_tasks/semantic_rl/corridor_semantics.py',path/'corridor_reference.py')
        save(path/'development.json',start)
        save(path/'probe_ids.json',sorted(panel))
        lab=path/'lab'
        lab.write_text(f'#!/bin/bash\ncd {base.ROOT} || exit 1\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec {previous.prior.PYTHON} -m tools.cap_semantic_library_pilot {arm} "$@"\n')
        lab.chmod(0o755)
        if not (OUT/arm/'campaign.json').exists():
            campaign=base.host.RoomCampaign.create(OUT/arm,dev)
            with campaign.locked():
                campaign.state['model'].update(name='gpt-6.2-sol',reasoning_effort='high',clean_context=True)
                campaign.state['author_protocol'].update(shared_initial='s025',max_library_policies=4,probes_per_arm=192)
                campaign._save()


def register(arm):
    path=WORK/arm/'conditions.json'
    if arm!='specialized':raise ValueError('conditions belong to specialist arm')
    rules=read(path)
    rows=read(OUT/'initial.json')
    if not 2<=len(rules)<=3:raise ValueError('choose 2 or 3 conditions')
    seen=set();info=[]
    for rule in rules:
        if not 1<=len(rule['all'])<=2 or any(k not in FIELDS or op not in OPS for k,op,v in rule['all']):
            raise ValueError('each condition uses 1 to 2 public comparisons')
        hit={k for k,r in rows.items() if matches(rule,r)}
        if len(hit)<12 or not hit-seen or any(hit==set(r['ids']) for r in info):
            raise ValueError('need >=12 rooms and distinct coverage')
        seen.update(hit)
        info.append(dict(name=rule['name'],ids=sorted(hit),n=len(hit),fail=sum(not rows[k]['base_success'] for k in hit)))
    lock=OUT/'conditions.json'
    if lock.exists() and read(lock)!=rules:raise ValueError('conditions already frozen')
    save(lock,rules)
    return [{k:v for k,v in r.items() if k!='ids'} for r in info]


def trial(arm,member,revision):
    if member not in (1,2,3) or revision not in (1,2):raise ValueError('invalid slot')
    if arm=='specialized' and not (OUT/'conditions.json').exists():raise ValueError('freeze scope first')
    src=(WORK/arm/f'm{member}v{revision}.py').read_text()
    names=next(ast.literal_eval(t.value) for t in ast.parse(src).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
    cfg=base.stream_config(src)
    groups=dict(candidate=dict.fromkeys(names,'A'),state=dict.fromkeys(cfg['state_names'],'A'))
    scope=OUT/arm
    log=scope/f'probe_m{member}v{revision}.json'
    if log.exists():
        rec=read(log)
        if rec['sha']!=base.digest(src):raise ValueError('immutable slot')
        return rec
    s=base.host.RoomSession(scope)
    sid=s.tool('submit',dict(source=src,summary=f'm{member}v{revision}',groups=groups))['id']
    ids=read(OUT/'probe.json');dev=read(OUT/'development.json')
    with s.c.locked():
        s.c._charge(attempts=len(ids),resets=len(ids),execution_ticks=1800*len(ids),feature_batches=301*len(ids));s.c._save()
    rows=base.run_batch(src,{k:dev[k] for k in ids},scope/'compact'/sid,6)
    with s.c.locked():
        s.c.state['used']['execution_ticks']-=1800*len(ids)-sum(r['ticks'] for r in rows)
        s.c.state['used']['feature_batches']-=301*len(ids)-sum(r['decisions']+int(bool(r['error'])) for r in rows)
        s.c._save()
    rec=dict(sid=sid,sha=base.digest(src),wins=[r['id'] for r in rows if r['success']],errors=[r for r in rows if not r['complete']],total=len(ids))
    save(log,rec)
    return rec


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    else:
        arm,cmd,*args=sys.argv[1:]
        assert arm in ARMS
        print(json.dumps(register(arm) if cmd=='freeze' else trial(arm,*map(int,args)) if cmd=='probe' else None))
