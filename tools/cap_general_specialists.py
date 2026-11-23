"""Develop one general CAP and up to four specialists on its observed failure families."""
import ast
import collections
import contextlib
import fcntl
import gzip
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

from tools import cap_focus_recovery_five as prior
from tools import cap_regional_five as base
from tools.cap_regional_evaluate import sha
from tools.cap_regional_extension import input_hash
from tools.cap_regional_token_meter import tail_usage

OUT=base.OUT.parent/'cap_general_specialists_62_20261120_01'
WORK=base.WS.parent/'general_specialists_62_01'
CAM=OUT/'author'
STATE=OUT/'state.json'
read,save=base.read,base.save
AGENT='/root/general62'
SECONDS,TOKENS=3600,40_000_000
SEED=66211201
REFS=('experiments/planar_tasks/relational_search/motion.py','experiments/planar_tasks/relational_search/target_binding.py',
      'experiments/planar_tasks/semantic_rl/environment.py','experiments/option_rl_v21/spec.py',
      'experiments/option_rl_v21/yield_skill.py','experiments/option_rl_v21/system.py','experiments/option_rl_v19/system.py',
      'experiments/option_rl_v19/spec.py','experiments/option_rl_v19/unknown_shape.py','experiments/option_rl_v13/controller.py',
      'experiments/option_rl_v13/push_controller.py','src/coimagine/random_start/effect_push.py',
      'src/coimagine/random_start/skills.py','src/coimagine/random_start/peer_clearance.py')


def prepare():
    if STATE.exists():return
    dev=read(base.WS/'regional/development_fixtures.json')
    assert base.digest(dev)==read(base.OUT/'protocol.json')['dev_sha']
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    save(OUT/'development.json',dev)
    held={'new_'+r['id']:r['fixture'] for r in base.rooms.generate(SEED)['cases']}
    used={input_hash(f) for f in dev.values()}
    for path in (base.OUT/'held.json',prior.OUT/'families.json',base.OUT.parent/'cap_regional_extension_20261119_01/held.json'):
        if path.name=='families.json':
            used.update(input_hash(f) for v in read(path).values() for f in v.values())
        else:used.update(input_hash(f) for f in read(path).values())
    used.update(read(base.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    heldkeys={input_hash(f) for f in held.values()}
    assert len(heldkeys)==200 and not heldkeys & used
    save(OUT/'held.json',held)
    c=base.host.RoomCampaign.create(CAM,dev)
    with c.locked():
        c.state['model'].update(name='gpt-6.2-sol',reasoning_effort='high',clean_context=True)
        c.state['author_protocol'].update(wall_seconds=SECONDS,aggregate_tokens=TOKENS,max_library_policies=5,source_reference='read-only option source v1')
        c.state['limits'].update(attempts=100000,resets=100000,feature_batches=30100000,execution_ticks=180000000)
        c._save()
    for i,name in enumerate(prior.GUIDES):
        dest=WORK/'guide'/f'{i}_{Path(name).name}'
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(base.ROOT/name,dest)
    hashes={}
    for name in REFS:
        src=base.ROOT/name
        dest=WORK/'option_source'/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dest)
        dest.chmod(0o444)
        hashes[name]=sha(dest)
    save(CAM/'knowledge/option_source_manifest.json',hashes)
    save(WORK/'option_source_manifest.json',hashes)
    lab=WORK/'lab'
    lab.write_text(f'#!/bin/bash\ncd {base.ROOT} || exit 1\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec {prior.PYTHON} -m tools.cap_general_specialists lab "$@"\n')
    lab.chmod(0o755)
    save(WORK/'development_ids.json',sorted(dev))
    save(WORK/'initial_public.json',{k:base.public(f) for k,f in dev.items()})
    save(OUT/'protocol.json',dict(model='gpt-6.2-sol',reasoning='high',tokens=TOKENS,seconds=SECONDS,
        general=[20_000_000,1800],specialist=[5_000_000,450],last='remaining budget',pool=base.digest(dev),
        held=base.digest(held),held_n=200,seed=SEED,reference_sha=base.digest(hashes),
        basis='natural 160-room generalist, highest completed full sweep, earliest tie',
        groups='public execution and progress windows on frozen generalist failures across distinct parent rooms; retest variants, include nearby successes',
        mix='GA directQ seed17, V 16000/Q 8000, from all committed member traces on original development pool and expanded hard set',
        evaluation='same 200 new natural rooms: basis, each specialist, union, chosen best single and directQ',
        comparison='single 6.2 pilot; historical 6.1 and additional author arms not primary comparisons'))
    save(STATE,dict(stage='ready',rounds=[],active=None,start=None))


def session_file():
    files=sorted((Path.home()/'.codex/sessions').glob('*/*/*/*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in files[:40]:
        with p.open() as f:meta=json.loads(f.readline())['payload']
        if meta.get('agent_path')==AGENT:return p


def tokens(s):
    p=s.get('telemetry') or session_file()
    return ((tail_usage(Path(p)) or {}).get('total_tokens',0) if p else 0)


def start(s,ids,topic,roots=None):
    now=time.time(); n=len(s['rounds'])+1
    if not s['start']:s['start']=now
    t=tokens(s)
    a=dict(n=n,topic=topic,ids=ids,roots=roots or [],trials=[],start=now,token_start=t,
           deadline=min(s['start']+SECONDS,now+(1800 if n==1 else 450)) if n<5 else s['start']+SECONDS,
           token_end=min(TOKENS,t+(20_000_000 if n==1 else 5_000_000)) if n<5 else TOKENS)
    s.update(stage='author',active=a)
    save(STATE,s)


def status(s):
    a=s['active'] or {}; t=tokens(s)
    return dict(stage=s['stage'],round=a.get('n'),topic=a.get('topic'),default_ids=a.get('ids'),
        seconds_left=round(max(0,a.get('deadline',time.time())-time.time())),tokens=t,
        tokens_left=max(0,a.get('token_end',0)-t),trials=a.get('trials'),committed=[(r['sid'],len(r['wins']),r['added']) for r in s['rounds']])


def submit(file,label):
    src=file.read_text()
    existing=next((r for r in read(CAM/'campaign.json')['submissions'] if r['sha256']==base.digest(src)),None)
    if existing:return existing['id']
    names=next(ast.literal_eval(t.value) for t in ast.parse(src).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
    cfg=base.stream_config(src)
    groups=dict(candidate=dict.fromkeys(names,'A'),state=dict.fromkeys(cfg['state_names'],'A'))
    return base.host.RoomSession(CAM).tool('submit',dict(source=src,summary=label,groups=groups))['id']


def measure(sid,fixtures):
    path=OUT/'runs'/sid
    missing={k:v for k,v in fixtures.items() if not (path/(k+'.json')).exists()}
    if missing:
        c=base.host.RoomCampaign(CAM)
        with c.locked():
            n=len(missing)
            c._charge(attempts=n,resets=n,execution_ticks=1800*n,feature_batches=301*n)
        rows=base.run_batch((CAM/'semantics'/(sid+'.py')).read_text(),missing,path,16)
        with c.locked():
            c.state['used']['execution_ticks']-=1800*n-sum(r['ticks'] for r in rows)
            c.state['used']['feature_batches']-=301*n-sum(r['decisions']+int(bool(r['error'])) for r in rows)
            c._save()
    return [read(path/(k+'.json')) for k in fixtures]


@contextlib.contextmanager
def lock():
    with (OUT/'lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX);yield


def available(s,a):
    return time.time()<a['deadline'] and tokens(s)<a['token_end']-180000


def lab(args):
    cmd,*args=args;s=read(STATE)
    if cmd=='status':return status(s)
    if not args:raise ValueError('round number required')
    n=int(args.pop(0));a=s['active']
    if s['stage']!='author' or not a or n!=a['n'] or not available(s,a):raise ValueError('round closed, read status')
    if cmd=='inspect':
        sid,cid=args[:2]
        allowed={r['sid'] for r in a['trials']}|{r['sid'] for r in s['rounds']}
        dev=read(OUT/'development.json')
        if sid not in allowed or cid not in set(a['ids'])|set(dev):raise ValueError('not a development record')
        with gzip.open(OUT/'runs'/sid/(cid+'.json.gz'),'rt') as f:rows=json.load(f)
        return rows[int(args[2]):int(args[3])] if len(args)>=4 else rows[-12:]
    if cmd=='finish':
        if not any(t['eligible'] for t in a['trials']):raise ValueError('obtain a complete eligible measurement first')
        a['note']=' '.join(args);s['stage']='select';save(STATE,s);return 'accepted'
    if cmd not in ('try','sweep'):raise ValueError('try ROUND FILE [ids...] / sweep ROUND FILE / inspect ROUND SID ID [begin end] / finish ROUND NOTE')
    file=(WORK/args.pop(0)).resolve()
    if not file.is_relative_to(WORK) or not file.is_file():raise ValueError('author workspace source required')
    sid=submit(file,file.name)
    if sid in {r['sid'] for r in s['rounds']}:raise ValueError('new revision required')
    dev=read(OUT/'development.json'); fixtures=dict(dev)
    if (OUT/'extra.json').exists():fixtures.update(read(OUT/'extra.json'))
    ids=sorted(dev) if cmd=='sweep' else (args or a['ids'])
    if len(set(ids))!=len(ids) or not set(ids)<=fixtures.keys():raise ValueError('invalid development IDs')
    if n>1 and cmd=='try' and not set(ids)<=set(a['ids']):raise ValueError('use this round family')
    save(OUT/'activity.json',dict(round=n,sid=sid,ids=ids,stage='running',started=time.time()))
    rows=measure(sid,{k:fixtures[k] for k in ids})
    errors=[r for r in rows if not r['complete']]
    valid=not errors and available(s,a)
    wins=sorted(r['id'] for r in rows if r['success'] and r['complete'])
    eligible=valid and (set(ids)==set(dev) if n==1 else set(ids)==set(a['ids']))
    rec=dict(sid=sid,kind=cmd,total=len(ids),ids=ids,score=len(wins),wins=wins,eligible=eligible,
             errors=len(errors),finished=time.time())
    a['trials'].append(rec);save(STATE,s);save(OUT/'activity.json',dict(stage='idle'))
    return dict(rec,error_examples=errors[:2])


def default_probe(dev):
    rng=random.Random(601)
    return [k for n in range(2,6) for k in rng.sample(sorted(k for k,v in dev.items() if len(v['movable_objects'])==n-1),6)]


def groups_and_expansion(s):
    from tools.cap_general_specialists_families import construct
    construct(s)


def next_topic(s):
    from tools.cap_general_specialists_families import choose
    return choose(s)


def advance():
    with lock():
        s=read(STATE)
        if s['stage'] in ('frozen','incomplete'):return
        if s['stage']=='ready':
            start(s,default_probe(read(OUT/'development.json')),'general');return
        if not s.get('telemetry'):
            path=session_file()
            if path:s['telemetry']=str(path);save(STATE,s)
        a=s['active']
        if s['stage']=='author' and available(s,a):return
        opts=[t for t in a['trials'] if t['eligible']]
        if not opts:
            s.update(stage='incomplete',error='no eligible completed submission in budget');save(STATE,s);return
        best=max(opts,key=lambda t:t['score']);sid=best['sid']
        s['stage']='census';save(STATE,s)
    dev=read(OUT/'development.json')
    rows=measure(sid,dev)
    if not all(r['complete'] for r in rows):raise RuntimeError('incomplete member census')
    wins=sorted(r['id'] for r in rows if r['success'])
    prev={k for r in s['rounds'] for k in r['wins']}
    s['rounds'].append(dict(sid=sid,n=a['n'],topic=a['topic'],wins=wins,added=len(set(wins)-prev),
        union=len(set(wins)|prev),ids=a['ids'],roots=a['roots'],trials=a['trials'],started=a['start'],ended=best['finished'],note=a.get('note','budget cutoff')))
    shutil.copyfile(CAM/'semantics'/(sid+'.py'),WORK/f"member_r{a['n']}.py")
    s['active']=None;s['stage']='grouping';save(STATE,s)
    if a['n']==1:groups_and_expansion(s)
    next_group=next_topic(s) if a['n']<5 and time.time()<s['start']+SECONDS and tokens(s)<TOKENS-180000 else None
    if next_group:
        ids,topic,roots=next_group;start(s,ids,topic,roots)
    else:
        s.update(stage='frozen',seal=base.host.RoomCampaign(CAM).freeze(),end_tokens=tokens(s),end_wall=time.time()-s['start']);save(STATE,s)


if __name__=='__main__':
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='lab':
        if sys.argv[2]=='status':print(json.dumps(status(read(STATE))))
        else:
            with lock():print(json.dumps(lab(sys.argv[2:])))
    elif sys.argv[1]=='monitor':
        while True:
            advance()
            if read(STATE)['stage'] in ('frozen','incomplete'):break
            time.sleep(2)
