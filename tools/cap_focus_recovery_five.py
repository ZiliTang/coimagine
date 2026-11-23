"""Five cold-start CAP rounds on one unresolved parent and four visible perturbations."""
import ast
import contextlib
import copy
import fcntl
import gzip
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2.cap_first_disagreement import legal_shift
from tools import cap_regional_five as base
from tools.cap_regional_extension import input_hash
from tools.cap_regional_token_meter import tail_usage

read,save=base.read,base.save
OUT=base.OUT.parent/'cap_focus_recovery_five_20261120_01'
WORK=base.WS.parent/'focus_recovery_five_01'
CAM=OUT/'author'
STATE=OUT/'state.json'
HOST=base.host.RoomCampaign
GUIDES=base.host.GUIDES
MAX_TOKENS=40_000_000
MAX_SECONDS=3600
ROUND_SECONDS=720
PYTHON=base.ROOT/'outputs/transfers/venv_py311/bin/python'
AGENT='/root/focus_sol'
REPORT=base.ROOT/'docs/research/key_findings/2026-11-20_focus_recovery_five'


def shifts(f,seed):
    rng=random.Random(seed)
    result={}
    for i in range(16):
        rad=(.01,.02,.03,.05)[i%4]
        for _ in range(2000):
            g=copy.deepcopy(f)
            for bot in g['robots']:
                ang=rng.uniform(-math.pi,math.pi)
                bot['x']+=rad*math.cos(ang)
                bot['y']+=rad*math.sin(ang)
            if legal_shift(g) and legal_shift(dict(g,robots=g['robots'][::-1])):
                break
        else:
            raise RuntimeError('failed to generate a valid paired robot shift')
        result[('v' if i<4 else 'h')+str(i if i<4 else i-4)]=g
    return result


def prepare():
    dev=read(base.WS/'regional/development_fixtures.json')
    assert base.digest(dev)==read(base.OUT/'protocol.json')['dev_sha']
    if STATE.exists():return
    OUT.mkdir(parents=True,exist_ok=True)
    WORK.mkdir(parents=True,exist_ok=True)
    save(OUT/'dev.json',dev)
    families={k:shifts(dev[k],11302000+i) for i,k in enumerate(sorted(dev))}
    hashes=[input_hash(f) for f in dev.values()]+[input_hash(f) for fam in families.values() for f in fam.values()]
    assert len(set(hashes))==len(hashes)
    save(OUT/'families.json',families)
    c=HOST.create(CAM,dev)
    with c.locked():
        c.state['limits'].update(attempts=100000,resets=100000,feature_batches=30100000,execution_ticks=180000000)
        c.state['author_protocol'].update(wall_seconds=MAX_SECONDS,aggregate_tokens=MAX_TOKENS,rounds=5,clean_start=True)
        c._save()
    for i,p in enumerate(GUIDES):
        dest=WORK/'guide'/f'{i}_{Path(p).name}'
        dest.parent.mkdir(exist_ok=True)
        shutil.copyfile(base.ROOT/p,dest)
    lab=WORK/'lab'
    lab.write_text(f'#!/bin/bash\ncd {base.ROOT} || exit 1\nexport OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1\nexec {PYTHON} -m tools.cap_focus_recovery_five lab "$@"\n')
    lab.chmod(0o755)
    save(OUT/'protocol.json',dict(dev=base.digest(dev),families=base.digest(families),model='gpt-6.1-sol high',rounds=5,seconds=MAX_SECONDS,tokens=MAX_TOKENS,first_four=[ROUND_SECONDS,8000000],last='all remaining time and tokens, total wall includes intermediate sweeps',visible_radii_cm=[1,2,3,5],hidden='three fresh paired robot translations per radius per parent; precomputed before first call',selection='author selects an actual uncovered not-yet-focused development root; sorted shortlist uses density of uncovered population then distance to past centers',target='single member solves center and all four visible perturbations, report incomplete local families honestly',readback='one 160-root census per committed version, author sees remaining choices only; hidden family test after library freeze',empty='no historical source or policies; only current campaign sources may be reused',deadline='best fully tested current-round candidate on time cap; distinct code required'))
    save(STATE,dict(rounds=[],active=None,start=None,stage='prepared'))
    print(WORK,flush=True)


def session():
    items=sorted((Path.home()/'.codex/sessions').glob('*/*/*/*.jsonl'),key=lambda p:p.stat().st_mtime,reverse=True)
    for p in items[:40]:
        with p.open() as f:
            m=json.loads(f.readline())['payload']
        if m.get('agent_path')==AGENT:
            return p
    return None


def used(s):
    file=s.get('telemetry') or session()
    return int(((tail_usage(Path(file)) if file else None) or {}).get('total_tokens') or 0)


def shortlist(s):
    regions=read(base.OUT/'regions.json')
    ids=regions['ids']; x=np.array(regions['x'])
    z=(x-x.mean(0))/np.maximum(.1,x.std(0))
    ds=np.linalg.norm(z[:,None]-z[None,:],axis=2)
    wins={k for r in s['rounds'] for k in r['wins']}
    prior=[ids.index(r['center']) for r in s['rounds']]
    eligible=[i for i,k in enumerate(ids) if k not in wins and i not in prior]
    ranking=[]
    for i in eligible:
        area=np.argsort(ds[i],kind='stable')[:24]
        density=sum(ids[j] not in wins for j in area)
        separation=min(ds[i,prior]) if prior else 0.
        ranking.append(((-density,-separation,max(ds[i,area]),i),dict(id=ids[i],open_among_24=density,public_geometry=x[i].round(2).tolist())))
    return [r for _,r in sorted(ranking)]


def start_round(s):
    now=time.time()
    if s['start'] is None:s['start']=now
    idx=len(s['rounds'])+1
    total=used(s)
    options=shortlist(s)
    if not options:raise RuntimeError('no remaining distinct unresolved center')
    s['active']=dict(number=idx,start=now,token_start=total,token_end=min(MAX_TOKENS,total+8000000) if idx<5 else MAX_TOKENS,
        deadline=min(s['start']+MAX_SECONDS,now+ROUND_SECONDS) if idx<5 else s['start']+MAX_SECONDS,
        options=options,center=None,trials=[])
    s['stage']='author'
    save(STATE,s)


def status(s):
    a=s['active']; tokens=used(s)
    return dict(stage=s['stage'],round=a['number'] if a else len(s['rounds']),tokens=tokens,
        seconds_left=round(max(0,a['deadline']-time.time()),1) if a else 0,
        tokens_left=max(0,a['token_end']-tokens) if a else 0,
        center=a['center'] if a else None,candidates=a['options'][:8] if a and not a['center'] else [],
        trials=a['trials'] if a else [],previous=[{k:r[k] for k in ('number','center','sid','visible','center_success','added','total')} for r in s['rounds']])


def family(center,hidden=False):
    allf=read(OUT/'families.json')[center]
    if hidden:return {k:v for k,v in allf.items() if k.startswith('h')}
    return {'center':read(OUT/'dev.json')[center],**{k:v for k,v in allf.items() if k.startswith('v')}}


def submit(file,label):
    src=file.read_text()
    names=next(ast.literal_eval(t.value) for t in ast.parse(src).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
    cfg=base.stream_config(src)
    group=dict(candidate={n:'A' for n in names},state={n:'A' for n in cfg['state_names']})
    for row in read(CAM/'campaign.json')['submissions']:
        if row['sha256']==base.digest(src):return row['id']
    return base.host.RoomSession(CAM).tool('submit',dict(source=src,groups=group,summary=label))['id']


def execute(sid,fixtures,directory,workers):
    return base.run_batch((CAM/'semantics'/(sid+'.py')).read_text(),fixtures,directory,workers)


def budget_ok(s,a):
    return time.time()<a['deadline'] and used(s)<a['token_end']


def lab(args):
    cmd,*args=args
    s=read(STATE)
    if cmd=='status':return status(s)
    a=s['active']
    stop=OUT/'stop_after.json'
    if stop.exists() and (not a or a['number']>read(stop)['after_round']):
        raise ValueError('authoring closed at the user-selected stopping round')
    if not a or int(args.pop(0))!=a['number'] or s['stage']!='author':raise ValueError('not the current author round')
    if not budget_ok(s,a):raise TimeoutError('round closed')
    n=a['number']
    if cmd=='choose':
        cid=args[0]
        if a['center'] or cid not in {r['id'] for r in a['options']}:raise ValueError('choose one eligible center once')
        a['center']=cid
        save(WORK/f'public_r{n}.json',base.public(read(OUT/'dev.json')[cid]))
        save(STATE,s);return status(s)
    if not a['center']:raise ValueError('choose a center first')
    if cmd=='inspect':
        sid,cid=args[:2]
        if sid not in {r['sid'] for r in a['trials']} or cid not in ['center','v0','v1','v2','v3']:raise ValueError('unknown completed trial')
        with gzip.open(OUT/'probes'/f'r{n}'/sid/(cid+'.json.gz'),'rt') as f:
            rows=json.load(f)
        return rows[int(args[2]):int(args[3])] if len(args)>3 else rows
    if cmd=='try':
        path=(WORK/args[0]).resolve()
        if not path.is_relative_to(WORK) or not path.is_file():raise ValueError('source file in author work directory required')
        sid=submit(path,' '.join(args[1:]) or path.name)
        if sid in {r['sid'] for r in s['rounds']}:raise ValueError('previously committed member')
        for trial in a['trials']:
            if trial['sid']==sid:return trial
        rows=execute(sid,family(a['center']),OUT/'probes'/f'r{n}'/sid,5)
        if any(not r['complete'] for r in rows):return dict(sid=sid,errors=rows)
        rec=dict(sid=sid,finished=time.time(),wins=[r['id'] for r in rows if r['success']],ticks={r['id']:r['ticks'] for r in rows})
        if not budget_ok(s,a):return dict(late=True,sid=sid)
        a['trials'].append(rec);save(STATE,s);return rec
    if cmd=='finish':
        sid=args[0]
        if sid not in [t['sid'] for t in a['trials']]:raise ValueError('requires a fully tested version')
        close=dict(at=time.time(),tokens=used(s),round=n,sid=sid)
        save(OUT/'author_closes'/f'r{n}.json',close)
        if n==5:save(OUT/'author_end.json',close)
        a['choice']=sid;a['reason']=' '.join(args[1:]);s['stage']='commit';save(STATE,s)
        return dict(accepted=sid,round=n)
    raise ValueError('status / choose ROUND ID / try ROUND FILE / inspect ROUND SID CASE START END / finish ROUND SID rationale')


@contextlib.contextmanager
def lock():
    with (OUT/'state.lock').open('a') as stream:
        fcntl.flock(stream,fcntl.LOCK_EX)
        yield


def advance():
    with lock():
        s=read(STATE);a=s['active']
        if s['stage']=='stopped':return
        if stop_requested(s):
            stop_after_census(s)
            return
        if s['stage']=='prepared':start_round(s);return
        if not s.get('telemetry'):
            path=session()
            if path:s['telemetry']=str(path);save(STATE,s)
        if s['stage']=='author' and budget_ok(s,a):return
        if s['stage'] in ('complete','incomplete','frozen'):return
        if not a['trials']:
            s['stage']='incomplete';s['error']='round closed without an on-time tested member';save(STATE,s);return
        preferred=max(a['trials'],key=lambda t:('center' in t['wins'],len(t['wins']),-sum(t['ticks'].values())))
        sid=a.get('choice',preferred['sid'])
        trial=next(t for t in a['trials'] if t['sid']==sid)
        s['stage']='census';save(STATE,s)
    rows=execute(sid,read(OUT/'dev.json'),OUT/'census'/sid,16)
    if not all(r['complete'] for r in rows):raise RuntimeError('incomplete census')
    wins=[r['id'] for r in rows if r['success']]
    union={k for r in s['rounds'] for k in r['wins']}
    assert (a['center'] in wins)==('center' in trial['wins'])
    r=dict(number=a['number'],sid=sid,center=a['center'],wins=wins,added=len(set(wins)-union),total=len(set(wins)|union),center_success='center' in trial['wins'],visible=len(trial['wins']),
        trials=a['trials'],closed_reason=a.get('reason','deadline; best completed trial'),started=a['start'],accepted=trial['finished'],token_start=a['token_start'],tokens_after_census=used(s))
    shutil.copyfile(CAM/'semantics'/(sid+'.py'),WORK/f"member_r{r['number']}.py")
    s['rounds'].append(r);s['active']=None
    if stop_requested(s):
        stop_after_census(s)
    elif len(s['rounds'])==5:
        s['stage']='frozen';s['seal']=HOST(CAM).freeze();save(STATE,s)
    elif time.time()>=s['start']+MAX_SECONDS:
        s['stage']='incomplete';save(STATE,s)
    else:start_round(s)


def stop_requested(s):
    marker=OUT/'stop_after.json'
    return marker.exists() and len(s['rounds'])>=read(marker)['after_round']


def stop_after_census(s):
    request=read(OUT/'stop_after.json')
    assert len(s['rounds'])==request['after_round']
    assert not (s.get('active') or {}).get('trials')
    c=HOST(CAM)
    seal=c.verify_frozen() if read(CAM/'campaign.json')['status']=='frozen' else c.freeze()
    s.update(stage='stopped',active=None,seal=seal,stopped_at=time.time(),cancelled=request['cancel'])
    save(STATE,s)


def monitor():
    while True:
        advance()
        s=read(STATE)
        if s['stage'] in ('incomplete','frozen','complete','stopped'):return
        time.sleep(1)


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':prepare()
    elif cmd=='monitor':monitor()
    elif cmd=='lab':
        if sys.argv[2]=='status':print(json.dumps(status(read(STATE)),ensure_ascii=False))
        else:
            with lock():print(json.dumps(lab(sys.argv[2:]),ensure_ascii=False))
