"""Five continuation rounds focused on the residual families of the regional CAP library."""
import ast
import fcntl
import gzip
import json
import random
import shutil
import sys
import time
from pathlib import Path

from tools import cap_regional_five as old
from tools.cap_regional_evaluate import sha
from tools.cap_regional_token_meter import tail_usage

OUT=old.OUT.parent/'cap_regional_extension_20261119_01'
WORK=old.WS.parent/'regional_extension_v1'
CAM=OUT/'author'
read,save=old.read,old.save
MAX_TOKENS=40_000_000
MAX_SECONDS=1800
DEV='development.json'
OLD=read(old.OUT/'regional/study.json')['sweeps']
SIDES=[r['sid'] for r in OLD]
PYTHON=old.ROOT/'outputs/transfers/venv_py311/bin/python'


def input_hash(f):
    return old.digest({k:v for k,v in f.items() if k!='scene_id'})


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);WORK.mkdir(parents=True,exist_ok=True)
    if not (OUT/'protocol.json').exists():
        dev=read(old.WS/'regional/development_fixtures.json')
        previous=read(old.OUT/'frozen.json')
        assert old.digest(dev)==read(old.OUT/'protocol.json')['dev_sha']
        assert all(previous['policies']['regional_'+s]==sha(old.OUT/'regional/semantics'/(s+'.py')) for s in SIDES)
        covered=set().union(*(r['wins'] for r in OLD)); parents=sorted(set(dev)-covered)
        assert len(parents)==45
        held={};extra={};families={}
        chosen=parents+random.Random(6111901).sample(parents,10)
        for i,k in enumerate(chosen):
            name=f'aug_{k.split("_")[1]}_{40+i:02}';families[name]=k
            extra[name]=old.variant(dict(case_id=k,fixture=dev[k]),6112000+i)
        for r in old.rooms.generate(6113000)['cases']:held['fresh_'+r['id']]=r['fixture']
        for i,k in enumerate(parents):
            for j in range(2):
                held[f'near_{k}_{j}']=old.variant(dict(case_id=k,fixture=dev[k]),6114000+i*2+j)
        used={r['content_sha256'] for r in read(old.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')}
        used.update(read(old.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
        used.update(input_hash(f) for f in [*dev.values(),*read(old.OUT/'held.json').values()])
        for pat in ('*/panel.json','*/evaluation_panel.json','*/batches/*_plan.json','cap_semantic_disagreement*/*_panel.json'):
            for p in old.OUT.parent.glob(pat):
                data=read(p); rows=data if isinstance(data,list) else data.get('cases',[])
                used.update(r['content_sha256'] for r in rows if isinstance(r,dict) and 'content_sha256' in r)
        for f in [*extra.values(),*held.values()]:
            key=input_hash(f);assert key not in used;used.add(key)
        save(OUT/DEV,dev|extra);save(OUT/'held.json',held)
        save(OUT/'protocol.json',dict(old_sids=SIDES,parents=parents,families=families,dev_sha=old.digest(dev|extra),
             held_sha=old.digest(held),time_limit=MAX_SECONDS,token_limit=MAX_TOKENS,model='gpt-6.1-sol high',
             extra_rounds=5,seed=17,train_episodes=2150,v_updates=16000,q_updates=8000,best='original 160; earliest tie',
             evaluation='fresh 200 and two new neighbors per original residual parent; old 5 versus expanded 10 and old versus new directQ'))
    p=read(OUT/'protocol.json');dev=read(OUT/DEV)
    assert old.digest(dev)==p['dev_sha']
    for r in OLD:
        sid=r['sid'];source=(old.OUT/'regional/semantics'/(sid+'.py')).read_text()
        rows=old.run_batch(source,{k:dev[k] for k in p['families']},OUT/'baseline'/sid,18)
        assert len(rows)==55 and all(r['complete'] for r in rows)
    if not (CAM/'campaign.json').exists():
        c=old.host.RoomCampaign.create(CAM,dev)
        with c.locked():
            c.state['limits'].update(attempts=100000,resets=100000,feature_batches=30100000,execution_ticks=180000000)
            c.state['author_protocol'].update(max_library_policies=5,wall_seconds=MAX_SECONDS)
            c._save()
    author_metadata()
    baseline={r['sid']:sorted(set(r['wins'])|{k for k in p['families'] if read(OUT/'baseline'/r['sid']/(k+'.json'))['success']}) for r in OLD}
    save(OUT/'baseline.json',baseline)
    shutil.copytree(old.WS/'regional/guide',WORK/'guide',dirs_exist_ok=True)
    for s in SIDES:shutil.copyfile(old.OUT/'regional/semantics'/(s+'.py'),WORK/(s+'.py'))
    save(WORK/'families.json',dict(parents=p['parents'],neighbor_to_parent=p['families'],old_wins=baseline))
    save(WORK/DEV,dev)
    lab=WORK/'lab';lab.write_text(f'#!/bin/bash\ncd {old.ROOT} || exit 1\nexport OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1\nexec {PYTHON} -m tools.cap_regional_extension lab "$@"\n');lab.chmod(0o755)
    save(OUT/'ready.json',dict(ready=True))


def author_metadata():
    c=old.host.RoomCampaign(CAM)
    with c.locked():
        c.state['model']['clean_context']=False
        c.state['author_protocol'].update(no_prior_semantics=False,previous_frozen_members=SIDES,
            context='same regional author; own five prior programs and public guides',
            wall_seconds=MAX_SECONDS,aggregate_tokens=MAX_TOKENS,
            token_accounting='delta of cumulative input including cache and output including reasoning',
            final='five additional distinct submitted sources with full 215-room sweeps')
        c._save()
        save(CAM/'knowledge/transfer_execution_protocol.json',dict(stage1=c.state['stage1'],
             limits=c.state['limits'],author_protocol=c.state['author_protocol']))


def clock_start():
    if (OUT/'clock.json').exists():raise FileExistsError('author clock already started')
    assert (OUT/'ready.json').exists()
    sid=read(old.OUT/'author_session_ids.json')['regional']
    path=next((Path.home()/'.codex/sessions').rglob('*'+sid+'.jsonl'))
    now=time.time(); baseline=tail_usage(path)
    save(OUT/'clock.json',dict(start=now,deadline=now+MAX_SECONDS,usage=baseline,session=str(path)))
    c=old.host.RoomCampaign(CAM)
    with c.locked():c.state['deadline']=now+MAX_SECONDS;c._save()
    save(OUT/'author_state.json',dict(sweeps=[],probes=0,active=None))


def usage():
    clock=read(OUT/'clock.json');last=tail_usage(Path(clock['session']))
    return dict(tokens=last['total_tokens']-clock['usage']['total_tokens'],seconds=max(0,time.time()-clock['start']),
                seconds_left=max(0,clock['deadline']-time.time()),deadline=clock['deadline'])


def validate_budget():
    b=usage()
    if b['seconds_left']<=0 or b['tokens']>=MAX_TOKENS-300000:raise TimeoutError('author budget closed')
    return b


def sourcepath(sid):
    if sid in SIDES:return old.OUT/'regional/semantics'/(sid+'.py')
    if sid.startswith('n_s') and sid[3:].isdigit():return CAM/'semantics'/(sid[2:]+'.py')
    raise ValueError('unknown submission')


def refs(sid,cid):
    if sid in SIDES:
        if cid.startswith('aug_'):return OUT/'baseline'/sid
        return old.OUT/'regional/logs'/next(r['key'] for r in OLD if r['sid']==sid)
    return OUT/'sweeps'/sid


def status(state):
    p=read(OUT/'protocol.json'); previous=read(OUT/'baseline.json')
    union=set().union(*previous.values(),*(r['wins'] for r in state['sweeps']))
    hard=set(p['parents'])|set(p['families'])
    return dict(budget=usage(),rounds=[{k:v for k,v in r.items() if k!='wins'} for r in state['sweeps']],
                open_parents=sorted(set(p['parents'])-union),open_neighbors=sorted(set(p['families'])-union),
                coverage_original=len([k for k in union if k.startswith('dev_')]),coverage_hard=len(union&hard),
                probes=state['probes'],active=state['active'])


def lab(args):
    cmd,*args=args;state=read(OUT/'author_state.json')
    if cmd=='status':return status(state)
    validate_budget()
    if len(state['sweeps'])==5:raise ValueError('five rounds sealed')
    if cmd=='inspect':
        sid,cid=args[:2]
        candidates=[refs(sid,cid),*sorted((OUT/'probes'/sid).glob('*'),reverse=True)]
        path=next(p/(cid+'.json.gz') for p in candidates if (p/(cid+'.json.gz')).exists())
        with gzip.open(path,'rt') as f:rows=json.load(f)
        return rows[int(args[2]) if len(args)>2 else 0:int(args[3]) if len(args)>3 else None]
    if cmd=='submit':
        file=(WORK/args[0]).resolve()
        if not file.is_relative_to(WORK.resolve()):raise ValueError('author workspace only')
        src=file.read_text()
        hashes={sha(sourcepath(s)) for s in SIDES}
        if sha(file) in hashes:raise ValueError('unchanged old member')
        for item in read(CAM/'campaign.json')['submissions']:
            if item['sha256']==old.digest(src):return dict(sid='n_'+item['id'])
        cfg=old.stream_config(src)
        names=next(ast.literal_eval(t.value) for t in ast.parse(src).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
        groups=dict(candidate={n:'A' for n in names},state={n:'A' for n in cfg['state_names']})
        result=old.host.RoomSession(CAM).tool('submit',dict(source=src,summary=' '.join(args[1:]) or file.name,groups=groups))
        return dict(sid='n_'+result['id'])
    if cmd not in ('test','sweep'):raise ValueError('status / submit FILE SUMMARY / test SID IDS / inspect SID ID [start end] / sweep SID')
    sid=args[0];dev=read(OUT/DEV)
    if cmd=='sweep' and (sid in SIDES or sid in [r['sid'] for r in state['sweeps']]):raise ValueError('requires a distinct new version')
    ids=args[1:] if cmd=='test' else sorted(dev)
    if not ids or len(set(ids))!=len(ids) or set(ids)-set(dev):raise ValueError('use distinct development IDs')
    source=sourcepath(sid).read_text()
    directory=OUT/'sweeps'/sid if cmd=='sweep' else OUT/'probes'/sid/str(time.time_ns())
    state['active']=dict(kind=cmd,sid=sid,total=len(ids),directory=str(directory),accepted=time.time())
    save(OUT/'author_state.json',state)
    c=old.host.RoomCampaign(CAM)
    with c.locked():
        c._check()
        c._charge(attempts=len(ids),resets=len(ids),feature_batches=301*len(ids),execution_ticks=1800*len(ids))
        c._save()
    rows=old.run_batch(source,{k:dev[k] for k in ids},directory,min(20,len(ids)))
    with c.locked():
        c.state['used']['execution_ticks']-=1800*len(ids)-sum(r['ticks'] for r in rows)
        c.state['used']['feature_batches']-=301*len(ids)-sum(r['decisions']+int(bool(r['error'])) for r in rows)
        c._save()
    if not all(r['complete'] for r in rows):raise RuntimeError('incomplete '+str(directory))
    wins=sorted(r['id'] for r in rows if r['success'])
    if cmd=='sweep':
        union=set().union(*read(OUT/'baseline.json').values(),*(r['wins'] for r in state['sweeps']))
        state['sweeps'].append(dict(sid=sid,wins=wins,original=sum(k.startswith('dev_') for k in wins),
            added_original=sum(k.startswith('dev_') for k in set(wins)-union),added_neighbors=sum(k.startswith('aug_') for k in set(wins)-union),
            accepted=state['active']['accepted'],finished=time.time(),hash=sha(sourcepath(sid))))
    else:state['probes']+=len(rows)
    state['active']=None;save(OUT/'author_state.json',state)
    return (status(state) if cmd=='sweep' else [dict(id=r['id'],success=r['success'],ticks=r['ticks'],decisions=r['decisions']) for r in rows])


if __name__=='__main__':
    cmd=sys.argv[1]
    if cmd=='prepare':prepare()
    elif cmd=='start':clock_start()
    else:
        with (OUT/'lab.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            print(json.dumps(lab(sys.argv[2:])))
