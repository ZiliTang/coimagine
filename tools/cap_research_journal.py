"""Read-only experiment journal views for the existing localhost dashboard."""
import json
import mimetypes
from functools import lru_cache
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
FINDINGS=ROOT/'docs/research/key_findings'
REGISTRY=ROOT/'docs/research/experiment_journal/entries.json'
EVENT_COMPARISON='abc-ppo-event-comparison'
EVENT_ARMS=('original','time','progress')


def load(path):
    return json.loads(path.read_text())


@lru_cache(maxsize=4096)
def event_metric(path, modified_ns, batch_size):
    row=load(path)
    episodes=int(path.stem[1:])*batch_size
    if path.parent.parent.name=='diagnostic':
        return dict(x=episodes,success=row['success'],n=row['n'])
    return dict(x=episodes,success=row['success'],n=row['episodes'],
                reward=row['mean_dense_reward'],entropy=row['entropy'],
                actor_loss=row['actor_loss'],critic_loss=row['critic_loss'],
                switches=row['reviews'].get('switched',0))


def event_curves(entry):
    root=ROOT/'outputs/semantic_rl'/entry['run']
    protocol=root/'protocol.json'
    plan=load(protocol) if protocol.is_file() else {}
    batch=plan.get('batch_size',50)

    def series(kind,arm):
        return [event_metric(path,path.stat().st_mtime_ns,batch)
                for path in sorted((root/kind/arm).glob('r*.json'))]

    return dict(updated=time.time(),budget=len(plan.get('room_seeds',[]))*batch,
                finished=(root/'finished.json').is_file(),error=(root/'ERROR.json').is_file(),
                arms={arm:dict(training=series('metrics',arm),diagnostic=series('diagnostic',arm))
                      for arm in EVENT_ARMS})


def progress(entry):
    root=ROOT/'outputs/semantic_rl'/entry.get('run','')
    rows=[]
    for step in entry.get('progress',[]):
        if 'glob' in step:
            done=sum(p.is_file() for p in root.glob(step['glob']))
        elif 'rows' in step:
            path=root/step['rows']
            done=len(load(path)) if path.is_file() else 0
        elif 'counter' in step:
            path=root/step['counter']
            done=int(load(path).get(step['key'],0)) if path.is_file() else 0
        else:
            path=(FINDINGS/entry['folder']/step['finding'] if 'finding' in step else root/step['file'])
            done=int(path.is_file())
        rows.append((step['label'],done,step['total']))
    return rows


def allowed(e):
    return [['README.md','报告'],*e.get('assets',[])]


def state():
    entries=load(REGISTRY)
    for e in entries:
        folder=FINDINGS/e['folder']
        e['links']=[dict(url='/journal-file/'+e['id']+'/'+f,title=t) for f,t in allowed(e) if (folder/f).is_file()]
        phases=progress(e)
        e['phases']=phases
        if phases:
            complete=all(n==limit for _,n,limit in phases)
            e['status']='已完成' if complete else ('汇总中' if all(v==n for _,v,n in phases[:-1]) else '进行中')
        else:e['status']=e.get('status','已完成')
        if e.get('run') and (ROOT/'outputs/semantic_rl'/e['run']/'ERROR.json').is_file():
            e['status']='运行异常'
    return dict(updated=time.time(),entries=entries)


def respond(path):
    if path=='/research-journal':
        return Path(__file__).with_suffix('.html').read_bytes(),'text/html'
    if path=='/research-journal-data':
        return json.dumps(state(),ensure_ascii=False).encode(),'application/json'
    parts=path.split('/')
    if len(parts)==4 and parts[1]=='journal-file':
        e=next((e for e in load(REGISTRY) if e['id']==parts[2]),None)
        if e and e['id']==EVENT_COMPARISON and parts[3]=='live-data.json':
            return json.dumps(event_curves(e),separators=(',',':')).encode(),'application/json'
        if e and parts[3] in {f for f,_ in allowed(e)}:
            p=FINDINGS/e['folder']/parts[3]
            if p.is_file():
                mime='text/plain' if p.suffix=='.md' else mimetypes.guess_type(p.name)[0] or 'application/octet-stream'
                return p.read_bytes(),mime
    return None
