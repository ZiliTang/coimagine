"""Export only author usage counters, file receipt times and public study milestones."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time

from tools.cap_cluster_module import OUT, PUBLIC, read, save
from tools.cap_cluster_module_report import DOC

FIELDS=('input_tokens','cached_input_tokens','output_tokens','reasoning_output_tokens','total_tokens')
def stamp(s):
    return datetime.fromisoformat(s.replace('Z','+00:00')).timestamp()


def telemetry(session, agent_path='/root/cluster_author'):
    """Identify the model response for the tool call in which a candidate file was written."""
    running={}
    calls=[]
    points=[]
    with session.open() as stream:
        metadata=json.loads(next(stream))['payload']
        if metadata.get('agent_path')!=agent_path:
            raise ValueError('expected only the cluster author session')
        for line in stream:
            try: e=json.loads(line)
            except ValueError: continue  # an active writer can have an incomplete last line
            t=e.get('payload',{})
            if e['type']=='token_usage_record':
                if t['thread_id']!=metadata['id']:raise ValueError('other thread usage')
                points.append((stamp(e['timestamp']),{k:t['thread_token_usage'][k] for k in FIELDS}))
            elif e['type']=='response_item':
                kind=t.get('type')
                if kind in ('custom_tool_call','function_call'):
                    running[t['call_id']]=stamp(e['timestamp'])
                elif kind in ('custom_tool_call_output','function_call_output') and t['call_id'] in running:
                    calls.append((running.pop(t['call_id']),stamp(e['timestamp'])))
    return points,calls


def paired_development(name):
    folder=OUT/'trials'/name/'packet/full'
    result={'tested':0,'rescued':0,'lost':0}
    for p in folder.glob('*.json'):
        new=read(p)['result']['success']
        old=read(OUT/'snapshots'/p.name)['result']['success']
        result['tested']+=1
        result['rescued']+=int(new>old)
        result['lost']+=int(new<old)
    return result


def export(session):
    points,calls=telemetry(session)
    budget=read(OUT/'author_budget.json')
    notes=read(OUT/'host_notes.json')
    if (PUBLIC/'notes.json').exists():
        try: author_notes=read(PUBLIC/'notes.json')
        except ValueError:author_notes={}
        for k,val in author_notes.items():
            if isinstance(val,str):notes.setdefault(k,val)
    previous={k:0 for k in FIELDS}
    receipts=[]
    for p in sorted(PUBLIC.glob('trial*.py'),key=lambda f:f.stat().st_mtime_ns):
        if not re.fullmatch(r'trial\d+',p.stem): continue
        when=p.stat().st_mtime
        containing=next(((a,b) for a,b in calls if a<=when<=b+.002),None)
        matching=[v for t,v in points if containing and containing[0]<=t<=containing[1]+.015]
        current=matching[0] if len(matching)==1 else None
        delta={k:current[k]-previous[k] for k in FIELDS} if current and previous else None
        receipts.append(dict(name=p.stem,submitted_at=when,usage=current,delta=delta,
            note=notes.get(p.stem),development=paired_development(p.stem)))
        previous=current
    lock=read(OUT/'locked.json') if (OUT/'locked.json').exists() else None
    milestones=read(OUT/'public_milestones.json')
    if lock:
        milestones.append(dict(at=lock['submitted'],title='作者交卷 · '+lock['name'],detail=lock['description']))
    obj=dict(start=stamp(budget['started_utc']),deadline=stamp(budget['deadline_utc']),
             usage_at=points[-1][0] if points else None,usage=points[-1][1] if points else None,
             candidates=receipts,locked=lock['name'] if lock else None,
             milestones=milestones,updated=time.time())
    save(DOC/'worklog.json',obj)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('session',type=Path)
    parser.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    while True:
        export(args.session)
        if not args.watch or (OUT/'finished.json').exists():break
        time.sleep(4)
