"""Seal the on-time choice, run its confirmation scan and publish the paired room audit."""
import csv
import json
import re
import sys
import time
import traceback
from pathlib import Path

from tools.cap_open_recovery_round import DOC,OUT,PUBLIC,dev_rooms,digest,read,run,save,seal
from tools.cap_cluster_module_report import JOURNAL
from tools.cap_cluster_worklog import telemetry,FIELDS

ID='open-recovery-cap'


def journal(**changes):
    data=read(JOURNAL)
    entry=next((r for r in data if r['id']==ID),None)
    if entry is None:
        entry=dict(id=ID,direction='cap-recovery',date='2026-11-24',title='让作者决定在哪一步干预、改母策略还是加补丁',
                   folder=DOC.name,run=OUT.name,motivation='同一类停滞可能可以提前避免，也可能需要发生后恢复。',
                   goal='上一轮 trial09 作为母程序。22 个开发案例，30 分钟自主开发。冻结选定程序后在另采的 200 间中对照。',
                   progress=[dict(label='开发房间时间线',glob='runs/dev/P/full/*.json',total=22),
                             dict(label='新面板母程序',glob='runs/confirm/P/full/*.json',total=200),
                             dict(label='作者定稿',file='locked.json',total=1),
                             dict(label='逐房间报告',file='finished.json',total=1)],
                   assets=[['worklog.html','作者开发记录与 token'],['rooms.csv','整局配对结果'],['local.csv','局部配对结果']])
        data.append(entry)
    entry.update(changes)
    # keep human-edited formatting of the other entries
    existing=JOURNAL.read_text()
    if not re.search('"id"\\s*:\\s*"'+ID+'"',existing):
        JOURNAL.write_text(existing.rstrip()[:-1].rstrip()+',\n  '+json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')+'\n]\n')
    else:
        start=existing.index('  {', existing.rfind('\n  {',0,existing.index('"id": "'+ID+'"')))
        _,end=json.JSONDecoder().raw_decode(existing[start+2:])
        JOURNAL.write_text(existing[:start]+'  '+json.dumps(entry,ensure_ascii=False,indent=2).replace('\n','\n  ')+existing[start+2+end:])


def usage(session):
    b=read(OUT/'budget.json')
    points,calls=telemetry(session,'/root/open_recovery_author')
    try:notes=read(PUBLIC/'notes.json')
    except (FileNotFoundError,ValueError):notes={}
    candidates=[]
    previous={k:0 for k in FIELDS}
    for path in sorted(PUBLIC.glob('trial*.py'),key=lambda p:p.stat().st_mtime):
        if not re.fullmatch('trial[0-9]{2,3}',path.stem):continue
        t=path.stat().st_mtime
        span=next(((a,z) for a,z in calls if a<=t<=z+.002),None)
        matches=[v for x,v in points if span and span[0]<=x<=span[1]+.015]
        current=matches[0] if len(matches)==1 else None
        diff={k:current[k]-previous[k] for k in FIELDS} if current and previous else None
        outcomes=[(read(p)['result']['success'],read(OUT/'runs/dev/P/full'/p.name)['result']['success'])
                  for p in (OUT/'runs/dev'/path.stem/'full').glob('*.json')]
        candidates.append(dict(name=path.stem,submitted_at=t,usage=current,delta=diff,note=notes.get(path.stem),
                               development=dict(tested=len(outcomes),rescued=sum(n>o for n,o in outcomes),
                                                lost=sum(n<o for n,o in outcomes))))
        previous=current
    lock=read(OUT/'locked.json') if (OUT/'locked.json').exists() else {}
    save(DOC/'worklog.json',dict(**b,usage=points[-1][1] if points else None,candidates=candidates,
         milestones=[dict(at=b['start'],title='开始开发',detail='母程序 trial09 已在全部开发房间逐项复现。')],
         locked=lock.get('name')))


def select():
    b=read(OUT/'budget.json')
    final=PUBLIC/'FINAL.json'
    if not final.is_file() or final.stat().st_mtime>b['deadline']:
        raise TimeoutError('no timely final selection')
    answer=read(final)
    name=answer['name']
    if not re.fullmatch('P|trial[0-9]{2,3}',name):raise ValueError('invalid candidate')
    if (PUBLIC/(name+'.py')).stat().st_mtime>b['deadline']: raise TimeoutError('late source')
    source=seal(name)
    lock=dict(answer,sha256=digest(source),elapsed_seconds=final.stat().st_mtime-b['start'])
    save(OUT/'locked.json',lock)
    return lock


def publish(lock):
    keys=read(OUT/'fixtures.json')
    records=[]
    for k in keys:
        base=read(OUT/'runs/confirm/P/full'/(k+'.json'))
        cand=read(OUT/'runs/confirm'/lock['name']/'full'/(k+'.json'))
        a,b=base['result'],cand['result']
        records.append(dict(room=k,cluster=int(base['anchor'] is not None),parent=int(a['success']),candidate=int(b['success']),
                            history_changed=int(a['history']!=b['history']),parent_ticks=a['ticks'],candidate_ticks=b['ticks']))
    with (DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    lines=[]
    for label,rows in [('全部',records),('簇内',[r for r in records if r['cluster']]),('簇外',[r for r in records if not r['cluster']])]:
        old=sum(r['parent'] for r in rows); new=sum(r['candidate'] for r in rows)
        rescue=sum(r['candidate']>r['parent'] for r in rows);loss=sum(r['candidate']<r['parent'] for r in rows)
        lines.append(f'| {label} | {len(rows)} | {old} | {new} | {rescue} | {loss} |')
    summary='整局配对，'+lines[0].replace('|',' ').strip()+'（依次为房间数、母程序成功、候选成功、救回、丢失）。'
    text='# 自主选择干预时机与实现的 CAP 开发\n\n母程序为前轮 trial09，另采同一生成规则下的 200 个房间。按母程序实际轨迹固定 staged/fixed 故障簇。开发与验收房间经内容哈希排重。\n\n'
    text+=f"作者选定 {lock['name']}，用时 {lock['elapsed_seconds']/60:.1f} 分钟。\n\n"
    text+='| 母程序分组 | 房间 | 母程序成功 | 候选成功 | 救回 | 丢失 |\n|---|---:|---:|---:|---:|---:|\n'+'\n'.join(lines)+'\n'
    (DOC/'README.md').write_text(text)
    save(OUT/'whole_room.json',dict(summary=summary,n=len(records)))
    journal(result=summary,analysis='整局扫描完成，局部承诺另行验收。')


def main():
    session=Path(sys.argv[1]);DOC.mkdir(parents=True,exist_ok=True)
    journal()
    prior=DOC.parent/'2026-11-24_single_cluster_cap'/'worklog.html'
    (DOC/'worklog.html').write_text(prior.read_text().replace('single-cluster-cap',ID).replace('失败 12 间、成功 10 间','失败 8 间、成功 14 间'))
    if not (DOC/'README.md').exists():
        (DOC/'README.md').write_text('# 开放式 CAP 修复开发中\n\n本轮让作者自行选择提前干预或事后恢复，修改母策略、编写补丁或两者合用。\n')
    while not (PUBLIC/'FINAL.json').exists() and time.time()<read(OUT/'budget.json')['deadline']:
        usage(session);time.sleep(4)
    lock=select()
    usage(session)
    run(lock['name'],'confirm',workers=14)
    # Existing parent scan is independently supervised; recover any missing room.
    run('P','confirm',workers=12)
    publish(lock)
    usage(session)


if __name__=='__main__':
    try:main()
    except Exception:
        save(OUT/'ERROR.json',dict(traceback=traceback.format_exc()));raise
