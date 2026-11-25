"""Seal the author's selected policies before the paired confirmation opens."""
import csv
from hashlib import sha256
import json
import re
import time
import traceback
from pathlib import Path
import numpy as np
from tools import cap_free_repair_round as r
from tools.cap_cluster_worklog import telemetry, FIELDS
from tools.cap_research_journal import REGISTRY

def worklog():
    pointer=r.OUT/'author_session.txt'
    points,calls=telemetry(Path(pointer.read_text().strip()),'/root/free_repair_author') if pointer.exists() else ([],[])
    notes_file=r.PUBLIC/'notes.json'
    notes=r.read(notes_file) if notes_file.exists() else {}
    previous_usage = {key: 0 for key in FIELDS}
    receipts = []
    batches = {}
    for path in sorted(r.PUBLIC.glob('trial*.py'), key=lambda f: f.stat().st_mtime):
        if path.stem == 'trial999':
            continue
        submitted = path.stat().st_mtime
        span = next(((a, b) for a, b in calls if a <= submitted <= b + .002), None)
        hits = [(t, counters) for t, counters in points if span and span[0] <= t <= span[1] + .015]
        point, usage = hits[0] if len(hits) == 1 else (None, None)
        outcomes = [
            (r.read(f)['result']['success'], r.read(r.OUT/'runs/dev/P/full'/f.name)['result']['success'])
            for f in (r.OUT/'runs/dev'/path.stem/'full').glob('*.json')
        ]
        receipt = dict(
            name=path.stem, submitted_at=submitted, usage=usage, delta=None,
            note=notes.get(path.stem) if isinstance(notes, dict) else None,
            development=dict(tested=len(outcomes), rescued=sum(a>b for a,b in outcomes),
                             lost=sum(a<b for a,b in outcomes)),
        )
        receipts.append(receipt)
        if point is not None:
            if point not in batches:
                delta = {k: usage[k] - previous_usage[k] for k in FIELDS} if previous_usage else None
                batches[point] = ([], delta)
            batches[point][0].append(receipt)
        previous_usage = usage

    shared_batches = []
    for members, delta in batches.values():
        if len(members) == 1:
            members[0]['delta'] = delta
        else:
            group = len(shared_batches)
            shared_batches.append(dict(candidates=[m['name'] for m in members], delta=delta))
            for m in members:
                m['shared_batch'] = group

    b = r.read(r.OUT/'budget.json')
    ready = r.read(r.OUT/'READY.json')
    pool = ready['dev']
    wins = ready['success']
    r.save(r.DOC/'worklog.json', dict(
        **b, usage=points[-1][1] if points else None, candidates=receipts,
        shared_batches=shared_batches, development_rooms=pool,
        locked=r.read(r.OUT/'locked.json')['name'] if (r.OUT/'locked.json').exists() else None,
        milestones=[dict(at=b['start'], title='作者开始开发',
                         detail=f'{pool} 个开发房间，{wins} 个成功、{pool-wins} 个失败。')],
    ))

def update_journal(**kw):
    data=r.read(REGISTRY)
    row=next(e for e in data if e['id']=='cap-free-repair')
    row.update(kw)
    REGISTRY.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def lock():
    b=r.read(r.OUT/'budget.json')
    final=r.PUBLIC/'FINAL.json'
    while not final.exists():
        worklog()
        if time.time()>b['deadline']+3:raise TimeoutError('No on-time author final selection')
        time.sleep(5)
    d=r.read(final)
    assert b['start']<=final.stat().st_mtime<=b['deadline']
    names={'selected':d['name']}
    if d['name']!='P':
        names['without_old_recovery']=d['without_old_recovery']
    if d.get('without_new_recovery'):names['without_new_recovery']=d['without_new_recovery']
    hashes={}
    for role,n in names.items():
        assert re.fullmatch(r'P|trial\d{2,3}',n) and n!='trial999'
        assert (r.PUBLIC/(n+'.py')).stat().st_mtime <= b['deadline']
        hashes[role]=r.host.digest(r.host.seal(n))
    assert hashes['selected']==d['source_sha256']
    r.immutable(r.OUT/'locked.json',dict(**d,arms=names,hashes=hashes))
    return names

def composition_report():
    """Publish the frozen arm identities alongside the original per-room export."""
    registry = r.read(r.DOC/'compositions.json')
    arms = registry['arms']
    with (r.DOC/'rooms.csv').open() as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames) != {'room', *arms}:
            raise ValueError('Registered arms and evaluated arms differ')
        rows = list(reader)
    panel = registry['panel']
    if sha256(Path(panel['fixtures']).read_bytes()).hexdigest() != panel['sha256']:
        raise ValueError('Panel identity mismatch')
    if {row['room'] for row in rows} != set(r.read(Path(panel['fixtures']))) or len(rows) != len({row['room'] for row in rows}):
        raise ValueError('Room set does not match the registered panel')
    labels = {k: arm['composition'] for k,arm in arms.items()}
    if len(set(labels.values())) != len(labels):
        raise ValueError('Duplicate composition names')
    for arm in arms.values():
        if sha256(Path(arm['source']).read_bytes()).hexdigest() != arm['sha256']:
            raise ValueError('Frozen CAP source mismatch: '+arm['composition'])
    with (r.DOC/'rooms_by_composition.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['room', *labels.values()])
        for row in rows:
            writer.writerow([row['room'], *(row[k] for k in arms)])
    lines = [f"面板 `{panel['id']}`，{len(rows)} 个新房间，从整局初态运行。各组采用同一程序交接规则 `G程序.v1`。",
             '', '| 组合 | 本轮文件 | 成功 |', '|---|---|---:|']
    for k,arm in arms.items():
        score = sum(int(row[k]) for row in rows)
        lines.append(f"| {labels[k].split('@')[0]} | `{arm['submission']}.py` | {score}/{len(rows)} |")
    lines.extend(['', 'P绕行含早期绕行能力。R换位、R侧推在本次消融中同时启停。R复接负责靠墙重建接触。旧恢复有提案或持有计划时优先。',
                  '', '[按组合命名的逐房间结果](rooms_by_composition.csv) · [文件与完整哈希](compositions.json) · [命名协议](../../CAP_组合命名.md)'])
    return '\n'.join(lines)

def assess():
    r.configure()
    names=lock()
    worklog()
    src=r.host.seal('P')
    gate="  if c and o['active_target']==tid and get_points(o,tid):"
    assert src.count(gate)==1
    (r.PUBLIC/'trial999.py').write_text(src.replace(gate,'  if False:'))
    names['parent_without_old_recovery']='trial999'
    names['parent']='P'
    fixtures=r.read(r.OUT/'fixtures.json')
    progress=next(e['progress'] for e in r.read(REGISTRY) if e['id']=='cap-free-repair')
    progress=[p for p in progress if p.get('glob')!='runs/confirm/P/full/*.json' and p.get('glob')!='evaluation/selected/*.json' and p['label']!='结果']
    labels={k:v['composition'] for k,v in r.read(r.DOC/'compositions.json')['arms'].items()}
    progress.extend(dict(label='新房间 · '+labels[role],glob='evaluation/'+role+'/*.json',total=200) for role in names)
    progress.append(dict(label='结果',file='finished.json',total=1))
    update_journal(progress=progress,result='作者已定稿，正在进行冻结配对评估。')
    for role,n in names.items():
        source=r.host.seal(n)
        jobs=[(k,f,True,None,'boundary',False,r.host.enabled(source),src) for k,f in fixtures.items()]
        r.host.old.collect(source,jobs,r.host.episode,16,r.OUT/'evaluation'/role)
    rows=[]
    for k in fixtures:
        row=dict(room=k)
        for role in names:
            row[role]=int(r.read(r.OUT/'evaluation'/role/(k+'.json'))['result']['success'])
        rows.append(row)
    with (r.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    score={role:sum(z[role] for z in rows) for role in names}
    diff=np.array([z['selected']-z['parent'] for z in rows])
    ci=np.quantile(np.random.default_rng(8361).choice(diff,size=(20000,len(diff))).mean(1)*100,[.025,.975]).tolist()
    summary=f"{labels['parent']} → {labels['selected']}，新房间 {score['parent']} → {score['selected']}/200，救回 {sum(diff>0)} 间、丢失 {sum(diff<0)} 间。"
    (r.DOC/'README.md').write_text('# 从剩余失败中自主选择修复方式\n\n'+summary+'\n\n配对房间自助法 95% 区间 '+str(ci)+' 个百分点。\n\n'+ composition_report()+'\n')
    update_journal(result=summary)
    r.save(r.OUT/'whole_room.json',dict(scores=score,ci=ci,summary=summary))
if __name__=='__main__':
    try:assess()
    except Exception:
        r.save(r.OUT/'ERROR.json',dict(error=traceback.format_exc()))
        raise
