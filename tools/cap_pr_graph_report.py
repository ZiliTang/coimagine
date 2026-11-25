"""Progress and paired audit of the graph-selected two-role CAP round."""
import csv
import json
import re
import time
from pathlib import Path
from tools import cap_pr_graph_round as r
from tools.cap_cluster_worklog import telemetry, FIELDS
from tools.cap_research_journal import REGISTRY

ID='pr-graph-round'
def journal(**changes):
    entries=r.read(REGISTRY)
    row=next((x for x in entries if x['id']==ID),None)
    if row is None:
        row=dict(id=ID,direction='cap-recovery',date='2026-11-24',title='从 P＋R 仍然失败的局面开发下一版',folder=r.DOC.name,run=r.OUT.name,
            motivation='提前接管在部分双败房间有效，多数失败房间仍缺少可用的替代动作。',
            goal='用冻结组合在 160 个开发房间中的失败与自然恢复前缀选一个簇；作者自由改 P、R 或交接条件，冻结后独立配对 200 个新房间。',
            result='开发与新房间验收进行中。',analysis='按原组合的轨迹确定簇成员，分别检查局部兑现和整局救回、丢失。',
            assets=[['worklog.html','作者工作记录与 token'],['worklog.json','作者计量明细'],['rooms.csv','逐房间结果'],['local.csv','同前缀局部结果'],['claim.csv','作者局部承诺'],['policy.py','冻结的完整程序']],
            progress=[dict(label='开发轨迹及故障图',glob='graph/*.json',total=160),dict(label='冻结目标簇',file='selection.json',total=1),
                      dict(label='作者材料和开发对照',glob='runs/dev/P/full/*.json',total=22),dict(label='冻结作者提交',file='locked.json',total=1),
                      dict(label='新房间原组合',glob='runs/confirm/P/full/*.json',total=200),dict(label='局部与整局配对报告',file='finished.json',total=1)])
        entries.append(row)
    row.update(changes)
    REGISTRY.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')

def worklog():
    path=r.OUT/'budget.json'
    if not path.exists():return
    b=r.read(path);points=[];calls=[]
    pointer=r.OUT/'author_session.txt'
    if pointer.exists():
        points,calls=telemetry(Path(pointer.read_text().strip()),'/root/pr_graph_author')
    notes_file=r.PUBLIC/'notes.json'
    notes=r.read(notes_file) if notes_file.exists() else {}
    previous={k:0 for k in FIELDS}
    receipts=[]
    for p in sorted(r.PUBLIC.glob('trial*.py'),key=lambda f:f.stat().st_mtime):
        if not re.fullmatch(r'trial\d+',p.stem):continue
        t=p.stat().st_mtime
        span=next(((a,b) for a,b in calls if a<=t<=b+.002),None)
        hit=[v for x,v in points if span and span[0]<=x<=span[1]+.015]
        use=hit[0] if len(hit)==1 else None
        delta={k:use[k]-previous[k] for k in FIELDS} if use and previous else None
        previous=use
        outcomes=[(r.read(f)['result']['success'],r.read(r.OUT/'runs/dev/P/full'/f.name)['result']['success']) for f in (r.OUT/'runs/dev'/p.stem/'full').glob('*.json')]
        receipts.append(dict(name=p.stem,submitted_at=t,usage=use,delta=delta,note=notes.get(p.stem),development=dict(tested=len(outcomes),rescued=sum(a>b for a,b in outcomes),lost=sum(a<b for a,b in outcomes))))
    lock=r.OUT/'locked.json'
    r.save(r.DOC/'worklog.json',dict(**b,usage=points[-1][1] if points else None,candidates=receipts,locked=r.read(lock)['name'] if lock.exists() else None,
        milestones=[dict(at=b['start'],title='作者开始开发',detail='冻结 P＋R 重放，按不同开发房间统计重复故障。')]))

def init():
    r.DOC.mkdir(exist_ok=True,parents=True)
    (r.DOC/'README.md').write_text('# 从 P＋R 剩余失败中选择一类故障\n\n从开发集构图并提供相似自然恢复对照，作者时限 30 分钟，token 不封顶。\n')
    html=(r.DOC.parent/'2026-11-24_open_recovery/worklog.html').read_text().replace('open-recovery-cap',ID).replace('母程序 trial09','冻结的 P＋R 组合').replace('失败 8 间、成功 14 间','22 个开发房间')
    (r.DOC/'worklog.html').write_text(html)
    journal()

def seal_final():
    r.configure()
    clock=r.read(r.OUT/'budget.json')
    final=r.PUBLIC/'FINAL.json'
    data=r.read(final)
    name=data['name']
    if not re.fullmatch(r'P|trial[0-9]{2,3}', name):
        raise ValueError('unexpected final policy')
    t=final.stat().st_mtime
    if not clock['start'] <= t <= clock['deadline'] or (r.PUBLIC/(name+'.py')).stat().st_mtime>clock['deadline']:
        raise ValueError('late submission')
    source=r.host.seal(name)
    lock=dict(data,sha256=r.host.digest(source),elapsed_seconds=t-clock['start'])
    r.immutable(r.OUT/'locked.json',lock)
    (r.DOC/'policy.py').write_text(source)
    return lock


def paired(rows):
    return dict(n=len(rows),old=sum(x['parent'] for x in rows),new=sum(x['candidate'] for x in rows),
                rescue=sum(x['candidate']>x['parent'] for x in rows),lost=sum(x['candidate']<x['parent'] for x in rows))


def publish(lock):
    import numpy as np
    from tools import cap_open_recovery_local as local
    name=lock['name']
    fixtures=r.read(r.OUT/'fixtures.json')
    selection=r.read(r.OUT/'selection.json')
    node=selection['node']
    records=[]
    for k in fixtures:
        a,b=[r.read(r.OUT/'runs/confirm'/tag/'full'/(k+'.json')) for tag in ('P',name)]
        def witness(rec):
            tr=[dict(public=dict(tick=x['tick']),node=x['node'],true_gap=x['gap']) for x in rec['effects']]
            return r.PREFIX._interventions(tr).get(node)
        ha,hb=witness(a),witness(b)
        assert a['anchor']==ha and b['anchor']==hb
        records.append(dict(room=k,cluster=bool(ha),candidate_cluster=bool(hb),
                            old_visited=any(x['node']==node for x in a['effects']),
                            parent=int(a['result']['success']),candidate=int(b['result']['success']),
                            parent_ticks=a['result']['ticks'],candidate_ticks=b['result']['ticks'],
                            history_changed=a['result']['history']!=b['result']['history']))
    with (r.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    groups={label:paired(rr) for label,rr in [
        ('全部',records),('原组合进入停滞簇',[v for v in records if v['cluster']]),
        ('原组合未进入停滞簇',[v for v in records if not v['cluster']]),
        ('原组合短暂经过同一节点且成功',[v for v in records if v['parent'] and not v['cluster'] and v['old_visited']])]}
    diffs=np.array([v['candidate']-v['parent'] for v in records])
    rng=np.random.default_rng(70341)
    ci=np.quantile(rng.choice(diffs,size=(20000,len(diffs))).mean(1)*100,[.025,.975]).tolist()
    visits=dict(avoided=sum(v['cluster'] and not v['candidate_cluster'] for v in records),
                introduced=sum(v['candidate_cluster'] and not v['cluster'] for v in records))
    dev_selected=[dict(parent=r.read(r.OUT/'runs/dev/P/full'/(k+'.json'))['result']['success'],
               candidate=r.read(r.OUT/'runs/dev'/name/'full'/(k+'.json'))['result']['success']) for k in r.dev_rooms()]
    r.save(r.OUT/'whole_room.json',dict(groups=groups,ci95_points=ci,development=paired(dev_selected),cluster_changes=visits))
    d=groups['全部']; cl=groups['原组合进入停滞簇'];sample=paired(dev_selected)
    summary=f"独立新房间 {d['old']} → {d['new']}/200，救回 {d['rescue']} 间、丢失 {d['lost']} 间。"
    lines=['# 从冻结 P＋R 的剩余故障开发下一版','',summary,'',
       f"在 160 个开发房间中按不同房间数构图。选中的 {node} 节点在 20 个最终失败房间出现持续停滞，0 个成功房间满足同样的停滞窗口。作者获得 12 个失败、10 个成功开发房间，后者有 2 间短暂经过同类节点后成功。",
       '',f"独立 GPT-6.2-Sol 作者限时 30 分钟，自主交卷，用时 {lock['elapsed_seconds']/60:.1f} 分钟。所选版本 {name}。开发材料上原组合 {sample['old']}/22，作者定稿 {sample['new']}/22。提供原有知识、事件时间线和同前缀运行工具；部署只读取公开历史。",'',
       '| 原组合定义的分组 | 房间 | 原组合成功 | 新组合成功 | 救回 | 丢失 |','|---|---:|---:|---:|---:|---:|']
    for label,v in groups.items():
        lines.append('| '+label+' | '+' | '.join(str(v[k]) for k in ('n','old','new','rescue','lost'))+' |')
    lines+=['',f"整组房间配对自助法 95% 区间为 [{ci[0]:+.1f}, {ci[1]:+.1f}] 个百分点。区间仅反映房间抽样。更换程序后，有 {visits['avoided']} 个房间避开原先的停滞窗口，另有 {visits['introduced']} 个新增进入窗口。",'',
        '该图节点表示目标方向推面后方的三个离线采样站位都无法容纳机器人，同时物体前方的短距离扫掠可通过。停滞是跨至少 120 ticks 回到同一节点、实际目标距离缩短不足 3.5 cm。图标签用于选择前缀和事后计量。','']
    (r.DOC/'README.md').write_text('\n'.join(lines))
    local.OUT,local.DOC=r.OUT,r.DOC
    msg=local.summarize()
    worklog()
    journal(result=summary,analysis=msg)
    return summary,msg

def supervise():
    r.configure()
    deadline=r.read(r.OUT/'budget.json')['deadline']
    while not (r.PUBLIC/'FINAL.json').exists():
        if time.time()>deadline+2:raise TimeoutError('Author did not submit a final selection by the deadline')
        worklog()
        time.sleep(5)
    lock=seal_final()
    name=lock['name']
    worklog()
    phase=next(x['progress'] for x in r.read(REGISTRY) if x['id']==ID)
    if not any(x['label']=='新房间定稿' for x in phase):
        phase.insert(-1,dict(label='新房间定稿',glob=f'runs/confirm/{name}/full/*.json',total=200))
    journal(progress=phase)
    r.host.run(name,'dev',workers=12)
    r.host.run(name,'confirm',workers=16)
    r.host.run('P','confirm',workers=12)
    summary,msg=publish(lock)
    r.save(r.OUT/'paired_study.json',dict(summary=summary,local=msg))


def write_csv(name,records):
    if not records:return
    with (r.DOC/name).open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=records[0])
        writer.writeheader()
        writer.writerows(records)


if __name__=='__main__':
    import argparse
    import traceback
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['init','supervise','worklog'])
    phase=p.parse_args().phase
    try:
        {'init':init,'supervise':supervise,'worklog':worklog}[phase]()
    except Exception:
        r.save(r.OUT/'ERROR.json',dict(traceback=traceback.format_exc()))
        raise
