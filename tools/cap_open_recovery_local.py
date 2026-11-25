"""Compare post-stall continuations at identical parent histories on the new rooms."""
import csv
from contextlib import contextmanager
from math import dist
from unittest.mock import patch
from tools import cap_open_recovery_round as study
from tools.cap_open_recovery_finish import journal

OUT,DOC=study.OUT,study.DOC
read,save=study.read,study.save


def matched(job):
    key,fixture,anchor,parent_src,hook=job
    prefix=study.old.prefix
    original=prefix.s.phase.world
    gaps=[]
    @contextmanager
    def traced(f):
        with original(f) as (world,other):
            body=world.env.env.backend.bodies[prefix.s.observe(world)['goal']['target_id']]
            goal=(fixture['goal']['x'],fixture['goal']['y'])
            def sample():
                assert len(gaps)==world.env.total_periods
                gaps.append(dist(body.position,goal))
            sample()
            update=world.detector.update
            def after(packet):
                value=update(packet)
                sample()
                return value
            world.detector.update=after
            yield world,other
    with patch.object(prefix.s.phase,'world',traced):
        _,row=study.episode((key,fixture,False,anchor,'boundary',False,hook,parent_src))
    end=min(anchor+120,len(gaps)-1)
    advance=gaps[anchor]-gaps[end]
    return key,dict(reached=row['reached'],result=row['result'],start=anchor,end=end,
                    advance=advance,met=bool(advance>=.035 or row['result']['success'] and end<anchor+120))


def summarize():
    fixtures=read(OUT/'fixtures.json')
    lock=read(OUT/'locked.json')
    parent=study.seal('P')
    population={}
    for k in fixtures:
        rec=read(OUT/'runs/confirm/P/full'/(k+'.json'))
        if rec['anchor'] is not None:
            population[k]=rec['effects'][rec['anchor']['window_end']]['tick']
    for tag in dict.fromkeys(['P',lock['name']]):
        src=study.seal(tag)
        jobs=[(k,fixtures[k],t,parent,study.enabled(src)) for k,t in population.items()]
        study.old.collect(src,jobs,matched,12,OUT/'local'/tag)
    rows=[]
    for k in population:
        a=read(OUT/'local/P'/(k+'.json'))
        b=read(OUT/'local'/lock['name']/(k+'.json'))
        assert a['reached']==b['reached']
        assert a['result']==read(OUT/'runs/confirm/P/full'/(k+'.json'))['result']
        rows.append(dict(room=k,anchor_tick=a['start'],prefix_sha256=a['reached']['history'],
                         parent_advance=a['advance'],candidate_advance=b['advance'],parent_local=int(a['met']),
                         candidate_local=int(b['met']),parent_win=a['result']['success'],candidate_win=b['result']['success']))
    with (DOC/'local.csv').open('w') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    pos=sum(r['candidate_local']>r['parent_local'] for r in rows);neg=sum(r['candidate_local']<r['parent_local'] for r in rows)
    report=(f"从母策略首次符合停滞定义的窗口末尾分叉，共 {len(rows)} 间。以 120 tick 内窗口终点的真实目标距离减少至少 3.5 cm，或此前完成任务为统一诊断。母策略 {sum(r['parent_local'] for r in rows)}/{len(rows)}，候选 {sum(r['candidate_local'] for r in rows)}/{len(rows)}，新增达标 {pos} 间，丢失达标 {neg} 间。")
    with (DOC/'README.md').open('a') as f:f.write('\n## 固定停滞窗口的同前缀续跑\n\n'+report+' 此项测量从故障形成后开始。作者自选时机的预防或恢复承诺另列。\n')
    save(OUT/'paired_local.json',dict(summary=report))
    return report


if __name__=='__main__':
    summarize()
