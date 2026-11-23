"""Publish paired option composition and transfer diagnostics to the research journal."""
import csv
import json
from pathlib import Path

import numpy as np
from tools import cap_option_handoff as exp
from tools import cap_option_handoff_report as report


def write_csv(path,rows):
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def diagnostic(data, x, y):
    groups=[]
    transitions=[]
    for r in data:
        a=r['arms']
        root=a['AA']['root']
        if not root:
            continue
        for first in 'AB':
            arm=a[first+'B']
            mid=arm['mid']
            before=a[first+'A']
            delta=arm['edges'][0]
            transitions.append(dict(room=r['key'],expert=first,root=root['node'],
                               dest=mid['node'] if mid else ('success' if arm['result']['success'] else 'end'),
                               ticks=delta['ticks'],gain=delta['gain']))
            if not (report.has(root,x) and not report.has(root,y) and report.has(mid,y) and mid['distinct']):
                continue
            b=arm['edges'][1]
            groups.append(dict(room=r['key'],origin=first,mid=mid['h'],recent=int('progress' in mid['flags']),
                               contact=int('touch' in mid['flags']),node=mid['node'],b_gain=b['gain'],
                               a_gain=before['edges'][1]['gain'],b_ticks=b['ticks'],
                               a_ticks=before['edges'][1]['ticks'],b_ok=report.action_effect(arm),
                               a_ok=report.action_effect(before),b_terminal=arm['result']['success'],
                               a_terminal=before['result']['success']))
    return groups,transitions


def display(label):
    return {'any':'所有起点','touch':'公开目标接触','close':'估计后侧站位在 45 cm 内',
            'clear':'站位通路估计净空至少 16 cm','progress':'最近三条命令观察到接近目标至少 2 cm'}.get(label,label.replace('node','公开拓扑簇 '))


def metric(value):
    v=value['delta']
    if v is None:
        return '无合资格样本'
    lo,hi=value['ci']
    return f"{value['first']}/{value['n']} 对 {value['second']}/{value['n']}，差 {100*v:+.1f} 个百分点 [{100*lo:+.1f}, {100*hi:+.1f}]"


def publish():
    lock=exp.read(exp.OUT/'frozen.json')
    import hashlib
    assert lock['spec_sha256']==hashlib.sha256((exp.DOC/'plan.md').read_bytes()).hexdigest()
    folds={k:report.cases(k) for k in ('dev','fresh')}
    assert not {r['identity'] for r in folds['dev']} & {r['identity'] for r in folds['fresh']}
    # An unsupported default contrast is only descriptive if no candidate has support.
    winner=lock['winner'] or dict(x='any',y='touch')
    x,y=winner['x'],winner['y']
    tables={}; extras=[]; edges=[]; rooms=[]
    for fold,data in folds.items():
        rows=report.rows_for(data,x,y)
        tables[fold]=report.summarize(rows,True)
        rooms.extend(dict(panel=fold,**r) for r in rows)
        grp,trans=diagnostic(data,x,y)
        extras.extend(dict(panel=fold,**r) for r in grp)
        edges.extend(dict(panel=fold,**r) for r in trans)
    new=tables['fresh']; dev=tables['dev']
    confirmed=lock['primary'] and all(new[k].get('ci',[0])[0]>0 for k in ('formation','local','terminal'))
    decision=('在新房间上，三个预定差值的区间下界均大于零，这条两步链获得确认。' if confirmed else
              '这次独立检验没有确认一条可提高整局成功率的两步链。')
    kind='正式规则' if lock['primary'] else '探索规则，正式策略保留 A'
    rule=f'从{display(x)}开始，A 新建立「{display(y)}」后交给 B 一次'
    write_csv(exp.DOC/'rooms.csv',rooms)
    write_csv(exp.DOC/'edges.csv',edges)
    if extras:write_csv(exp.DOC/'handoffs.csv',extras)
    exp.save(exp.DOC/'metrics.json',dict(rule=rule,kind=kind,confirmed=confirmed,folds=tables))
    labels={'formation':'同一 X 中 A 与 B 各执行一条而建立 Y','local':'A 送到 Y 后 B 与 A 各执行一条而推进 ≥2 cm',
            'terminal':'条件交接与整局 A 的成功','vs_bprep':'条件交接与同条件改用 B 准备的成功','vs_b':'条件交接与整局 B 的成功'}
    exposure=[]
    for label in report.CONDITIONS:
        rec={'Y':label}
        for fold,data in folds.items():
            rs=report.rows_for(data,'any',label)
            for key in ('eligible','a_hit','b_hit','switch','switch_b'):
                rec[fold+'_'+key]=sum(r[key] for r in rs)
        exposure.append(rec)
    write_csv(exp.DOC/'support.csv',exposure)
    comp='\n'.join(f"| {name} | {metric(dev[k])} | {metric(new[k])} |" for k,name in labels.items())
    t=new['terminal']; b=new['vs_b']; bp=new['vs_bprep']; at=new['paired_at_y']
    d=new['b_progress_by_origin']
    stages=[]
    for f,data in folds.items():
        diffs=sum(r['arms']['AA']['root'] is not None for r in data)
        twodiff=sum(bool(r['arms']['AA']['mid'] and r['arms']['AA']['mid']['distinct']) for r in data)
        two=sum(r['arms']['AA']['mid'] is not None for r in data)
        stages.append(f'{f} 面板 200 个房间中有 {diffs} 个到达首次分歧，A 首步后 {two} 个仍可行动，{twodiff} 个两位专家再次给出不同合法提案。')
    desc=[]
    for who in 'AB':
        buf=[r for r in extras if r['panel']=='fresh' and r['origin']==who]
        if not buf:continue
        desc.append(f"经 {who} 到达后 B 的单命令推进达标 {sum(r['b_ok'] for r in buf)}/{len(buf)}，平均实际执行 {np.mean([r['b_ticks'] for r in buf]):.1f} tick，平均终距减少 {100*np.mean([r['b_gain'] for r in buf]):.2f} cm。")
    thirds=[]
    for cat in ('recent','contact'):
        parts=[]
        for bit in (0,1):
            buf=[r for r in extras if r['panel']=='fresh' and r['origin']=='A' and r[cat]==bit]
            parts.append(f"{sum(r['b_ok'] for r in buf)}/{len(buf)}")
        thirds.append(f"{'近期公开推进' if cat=='recent' else '公开目标接触'}无／有时，B 分别为 {'、'.join(parts)}")
    ok = f"正式规则在新房间为 {t['first']}/200。" if lock['primary'] else f"未选出正式切换规则，保留 A 得 {t['second']}/200。"
    interpretation=f"{decision} {ok} 冻结条件相对 A 救回 {t['plus']} 例、丢失 {t['minus']} 例。"
    md=f"""# CAP 的一条准备命令能否与另一位专家的一条接手命令组合

**{decision}** {rule}。这条是{kind}。开发阶段 {lock['candidate_count']} 个条件组合达到最低房间数，{lock['passing']} 个同时满足三个正向差值。用 200 个开发房间冻结规则，在另 200 个不同内容的新房间作一次确认。

| 配对比较 | 开发 | 独立确认 |
|---|---|---|
{comp}

每一列的比较顺序与左栏一致，区间为房间配对重抽 95% 百分位区间。每房间只有一个预定分叉机会。{' '.join(stages)} 成功、超时和无分歧均保留在整局分母。局部推进使用目标物体到目标点的实际距离变化，记录的 option 时长为实际值。路由只读取当时公开信号。因两条提案的动作和时长可不同，局部效应描述整条提案的效果。

## 真实接手与整局成败

新房间中该条件实际触发 {new['active']} 次；在这些相同历史上，交 B 与留 A 的整局成功为 {at['first']} 与 {at['second']}，救回 {t['plus']}、丢失 {t['minus']}。从初态全 A 成功 {t['second']}/200，全 B 成功 {b['second']}/200；两个原专家从初态都失败的 {new['both_failed']} 个房间中，该组合救回 {new['both_rescued']} 个。相同触发逻辑下在合资格起点先用 B 准备得 {bp['second']}/200。{ok}

完全不看触发条件，从同前缀固定执行 AA、AB、BA、BB 两条命令再还给 A，各有 {', '.join(str(new['unconditional'][k]) for k in exp.ARMS)}/200 成功。逐房间事后在四条分支任取成功者为 {new['hindsight_four']}/200。BB 后还给 A，与全程 B 的数据单列。

## 同一个 Y 的来路

{' '.join(desc)} 同一房间可贡献两种来路，这两组的成员并不完全相同。仅凭组均值无法把差异归因于来路本身。针对 A 真正送到的相同前缀，上表的 B 对 A 才是配对的接手效应。{'；'.join(thirds)}，这些分组只作描述，小分母不支持新的路由规则。

[设计与成功判据](plan.md) · [开发候选](dev_candidates.json) · [各节点实际准备与接手机会数](support.csv) · [新旧面板统计](metrics.json) · [逐房间配对](rooms.csv) · [单 option 边](edges.csv) · [接手前缀](handoffs.csv)

本次验证限定为 A 和 B、公开节点条件、一个预定分叉机会、同一后续专家 A。单条边对应一条上层 CAP 命令，原有 PUSH 接触与推进执行器可在其内部更新多个物理决策边界。稀有条件会被最低支持数排除。这轮没有估计任意深度重规划的成功率。
"""
    lookup={r['key']:r for r in folds['fresh']}
    examples=[]
    for delta in (1,-1,0):
        candidates=[r for r in rooms if r['panel']=='fresh' and r['switch'] and r['chosen']-r['AA']==delta]
        if candidates:
            r=candidates[0]
            aa=lookup[r['room']]['arms']['AA']
            ab=lookup[r['room']]['arms']['AB']
            examples.append(dict(room=r['room'],root=aa['root'],mid=aa['mid'],prepare=aa['edges'][0],
                                 continue_A=aa['edges'][1],take_B=ab['edges'][1],
                                 final_A=aa['result'],final_AB=ab['result'],single_B=r['single_b']))
    exp.save(exp.DOC/'examples.json',examples)
    from tools.cap_option_handoff_figure import plot
    plot(tables)
    md+='\n![开发与新房间的配对转移](handoff.png)\n\n[具体分叉的公开输入、命令、实际推进与终局](examples.json)\n'
    (exp.DOC/'README.md').write_text(md)
    registry=Path('docs/research/experiment_journal/entries.json')
    entries=json.loads(registry.read_text())
    entry=next(e for e in entries if e['id']=='option-handoff')
    entry['result']=f"全新 200 房间。{rule}，条件触发 {new['active']} 个，条件交接 {t['first']}/200，全 A {t['second']}/200，全 B {b['second']}/200。"
    entry['analysis']=interpretation
    entry['assets']=[['plan.md','验证设计'],['handoff.png','局部转移与接手'],['metrics.json','配对统计'],['rooms.csv','逐房间得失'],['examples.json','具体分叉']]
    tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n');tmp.replace(registry)
    exp.save(exp.OUT/'FINISHED.json',dict(confirmed=confirmed,dev=200,fresh=200,rule=rule))
    print(interpretation,flush=True)
