"""Summarize paired room outcomes, immutable route selection and authoring costs."""
import csv
from datetime import datetime
import json
from pathlib import Path

from tools import cap_authored_router as exp
from tools.cap_option_handoff_opportunity_report import paired
from tools.cap_regional_token_meter import tail_usage


def publish():
    e = exp.e
    read, save = e.read, e.save
    seal = e.base.host.RoomCampaign(exp.CAM).verify_frozen()
    assert read(exp.OUT/'frozen.json')['seal'] == seal
    ids = read(exp.OUT/'confirmation_ids.json')
    rows = []
    for k, identity in sorted(ids.items()):
        arms = read(exp.OUT/'test'/(k+'.json'))
        row = dict(room=k, content_sha256=identity, trigger=int(bool(arms['A']['hit'])))
        for m in exp.ARMS:
            row[m] = arms[m]['result']['success']
            row[m+'_ticks'] = arms[m]['result']['ticks']
            row[m+'_cpu_s'] = arms[m]['cpu_seconds']
        r = arms['router']
        row.update(switches=r['switches'],proposals=r['proposals'],b_selections=r['b'],
                   changed_actions=r['different'],history_equals_A=r['result']['history']==arms['A']['result']['history'])
        rows.append(row)
    assert len(rows)==len(set(ids.values()))==200
    with (exp.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scores={m:sum(r[m] for r in rows) for m in exp.ARMS}
    comparisons={m:paired([r['router'] for r in rows],[r[m] for r in rows]) for m in exp.ARMS[:-1]}
    groups={}
    for a,b in ((1,1),(1,0),(0,1),(0,0)):
        group=[r for r in rows if r['A']==a and r['B']==b]
        groups[f'{a}{b}']=dict(rooms=len(group),router=sum(r['router'] for r in group),graph=sum(r['graph'] for r in group))
    probes=[read(p) for p in sorted(exp.CAM.glob('probe?.json'))]
    selection=read(exp.CAM/'selection.json')
    tokens=None
    day=Path.home()/'.codex/sessions/2026/11/23'
    for path in sorted(day.glob('*.jsonl')):
        with path.open() as f:
            meta=json.loads(f.readline())['payload']
        if meta.get('agent_path')=='/root/router_author' and abs(datetime.fromisoformat(meta['timestamp'].replace('Z','+00:00')).timestamp()-exp.START)<5:
            tokens=tail_usage(path)
            break
    c=read(exp.CAM/'campaign.json')
    timing=read(exp.OUT/'evaluation_time.json')
    dev=read(exp.WORK/'dev.json')
    devscores={m:sum(r['arms'][m]['success'] for r in dev) for m in ('A','B','lock','mixed')}
    assert devscores['A']>=devscores['B']
    author_seconds=(exp.WORK/'selection.json').stat().st_mtime-exp.START
    trigger=sum(r['trigger'] for r in rows)
    modified=sum(bool(r['changed_actions']) for r in rows)
    cpu={m:round(sum(r[m+'_cpu_s'] for r in rows),2) for m in exp.ARMS}
    result=dict(rooms=len(rows),scores=scores,comparisons=comparisons,groups=groups,development=devscores,
                probes=probes,frozen=selection,seal=seal,author_tokens=tokens,author_seconds=author_seconds,
                author_metered=c['used'],triggered=trigger,changed_rooms=modified,cpu_seconds=cpu,
                evaluation_wall_seconds=timing['end']-timing['start'])
    score_lines='\n'.join(f"| {m} | {scores[m]}/200 | {scores[m]/2:.1f}% |" for m in exp.ARMS)
    pair_lines='\n'.join(f"| {m} | {v['plus']} | {v['minus']} | {(v['plus']-v['minus'])/2:+.1f} | [{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}] | {v['mcnemar']:.3f} |" for m,v in comparisons.items())
    grp_lines='\n'.join(f"| {name} | {groups[key]['rooms']} | {groups[key]['router']} | {groups[key]['graph']} |" for key,name in [('11','两者成功'),('10','仅 A 成功'),('01','仅 B 成功'),('00','两者均失败')])
    probe_lines='\n'.join(f"| v{p['revision']} | {p['success']}/{p['rooms']} | {p['ticks']:,} | {p['seconds']:.1f} |" for p in probes)
    txt=(f"累计 {tokens['total_tokens']:,} tokens，输入 {tokens['input_tokens']:,}（其中缓存 {tokens['cached_input_tokens']:,}），输出 {tokens['output_tokens']:,}" if tokens else '作者 token 遥测不可用')
    union=200-groups['00']['rooms']
    active=[r for r in rows if r['trigger']]
    active_a=paired([r['router'] for r in active],[r['A'] for r in active])
    assert all(r['router']==r['A']==r['lock']==r['graph'] for r in rows if not r['trigger'])
    result['triggered_router_vs_A']=active_a
    save(exp.OUT/'FINISHED.json',result)
    (exp.DOC/'README.md').write_text(f"""# 6.2sol 编写冻结 A、B 的路由

200 个新随机房间，逐房间同初态比较。作者只看两份冻结源码、公共指南、176 个开发房间的记录，以及固定 24 房间的试跑反馈。该作者只参加一轮开发。测量对象是这份程序在新房间上的效果。

| 执行策略 | 成功 | 成功率 |
|---|---:|---:|
{score_lines}

开发池 A 为 {devscores['A']}/176，B 为 {devscores['B']}/176，预先指定 A 为单策略主对照。测试中，{trigger}/200 个房间到达首次公开停滞条件，{modified}/200 个房间中路由选择过与当时 A 不同的动作。A、B 从同一条真实公开历史更新记忆，动作和执行器均冻结。lock、graph、router 使用同一交接窗口，触发前由 A 执行。lock 触发后持续执行 B，graph 查冻结转移表，router 读取两条公开提案、观测和自己的记忆反复选择。

| 相对策略 | 作者救回 | 作者丢失 | 净差（百分点） | 配对 95% 区间 | 双侧 McNemar p |
|---|---:|---:|---:|---|---:|
{pair_lines}

单列满足交接条件的 {trigger} 个新房间，作者成功 {active_a['first']} 个，A 成功 {active_a['second']} 个。其余 {200-trigger} 个房间里的三种路由均随 A 完成。开发试跑从满足条件的房间取样，且在同一批 24 房间反复选版。

区间按房间成对自助抽样 4,000 次，附精确配对检验。图、固定切换和 B 为次要比较，区间均未作多重校正。

| 单策略结果 | 房间 | 作者成功 | 图成功 |
|---|---:|---:|---:|
{grp_lines}

A、B 各自从头执行的成功并集为 {union}/200。作者在并集内保住 {scores['router']-groups['00']['router']}/{union}，并集外额外完成 {groups['00']['router']} 例。并集需要知道整局结果后才能逐房间选择，动态切换还可能走出两条原轨迹。

两位专家中恰有一位成功的房间共 {groups['10']['rooms']+groups['01']['rooms']} 个。作者保住 {groups['10']['router']+groups['01']['router']} 个，图保住 {groups['10']['graph']+groups['01']['graph']} 个。两者在双方都成功的房间各完成 {groups['11']['router']} 和 {groups['11']['graph']} 个。这次对照中，显式利用提案方向、持续时间以及近期推进情况的程序，主要改善了专家独有能力的保留。单条判断规则的因果贡献尚未拆分。

## 开发预算

最多 12 分钟、3 次提交、每次相同的 24 个可交接开发房间，单局最多 1,800 ticks 和 300 次内部决策。按房间哈希选取试跑集。三种资源独立列账。作者至最终选版耗时 {author_seconds:.1f} 秒；{txt}。实际 {c['used']['attempts']} 次开发执行、{c['used']['execution_ticks']:,} ticks。冻结的图此前使用同一批 176 个开发房间上四种执行策略的轨迹，其中重复边界和专家选择按历史去重。作者同样取得了这些轨迹的转移摘要，并有额外闭环试跑。二者开发计算成本没有配平。

| 提交 | 开发成功 | 实际 ticks | 批次墙钟秒 |
|---|---:|---:|---:|
{probe_lines}

选择 v{selection['revision']}。作者原始理由如下。

> {selection['reason']}

本次统一测试批次墙钟 {result['evaluation_wall_seconds']:.1f} 秒，18 个 worker。各策略实际仿真进程 CPU 秒为 {cpu}。未触发交接时，lock 和 graph 直接复用已验证共同前缀 A 的完整结果。该 CPU 账包括策略调用和物理仿真。作者拥有更细的可见量和提案参数，图使用 48 个粗粒度公开节点与剩余预算，差异包含这项表征选择。换作者、预算或其他专家对的效果仍待验证。

[逐房间结果](rooms.csv) · [冻结路由](router.py) · [作者开发笔记](author_notes.md)
""")
    p=exp.DOC.parents[1]/'experiment_journal/entries.json'
    items=read(p)
    entry=next(q for q in items if q['id']=='authored-router')
    entry.update(result='，'.join(f'{m} {n}/200' for m,n in scores.items())+'。',analysis=f"对开发最强单策略 A，救回 {comparisons['A']['plus']} 个、丢失 {comparisons['A']['minus']} 个；对图救回 {comparisons['graph']['plus']} 个、丢失 {comparisons['graph']['minus']} 个。")
    p.write_text(json.dumps(items,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    publish()
