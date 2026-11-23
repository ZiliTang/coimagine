"""Publish the frozen-teacher imitation comparison after all rooms complete."""
import csv
import json
from pathlib import Path

from tools import cap_router_distill as d

LABEL={'A':'单 CAP A','B':'单 CAP B','teacher':'规则教师','current':'几何＋提案','history':'再加历史','semantic':'再加精简语义','relabel':'精简语义＋学生采样重标注'}


def publish():
    read,save=d.e.read,d.e.save
    panel=read(d.OUT/'protocol.json')['rooms'][2]
    fits=read(d.OUT/'lock.json')
    rows=[]
    for k,sha in sorted(panel.items()):
        v=read(d.OUT/'test'/(k+'.json'))
        rows.append(dict(room=k,content_sha256=sha,**v['outcomes'],differing_commands=v['n'],
                         **{arm+'_correct':val for arm,val in v['correct'].items()}))
    assert len(set(panel.values()))==len(rows)==200
    with (d.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scores={m:sum(r[m] for r in rows) for m in LABEL}
    def pair(a,b):
        return d.paired([r[a] for r in rows],[r[b] for r in rows])
    vs_teacher={m:pair(m,'teacher') for m in d.ARMS}
    contrasts={k:pair(a,b) for k,a,b in [('history_current','history','current'),('semantic_history','semantic','history'),('relabel_semantic','relabel','semantic')]}
    groups={}
    for a,b in ((1,1),(1,0),(0,1),(0,0)):
        subset=[r for r in rows if (r['A'],r['B'])==(a,b)]
        groups[f'{a}{b}']=dict(n=len(subset),**{m:sum(r[m] for r in subset) for m in ('teacher',*d.ARMS)})
    n=sum(r['differing_commands'] for r in rows)
    agreement={m:sum(r[m+'_correct'] for r in rows) for m in d.ARMS}
    def delta(p):
        return f"{(p['plus']-p['minus'])/2:+.1f} [{p['ci'][0]*100:+.1f}, {p['ci'][1]*100:+.1f}]"
    exact={m:sum(r[m+'_correct']==r['differing_commands'] for r in rows) for m in d.ARMS}
    result=dict(exact_rooms=exact,scores=scores,vs_teacher=vs_teacher,contrasts=contrasts,groups=groups,
                teacher_test_boundaries=n,correct=agreement,fit=fits)
    save(d.OUT/'FINISHED.json',result)
    lines=['| 方法 | 训练集一致率 | 新房间教师轨迹上的一致率 | 闭环成功 | 相对教师救回 / 丢失 | 净差与配对 95% 区间（百分点） |',
           '|---|---:|---:|---:|---:|---:|']
    for m in LABEL:
        if m in fits:
            p=vs_teacher[m]
            train=f"{fits[m]['correct']}/{fits[m]['boundaries']} ({100*fits[m]['correct']/fits[m]['boundaries']:.1f}%)"
            lines.append(f"| {LABEL[m]} | {train} | {agreement[m]}/{n} ({agreement[m]/n*100:.1f}%) | {scores[m]}/200 | {p['plus']} / {p['minus']} | {delta(p)} |")
        else:
            lines.append(f"| {LABEL[m]} | — | — | {scores[m]}/200 | — | — |")
    cols=['teacher',*d.ARMS]
    group_lines=['| 专家结果 | 房间数 | '+' | '.join(LABEL[m] for m in cols)+' |','|---|---:|'+'---:|'*len(cols)]
    for name,key in [('A、B 均成功','11'),('仅 A 成功','10'),('仅 B 成功','01'),('A、B 均失败','00')]:
        g=groups[key]
        group_lines.append('| '+name+' | '+str(g['n'])+' | '+' | '.join(str(g[m]) for m in cols)+' |')
    text='\n'.join(lines)
    group_text='\n'.join(group_lines)
    s=vs_teacher['semantic']
    report=f"""# 冻结路由教师的双 CAP actor 蒸馏

200 个新随机房间使用同一组初态，各自运行到终止。规则教师成功 {scores['teacher']}/200；主对照的精简语义 actor 成功 {scores['semantic']}/200，相对教师救回 {s['plus']} 个、丢失 {s['minus']} 个。所有方法都沿用停滞触发前固定 A 的规则。

{text}

一致率的分母仅包括触发后两份完整命令不同的决策，时长不同也计入。测试一致率在教师实际访问的前缀上计算；闭环使用各 actor 自己访问的前缀，每个 actor 完整执行 200 局。训练为单个预先指定的随机种子 17。配对区间以房间为单位重采样 4,000 次，未预设非劣界值。

主对照中，精简语义 actor 在 {exact['semantic']}/200 个房间的整条教师轨迹上没有一次实际命令分歧，包含没有可改选动作的房间。所有完全一致的房间都另行执行 actor 并核对终局历史哈希。

## 能力保留

{group_text}

历史相对当前输入的闭环净差为 {delta(contrasts['history_current'])} 个百分点，精简语义相对历史为 {delta(contrasts['semantic_history'])} 个百分点。学生采样重标注相对同结构教师轨迹训练为 {delta(contrasts['relabel_semantic'])} 个百分点。

## 监督与交互

教师来自上一轮 6.2sol 作者的三版闭环开发，冻结后一直可以对当前到达的公开前缀给出 A/B 选择。本轮采集覆盖成功和失败的整局。标签是教师当步实际推荐，学生不会读到教师内部控制模式、最终成败或完整程序 trace。

前三个主模型共用 400 个房间的教师轨迹和逐边界标签。当前输入包括公开几何、接触、两个机器人的候选动作编码、目标绑定和命令时长。历史组额外读到上一步执行身份、从实际分歧选择推断的持续专家、该选择持续时间、首次触发后时间、上一轮明显推进以来的时间和最近四条命令的推进量。持续身份和推进记录均由学生自己的执行历史维护。精简语义组额外给出 9 个提案比较量，涉及同动作组合、时长差、是否推动、推动方向是否保持、等待与让位的区别。动作相同时统一执行 A 的同一命令，不拿专家编号差异计错。

附加对照先在首批 200 房间训练一个精简语义学生，让它在第二批 200 房间实际执行，教师在同一条学生访问的历史上更新自己的记忆并重标注。把这些标签与首批教师数据合并后重新拟合。第二批房间完全相同，每批实际分歧边界数由执行策略决定。附加组多用了一次先导拟合。教师采样和学生采样都没有用奖励更新路由参数。

四个最终 actor 都是同形 64×64 ReLU MLP，输出 A/B 的二元选择，用相同初始化、Adam、步长 0.001、批量 128、4,000 次二元交叉熵更新。对未使用的输入列置零使参数数目一致。模型一次冻结后才开始闭环测试；测试房间内容哈希与开发、旧确认面板核对。此前已查看的 118/200 属于另一批房间。

同一位教师用于所有组，此实验比较教师知识转移方式。要把收益单独归因于作者的试跑迭代，还需要对照没有试跑反馈的作者。与旧的全量 trace＋attention 方法相比，教师、数据分布、监督目标和表征均有差异。

[逐房间闭环和动作一致数](rooms.csv) · [规则教师](../2026-11-23_authored_router/router.py)
"""
    (d.DOC/'README.md').write_text(report)
    registry=d.DOC.parents[1]/'experiment_journal/entries.json'
    entries=read(registry)
    entry=next(x for x in entries if x['id']=='router-distill')
    entry.update(result='，'.join(f'{LABEL[m]} {scores[m]}/200' for m in ('teacher',*d.ARMS))+'。',
                 analysis=f"精简语义学生相对冻结教师逐房间救回 {s['plus']} 个、丢失 {s['minus']} 个。")
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    publish()
