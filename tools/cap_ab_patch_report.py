"""Summarize the conditional programs and matched counterfactual trajectories."""
from collections import Counter
import csv

from tools import cap_ab_patch_gate as gate


def publish():
    train,rows=gate.rows('train'),gate.rows('test')
    lock=gate.exp.read(gate.OUT/'frozen.json')
    assert len(train)==176 and len(rows)==200
    rule=lock['rule']
    sheet=[]
    by_outcome=Counter()
    first_action=Counter()
    for r in rows:
        a,b=r['a'],r['b']
        control=gate.exp.read(gate.OUT/'controls'/(r['room']+'.json'))
        rec=control['recovery']['success']
        default=control['default']['success']
        f=r['first']
        by_outcome[a,b,'recovery' if f and f['recovery'] else 'timing' if f else 'agree']+=1
        extra={}
        if a!=b:
            assert r['suffix']['AA']['success']==a and r['suffix']['BB']['success']==b
            for k in ('AB','BA'):
                extra[k]=r['suffix'][k]['success']
                first_action[a,b,k]+=extra[k]
        sheet.append(dict(room=r['room'],A=a,B=b,one_gate=gate.choice(r,rule),every_recovery=rec,
                          every_default=default,first_tick=f['tick'] if f else '',
                          source=('recovery' if f['recovery'] else 'timing') if f else 'agree',**extra))
    names=['A','B','one_gate','every_recovery','every_default']
    groups={(a,b):[r for r in sheet if (r['A'],r['B'])==(a,b)] for a in (0,1) for b in (0,1)}
    lines=[];metrics={}
    for k in names:
        s=[sum(r[k] for r in groups[a,b]) for a,b in [(1,1),(1,0),(0,1),(0,0)]]
        metrics[k]=s
        lines.append(f"| {k} | {sum(s)}/200 | {s[0]}/77 | {s[1]}/21 | {s[2]}/29 | {s[3]}/73 |")
    assert [len(groups[t]) for t in [(1,1),(1,0),(0,1),(0,0)]]==[77,21,29,73]
    loss=groups[1,0]
    cf=Counter((r['AB'],r['BA']) for r in loss)
    tables='\n'.join(lines)
    examples=['recovery_662316119_dev_4_01','recovery_662316119_dev_2_00','recovery_662316119_test_2_48']
    out_examples=[r for r in rows if r['room'] in examples]
    gate.exp.save(gate.DOC/'patch_examples.json',out_examples)
    with (gate.DOC/'patch_rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['room','A','B','one_gate','every_recovery','every_default','first_tick','source','AB','BA'])
        w.writeheader();w.writerows(sheet)
    text=f"""# 代码条件能否保住 A 与 B 的成功并集

两位专家从同一初态运行，交集 77、仅 A 21、仅 B 29、都失败 73。一次选择后保持所选专家，要取得 127/200，需要在 50 个成败分歧房间选对全部。一开始两位专家若提出相同命令可以延迟选择，直到首次真正不同的提案。此时两位专家的记忆都沿相同公开历史更新，选定后所得整局成功可直接由真实 A、B 轨迹读取。

在原先没有给 B 作者的 176 个开发房间扫描 10 个公开布尔条件及其相反方向，按整局最多成功择一，平局按预设枚举顺序。开发集全 A 99、全 B 98，选出的条件成功 104。冻结的程序为首次不同提案时，B 有明确恢复提案则保持 A，否则保持 B。另真实运行两条每边界重选的对照：every_recovery 只在 B 有恢复提案时采用 B，every_default 只在 B 没有恢复提案时采用 B，其余均采用 A。普通分支唯一可能的变化是近目标的推动时长。以下复用已经看过上一轮总成绩的 200 房间，属于诊断。

| 程序 | 总成功 | 两者原本都赢时保留 | A 独有保留 | B 独有保留 | 两者都输时新增 |
|---|---:|---:|---:|---:|---:|
{tables}

首次分歧来自恢复提案的，A 独赢 {by_outcome[1,0,'recovery']}、B 独赢 {by_outcome[0,1,'recovery']}；没有恢复提案、只缩短原本推动时长的，A 独赢 {by_outcome[1,0,'timing']}、B 独赢 {by_outcome[0,1,'timing']}。同类修改可导致两个方向的成败翻转。

在所有 50 个分歧房间从初态重放，逐一核对 AA 和 BB 终局历史哈希。AB 表示在首次分歧执行一条 A 提案，此后全部按 B；BA 表示首次执行 B，此后全部按 A。在 21 个 A 独赢房间，BA 成功 {first_action[1,0,'BA']}/21，AB 成功 {first_action[1,0,'AB']}/21。两者都成功 {cf[1,1]}、都失败 {cf[0,0]}、仅 AB 成功 {cf[1,0]}、仅 BA 成功 {cf[0,1]}。这能区分局部替换和后续策略的作用，不能把每次失败都归于一种物理机制。

在新面板的 dev_4_01，第 425 tick 两机器人尚未接触目标。A 给两个机器人都发 MOVE、时长 30 tick，B 让其中一个停住、另一个 PUSH 40 tick。若先用 A，后续无论用 A 或 B，都成功（1111、1319 tick）；若先用 B，后续用 A 或 B 均在 1800 tick 超时。B 首条执行后的实际终距反而比用 A 后更小（2.59 m 对 2.86 m）。这项干预定位到首次提案，对更细的接触或碰撞原因需要另外的证据。

[三个实例的命令、首步物体终距与真实终局](patch_examples.json) · [逐房间表](patch_rooms.csv)
"""
    (gate.DOC/'patch_gate.md').write_text(text)
    print(text)


if __name__=='__main__':publish()
