"""Summarize the actual first eligible handoff along each trajectory."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from tools import cap_option_handoff as exp


def paired(v,w):
    v,w=np.asarray(v,int),np.asarray(w,int)
    d=v-w
    wins,losses=int(sum(d>0)),int(sum(d<0))
    n=wins+losses
    p=min(1.,2*sum(math.comb(n,k)*.5**n for k in range(min(wins,losses)+1)))
    rng=np.random.default_rng(38013)
    ci=np.quantile(d[rng.integers(len(d),size=(4000,len(d)))].mean(1),[.025,.975]).tolist() if len(d) else None
    return dict(n=len(d),first=int(v.sum()),second=int(w.sum()),plus=wins,minus=losses,
                ci=ci,mcnemar=p)


def main():
    data=[exp.read(p) for p in sorted((exp.OUT/'opportunities').glob('*.json'))]
    assert len({r['room'] for r in data})==len(data)==200
    rows=[]
    for r in data:
        a,b=r['a'],r['b']
        row=dict(room=r['room'],active=int(bool(a['event'])),base=a['result']['success'],
                 chosen=b['result']['success'],single_b=r['single_b'])
        for name,v in [('a',a),('b',b)]:
            e=v['event']
            row[name+'_gain']=e['gain'] if e else None
            row[name+'_ticks']=e['ticks'] if e else None
        own=r['b_own']['event']
        row['b_own_gain']=own['gain'] if own else None
        row['b_own_ticks']=own['ticks'] if own else None
        row['index']=a['event']['n'] if a['event'] else None
        rows.append(row)
    with (exp.DOC/'opportunities.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    active=[r for r in rows if r['active']]
    t=paired([r['chosen'] for r in rows],[r['base'] for r in rows])
    local=paired([r['b_gain']>=.02 for r in active],[r['a_gain']>=.02 for r in active])
    baseb=paired([r['chosen'] for r in rows],[r['single_b'] for r in rows])
    double=[r for r in rows if not(r['base'] or r['single_b'])]
    a_gain=np.array([r['a_gain'] for r in active]);b_gain=np.array([r['b_gain'] for r in active])
    gap=100*(b_gain-a_gain)
    ci=np.quantile(np.random.default_rng(84).choice(gap,(4000,len(gap))).mean(1),[.025,.975])
    msg=f"新 200 房间中实际触发 {len(active)} 次，B 接手整局 {t['first']}/200，继续 A 为 {t['second']}/200；救回 {t['plus']}、丢失 {t['minus']}。"
    local_text=f"同一接手点只执行一条提案，B 达到 2 cm 推进的有 {local['first']}/{len(active)}，A 为 {local['second']}/{len(active)}。"
    def interval(v):
        return f"[{100*v['ci'][0]:+.1f}, {100*v['ci'][1]:+.1f}]"
    own=[r for r in rows if r['b_own_gain'] is not None]
    both=[r for r in active if r['b_own_gain'] is not None]
    own_ok=sum(r['b_own_gain']>=.02 for r in own)
    within_a=sum(r['b_gain']>=.02 for r in both)
    within_b=sum(r['b_own_gain']>=.02 for r in both)
    text=f"""# 沿 A 的真实轨迹逐步检查冻结的交接条件

**{msg}** {local_text} 该补充使用主实验相同的 200 个确认房间。两次实验的结果共享初态。每房间从预定时点起检查每条命令，只在首次符合条件时分叉。A 与 B 的准备命令不同，A 从公开簇 4 以外进入公开簇 4，到达后两位专家仍提出不同合法命令。之后 B 或 A 执行一次，统一由 A 执行到底。

| 指标 | B 接手 | A 继续 | B 单独达标 / A 单独达标 | 配对差的 95% 区间，百分点 |
|---|---:|---:|---:|---:|
| 实际接手点单条提案使物体向目标推进至少 2 cm | {local['first']}/{len(active)} | {local['second']}/{len(active)} | {local['plus']} / {local['minus']} | {interval(local)} |
| 原预算内整局成功，含未触发 | {t['first']}/200 | {t['second']}/200 | {t['plus']} / {t['minus']} | {interval(t)} |

该节点在旧物理图中指后侧站位可用、直达条件不满足；这里使用公开模型输出的簇号。冻结规则下每个触发房间的前缀与中间历史在两个分支逐项一致，200 条整局 A 的终局哈希也与主实验相同。

实际触发时，B 与 A 平均单条命令推进 {100*b_gain.mean():.2f} cm 和 {100*a_gain.mean():.2f} cm，配对均值差 {gap.mean():+.2f} cm，区间 [{ci[0]:+.2f}, {ci[1]:+.2f}] cm。局部二元推进和整局成功的双侧配对精确检验分别为 p={local['mcnemar']:.3f}、p={t['mcnemar']:.3f}。区间按房间重抽；若所有观察配对相同，零宽的经验区间不表示真实差异已知为零。

另沿 B 自己从初态产生的整条轨迹检查同一个条件，命中 {len(own)} 个房间，其中 B 自己接着执行的一条提案达到 2 cm 推进 {own_ok}/{len(own)}；在 A 送来的边界是 {local['first']}/{len(active)}。共有 {len(both)} 个房间在两种来路上都命中，在这些相同房间中 B 的推进数分别为 A 送来 {within_a}、B 自己送来 {within_b}。同房间的两个边界可以发生在不同时刻、不同具体几何与剩余预算；这项分布对照不能归因为准备者的独立因果效应。所有 B 自然轨迹的终局历史哈希也与主实验的单 B 对照一致。

全程 B 得 {baseb['second']}/200。两个单专家均失败的 {len(double)} 个房间里，接手方案救回 {sum(r['chosen'] for r in double)} 个。这个观察回答了放宽交接时机后可见多少次机会。按 A 已到达来选择故障前缀，会偏向 A 成功进入该节点的状态，因此不能用于估计 A 和 B 创造该条件的相对能力。

[补充设计](dense_plan.md) · [逐房间结果](opportunities.csv) · [主实验](README.md)
"""
    exp.save(exp.DOC/'opportunity_metrics.json',dict(local=local,terminal=t,vs_b=baseb,active=len(active),b_own=[own_ok,len(own)],same_rooms_by_origin=[within_a,within_b,len(both)]))
    (exp.DOC/'opportunities.md').write_text(text)
    path=exp.DOC/'README.md'
    main_text=path.read_text()
    if '[逐边界补充验证]' not in main_text:
        path.write_text(main_text+'\n'+msg+local_text+'[逐边界补充验证](opportunities.md)\n')
    registry=Path('docs/research/experiment_journal/entries.json')
    es=json.loads(registry.read_text())
    entry=next(e for e in es if e['id']=='option-handoff-opportunities')
    entry['result']=msg;entry['analysis']=local_text+' 两组共用同一后续 A，200 个基线哈希一致。'
    tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(es,ensure_ascii=False,indent=2)+'\n');tmp.replace(registry)
    index=exp.DOC.parent/'README.md'
    index_text=index.read_text();link=f'{exp.DOC.name}/README.md'
    if link not in index_text:
        ref=f"| 2026-11-23 | [逐 option 验证跨 CAP 准备与接手]({link}) | 在 200 个新房间逐步检查冻结条件，交接 {len(active)} 次，整局成功 {t['first']} 对 {t['second']}，救回 {t['plus']}、丢失 {t['minus']} |"
        index.write_text(index_text.replace('|---|---|---|','|---|---|---|\n'+ref,1))
    print(msg,local_text,flush=True)


if __name__=='__main__':main()
