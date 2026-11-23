"""Paired independent confirmation of authored recovery and graph readout."""
import csv
import json
from pathlib import Path

import numpy as np

from tools import cap_authored_handoff as e
from tools.cap_option_handoff_opportunity_report import paired


def text(p):
    lo,hi=p['ci']
    return (f"{p['first']}/{p['n']} 对 {p['second']}/{p['n']}，救回 {p['plus']}、丢失 {p['minus']}，"
            f"差 {100*(p['first']-p['second'])/p['n']:+.1f} 个百分点 [{100*lo:+.1f}, {100*hi:+.1f}]")


def publish():
    ids=e.read(e.OUT/'confirmation_ids.json')
    surfaces=[np.load(e.OUT/f'commit{i}.npy',mmap_mode='r') for i in (0,1)]
    rows=[]
    for k in sorted(ids):
        modes=e.read(e.OUT/'test'/(k+'.json'))
        a,b,hard,graph=[modes[m] for m in ('A','B','lock','graph')]
        r=dict(room=k,content_sha256=ids[k],trigger=int(a['hit'] is not None),
               trigger_tick=a['hit']['tick'] if a['hit'] else '',trigger_hash=a['hit']['h'] if a['hit'] else '',trigger_node=a['hit']['z'] if a['hit'] else '',
               lock_equals_B=int(hard['result']['history']==b['result']['history']),
               graph_equals_A=int(graph['result']['history']==a['result']['history']),
               graph_equals_lock=int(graph['result']['history']==hard['result']['history']),
               graph_switched=int(graph['first_take'] is not None),graph_changes=graph['changed'],
               graph_switches=graph['switches'],lock_changes=hard['changed'],
               graph_B=graph['counts'].get('1',0),graph_A=graph['counts'].get('0',0))
        for m,v in modes.items():
            r[m]=v['result']['success'];r[m+'_ticks']=v['result']['ticks']
        r['estimate_A']=r['estimate_B']=''
        pick=False
        if a['hit']:
            h=a['hit']; idx=((1800-h['tick'])//5,h['z'],300-a['hit_decisions'])
            r['estimate_A'],r['estimate_B']=[float(v[idx]) for v in surfaces]
            pick=r['estimate_B']>r['estimate_A']+1e-6
        r['graph_once']=r['lock'] if pick else r['A']
        r['once_chose_B']=int(pick)
        rows.append(r)
    assert len(rows)==len({r['content_sha256'] for r in rows})==200
    trainids={e.s.phase.input_hash(f) for f in e.dev().values()}
    assert not trainids & set(ids.values())
    with (e.DOC/'routes.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    def comp(first,second,group):
        return paired([r[first] for r in group],[r[second] for r in group])
    triggered=[r for r in rows if r['trigger']]
    both=[r for r in rows if not r['A'] and not r['B']]
    scores={mode:sum(r[mode] for r in rows) for mode in ('A','B','lock','graph')}
    key_comps={n:comp(x,y,rows) for n,x,y in [('lock_A','lock','A'),('graph_A','graph','A'),('graph_lock','graph','lock'),('graph_B','graph','B'),('B_A','B','A'),('lock_B','lock','B')]}
    active={n:comp(x,y,triggered) for n,x,y in [('lock_A','lock','A'),('graph_A','graph','A'),('graph_lock','graph','lock')]}
    extras={x:sum(r[x] for r in both) for x in ['lock','graph']}
    en=e.read(e.OUT/'graph.json');sel=e.read(e.OUT/'member.json');ref=e.read(e.OUT/f"probe{sel['revision']}.json")
    results=dict(scores=scores,paired=key_comps,triggered=len(triggered),triggered_paired=active,
                 two_experts_failed=len(both),outside_success=extras,
                 reached=200-len(both),graph_distinct_rooms=sum(r['graph_changes']>0 for r in rows),
                 returned_or_switched_again=sum(r['graph_switches']>=2 for r in rows))
    different=[r for r in triggered if r['A']!=r['lock']]
    correctly=sum((r['estimate_B']-r['estimate_A'])*(r['lock']-r['A'])>1e-6 for r in different)
    ties=sum(abs(r['estimate_B']-r['estimate_A'])<=1e-6 for r in different)
    once=comp('graph_once','A',rows)
    results['hindsight_at_X']=sum(max(r['A'],r['lock']) for r in rows)
    results['never_trigger_A']=[sum(r['A'] for r in rows if not r['trigger']),len(rows)-len(triggered)]
    results['once']=dict(versus_A=once,n_discordant=len(different),correct=correctly,ties=ties,
                        chose_B=sum(r['once_chose_B'] for r in rows))
    subgroups=[]
    for label,predicate in [('近目标（公开估计 ≤1 m）',lambda z:1<=z//8<=3),('较远（>1 m）',lambda z:z//8>=4),
                            ('触发时有目标接触',lambda z:(z//4)%2==1),('触发时无目标接触',lambda z:(z//4)%2==0)]:
        subset=[r for r in triggered if predicate(r['trigger_node'])]
        if subset:subgroups.append(dict(label=label,n=len(subset),lock_A=comp('lock','A',subset),graph_lock=comp('graph','lock',subset)))
    results['trigger_groups']=subgroups
    group_rows='\n'.join(f"| {v['label']} | {v['n']} | {v['lock_A']['plus']} / {v['lock_A']['minus']} | {v['graph_lock']['plus']} / {v['graph_lock']['minus']} |" for v in subgroups)
    e.save(e.DOC/'metrics.json',results)
    for r in rows:
        if not r['trigger']:assert r['A']==r['graph']==r['lock']
    conclusion=[]
    for label,key_name in [('冻结条件下接手 B 相对 A','lock_A'),('图相对简单触发','graph_lock')]:
        row=key_comps[key_name]
        lo,hi=row['ci']
        conclusion.append(f"{label}"+('在独立面板观察到正增益，区间下界大于零' if lo>0 else '在独立面板观察到退化，区间上界小于零' if hi<0 else '的配对区间包含零'))
    summary='；'.join(conclusion)+'。'
    rule=e.read(e.OUT/'family.json')
    family=next(v for v in rule['baseline_candidates'] if v['family']==rule['family'])
    stats='\n'.join(f"| {name} | {text(key_comps[k])} |" for k,name in [
        ('lock_A','首次 X 后 B 执行到底，相对全 A'),('graph_lock','图相对首次 X 后保持 B'),
        ('graph_A','图相对全 A'),('graph_B','图相对全 B'),('B_A','全 B 相对全 A'),('lock_B','A 准备后交 B，相对全程 B')])
    totals='，'.join(f"{k} {scores[k]}/200" for k in scores)
    triggers=np.asarray([r['trigger_tick'] for r in triggered])
    md=f"""# 专家 B 能否从 A 的停滞前缀接手，图能否进一步改善选择

**{totals}。** {summary} 本轮为一个通用母本、一个定向作者和一套冻结图的配对实验。

## 作者和交接点

冻结通用 A（usual s003），gpt-6.2-sol 以 A 代码为母本独立开发 B。沿旧开发面板 A 的真实轨迹在至少五条命令、180 至 1150 tick 时检查，目标可见且最近四条命令估计终距减少不足 3.5 cm 时触发停滞。开发面板中 {family['n']}/200 个房间出现该条件。作者可用 16 个 A 最终失败和 8 个实际自行恢复的前缀。每个 B 在整个 A 前缀上同步从公开事件恢复记忆；接手时没有重置时间、决策数或物理状态。作者冻结第 {sel['revision']} 版，作者案例救回 {ref['rescued']}/16、丢失 {ref['lost']}/8。

全新面板有 {len(triggered)}/200 个房间首次触发，触发 tick 中位数 {np.median(triggers):.0f}。从这些相同前缀让 B 继续至终止，与 A 原后缀配对为 **{text(active['lock_A'])}**。所有未触发房间在两种路由中沿用 A。若事后逐房间从首次 X 的两条实测后缀中选胜者，成功可到 {results['hindsight_at_X']}/200。它仅说明这个接手点有多少可利用的互补机会。

## 独立配对确认

| 全部 200 房间 | 成功数、胜负差及 95% 房间重抽区间 |
|---|---|
{stats}

A、B 各自从头运行的成功并集为 {200-len(both)}/200。余下 {len(both)} 个两位单专家均失败的房间，简单触发实际救回 {extras['lock']} 个，图救回 {extras['graph']} 个。并集只作为事后可比读数。另有 {sum(r['lock_equals_B'] for r in rows)}/200 个房间，首次 X 后换 B 与全程 B 的完整历史哈希相同。图与全 A 的完整历史相同 {sum(r['graph_equals_A'] for r in rows)} 个，与简单触发完全相同 {sum(r['graph_equals_lock'] for r in rows)} 个。配对区间来自冻结策略在独立房间中的重抽（4000 次）；每房间只有一个独立样本。

## 在同一个接手点，图能分辨继续 A 与继续 B 吗

补充诊断在图上分别强制两个专家保持到终止，给同一个首次 X 估计两次成功可达概率，直接与该前缀两条真实后缀对照。A 和 B 真实一胜一负的前缀共 {len(different)} 个，预测严格排序正确 {correctly} 个，持平 {ties} 个。在所有房间只允许按这个估计在 X 选择一次，选 B 共 {results['once']['chose_B']} 次，整局相对 A 是 {text(once)}。此处与主图反复重规划共享训练数据、转移表和确认房间。

### 首次 X 中的公开局面

| 起点 | 房间数 | 换 B 相对 A 救回 / 丢失 | 图相对简单触发救回 / 丢失 |
|---|---:|---:|---:|
{group_rows}

这些组只描述同一面板，接触组与距离组交叉。小分母的方向变化不能单独确立可部署的细分条件。

## 图如何决定选谁

从作者未看过的另外 176 个开发房间采集全 A、全 B、首次 X 后保持 B 和随机交替路径。按真实历史哈希和专家编号合并重复边，得到 {en['transitions']} 条局部转移。48 个公开节点按目标距离/可见性、接触、四命令推进、静态接触划分，{en['enabled_nodes']} 个节点为两个专家均积累了至少 20 次实测转移。在每种节点、每位专家下，后继及执行时间/决策数取联合经验分布，并以该节点两位专家的混合分布提供 16 个平滑样本。图对成功吸收概率做有限预算动态规划，每个候选的所有后继按概率求和。实际耗时向上取整到 5 tick、决策次数按真实整数计，运行仍按实际预算结算。低支持节点保持 A。首次 X 之前路由都执行 A；之后图每个上层 option 边界可重选，简单触发一直保持 B。

图在 {sum(r['graph_switched'] for r in rows)} 个房间调用过 B，在 {results['graph_distinct_rooms']} 个房间实际执行过不同于当时 A 提案的动作，{results['returned_or_switched_again']} 个房间有第二次专家切换。具体终局及每房间调用/动作差异见配对表。图的局限包括节点内来路混合，以及开发分布和图自身执行分布可能不同；失败/预算耗尽均进入失败吸收端。两种方案都沿用原物理和成功定义。

[实验设计](plan.md) · [作者与 3 个候选槽](author.md) · [逐房间结果](routes.csv) · [配对统计](metrics.json)
"""
    failures_at_x=sum(not r['A'] for r in triggered)
    md+=f"\n## 研究判断\n\n停滞条件覆盖了 A 的 {failures_at_x}/{200-scores['A']} 个终局失败，也覆盖了 {len(triggered)-failures_at_x} 个原本会成功的房间。简单触发与全程 B 有 {sum(r['lock_equals_B'] for r in rows)} 个房间历史完全相同。因此，应将 B 作为独立策略的恢复能力，与需要 A 准备后才能获得的接手增益分别评价。\n"
    md+=f"\n图救回了 {extras['graph']} 个两位单专家都失败的房间；相对简单触发又救回 {key_comps['graph_lock']['plus']}、丢失 {key_comps['graph_lock']['minus']}。固定后缀的真实分歧排序为 {correctly}/{len(different)}。能组合出成功路线与稳定识别合适的路线，是两个分别可测的能力。\n"
    from tools.cap_authored_handoff_figure import plot
    plot(rows,key_comps,e.DOC/'routing.png')
    md+='\n![配对救回与丢失、锁定专家的优势排序](routing.png)\n'
    if (e.DOC/'examples.md').exists():md+='\n[成败翻转案例](examples.md) · [实际公开距离曲线](examples.png)\n'
    (e.DOC/'README.md').write_text(md)
    registry=Path('docs/research/experiment_journal/entries.json');es=json.loads(registry.read_text())
    entry=next(r for r in es if r['id']=='authored-handoff')
    entry['result']=totals+'。';entry['analysis']=summary
    if not any(x[0]=='routing.png' for x in entry['assets']):entry['assets'].append(['routing.png','图路由与后缀估值'])
    tmp=registry.with_suffix('.tmp');tmp.write_text(json.dumps(es,ensure_ascii=False,indent=2)+'\n');tmp.replace(registry)
    idx=e.DOC.parent/'README.md';body=idx.read_text();link=e.DOC.name+'/README.md'
    if link not in body:idx.write_text(body.replace('|---|---|---|','|---|---|---|\n'+f'| 2026-11-23 | [从 A 的失败前缀开发 B 并验证图路由]({link}) | 独立 200 房间，{totals} |',1))
    e.save(e.OUT/'FINISHED.json',dict(rooms=200,seal=sel['seal'],scores=scores))
    print(summary,totals,flush=True)


if __name__=='__main__':publish()
