"""Summarize the paired closed-loop continuation-target ablation."""
import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from experiments.planar_tasks.authoring_v2.cap_q_continuation import OUT, Q, protocol, scale
from experiments.planar_tasks.authoring_v2.cap_temporal_compare import ARCHIVE, add_entry, archive
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_inverse_scale_report import estimates
from tools.cap_trace_value_long_report import write_csv


def paired_actions(case_id):
    replays=[]
    for root in (scale.OUT/'evaluation/q1001',OUT/'evaluation'):
        with gzip.open(root/case_id/'result.json.gz','rt') as stream:
            replays.append(json.load(stream))
    a,b=replays
    for i,(x,y) in enumerate(zip(a['decisions'],b['decisions'])):
        assert a['public_rows'][i]['history_hash']==b['public_rows'][i]['history_hash']
        if x['action']!=y['action']:
            return i
    assert len(a['decisions'])==len(b['decisions'])
    assert a['public_rows'][-1]['history_hash']==b['public_rows'][-1]['history_hash']
    assert a['summary']['success']==b['summary']['success']
    return None


def main():
    protocol()
    assert read(OUT/'reproduction.json')['exact_state_dict']
    assert read(OUT/'baseline_recheck.json')['exact_public_histories_and_scored_actions']
    assert read(OUT/'own_frozen.json')['sha256']==sha(Q)
    with (scale.OUT/'per_scene.csv').open() as f:
        baseline=list(csv.DictReader(f))
    new=read(OUT/'evaluation.json')
    outcomes={r['case_id']:r for r in new}
    cases={c['case_id'] for c in read(scale.OUT/'evaluation_panel.json')}
    assert len(new)==len(baseline)==len(cases)==600
    assert set(outcomes)==cases=={r['case_id'] for r in baseline}
    rows=[]
    for b in baseline:
        s=outcomes[b['case_id']]
        rows.append(dict(case_id=b['case_id'],movable_count=int(b['movable_count']),expert_successes=int(b['ncap']),
                         max_success=int(b['q1001']),own_success=int(s['success']),own_ticks=s['ticks'],
                         own_decisions=s['decisions'],own_stop=s['stop']))
    with ProcessPoolExecutor(4) as pool:
        divergences=list(pool.map(paired_actions,[r['case_id'] for r in rows],chunksize=4))
    for row,first in zip(rows,divergences,strict=True):
        row['first_different_action_index']=first
    opening=sum(i==0 for i in divergences)
    later=sum(i is not None and i>0 for i in divergences)
    unchanged=divergences.count(None)
    write_csv(OUT/'per_scene.csv',rows)
    ncap,maximum,own,size=(np.array([r[k] for r in rows]) for k in
                           ('expert_successes','max_success','own_success','movable_count'))
    rng=np.random.default_rng(scale.SEED)
    draws=np.concatenate([rng.choice(np.flatnonzero(size==k),(10000,(size==k).sum())) for k in np.unique(size)],axis=1)
    groups=[('全部',ncap>=0),('仅一个 CAP 成功',ncap==1),('至少一个 CAP 成功',ncap>0),('七个 CAP 均失败',ncap==0)]
    table=[]
    for name,keep in groups:
        table.append(dict(group=name,n=int(keep.sum()),max_success=int(maximum[keep].sum()),
                          own_success=int(own[keep].sum()),max_only=int(((maximum>own)&keep).sum()),
                          own_only=int(((own>maximum)&keep).sum()),**estimates(maximum-own,keep,draws)))
    t=read(OUT/'targets.json')
    save(OUT/'summary.json',dict(paired=table,primary='all scenes, max minus own',targets=t,action_paths=dict(first_action_changed=opening,later_changed=later,identical=unchanged)))
    overall=table[0]
    lo,hi=overall['ci_pp']
    if lo>0:
        interpretation='本面板支持把下一步的专家选择纳入离线监督能提高 GA-Q 的闭环成功率。'
    elif hi<0:
        interpretation='本面板中，使用原专家的续接值获得了更高的闭环成功率。'
    else:
        interpretation='总体配对区间包含零，本面板未能确定跨专家续接标签带来净提升；两种监督的等效性也尚待检验。'
    lines=['# GA-Q 的跨专家续接标签消融','',interpretation,'',
           '最高增广剂量、seed17，复用冻结专家 V。两组使用相同训练转移、归一化、动作、Q 初始化、逐批抽样和优化器，均训练 8,000 步。max 组用后继最高合法专家 V，own 组用原轨迹专家在同一后继的 V。终止标签完全相同。部署时两组均对同一合法候选池贪心重选。','',
           '| 场景 | 分母 | max 成功 | own 成功 | 仅 max / 仅 own | max − own，百分点（95% 区间） |',
           '|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['group']} | {r['n']} | {r['max_success']} | {r['own_success']} | {r['max_only']} / {r['own_only']} | {r['pp']:+.2f} [{r['ci_pp'][0]:+.2f}, {r['ci_pp'][1]:+.2f}] |")
    lines+=['',f"训练转移共 {t['boundaries']:,} 条。{t['terminal']:,} 条终止转移的目标相同。在 {t['nonterminal']:,} 条非终止转移中，{t['positively_changed']:,} 条的 max 目标较高，{t['change_above_01']:,} 条差值超过 0.1。目标升幅均值 {t['mean_gap']:.4f}、中位数 {t['median_gap']:.4f}、第 90 百分位 {t['p90_gap']:.4f}。",'',
            f'逐场景比较实际选中的动作，{opening} 例从第一步就不同，{later} 例在相同开局动作之后分叉，{unchanged} 例全程动作一致。分叉前两条轨迹的公开历史哈希逐步相符。','',
            '归档 max 标签逐条一致，以原标签重训所得模型全部权重逐项一致。每种可动物体数选取第一个场景重放 max 基线，公开历史与带评分的动作记录完全一致。600 个 max 结果复用原归档，own 的 600 个结果来自本次完整闭环。','',
            '主比较为全场景成功率。区间按可动物体数分层，对配对场景重抽 10,000 次，条件于一个训练种子和一份数据。这是已查看的诊断面板，各子组为描述性结果。两组都能切换，差异只检验训练标签；跨专家后缀的真实成功率未在本轮测量。','',
            '[逐场景数据](per_scene.csv) · [冻结协议](protocol.json) · [标签分布](targets.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['REPORT.md','protocol.json','summary.json','per_scene.csv','targets.json','reproduction.json','baseline_recheck.json','own_frozen.json']
    archive(OUT,ARCHIVE/'2026-11-19_q_continuation',files)
    add_entry('| 2026-11-19 | [GA-Q 最高专家与原专家续接标签消融](2026-11-19_q_continuation/README.md) | 同一 600 场景、最高剂量、seed17，重放基线权重逐项一致 |')
    save(OUT/'FINISHED.json',dict(complete=True,paired_scenes=600,files={f:sha(OUT/f) for f in files}))


if __name__=='__main__':
    main()
