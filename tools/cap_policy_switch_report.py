"""Per-scene paired comparison of the highest-dose frozen GA V and Q."""
import csv
import time

import numpy as np

from experiments.planar_tasks.authoring_v2 import cap_inverse_scale as scale
from experiments.planar_tasks.authoring_v2.cap_policy_switch import OUT
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.authoring_v2.cap_temporal_compare import archive, add_entry, ARCHIVE
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_inverse_scale_report import estimates
from tools.cap_trace_value_long_report import write_csv


def report():
    while not (scale.OUT/'FINISHED.json').exists():
        if (scale.OUT/'pipeline_error.json').exists():
            raise RuntimeError(read(scale.OUT/'pipeline_error.json'))
        time.sleep(10)
    protocol=read(OUT/'protocol.json')
    assert sha(scale.OUT/'evaluation_panel.json')==protocol['panel_sha256']
    assert read(scale.OUT/'frozen.json')['1001']==protocol['q_sha256']
    assert sha(scale.OUT/'per_scene.csv')==read(scale.OUT/'FINISHED.json')['files']['per_scene.csv']
    results={r['case_id']:r for r in read(OUT/'evaluation.json')}
    with (scale.OUT/'per_scene.csv').open() as f:
        rows=list(csv.DictReader(f))
    assert len(results)==len(rows)==600
    for r in rows:
        v=results[r['case_id']]
        expert=v['initial_expert']
        r.update(v_switch=v['success'],initial_cap=None if expert is None else scale.CAPS[expert],
                 commit=None if expert is None else int(r[scale.CAPS[expert]]))
    write_csv(OUT/'per_scene.csv',rows)
    v=np.array([r['v_switch'] for r in rows])
    q=np.array([int(r['q1001']) for r in rows])
    ncap=np.array([int(r['ncap']) for r in rows])
    commit=np.array([-1 if r['commit'] is None else r['commit'] for r in rows])
    size=np.array([int(r['movable_count']) for r in rows])
    rng=np.random.default_rng(scale.SEED)
    draw=np.concatenate([rng.choice(np.flatnonzero(size==i),(10000,150)) for i in range(2,6)],axis=1)
    table=[]
    for name,mask in [('all',ncap>=0),('unique',ncap==1),('union',ncap>0),('outside',ncap==0)]:
        for reference,vec in [('GA-Q',q),('commit',commit)]:
            keep=mask&(vec>=0)
            if not keep.any():
                continue
            table.append(dict(group=name,reference=reference,n=int(keep.sum()),switch_success=int(v[keep].sum()),
                         reference_success=int(vec[keep].sum()),gain=int(((v>vec)&keep).sum()),loss=int(((v<vec)&keep).sum()),
                         **estimates(v-vec,keep,draw)))
    counts={cap:sum(int(r[cap]) for r in rows) for cap in scale.CAPS}
    save(OUT/'summary.json',dict(paired=table,single_caps=counts))
    lines=['# 七 CAP 逐步按专家 V 切换与动作 Q 的闭环表现','',
           '冻结同一最高剂量 seed17 GA 模型的 V 与 Q。V 在每个决策边界选择预测续接成功率最高的合法专家并执行一次动作，再于下一边界重选。Q 对合法去重动作排序。首步固定专家的结果直接按初始 V 选中的 CAP 查该场景已有单 CAP 闭环。','',
           '| 范围 | 参照 | 初态 | V 切换成功 | 参照成功 | 新增 / 丢失 | 差值（百分点，95% 区间） |','|---|---|---:|---:|---:|---:|---:|']
    for r in table:
        lines.append(f"| {r['group']} | {r['reference']} | {r['n']} | {r['switch_success']} | {r['reference_success']} | {r['gain']} / {r['loss']} | {r['pp']:+.2f} [{r['ci_pp'][0]:+.2f}, {r['ci_pp'][1]:+.2f}] |")
    lines.extend(['','单 CAP 在同一 600 场景中的成功数为 '+ '、'.join(f"{k} {v}" for k,v in counts.items())+'。',
        '','区间基于按可动物体数分层的 10,000 次场景配对重抽，条件于固定训练种子和采集数据。此处检验新场景上的经验表现。固定专家库、同分布场景与近似学习价值限制了泛化范围。',
        '','经典 policy switching 提供选专家的规则。表征或架构的贡献需要在固定切换规则和候选支持下另行比较，并在新冻结面板复验。[Chang、Givan 和 Chong](https://engineering.purdue.edu/~givan/papers/deds01.pdf)',
        '','[逐场景结果](per_scene.csv) · [冻结实验设定](protocol.json)'])
    from tools.cap_policy_switch_behavior import main as measure
    behavior=read(OUT/'behavior.json') if (OUT/'behavior.json').exists() else measure()
    lines.extend(['', '开局各专家选择次数为 '+ '、'.join(f"{k} {v}" for k,v in behavior['initial_choices'].items())+'。'
                  +f" 600 条轨迹中，{behavior['differing_episodes']} 条至少执行过一次与开局专家当前提案不同的动作，合计 {behavior['differing_actions']}/{behavior['decisions']} 个决策边界。多个专家推荐同一动作时，此口径只检查实际动作是否改变。",
                  f" 开局最高合法专家与次高者的 V 分差中位数为 {behavior['opening_top_margin_median']:.5f}，最高分精确并列 {behavior['opening_top_margin_zero']} 次。"])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    files=['REPORT.md','protocol.json','summary.json','per_scene.csv','behavior.json']
    archive(OUT,ARCHIVE/'2026-11-19_cap_policy_switch',files)
    add_entry('| 2026-11-19 | [七 CAP 的专家 V 切换与动作 Q](2026-11-19_cap_policy_switch/README.md) | 同一组 600 个初态和最高剂量 seed17 权重，含首步选定专家对照 |')
    save(OUT/'FINISHED.json',dict(complete=True,episodes=len(results),files={f:sha(OUT/f) for f in files}))


if __name__=='__main__':report()
