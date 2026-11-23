"""Scene-paired dose-response curves for inverse parent sampling."""
import csv
import os
import shutil
from collections import Counter, defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.planar_tasks.authoring_v2 import cap_inverse_scale as study
from experiments.planar_tasks.authoring_v2.cap_inverse_fit import rows_for
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_trace_value_long_report import write_csv

OUT=study.OUT
LEVELS=(0,*study.DOSES)


def estimates(vec,mask,draws):
    denominator=mask[draws].sum(1)
    means=100*((vec*mask)[draws].sum(1)[denominator>0]/denominator[denominator>0])
    return dict(pp=float(100*vec[mask].mean()),ci_pp=np.quantile(means,[.025,.975]).tolist())


def figure(table):
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),sharex=True,layout='constrained')
    for ax,(k,title,color) in zip(axes,[('all','All new scenes','#086b59'),('unique','Exactly one successful CAP','#a64c18')],strict=True):
        rows=[r for r in table if r['scope']==k]
        x=np.array([r['distinct'] for r in rows])
        y=np.array([r['rate'] for r in rows])
        lower=[r['rate_low'] for r in rows]; upper=[r['rate_high'] for r in rows]
        ax.errorbar(x,y,yerr=[y-lower,upper-y],color=color,lw=1.8,marker='o',capsize=4)
        ax.set_title(f"{title} (n={rows[0]['n']})")
        ax.set_xticks(x,labels=[str(v) for v in x]);ax.set_ylim(0,100)
        ax.set_xlabel('Distinct new successful action sequences')
        ax.set_ylabel('Greedy closed-loop success (%)')
        ax.grid(axis='y',alpha=.22)
        for xi,yi,r in zip(x,y,rows):
            ax.annotate(str(r['wins']),(xi,yi),xytext=(5,8),textcoords='offset points',color=color)
    fig.suptitle('Offline GA-Q, inverse parent sampling, fixed seed 17')
    fig.savefig(OUT/'curve.png',dpi=180)
    plt.close(fig)


def report():
    cases=read(OUT/'evaluation_panel.json')
    ev=read(OUT/'evaluation.json')
    outcomes=defaultdict(dict)
    for r in ev:outcomes[r['case_id']][r['condition']]=r['success']
    rows=[]
    for c in cases:
        o=outcomes[c['case_id']]
        assert len(o)==len(study.CAPS)+len(LEVELS)
        rows.append(dict(case_id=c['case_id'],movable_count=c['movable_count'],
                         ncap=sum(o[k] for k in study.CAPS),**o))
    write_csv(OUT/'per_scene.csv',rows)
    k=np.array([r['ncap'] for r in rows])
    movable=np.array([r['movable_count'] for r in rows])
    rng=np.random.default_rng(study.SEED)
    strata=[np.flatnonzero(movable==n) for n in range(2,6)]
    assert all(len(i)==150 for i in strata)
    draws=np.concatenate([rng.choice(i,(10000,len(i))) for i in strata],axis=1)
    exposure={0:dict(neighbors=0,distinct=0,success=0,runs=0,families=0,boundaries=0)}
    for d in LEVELS[1:]: exposure[d]=read(OUT/'doses'/f'{d}.json')
    table=[]; groups={'all':k>=0,'unique':k==1,'union':k>0,'multiple':k>1,'outside':k==0}
    for scope,mask in groups.items():
        for i,d in enumerate(LEVELS):
            y=np.array([r[f'q{d}'] for r in rows]);prev=np.array([r[f'q{LEVELS[max(i-1,0)]}'] for r in rows])
            base=np.array([r['q0'] for r in rows])
            point=estimates(y,mask,draws);ref=estimates(y-base,mask,draws);step=estimates(y-prev,mask,draws)
            table.append(dict(scope=scope,target=d,distinct=exposure[d]['distinct'],n=int(mask.sum()),wins=int(y[mask].sum()),
                rate=point['pp'],rate_low=point['ci_pp'][0],rate_high=point['ci_pp'][1],
                delta0=ref['pp'],delta0_low=ref['ci_pp'][0],delta0_high=ref['ci_pp'][1],
                delta_previous=step['pp'],previous_low=step['ci_pp'][0],previous_high=step['ci_pp'][1],
                gained0=int(((y>base)&mask).sum()),lost0=int(((y<base)&mask).sum())))
    write_csv(OUT/'results.csv',table)
    full=[]
    for b in range(1,read(OUT/'collection.json')['batches']+1):
        full.extend(read(OUT/'batches'/f'{b:03d}_results.json'))
    anchor_count={p['case_id']:len(p['winners']) for p in read(OUT/'anchors.json')}
    write_csv(OUT/'collection.csv',[dict(index=r['index'],case_id=r['case_id'],parent=r['parent'],k=anchor_count[r['parent']],
                   cap=r['cap'],success=int(r['success']),boundaries=r['shard']['boundaries'],sequence=r['sequence'],
                   replay_sha256=r['sha256']) for r in full])
    used=rows_for(LEVELS[-1])
    counts=Counter(r['parent'] for r in used if r['cap']==study.CAPS[0])
    collection=study.stats(full)
    save(OUT/'summary.json',dict(doses=exposure,whole_collection=collection,sampled_parents=dict(counts),closed_loop=table,
                                 complete_episodes=len(ev),fixed_training_seed=17))
    figure(table)
    lines=['# 原场景成功数反比采样的数据规模曲线','',
           '从同一批 200 个可解训练初态中有放回抽取基础族。原场景有 k 个成功 CAP 时，抽取权重为 1/k。原先零成功的场景不生成邻居，其既有失败轨迹随全量原训练集保留。每个受扰动初态完整执行七个 CAP。','',
           '| 不同成功序列 | 新邻居 | 来源基础族 | 新增成功 / 失败 | 总训练轨迹 | 总训练决策边界 |',
           '|---:|---:|---:|---:|---:|---:|']
    for d in LEVELS:
        e=exposure[d];lines.append(f"| {e['distinct']} | {e['neighbors']} | {e['families']} | {e['success']} / {e['runs']-e['success']} | {4256+e['runs']} | {294285+e['boundaries']} |")
    lines+=['','各剂量按预定的初态生成顺序嵌套。阈值只对完整成功轨迹的动作序列去重计数；训练保留截至该邻居的所有真实运行。各模型共用原始归一化和训练 seed17，从头训练 16,000 次 V 与 8,000 次 Q；零剂量复用冻结的原离线 Q。各剂量的更新数相同，单条样本的平均训练曝光随数据量改变。','',
            '| 不同成功序列 | 全面板成功 | 相对零剂量（百分点，95% 区间） | 独有场景成功 | 相对零剂量（百分点，95% 区间） |',
            '|---:|---:|---:|---:|---:|']
    for d in LEVELS:
        cells=[str(exposure[d]['distinct'])]
        for s in ['all','unique']:
            r=next(x for x in table if x['scope']==s and x['target']==d)
            cells+=[f"{r['wins']}/{r['n']}",f"{r['delta0']:+.2f} [{r['delta0_low']:+.2f}, {r['delta0_high']:+.2f}]"]
        lines.append('| '+' | '.join(cells)+' |')
    lines+=['','![闭环数据规模曲线](curve.png)','',
            '在固定的 600 个新初态上执行七个单 CAP 与五个冻结模型，共 7,200 次完整闭环。主比较为最高剂量相对零剂量在恰一个 CAP 成功场景中的保留率差。对初态按物体数分层、配对重抽 10,000 次。区间条件于本采集序列、单训练种子；中间点和相邻剂量差用于描述曲线形状。该曲线衡量反比采样整套方法在固定训练预算下的表现；本轮没有同规模均匀增广臂。','',
            '[逐场景结果](per_scene.csv) · [各档统计及相邻点差值](results.csv) · [采集明细](collection.csv) · [冻结协议](protocol.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    for f in ['curve.png','results.csv']:shutil.copyfile(OUT/f,OUT/'live'/f)
    save(OUT/'FINISHED.json',dict(complete=True,episodes=len(ev),files={k:sha(OUT/k) for k in
         ['protocol.json','summary.json','per_scene.csv','results.csv','collection.csv','frozen.json','curve.png','REPORT.md']}))


if __name__=='__main__':report()
