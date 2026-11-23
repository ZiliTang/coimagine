"""Freeze, queue and compare the highest-dose GLOBAL-attention checkpoint."""
import csv
import multiprocessing as mp
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import cap_temporal_attention as attention
from . import cap_inverse_scale as scale
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import temporal_data, temporal_q
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save
from tools.cap_inverse_scale_report import estimates
from tools.cap_trace_value_long_report import write_csv

OUT=attention.OUT
ARCHIVE=Path('docs/research/key_findings')
SCALE_ARCHIVE=ARCHIVE/'2026-11-19_inverse_scale'
ATTENTION_ARCHIVE=ARCHIVE/'2026-11-19_global_attention'


def freeze():
    path=OUT/'protocol.json'
    spec=dict(seed=17,dose=1001, panel_sha256=sha(scale.OUT/'evaluation_panel.json'),
              epochs='16000 expert V, 8000 candidate Q, fixed last checkpoint',batch=64,
              learning='3e-4 Adam with gradient norm clip 1, uniform decision boundaries and original normalizers',
              norm_sha256=sha(scale.PARENT/'normalizers.npz'),dose_train='same original and four increment partitions as highest GA-Q, all outcomes',
              model='initial geometry and latest 8 frames, 8 tokens per valid time, 32 dimensions, 4 attention heads, shared 64x64 scoring head',
              inputs='public geometry, budget, seven proposals and traces, executed previous action in geometry, its source support and elapsed time',
              historical_Q_input=False,ppo=False,
              targets='V regresses terminal success of executing CAP; Q regresses reward plus max legal frozen V at actual successor for nonterminal only',
              comparison='600 paired scenes with highest dose GA-Q. Overall and exactly one successful CAP, union and outside union; stratified paired 95% intervals, one seed.',
              scope='combined model and input change; no trace-only attribution',
              source={Path(m.__file__).name:sha(m.__file__) for m in (attention,temporal_data,temporal_q)})
    if path.exists():assert read(path)==spec
    else:save(path,spec)


def graph(folder):
    fig,axes=plt.subplots(1,2,figsize=(9,3.6),layout='constrained')
    for ax,stage in zip(axes,['V','Q'],strict=True):
        curve=read(folder/(stage+'_curve.json'))
        for name,color in [('train','#137467'),('validation','#9c4c1e')]:
            ax.plot([r['update'] for r in curve],[r[name]['mse'] for r in curve],marker='.',label=name,color=color)
        ax.set(xlabel='Optimizer updates',ylabel='Fixed-probe MSE',title=stage)
        ax.legend()
        ax.grid(axis='y',alpha=.2)
    fig.savefig(folder/'loss.png',dpi=160)
    plt.close(fig)


def archive(folder,dest,files):
    dest.mkdir(parents=True,exist_ok=True)
    for filename in files:
        shutil.copyfile(folder/filename,dest/('README.md' if filename=='REPORT.md' else filename))


def add_entry(line):
    index=ARCHIVE/'README.md'
    s=index.read_text()
    if line not in s:
        index.write_text(s.replace('\n尚待验证',line+'\n\n尚待验证'))


def compare(rows):
    with (scale.OUT/'per_scene.csv').open() as f:
        baseline=list(csv.DictReader(f))
    results={r['case_id']:r for r in rows}
    assert len(baseline)==len(results)==len(rows)==600
    for r in baseline:
        r['attention']=int(results[r['case_id']]['success'])
    write_csv(OUT/'per_scene.csv',baseline)
    ga=np.array([int(r['q1001']) for r in baseline])
    att=np.array([r['attention'] for r in baseline])
    caps=np.array([int(r['ncap']) for r in baseline])
    movable=np.array([int(r['movable_count']) for r in baseline])
    rng=np.random.default_rng(scale.SEED)
    draws=np.concatenate([rng.choice(np.flatnonzero(movable==k),(10000,150)) for k in range(2,6)],axis=1)
    groups=[('全部',caps>=0),('恰一个 CAP 成功',caps==1),('七 CAP 成功并集',caps>0),('并集以外',caps==0)]
    table=[]
    lines=['# 最高剂量 GLOBAL attention 与 GA-Q 的闭环对照','',
           '同一组完整训练轨迹、归一化、候选动作及训练预算。两组均使用 seed17，在同一组 600 个初态上贪心执行。正差值代表 attention 成功率更高。','',
           '| 场景 | 初态数 | GA-Q 成功 | Attention 成功 | 新增 / 丢失 | 差值（百分点，95% 区间） |','|---|---:|---:|---:|---:|---:|']
    for name,mask in groups:
        pair=estimates(att-ga,mask,draws)
        r=dict(group=name,n=int(mask.sum()),ga=int(ga[mask].sum()),attention=int(att[mask].sum()),
               gained=int(((att>ga)&mask).sum()),lost=int(((att<ga)&mask).sum()),**pair)
        table.append(r)
        lines.append(f"| {name} | {r['n']} | {r['ga']} | {r['attention']} | {r['gained']} / {r['lost']} | {r['pp']:+.2f} [{r['ci_pp'][0]:+.2f}, {r['ci_pp'][1]:+.2f}] |")
    lines.extend(['','区间按场景配对、按可动物体数分层重抽 10,000 次，条件于这一训练种子。差异同时包含时间窗口内七专家 trace 和注意力架构的影响。','',
                  '![固定样本训练与验证损失](loss.png)','',
                  '[逐场景配对](per_scene.csv) · [训练和评估设定](protocol.json) · [冻结权重摘要](frozen.json)'])
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'summary.json',table)
    graph(OUT)
    files=['REPORT.md','summary.json','protocol.json','frozen.json','per_scene.csv','loss.png']
    archive(OUT,ATTENTION_ARCHIVE,files)
    add_entry('| 2026-11-19 | [最高剂量 GLOBAL attention 与 GA-Q](2026-11-19_global_attention/README.md) | 同一 600 场景、固定 seed17 的闭环配对 |')
    save(OUT/'FINISHED.json',dict(episodes=len(results),files={f:sha(OUT/f) for f in files}))


def full():
    freeze()
    save(OUT/'state.json',dict(stage='waiting_for_GA_scale_curve'))
    while not (scale.OUT/'FINISHED.json').exists():
        if (scale.OUT/'pipeline_error.json').exists():raise RuntimeError(read(scale.OUT/'pipeline_error.json'))
        time.sleep(20)
    assert read(OUT/'smoke/result.json')['parity']['max_absolute_score_difference']<=1e-6
    receipt=read(scale.OUT/'FINISHED.json')
    assert receipt['complete'] and sha(scale.OUT/'per_scene.csv')==receipt['files']['per_scene.csv']
    archive(scale.OUT,SCALE_ARCHIVE,['REPORT.md','summary.json','results.csv','collection.csv','per_scene.csv','protocol.json','curve.png'])
    add_entry('| 2026-11-19 | [反比采样的 GA-Q 增广规模曲线](2026-11-19_inverse_scale/README.md) | 五档、同一 600 初态共 7,200 次闭环、固定 seed17 |')
    index=Path('docs/agent_exchange/CURRENT.md')
    text=index.read_text().replace('按原场景成功数反比采样的千条规模曲线运行中。','[按原场景成功数反比采样的千条规模曲线已完成](../research/key_findings/2026-11-19_inverse_scale/README.md)。')
    index.write_text(text)
    save(OUT/'state.json',dict(stage='fitting'))
    if not (OUT/'fit.json').exists():attention.fit(OUT,False)
    checkpoint=str(OUT/'Q.pt')
    seals=dict(attention=sha(checkpoint),ga=read(scale.OUT/'frozen.json')['1001'],
               dose=sha(scale.OUT/'doses/1001.json'),panel=sha(scale.OUT/'evaluation_panel.json'))
    assert seals['panel']==read(OUT/'protocol.json')['panel_sha256']
    if (OUT/'frozen.json').exists():assert read(OUT/'frozen.json')==seals
    else:save(OUT/'frozen.json',seals)
    cases=read(scale.OUT/'evaluation_panel.json')
    rows=[]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=attention.initialize_worker,
                             initargs=(checkpoint,)) as pool:
        jobs=[dict(case=c,condition='global_attention',epsilon=0.,checkpoint=checkpoint,
                   directory=str(OUT/'evaluation'/c['case_id'])) for c in cases]
        for future in as_completed([pool.submit(attention.evaluate_case,j) for j in jobs]):
            row=future.result()
            assert row['complete'] and row['error'] is None
            rows.append({k:row[k] for k in ('case_id','success','stop','decisions','ticks')})
            if len(rows)%10==0:
                save(OUT/'state.json',dict(stage='evaluating',completed=len(rows),total=600))
    save(OUT/'evaluation.json',rows)
    compare(rows)
    s=index.read_text()
    marker='[未来研究方向讨论]'
    if '2026-11-19_global_attention/README' not in s:
        index.write_text(s.replace(marker,'[最高剂量 GLOBAL attention 与 GA-Q 的 600 场景配对结果](../research/key_findings/2026-11-19_global_attention/README.md)\n\n'+marker,1))
    save(OUT/'state.json',dict(stage='complete'))


if __name__=='__main__':
    try:
        full()
    except BaseException:
        save(OUT/'error.json',dict(error=traceback.format_exc()))
        raise
