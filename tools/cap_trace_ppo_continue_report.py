"""Fixed-budget, paired endpoint and change-in-gap report for continued PPO."""
from pathlib import Path
import shutil
import numpy as np
from tools.cap_trace_value_long_report import write_csv
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save

OLD=Path('outputs/semantic_rl/cap_trace_ppo_confirm_20261118_01')


def curves(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for arm in ('GA','GLOBAL'):
        h=read(out/arm/'history.json')
        for ax,key in zip(axes.flat,('success','entropy','value_mse')):
            ax.plot([r['round']*50 for r in h],[r[key] for r in h],label=arm)
            ax.set_title({'success':'Training successes / 50','entropy':'PPO policy entropy','value_mse':'Critic MSE'}[key])
        points=[]
        for r in (40,60,80):
            p=OLD/'confirmation'/f'{arm}_s17_post'/'results.json' if r==40 else out/'diagnostic'/f'{arm}_r{r}'/'results.json'
            if p.exists():points.append((r*50,sum(s['success'] for s in read(p))))
        axes.flat[3].plot(*zip(*points),'o-',label=arm)
    axes.flat[3].set_title('Greedy successes / 200 (previous panel)')
    for ax in axes.flat:
        ax.legend();ax.set_xlabel('Cumulative online episodes');ax.grid(alpha=.2)
    fig.savefig(out/'curves.tmp.png',dpi=120);(out/'curves.tmp.png').replace(out/'curves.png');plt.close(fig)


def report(out):
    panel=read(out/'confirmation.json')['cases']
    ids=[c['case_id'] for c in panel]
    data={}
    for arm in ('GA','GLOBAL'):
        for r in (40,80):
            rows=read(out/'confirmation'/f'{arm}_r{r}'/'results.json')
            assert len(rows)==200 and all(row['complete'] for row in rows)
            scores={row['case_id']:int(row['success']) for row in rows}
            data[f'{arm}_{r*50}']=np.array([scores[c] for c in ids])
    records=[dict(case_id=c['case_id'],movable_count=c['movable_count'],content_sha256=c['content_sha256'],
        **{k:int(v[i]) for k,v in data.items()}) for i,c in enumerate(panel)]
    write_csv(out/'per_scene.csv',records)
    rng=np.random.default_rng(20261118)
    indices=np.concatenate([rng.choice([i for i,c in enumerate(panel) if c['movable_count']==n],
        size=(10000,50),replace=True) for n in range(2,6)],axis=1)
    pre=data['GLOBAL_2000']-data['GA_2000']
    post=data['GLOBAL_4000']-data['GA_4000']
    comparisons={}
    for k,d in dict(change_in_gap=post-pre,start_gap=pre,end_gap=post,
        GLOBAL_gain=data['GLOBAL_4000']-data['GLOBAL_2000'],GA_gain=data['GA_4000']-data['GA_2000']).items():
        comparisons[k]=dict(pp=100*float(d.mean()),ci95=(100*np.quantile(d[indices].mean(1),[.025,.975])).tolist())
        if k!='change_in_gap':comparisons[k].update(gains=int((d>0).sum()),losses=int((d<0).sum()))
    counts={k:int(v.sum()) for k,v in data.items()}
    summary=dict(counts=counts,comparisons=comparisons,independent_scenes=200,seed=17)
    save(out/'summary.json',summary)
    c=comparisons['change_in_gap'];lo,hi=c['ci95']
    verdict=('这次单种子对照支持续训时差距进一步扩大。' if lo>0 else
             '这次单种子对照显示续训时差距缩小。' if hi<0 else
             '这次单种子对照未确认续训时差距进一步扩大。')
    lines=['# GA 与 GLOBAL 再训练 2000 episode', '',verdict,'','|新面板|起点|终点|','|---|---:|---:|']
    for arm in ('GA','GLOBAL'):
        lines.append(f'|{arm}|{counts[arm+"_2000"]}/200|{counts[arm+"_4000"]}/200|')
    lines.extend(['','|比较|差值（百分点）|95% 场景配对区间|','|---|---:|---:|'])
    for k,label in [('change_in_gap','差距变化，主指标'),('start_gap','初始 GLOBAL − GA'),('end_gap','最终 GLOBAL − GA'),('GLOBAL_gain','GLOBAL 续训收益'),('GA_gain','GA 续训收益')]:
        x=comparisons[k];a,b=x['ci95']
        lines.append(f'|{label}|{x["pp"]:+.1f}|[{a:+.1f}, {b:+.1f}]|')
    lines+=['','两组采用同一 seed17，来自上一确认面板并列最优的检查点。各重放36批历史更新，每批权重逐张量复现，再用恢复的Adam和小批次随机数状态继续40批。两组共4000次新增训练，800次已查看面板诊断，800次新面板评估。每200次访问同一批200训练初态一次，场景顺序与动作采样种子成对。固定最终检查点，同一批200个新场景评估起点与终点。区间条件于本次训练链，按物体数分层、配对重抽场景10000次。','','[逐场景结果](per_scene.csv) · [配对统计](summary.json) · [训练与诊断曲线](curves.png)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    assets=Path('docs/agent_exchange/assets')/out.name
    assets.mkdir(exist_ok=True)
    for name in ('REPORT.md','per_scene.csv','summary.json','curves.png','protocol.json','FROZEN.json'):
        shutil.copyfile(out/name,assets/name)
    save(out/'FINISHED.json',dict(complete=True,new_training=4000,diagnostic=800,new_panel=800,**summary))
