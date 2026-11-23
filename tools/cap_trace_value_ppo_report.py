"""Report seed17 PPO against completed paired DQN rollouts."""
import csv
from pathlib import Path
import numpy as np
from tools.cap_trace_value_long_report import read, save, write_csv
from tools.cap_trace_value_shuffle_report import paired


def render(out, final=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    inputs = read(out/'inputs.json')
    baseline = Path(inputs['baseline'])
    with (baseline/'report/per_case_results.csv').open() as stream:
        cases = list(csv.DictReader(stream))
    history = read(out/'history.json')
    with (baseline/'report/training_returns.csv').open() as stream:
        old = list(csv.DictReader(stream))
    fig, axes = plt.subplots(2,3,figsize=(14,7),layout='constrained')
    x = [r['episodes'] for r in history]
    for ax,key,label in zip(axes.flat,('success','actor_loss','value_mse','behavior_entropy','kl'),('Training successes / 50','PPO actor loss','Critic MSE','Sampling entropy (nats)','Estimated policy KL')):
        ax.plot(x,[r[key] for r in history],'.-',label='PPO')
        ax.set(title=label,xlabel='Cumulative online episodes')
    axes.flat[0].plot([int(r['episodes']) for r in old],[int(r['success']) for r in old],alpha=.45,label='Shuffle DQN')
    axes.flat[0].legend()
    evaluations = []
    ax = axes.flat[5]
    for label in ('monitor','diagnostic'):
        with (baseline/'report/evaluation_curve.csv').open() as stream:
            reference=[r for r in csv.DictReader(stream) if r['panel']==label]
        first=next(r for r in reference if int(r['episodes'])==200)
        curve=[dict(panel=label,episodes=200,success=int(first['success']),count=int(first['count']))]
        for path in sorted((out/label).glob('*/results.json'),key=lambda p:int(p.parent.name[1:])):
            rows=read(path)
            curve.append(dict(panel=label,episodes=int(path.parent.name[1:]),success=sum(r['success'] for r in rows),count=len(rows)))
        evaluations.extend(curve)
        ax.plot([r['episodes'] for r in curve],[r['success']/r['count'] for r in curve],'.-',label='PPO '+label)
        if label=='diagnostic':
            ax.plot([int(r['episodes']) for r in reference],[int(r['success'])/int(r['count']) for r in reference],'--',label='Shuffle DQN diagnostic')
    ax.legend(fontsize=8)
    ax.set(title='Greedy success rate',xlabel='Cumulative online episodes',ylim=(0,1))
    fig.savefig(out/'curves.png',dpi=120)
    plt.close(fig)
    write_csv(out/'evaluation.csv',evaluations)
    if not final: return
    rows=read(out/'diagnostic/e2000/results.json')
    by_id={r['case_id']:int(r['success']) for r in rows}
    assert len(by_id)==len(cases)==200 and len(history)==36
    for case in cases:
        case['movable_count']=int(case['movable_count'])
        case['PPO']=by_id[case['case_id']]
    write_csv(out/'per_case.csv',cases)
    v=np.array([c['PPO'] for c in cases])
    comparisons={}
    for name in ('initial200','shuffle500_final2000','fixed1500_final2000','CAP_003'):
        comparisons[name]=paired(v,[int(c[name]) for c in cases],cases)
    within=np.array([bool(int(c['union7'])) for c in cases])
    counts={}
    for label, expected in [('training',1800),('monitor',288),('diagnostic',800)]:
        runs=[r for path in (out/label).glob('*/results.json') for r in read(path)]
        assert len(runs)==expected and all(r['complete'] for r in runs)
        counts[label]=len(runs)
    summary=dict(success=int(v.sum()),within_union7=int(v[within].sum()),outside_union7=int(v[~within].sum()),
        comparisons=comparisons,counts=counts,updates=sum(r['updates'] for r in history),confirmation=0)
    save(out/'SUMMARY.json',summary)
    delta=comparisons['shuffle500_final2000']
    text=f"""# CAP候选动作上的PPO对照

固定seed17与累计2000 episode，PPO成功 {v.sum()}/200，同场景顺序的Double DQN成功93/200，共同起点112/200。PPO相对配对DQN差 {delta['difference_pp']:+.1f} 个百分点，按场景分层bootstrap 95%区间 {delta['ci95_pp']}。新增成功{delta['gains']}例，丢失{delta['losses']}例。7CAP并集内成功{summary['within_union7']}/151，并集外成功{summary['outside_union7']}/49。

起点为相同的GLOBAL模型。训练时PPO使用合法候选上的softmax(Q/0.03)和独立状态critic，仅使用当批轨迹估计优势。DQN使用ε-greedy与历史回放。两组共享场景、采集量、奖励和输入。单种子、已查看诊断面板及两种不同的探索机制限制了归因。实际新增1800条训练轨迹、288次开发监测、800次诊断。固定最终模型，确认集继续封存。

[曲线](curves.png) · [逐场景结果](per_case.csv) · [配对比较](SUMMARY.json) · [训练记录](history.json) · [配置与源码哈希](inputs.json)
"""
    (out/'REPORT.md').write_text(text)
    save(out/'FINISHED.json',dict(complete=True,**summary))
