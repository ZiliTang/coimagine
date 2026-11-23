"""Paired, fixed-budget diagnostic report for cyclic shuffle plus fewer Q updates."""
import base64
import hashlib
import html
from pathlib import Path
import shutil

import numpy as np

from tools import cap_trace_value_long_report as L

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT/'docs/agent_exchange/assets/cap_trace_value_shuffle500_20261118'
REVIEW = ROOT/'docs/agent_exchange/reviews/CAP_Trace_Value_Shuffle500_20261118.md'


def paired(a, b, cases):
    delta = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    rng = np.random.default_rng(2026111813)
    indices = np.concatenate([rng.choice([i for i,c in enumerate(cases) if c['movable_count']==n],
        (10000, 50), replace=True) for n in range(2,6)], axis=1)
    lo, hi = np.quantile(delta[indices].mean(1), [.025,.975])*100
    return dict(difference_pp=float(delta.mean()*100), ci95_pp=[float(lo),float(hi)],
        gains=int((delta>0).sum()), losses=int((delta<0).sum()))


def coverage(out):
    inputs = L.read(out/'inputs.json')
    parent, baseline = Path(inputs['parent']), Path(inputs['baseline'])
    cases = L.read(parent/'panels/final_test.json')['cases']
    ids = [c['case_id'] for c in cases]
    original = L.read(parent/'final_test.json')['results']
    table = {(r['condition'],r['case_id']): r for r in original}
    def vector(condition):
        rows = [table[condition,c] for c in ids]
        assert all(r['complete'] and not r.get('error') for r in rows)
        return np.asarray([r['success'] for r in rows], dtype=bool)
    caps = {n: vector(f'CAP-RR-{n:03d}') for n in range(1,11)}
    union7 = np.logical_or.reduce([caps[n] for n in (9,3,1,10,6,7,2)])
    union10 = np.logical_or.reduce(list(caps.values()))
    methods = {'initial200': vector('GLOBAL_s17_online')}
    for label, path in [('fixed1500_final2000', baseline), ('shuffle500_final2000', out)]:
        result = L.read(path/'diagnostic/GLOBAL_s17_e2000.json')
        assert result['complete'] and len(result['results']) == 200
        rows = {r['case_id']: r for r in result['results']}
        assert set(rows) == set(ids) and all(r['complete'] and not r.get('error') for r in rows.values())
        methods[label] = np.asarray([rows[c]['success'] for c in ids], dtype=bool)
    groups = {name: dict(success=int(v.sum()), count=200,
        within_union7_success=int(v[union7].sum()), within_union7_count=int(union7.sum()),
        outside_union7_success=int(v[~union7].sum()), outside_union7_count=int((~union7).sum()),
        missed_union7=int((~v[union7]).sum()),
        outside_union10_success=int(v[~union10].sum()), outside_union10_count=int((~union10).sum()))
        for name, v in methods.items()}
    comparisons = {name: paired(methods['shuffle500_final2000'], v, cases)
        for name,v in methods.items() if name != 'shuffle500_final2000'}
    records = [dict(case_id=c, movable_count=cases[i]['movable_count'], union7=int(union7[i]),
        union10=int(union10[i]), **{name:int(v[i]) for name,v in methods.items()},
        **{f'CAP_{n:03d}':int(v[i]) for n,v in caps.items()}) for i,c in enumerate(ids)]
    return dict(groups=groups, comparisons=comparisons,
        cap_successes={f'CAP_{n:03d}':int(v.sum()) for n,v in caps.items()},
        uncertainty='10000 stratified paired root bootstrap draws; one training seed; viewed diagnostic panel'), records


def render_live(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    L.SEEDS = (17,)
    inputs = L.read(out/'inputs.json')
    baseline = Path(inputs['baseline'])
    report = out/'report'
    report.mkdir(exist_ok=True)
    current = L.gather(out)
    old = L.gather(baseline)
    cycles = []
    for label, data in [('fixed1500',old),('shuffle500',current)]:
        for cycle in range(10):
            rows = [r for r in data[0] if cycle*200 < r['episodes'] <= (cycle+1)*200]
            if len(rows)==4:
                cycles.append(dict(method=label, episodes=(cycle+1)*200,
                    success=sum(r['success'] for r in rows), count=200))
    L.write_csv(report/'training_cycles.csv',cycles)
    for name, rows in zip(('training_returns','loss_metrics','evaluation_curve'), current):
        L.write_csv(report/(name+'.csv'), rows)
    L.save(report/'curves.json', dict(training=current[0], loss=current[1], evaluation=current[2],
                                     baseline=str(baseline)))
    fig, axes = plt.subplots(3, 3, figsize=(17,12), constrained_layout=True)
    axes = axes.ravel()
    for label, color, (training, losses, evaluation) in [
        ('Fixed order / 1500 updates', '#94a3b8', old),
        ('Cyclic shuffle / 500 updates', '#2563eb', current)]:
        axes[0].plot([r['episodes'] for r in training], [r['mean_return'] for r in training], '.-', color=color,label=label)
        for ax,panel in [(axes[1],'monitor'),(axes[2],'diagnostic')]:
            rows = [r for r in evaluation if r['panel']==panel]
            ax.plot([r['episodes'] for r in rows], [r['success']/r['count'] for r in rows], 'o-',color=color)
        for ax,key in [(axes[3],'td_mse'),(axes[4],'offline_td_mse'),(axes[5],'online_td_mse'),(axes[7],'grad_norm_before_clip')]:
            # Show round means against episodes so budgets are directly comparable.
            rounds = sorted({r['round'] for r in losses})
            rows = [dict(episodes=r*50,value=float(np.mean([v[key] for v in losses if v['round']==r]))) for r in rounds]
            ax.plot([r['episodes'] for r in rows],[r['value'] for r in rows],color=color)
        for ax,key in [(axes[6],'entropy'),(axes[8],'mean_q_gap')]:
            rows = [r for r in training if key in r]
            ax.plot([r['episodes'] for r in rows],[r[key] for r in rows],'.-',color=color)
    titles = ['Training return: 50-episode batches', 'Greedy development monitor (32 roots)',
        'Viewed diagnostic (200 roots)', 'TD MSE: round mean', 'Offline-half TD MSE',
        'Online-half TD MSE', 'Actual epsilon-greedy entropy (nats)', 'Gradient norm before clipping',
        'Rollout top-two Q gap (K >= 2)']
    for ax,title in zip(axes,titles):
        ax.set_title(title); ax.set_xlabel('Cumulative online episodes'); ax.grid(alpha=.2)
        ax.axvline(200, color='#64748b', ls=':', lw=1)
    for i in (0,1,2): axes[i].set_ylim(0,1)
    axes[0].legend(fontsize=8)
    fig.suptitle('GLOBAL + Double DQN, seed17: shuffle + lower update budget\nSame initial checkpoint, sparse reward, roots, epsilon, optimizer and replay ratio')
    fig.savefig(report/'learning_curves.png',dpi=130); plt.close(fig)
    encoded = base64.b64encode((report/'learning_curves.png').read_bytes()).decode()
    progress = L.read(out/'progress.json') if (out/'progress.json').exists() else {}
    rows = ''.join(f'<tr><td>{r["panel"]}</td><td>{r["episodes"]}</td><td>{r["success"]}/{r["count"]}</td></tr>' for r in current[2])
    complete = (out/'REPORT_FINISHED.json').exists()
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>随机场景批次＋500次更新</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:24px auto;padding:20px}}img{{width:100%}}td{{padding:6px 20px}}pre{{white-space:pre-wrap}}</style>
<h1>seed17：每周期分层洗牌＋每批500次Q更新</h1>
<p>状态：{'完成' if complete else '运行中'}。从同一个200 episode／6000更新检查点开始；最终2000 episode／24000更新。</p>
<p>同200个训练初态，每周期无放回覆盖一次；成功1、否则0，ε=.05。两个因素同时改变，不能独立归因。
32开发监测与200已查看诊断场景，不用于挑选checkpoint。新确认集封存。本轮单seed区间不覆盖训练随机性。</p>
<p>训练return含探索；行为熵来自ε-greedy候选分布，不是actor熵。灰线为已完成的固定顺序／1500更新对照。
loss曲线是每轮均值，逐100更新记录见CSV；训练批次组成不同，单批return不直接配对。</p>
<pre>{html.escape(str(progress))}</pre><img src="data:image/png;base64,{encoded}">
<p><a href="training_returns.csv">reward/熵 CSV</a> · <a href="loss_metrics.csv">loss CSV</a> ·
<a href="evaluation_curve.csv">评估 CSV</a> · <a href="training_cycles.csv">完整200周期 CSV</a> · <a href="REPORT.md">最终报告</a></p>
<table><tr><th>面板</th><th>episode</th><th>成功</th></tr>{rows}</table>
{'<p>已完成；停止，不自动扩预算。</p>' if complete else '<script>setTimeout(()=>location.reload(),60000)</script>'}</html>'''
    pending = report/'dashboard.pending.html'; pending.write_text(page); pending.replace(report/'dashboard.html')


def final_report(out):
    out = Path(out).resolve()
    L.SEEDS = (17,)
    finished = L.read(out/'FINISHED.json')
    assert finished['complete'] and finished['online_updates']==24000
    training, losses, evaluation = L.gather(out)
    assert len(training)==40 and max(r['updates'] for r in losses)==24000
    physical = []
    for pattern in ('online/GLOBAL_s17/r*.json','monitor/*.json','diagnostic/*.json'):
        for path in out.glob(pattern):
            if path.name.endswith('_progress.json'): continue
            data = L.read(path)
            assert data['complete']
            physical.extend(data['results'])
    assert len(physical)==2888 and all(r['complete'] and not r.get('error') for r in physical)
    frozen = L.read(out/'ALL_MODELS_FROZEN.json')
    assert all(hashlib.sha256((out/p).read_bytes()).hexdigest()==v for p,v in frozen['files'].items())
    result, records = coverage(out)
    summary = dict(complete=True, seed=17, diagnostic=result, budget=finished,
        new_physical_executions=len(physical), errors=0, confirmation_executions=0,
        new_decisions=sum(r['decisions'] for r in physical), new_ticks=sum(r['ticks'] for r in physical),
        final_resident_memory=L.read(out/'memory.json'))
    L.save(out/'report/SUMMARY.json',summary)
    L.write_csv(out/'report/per_case_results.csv',records)
    groups=result['groups']; delta=result['comparisons']['fixed1500_final2000']; low,high=delta['ci95_pp']
    conclusion = ('出现正向配对收益，可作为后续独立确认的候选。' if low>0 else
                  '出现负向配对收益，不支持这个组合。' if high<0 else
                  '配对区间跨零，本轮不足以确认这个组合带来稳定收益。')
    rows='\n'.join(f'|{name}|{g["success"]}/200|{g["within_union7_success"]}/{g["within_union7_count"]}|'
        f'{g["outside_union7_success"]}/{g["outside_union7_count"]}|{g["missed_union7"]}|'
        f'{g["outside_union10_success"]}/{g["outside_union10_count"]}|' for name,g in groups.items())
    curve='\n'.join(f'|{r["episodes"]}|{r["success"]}/{r["count"]}|' for r in evaluation if r['panel']=='diagnostic')
    text=f'''# seed17：分层洗牌＋降低Q更新次数（2026-11-18）

已完成。新组合最终 **{groups['shuffle500_final2000']['success']}/200**；旧固定顺序／1500更新对照 **{groups['fixed1500_final2000']['success']}/200**。
差 **{delta['difference_pp']:+.2f}pp**，按物体数量分层的配对场景bootstrap95%区间 **[{low:.2f},{high:.2f}]**；新增成功{delta['gains']}例、丢失成功{delta['losses']}例。
{conclusion}这是单seed、已查看诊断面板上的两因素组合筛查，不能分辨洗牌与更新次数各自的效应，也不是独立盲测。

## CAP覆盖与能力保留

|方法|总成功|7CAP并集内成功|7CAP并集外成功|漏掉并集内|10CAP并集外成功|
|---|---:|---:|---:|---:|---:|
{rows}

单CAP并集来自各CAP从初态执行到底；不构成逐决策切换策略的成功上界。并集外成功不自动证明trace有效，需要对应无trace对照。
在线训练200初态由同一生成器新生成并按内容去重，未按成功筛选；没有在训练200上运行完整单CAP矩阵，训练并集覆盖未知。

## 固定预算诊断曲线

|累计在线episode|成功|
|---:|---:|
{curve}

[曲线面板](../assets/cap_trace_value_shuffle500_20261118/dashboard.html) · [完整统计](../assets/cap_trace_value_shuffle500_20261118/SUMMARY.json) · [逐场景配对](../assets/cap_trace_value_shuffle500_20261118/per_case_results.csv)。

两组从同一原200 episode／6000更新checkpoint恢复模型、Adam、目标网络、NumPy与Torch RNG，以及相同原200在线回放。
新组每200次采集按2/3/4/5物体数量分层洗牌，再拆四批50；每批每层12或13，周期内无放回、覆盖全部200根。
调度RNG独立固定2026111817；动作探索随机数按场景和访问次数与旧组配对。在线回放仍均匀按决策边界抽样。
每批更新500而非1500，batch64（32离线＋32自身在线）、Adam3e-4、目标同步每250更新、clip1、ε=.05、success1/otherwise0、gamma1、网络与normalizer均不变。
因此目标同步相对采集频率也随更新数减少，不声称只改变梯度大小或纯场景顺序。
累计在线2000 episode／24000更新（旧组60000）；未继承旧长程采集的回放，不从旧2000模型继续训练。

新增正式执行2888＝在线1800＋32根监测288＋200根诊断800；全部完整、错误0。实际墙钟{finished['seconds']/60:.1f}分钟。
12采样worker各1线程，learner4线程；沿用已验证的无损驻内存回放和传感/历史缓存优化。
最终固定2000预算；32根监测不用于早停或挑模型。新确认集0次执行，其他seed没有续跑。
配对区间只反映这200场景的条件抽样不确定性，不覆盖训练seed变化，也不能消除多次查看诊断面板的偏差。
训练曲线批次组成已变，不能把单个50例batch下降当成策略退化；报告完整200周期与固定贪心诊断。
针对性调度测试验证确定性、每周期覆盖、分层平衡和探索种子配对；未重跑全仓库测试。
原始输出：`{out}`。源码、采样表、父checkpoint哈希、原始轨迹、loss与回放均保留。完成后停止，不自动追加实验。
'''
    (out/'report/REPORT.md').write_text(text); REVIEW.write_text(text)
    L.save(out/'REPORT_FINISHED.json',dict(complete=True,review=str(REVIEW),assets=str(ASSETS)))
    render_live(out)
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name in ('dashboard.html','learning_curves.png','training_returns.csv','loss_metrics.csv',
                 'evaluation_curve.csv','training_cycles.csv','curves.json','SUMMARY.json','per_case_results.csv','REPORT.md'):
        shutil.copyfile(out/'report'/name,ASSETS/name)
    for name in ('inputs.json','schedule.json','FINISHED.json','ALL_MODELS_FROZEN.json'):
        shutil.copyfile(out/name,ASSETS/name)
    L.save(ASSETS/'manifest.json',{p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
        for p in ASSETS.iterdir() if p.is_file() and p.name!='manifest.json'})
    path=ROOT/'docs/agent_exchange/CURRENT.md'; content=path.read_text()
    start,end='<!-- LONG_ONLINE_START -->','<!-- LONG_ONLINE_END -->'
    a,b=content.split(start,1); _,c=b.split(end,1)
    body=f'\n**分层洗牌＋500更新筛查已完成。** 新组{groups["shuffle500_final2000"]["success"]}/200，旧组{groups["fixed1500_final2000"]["success"]}/200，差{delta["difference_pp"]:+.2f}pp。\n[报告](reviews/CAP_Trace_Value_Shuffle500_20261118.md)；[曲线](assets/cap_trace_value_shuffle500_20261118/dashboard.html)。单seed、已查看诊断，两因素组合不能独立归因。2888新增执行完成，新确认集0次；停止，不自动扩预算。\n'
    path.write_text(a+start+body+end+c)
    print('SHUFFLE_REPORT_COMPLETE',delta,flush=True)
