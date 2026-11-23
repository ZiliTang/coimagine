"""Live curves and fixed-budget paired confirmation; no training decisions."""
import argparse
import base64
import csv
import hashlib
import html
import json
from pathlib import Path
import shutil
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

SEEDS = (17, 29, 43)
COLORS = ('#2563eb', '#d97706', '#059669')
ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT/'docs/agent_exchange/assets/cap_trace_value_long_20261118'
REVIEW = ROOT/'docs/agent_exchange/reviews/CAP_Trace_Value_Long_20261118.md'


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def gather(out):
    parent = Path(read(out/'inputs.json')['parent'])
    training, losses, evaluations = [], [], []
    for seed in SEEDS:
        for round_number in range(1, 41):
            base = parent if round_number <= 4 else out
            path = base/'online'/f'GLOBAL_s{seed}'/f'r{round_number}.json'
            if not path.exists():
                continue
            data = read(path)
            rows = data['results']
            assert data['complete'] and len(rows) == 50
            row = dict(seed=seed, round=round_number, episodes=round_number*50,
                success=sum(r['success'] for r in rows), count=len(rows),
                mean_return=float(np.mean([r['success'] for r in rows])),
                mean_ticks=float(np.mean([r['ticks'] for r in rows])),
                epsilon=rows[0]['epsilon'], boundaries=sum(r['decisions'] for r in rows))
            if all('behavior' in r for r in rows):
                n = row['boundaries']
                for key in ('entropy', 'normalized_entropy', 'candidate_count', 'chosen_q'):
                    row[key] = sum(r['behavior'][key+'_sum'] for r in rows)/n
                row['nongreedy_fraction'] = sum(r['behavior']['nongreedy_decisions'] for r in rows)/n
                ng = sum(r['behavior']['q_gap_count'] for r in rows)
                row['mean_q_gap'] = sum(r['behavior']['q_gap_sum'] for r in rows)/max(ng,1)
                row['q_saturated_fraction'] = sum(r['behavior']['chosen_q_saturated'] for r in rows)/n
            training.append(row)
        for path in sorted((out/'models'/f'GLOBAL_s{seed}').glob('online_r*_metrics.jsonl')):
            for line in path.read_text().splitlines():
                losses.append(json.loads(line))
        for label, episodes in [('online200',200)]:
            path = parent/'monitor'/f'GLOBAL_s{seed}'/(label+'.json')
            rows = read(path)['results']
            evaluations.append(dict(panel='monitor', seed=seed, episodes=episodes,
                success=sum(r['success'] for r in rows), count=len(rows)))
        old = read(parent/'final_test.json')['results']
        rows = [r for r in old if r['condition'] == f'GLOBAL_s{seed}_online']
        evaluations.append(dict(panel='diagnostic', seed=seed, episodes=200,
                                success=sum(r['success'] for r in rows), count=len(rows)))
        for panel in ('monitor', 'diagnostic', 'confirmation'):
            for path in (out/panel).glob(f'GLOBAL_s{seed}_e*.json'):
                if path.name.endswith('_progress.json'):
                    continue
                episodes = int(path.stem.split('_e')[-1])
                rows = read(path)['results']
                evaluations.append(dict(panel=panel, seed=seed, episodes=episodes,
                    success=sum(r['success'] for r in rows), count=len(rows)))
    training.sort(key=lambda r:(r['seed'],r['episodes']))
    losses.sort(key=lambda r:(r['seed'],r['updates']))
    evaluations.sort(key=lambda r:(r['panel'],r['seed'],r['episodes']))
    return training, losses, evaluations


def render_live(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = Path(out)
    destination = out/'report'
    destination.mkdir(exist_ok=True)
    training, losses, evaluations = gather(out)
    write_csv(destination/'training_returns.csv', training)
    write_csv(destination/'loss_metrics.csv', losses)
    write_csv(destination/'evaluation_curve.csv', evaluations)
    save(destination/'curves.json', dict(training=training, loss=losses, evaluation=evaluations))
    fig, axes = plt.subplots(4, 3, figsize=(17, 15), constrained_layout=True)
    axes = axes.ravel()
    for seed,color in zip(SEEDS,COLORS):
        tr = [r for r in training if r['seed'] == seed]
        lo = [r for r in losses if r['seed'] == seed]
        axes[0].plot([r['episodes'] for r in tr], [r['mean_return'] for r in tr], '.-',
                     color=color, label=f'seed {seed}', alpha=.8)
        for ax,panel in [(axes[1],'monitor'), (axes[2],'diagnostic')]:
            rows = [r for r in evaluations if r['seed'] == seed and r['panel'] == panel]
            ax.plot([r['episodes'] for r in rows], [r['success']/r['count'] for r in rows],
                    'o-', color=color, label=f'seed {seed}')
        for ax,key in [(axes[3],'td_mse'), (axes[4],'offline_td_mse'), (axes[5],'online_td_mse'),
                       (axes[8],'grad_norm_before_clip')]:
            ax.plot([r['updates'] for r in lo], [r[key] for r in lo], color=color, alpha=.7)
        axes[7].plot([r['updates'] for r in lo], [r['mean_q'] for r in lo], color=color)
        axes[7].plot([r['updates'] for r in lo], [r['mean_target'] for r in lo], color=color, ls='--', alpha=.6)
        for ax,key in [(axes[6],'entropy'), (axes[9],'mean_q_gap'),
                       (axes[10],'q_saturated_fraction'), (axes[11],'nongreedy_fraction')]:
            rows = [r for r in tr if key in r]
            ax.plot([r['episodes'] for r in rows], [r[key] for r in rows], '.-', color=color)
    titles = ['Training return: mean of 50 episodes', 'Greedy monitor: fixed 32 development roots',
        'Viewed diagnostic: fixed 200 roots', 'TD MSE: 100-update means', 'Offline-half TD MSE',
        'Online-half TD MSE', 'Actual epsilon-greedy entropy (nats)', 'Replay Q (solid) / target (dashed)',
        'Gradient norm before clipping at 1', 'Rollout top-two Q gap (K >= 2)',
        'Rollout chosen Q < .01 or > .99', 'Observed nongreedy decision fraction']
    update_axes = {3,4,5,7,8}
    for i,ax in enumerate(axes):
        ax.set_title(titles[i], fontsize=11)
        ax.set_xlabel('Cumulative online gradient updates' if i in update_axes else 'Cumulative online episodes')
        ax.grid(alpha=.2)
        if i not in update_axes:
            ax.axvline(200, color='#64748b', ls=':', lw=1)
    for i in (0,1,2,7,10,11):
        axes[i].set_ylim(0,1)
    axes[0].legend(loc='best')
    fig.suptitle('GLOBAL + Double DQN: fixed-root 10x budget continuation\n'
                 'Same reward, optimizer, replay ratio and 200 online roots; continuation epsilon = 0.05', fontsize=15)
    figure = destination/'learning_curves.png'
    fig.savefig(figure, dpi=135)
    plt.close(fig)
    progress = read(out/'progress.json') if (out/'progress.json').exists() else {'stage':'prepared'}
    completed = (out/'FINISHED.json').exists()
    timestamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    rows = ''.join(f'<tr><td>{r["panel"]}</td><td>{r["seed"]}</td><td>{r["episodes"]}</td>'
                   f'<td>{r["success"]}/{r["count"]}</td><td>{100*r["success"]/r["count"]:.1f}%</td></tr>'
                   for r in evaluations)
    encoded = base64.b64encode(figure.read_bytes()).decode()
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GLOBAL 在线预算对照</title>
<style>body{{font:16px system-ui;max-width:1500px;margin:24px auto;padding:0 20px;background:#f8fafc;color:#172033}}
img{{width:100%;background:white;border:1px solid #ddd}}table{{border-collapse:collapse}}td,th{{padding:6px 18px;border-bottom:1px solid #ddd}}
.note{{background:#e2e8f0;padding:14px;border-radius:8px}}button{{padding:8px;margin:5px}}pre{{white-space:pre-wrap}}</style>
<h1>GLOBAL＋RL：200 → 2,000 episode 预算对照</h1>
<p>更新时间（上海）：{timestamp}；状态：{'全部执行完成' if completed else '运行中'}</p>
<div class="note">三种子17/29/43；同一200训练初态重复采样。reward为成功1、否则0，无dense reward。
熵是实际ε-greedy候选动作分布的熵，不是actor熵或训练奖励。训练return含探索，独立监测采用贪心动作。
旧200测试已作为诊断面板；全新200确认集只在全部模型冻结后运行。固定报告最终2,000预算，不选最好checkpoint。</div>
<p>曲线每完成一轮训练/评估更新；浏览器每60秒重载。原200预算之前未保存的细粒度loss/熵不补造。</p>
<button onclick="location.reload()">立即刷新</button><button onclick="document.getElementById('details').hidden=!document.getElementById('details').hidden">显示/隐藏逐种子结果</button>
<pre>{html.escape(json.dumps(progress,ensure_ascii=False,indent=2))}</pre>
<a href="learning_curves.png">下载曲线PNG</a> · <a href="loss_metrics.csv">loss CSV</a> ·
<a href="training_returns.csv">reward CSV</a> · <a href="evaluation_curve.csv">评估CSV</a>
<img src="data:image/png;base64,{encoded}" alt="reward loss entropy Q gradient curves">
<div id="details"><table><tr><th>面板</th><th>seed</th><th>累计episode</th><th>成功数</th><th>成功率</th></tr>{rows}</table></div>
{'<p>完整结论见同目录 REPORT.md 与 SUMMARY.json。</p>' if completed else '<script>setTimeout(()=>location.reload(),60000)</script>'}</html>'''
    pending = destination/'dashboard.pending.html'
    pending.write_text(page)
    pending.replace(destination/'dashboard.html')


def analyze_panel(out, panel):
    parent = Path(read(out/'inputs.json')['parent'])
    cases = read((out/'panels/confirmation.json') if panel == 'confirmation' else parent/'panels/final_test.json')['cases']
    ids = [c['case_id'] for c in cases]
    old = read(parent/'final_test.json')['results'] if panel == 'diagnostic' else []
    rows = list(old)
    for path in (out/panel).glob('*.json'):
        if path.name.endswith('_progress.json'):
            continue
        data = read(path)
        assert data['complete']
        rows.extend(data['results'])
    table = {(r['condition'],r['case_id']):r for r in rows}
    assert all(r['complete'] and not r.get('error') for r in rows)
    def scores(condition):
        return np.array([table[condition,c]['success'] for c in ids], dtype=float)
    short = np.array([scores(f'GLOBAL_s{s}_online' if panel == 'diagnostic' else f'GLOBAL_s{s}_e200') for s in SEEDS])
    long = np.array([scores(f'GLOBAL_s{s}_e2000') for s in SEEDS])
    caps = {f'CAP-RR-{n:03d}':scores(f'CAP-RR-{n:03d}') for n in range(1,11)}
    best = min(caps, key=lambda k:(-caps[k].sum(),k))
    selected = [f'CAP-RR-{n:03d}' for n in (9,3,1,10,6,7,2)]
    union7 = np.max([caps[k] for k in selected], axis=0).astype(bool)
    union10 = np.max(list(caps.values()), axis=0).astype(bool)
    rng = np.random.default_rng(2026111813)
    indices = np.concatenate([rng.choice([i for i,c in enumerate(cases) if c['movable_count']==n],
                             (10000,50), replace=True) for n in range(2,6)], axis=1)
    def compare(a,b):
        delta = a-b
        low,high = np.quantile(delta[indices].mean(1), [.025,.975])
        return dict(difference_pp=float(100*delta.mean()), ci95_pp=[float(100*low),float(100*high)])
    def group(matrix):
        return dict(successes=matrix.sum(1).astype(int).tolist(), denominator_per_seed=200,
                    mean_success_rate=float(matrix.mean()),
                    retained_union7=(matrix[:,union7]).sum(1).astype(int).tolist(),
                    missed_union7=(1-matrix[:,union7]).sum(1).astype(int).tolist(),
                    retained_union10=(matrix[:,union10]).sum(1).astype(int).tolist(),
                    missed_union10=(1-matrix[:,union10]).sum(1).astype(int).tolist(),
                    outside_union10=(matrix[:,~union10]).sum(1).astype(int).tolist(),
                    versus_fixed_cap009=compare(matrix.mean(0),caps['CAP-RR-009']),
                    versus_test_best_descriptive=compare(matrix.mean(0),caps[best]))
    result = dict(panel=panel, unique_roots=200, short=group(short), long=group(long),
         long_minus_short=compare(long.mean(0), short.mean(0)),
         gains_per_seed=((long==1)&(short==0)).sum(1).tolist(),
         losses_per_seed=((long==0)&(short==1)).sum(1).tolist(),
         cap_successes={k:int(v.sum()) for k,v in caps.items()}, best_cap_descriptive=best,
         union7=int(union7.sum()), union10=int(union10.sum()),
         uncertainty='10000 strata-balanced root bootstrap draws; conditional on these 3 training seeds; not 600 independent scenes')
    csv_rows = [dict(panel=panel, condition=k, case_id=c, success=int(r['success']), ticks=r['ticks'],
                     decisions=r['decisions'], stop=r['stop']) for (k,c),r in sorted(table.items())]
    return result,csv_rows


def final_report(out):
    out = Path(out).resolve()
    assert read(out/'FINISHED.json')['complete']
    frozen = read(out/'ALL_MODELS_FROZEN.json')
    for path,expected in frozen['files'].items():
        assert hashlib.sha256((out/path).read_bytes()).hexdigest() == expected
    summary, records = {}, []
    for panel in ('diagnostic','confirmation'):
        summary[panel], rows = analyze_panel(out,panel)
        records.extend(rows)
    training, losses, evaluations = gather(out)
    assert len(training) == 120 and len(losses) == 1620
    for seed in SEEDS:
        assert max(r['updates'] for r in losses if r['seed']==seed) == 60000
    physical = []
    for pattern in ('online/*/r*.json', 'monitor/*.json', 'diagnostic/*.json', 'confirmation/*.json'):
        for path in out.glob(pattern):
            if path.name.endswith('_progress.json'):
                continue
            data = read(path)
            if 'results' in data:
                assert data['complete']
                physical.extend(data['results'])
    assert len(physical) == 11864
    assert all(r['complete'] and not r.get('error') for r in physical)
    summary['complete'] = True
    summary['physical'] = dict(executions=len(physical), ticks=sum(r['ticks'] for r in physical),
         decisions=sum(r['decisions'] for r in physical), errors=sum(bool(r.get('error')) for r in physical),
         invalid_proposals=sum(r.get('invalid_proposals',0) for r in physical),
         fallback_boundaries=sum(r.get('fallback_boundaries',0) for r in physical))
    summary['training_last200'] = {str(s):dict(success=sum(r['success'] for r in training if r['seed']==s and r['episodes']>1800),
        denominator=200) for s in SEEDS}
    summary['budget'] = read(out/'FINISHED.json')
    save(out/'report/SUMMARY.json', summary)
    write_csv(out/'report/per_case_results.csv', records)
    render_live(out)
    confirm = summary['confirmation']
    diag = summary['diagnostic']
    delta = confirm['long_minus_short']
    lo,hi = delta['ci95_pp']
    verdict = ('支持在本轮固定初态训练池上增加在线预算带来泛化收益。' if lo > 0 else
               '本轮长预算降低了新确认集表现。' if hi < 0 else
               '当前区间跨零，尚不能确证增加在线预算带来稳定泛化收益。')
    def counts(row):return '/'.join(map(str,row['successes']))
    text = f'''# GLOBAL＋RL 长程在线预算对照（2026-11-18）

已完成固定预算对照：三个seed从200在线episode/6000更新续训至2000episode/60000更新。
新确认200场景上，原预算成功数 **{counts(confirm['short'])}**，长预算 **{counts(confirm['long'])}**（seed17/29/43，各/200）。
平均成功率 **{100*confirm['short']['mean_success_rate']:.2f}% → {100*confirm['long']['mean_success_rate']:.2f}%**；
配对差 **{delta['difference_pp']:+.2f}个百分点**，95%根场景bootstrap区间 **[{lo:.2f}, {hi:.2f}]**。
{verdict}

## 曲线与原始证据

- [可交互查看的曲线面板](../assets/cap_trace_value_long_20261118/dashboard.html)
- [12面板曲线PNG](../assets/cap_trace_value_long_20261118/learning_curves.png)
- [逐100更新loss/Q/梯度CSV](../assets/cap_trace_value_long_20261118/loss_metrics.csv)
- [每50任务的训练reward/熵CSV](../assets/cap_trace_value_long_20261118/training_returns.csv)
- [固定节点独立评估曲线CSV](../assets/cap_trace_value_long_20261118/evaluation_curve.csv)
- [逐场景结果CSV](../assets/cap_trace_value_long_20261118/per_case_results.csv)
- [完整统计JSON](../assets/cap_trace_value_long_20261118/SUMMARY.json)

## 隔离了什么

同三个GLOBAL在线200检查点续接，恢复Adam、目标网络、NumPy/Torch RNG和各自全部在线回放；
相同7CAP、trace、Q网络、离线normalizer、Adam3e-4、batch64（32离线＋32自身在线）、target每250更新同步、梯度裁剪1。
沿用末轮epsilon=.05。每50真实episode后更新1500次。原200在线初态按原seed顺序循环共10轮；
新增5400采集执行但没有新增独立训练初态，不用新场景分布解释收益。
回放按全部决策边界均匀采样，轨迹长度影响有效样本权重。
同时增加真实交互与梯度更新，固定二者比例；不能将结果单独归因于更多数据或更多优化。

奖励仍为任务成功终止1、其他0，无dense reward，gamma=1；1800ticks或300决策为有限任务终止。
失败和截断保留。熵是实际epsilon-greedy动作分布的解析熵，K个去重候选中贪心概率1-epsilon+epsilon/K，
其他epsilon/K；它不是actor熵，也没有新增熵奖励。训练reward含探索，监测采用冻结模型贪心动作。
低TD loss、熵稳定或训练成功升高均不能独立证明新场景成功率提升。
原预算未保存的细粒度loss/熵不补造，图中此类曲线从续训开始。

## 能力覆盖与配对得失

新确认集最好单CAP（事后描述性参照）为 **{confirm['best_cap_descriptive']}**，
成功 **{confirm['cap_successes'][confirm['best_cap_descriptive']]}/200**；固定开发选定CAP009为 **{confirm['cap_successes']['CAP-RR-009']}/200**。
7CAP并集 **{confirm['union7']}/200**，10CAP并集 **{confirm['union10']}/200**，并集不是切换策略的理论上限。

|新确认200场景|seed17|seed29|seed43|
|---|---:|---:|---:|
|长预算相对短预算新增成功|{'|'.join(map(str,confirm['gains_per_seed']))}|
|长预算相对短预算丢失成功|{'|'.join(map(str,confirm['losses_per_seed']))}|
|长预算遗漏10CAP已知可解场景|{'|'.join(map(str,confirm['long']['missed_union10']))}|
|长预算解决10CAP均失败场景|{'|'.join(map(str,confirm['long']['outside_union10']))}|

旧200已查看面板降格为诊断：短预算 **{counts(diag['short'])}**，长预算 **{counts(diag['long'])}**，
均值差 **{diag['long_minus_short']['difference_pp']:+.2f}个百分点**。各中间节点全部报告，不挑最好checkpoint或seed。
32开发根监测每200episode运行；新确认集只在全部模型冻结后运行，不用于调参、早停或追加预算。

新确认200按2/3/4/5物体各50例，冻结于续训前，与已知历史及前轮在线/测试初态按ID和内容hash去重。
三个seed共享200场景；统计区间按根场景分层配对bootstrap10000次，条件于这3个训练seed，不是600个独立场景。
测试最佳单CAP比较未经事后选择校正；先看预指定的长预算−短预算配对比较。

## 执行与限制

新增执行共11864：在线5400、开发监测864、旧面板诊断2400、新确认3200（6模型×200＋10CAP×200）。
全部完成，无宿主错误；实际ticks **{summary['physical']['ticks']}**、决策 **{summary['physical']['decisions']}**。
非法CAP建议 **{summary['physical']['invalid_proposals']}**，fallback边界 **{summary['physical']['fallback_boundaries']}**。
固定最终60000次在线更新，不以曲线表现改变预算。本轮没有重跑GA/LOCAL，所以不能用它单独确证trace的独立贡献。
同200训练根反复交互可检验固定池上的预算不足，也可能过拟合；不能外推为更丰富在线训练分布的效果。
实现校验包含分块回放与原整表下一状态逐值等价、原训练器与带指标续训一步参数/优化器/目标网络/RNG一致、实际epsilon-greedy熵边界。
未运行全仓库测试。完整配置、父检查点hash、源码快照、全部训练原始回放与checkpoint保存在 `{out}`。
历史完整实验与其冻结源码未改写。报告为固定协议的自动统计；不继续追加训练。
'''
    (out/'report/REPORT.md').write_text(text)
    REVIEW.parent.mkdir(parents=True, exist_ok=True)
    REVIEW.write_text(text)
    ASSET.mkdir(parents=True, exist_ok=True)
    for name in ('SUMMARY.json','per_case_results.csv','learning_curves.png','dashboard.html',
                 'loss_metrics.csv','training_returns.csv','evaluation_curve.csv','curves.json'):
        shutil.copyfile(out/'report'/name, ASSET/name)
    for name in ('inputs.json','FINISHED.json','ALL_MODELS_FROZEN.json'):
        shutil.copyfile(out/name, ASSET/name)
    save(ASSET/'manifest.json', {p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
                               for p in ASSET.iterdir() if p.is_file() and p.name!='manifest.json'})
    current = ROOT/'docs/agent_exchange/CURRENT.md'
    content = current.read_text()
    start, end = '<!-- LONG_ONLINE_START -->', '<!-- LONG_ONLINE_END -->'
    if start in content and end in content:
        a,b = content.split(start,1)
        _,c = b.split(end,1)
        body = f'''\n**GLOBAL长程预算对照已完成。** 新确认200：短预算{counts(confirm['short'])}，长预算{counts(confirm['long'])}；
平均差{delta['difference_pp']:+.2f}pp，95%区间[{lo:.2f},{hi:.2f}]。{verdict}
[报告](reviews/CAP_Trace_Value_Long_20261118.md)；[曲线](assets/cap_trace_value_long_20261118/dashboard.html)。
11864新增执行完成，无错误。固定最终预算，无剩余训练；停止，不自动追加。\n'''
        current.write_text(a+start+body+end+c)
    save(out/'REPORT_FINISHED.json', dict(complete=True, review=str(REVIEW), assets=str(ASSET)))
    print('LONG_REPORT_COMPLETE', summary['confirmation']['long_minus_short'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    (render_live if args.live else final_report)(Path(args.out))
