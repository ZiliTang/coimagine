"""Scene-paired comparison of frozen offline GA-Q and single-teacher GA."""
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2 import cap_offline_q_closedloop as study
from tools.cap_actor_closedloop_report import read_rows
from tools.cap_actor_distillation_report import record, key
from tools.cap_trace_value_long_report import write_csv
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save

OUT, ACTOR = study.OUT, study.baseline.OUT
SEEDS = study.SEEDS


def inspect(prior):
    case = prior['case_id']
    caps = prior['successful_caps'].split()
    row = dict(case_id=case, movable_count=int(prior['movable_count']), ncap=len(caps), successful_caps=' '.join(caps))
    assert len(caps) == int(prior['ncap'])
    for seed in SEEDS:
        name = f'offline_s{seed}'
        path = OUT/'evaluation'/name/case
        q = record(path)
        actor = record(ACTOR/'evaluation'/f'single_s{seed}'/case)
        s = q['summary']
        assert s == read(path/'summary.json') and s['complete'] and s['error'] is None
        assert s['case_id'] == case and s['condition'] == name and s['epsilon'] == 0
        assert Path(q['checkpoint']) == study.checkpoint(seed)
        assert len(q['events']) == len(q['decisions']) == s['decisions'] and len(q['public_rows']) == s['decisions']+1
        assert 0 < s['ticks'] <= 1800 and 0 < s['decisions'] <= 300
        assert actor['summary']['complete'] and actor['summary']['success'] == bool(int(prior[f'24k_s{seed}']))
        assert actor['summary']['case_id'] == case
        for dec, event in zip(q['decisions'], q['events'], strict=True):
            i = dec['chosen']
            assert i == dec['greedy'] and dec['action'] == dec['candidates'][i]
            assert key(dec['action']) == key(event['requested'])
        common = 0
        for i, (aq, aa) in enumerate(zip(q['decisions'], actor['decisions'])):
            assert actor['public_rows'][i]['history_hash'] == q['public_rows'][i]['history_hash']
            assert aa['candidates'] == aq['candidates'] and aa['sources'] == aq['sources']
            if key(aa['action']) != key(aq['action']):
                break
            common += 1
        if common == min(s['decisions'], len(actor['decisions'])):
            assert len(actor['decisions']) == s['decisions'] and actor['summary']['success'] == s['success']
        row.update({f'actor_s{seed}': int(actor['summary']['success']), f'q_s{seed}': int(s['success']),
                    f'common_actions_s{seed}': common, f'q_ticks_s{seed}': s['ticks'],
                    f'q_stop_s{seed}': s['stop'], f'q_invalid_s{seed}': s['invalid_proposals'],
                    f'q_fallback_s{seed}': s['fallback_boundaries']})
    return row


def compare(rows, mask, draws, reference=None):
    pairs = {}
    differences = []
    for seed in SEEDS:
        a = np.array([r[f'actor_s{seed}'] for r in rows]) if reference is None else reference
        q = np.array([r[f'q_s{seed}'] for r in rows])
        pairs[str(seed)] = dict(actor_or_reference=int(a[mask].sum()), q=int(q[mask].sum()),
                                gained=int(((q > a) & mask).sum()), lost=int(((q < a) & mask).sum()))
        differences.append(q - a)
    delta = np.mean(differences, axis=0)
    counts = mask[draws].sum(axis=1)
    samples = (delta*mask)[draws].sum(axis=1)[counts > 0] / counts[counts > 0]
    return dict(n=int(mask.sum()), seeds=pairs, delta_pp=float(100*delta[mask].mean()),
                ci_pp=np.quantile(100*samples, [.025,.975]).tolist())


def report():
    protocol = read(OUT/'protocol.json')
    assert protocol['actor_per_scene_sha256'] == study.sha(ACTOR/'per_scene.csv')
    assert protocol['panel_sha256'] == study.sha(study.PANEL)
    cases = [c['case_id'] for c in read(study.PANEL)['cases']]
    priors = read_rows(ACTOR/'per_scene.csv')
    assert set(cases) == priors.keys() and len(cases) == 200
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(inspect, [priors[c] for c in cases]))
    write_csv(OUT/'per_scene.csv', rows)
    movable = np.array([r['movable_count'] for r in rows])
    ncap = np.array([r['ncap'] for r in rows])
    rng = np.random.default_rng(20261119)
    strata = [np.flatnonzero(movable == n) for n in np.unique(movable)]
    draws = np.concatenate([rng.choice(idx, (10000,len(idx))) for idx in strata], axis=1)
    scopes = {'all': ncap >= 0, 'union': ncap > 0, 'unique': ncap == 1,
              'multiple': ncap > 1, 'outside': ncap == 0}
    metrics = {k:compare(rows, mask, draws) for k, mask in scopes.items()}
    caps = read_rows(study.baseline.actor.ARCHIVE/'per_scene.csv')
    best = max(read(ACTOR/'summary.json')['single_cap_success'], key=lambda cap:sum(int(caps[c][cap]) for c in cases))
    best_outcomes = np.array([int(caps[c][best]) for c in cases])
    best_comparison = compare(rows, scopes['all'], draws, reference=best_outcomes)
    diagnostics = {str(seed):dict(shared_first=sum(r[f'common_actions_s{seed}'] > 0 for r in rows),
                                  invalid=sum(r[f'q_invalid_s{seed}'] for r in rows),
                                  fallback=sum(r[f'q_fallback_s{seed}'] for r in rows)) for seed in SEEDS}
    save(OUT/'summary.json', dict(comparisons=metrics, panel_best=best, q_vs_best=best_comparison, diagnostics=diagnostics))
    lines = ['# 同面板离线 GA-Q 与单教师 GA 的闭环对照', '',
             '采用上一轮已查看的 200 个场景。seed17、29 各用冻结的 offline.pt 贪心执行一次，配对复用同种子 24,000 步 actor 的归档执行。两组每个边界都从相同七 CAP 的合法动作建议中重选，均屏蔽 trace。', '',
             '| 场景 | 例数 | seed17 actor → Q | Q 新增 / 失去 | seed29 actor → Q | Q 新增 / 失去 | 平均差及配对 95% 区间（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for label, k in [('七 CAP 并集','union'), ('全部','all'), ('恰一 CAP 成功','unique'),
                     ('至少两个 CAP 成功','multiple'), ('并集外','outside')]:
        m = metrics[k]
        cells = [label, str(m['n'])]
        for p in m['seeds'].values():
            cells += [f"{p['actor_or_reference']} → {p['q']}", f"{p['gained']} / {p['lost']}"]
        lo, hi = m['ci_pp']
        cells.append(f"{m['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]")
        lines.append('| '+' | '.join(cells)+' |')
    lo, hi = best_comparison['ci_pp']
    lines += ['', '主指标为七 CAP 成功并集上的保留率。按物体数分层、以场景为单位配对重抽 10,000 次，每次保留两个固定种子的全部结果。分组为描述性比较。', '',
              f"面板上十个单 CAP 中成功数最高的是 {best}（{best_outcomes.sum()}/200）。Q 相对这个固定参照平均差 {best_comparison['delta_pp']:+.2f} 个百分点，区间 [{lo:+.2f}, {hi:+.2f}]，条件于本面板选出的 CAP。", '',
              '离线 Q 利用 608 个训练初态的成功和失败决策，16,000 次 V 更新后做 8,000 次一步自举 Q 更新。actor 只从有优先成功教师的初态抽取该教师的动作，做 24,000 次模仿更新。目标、示范范围和抽样权重同时变化，这个实验比较两套完整训练方法。', '',
              '## 配对回放', '', '在 28 个独有场景中，按场景 ID 和种子顺序各取首例。']
    for gain, label in [(1, 'Q 救回'), (-1, 'Q 失去')]:
        examples = [(r['case_id'], seed) for r in rows for seed in SEEDS
                    if r['ncap'] == 1 and r[f'q_s{seed}'] - r[f'actor_s{seed}'] == gain]
        if examples:
            case, seed = min(examples)
            a = ACTOR/'evaluation'/f'single_s{seed}'/case/'result.json.gz'
            q = Path('evaluation')/f'offline_s{seed}'/case/'result.json.gz'
            lines += ['', f'- {label}，{case}，seed{seed}。[actor 回放]({os.path.relpath(a, OUT)}) · [Q 回放]({q})']
    lines += ['', '[逐场景成败](per_scene.csv) · [冻结权重与评估协议](protocol.json) · [机器可读结果](summary.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'FINISHED.json', dict(complete=True, new_episodes=len(rows)*len(SEEDS),
                                  files={name:study.sha(OUT/name) for name in ('REPORT.md','summary.json','per_scene.csv','protocol.json')}))


if __name__ == '__main__':
    report()
