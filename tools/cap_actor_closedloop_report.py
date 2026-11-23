"""Audit paired rollouts and report the incremental effect of the supervision budget."""
import csv
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2 import cap_actor_closedloop as study
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_actor_distillation_report import key, record
from tools.cap_trace_value_long_report import write_csv

OUT, EARLY, ARCHIVE = study.OUT, study.EARLY, study.actor.ARCHIVE
SEEDS = study.actor.SEEDS


def read_rows(path):
    with path.open() as stream:
        return {r['case_id']: r for r in csv.DictReader(stream)}


def inspect_case(args):
    case, early, caps = args
    successful = [cap for cap in study.actor.CAPS if int(caps[cap])]
    assert len(successful) == int(early['ncap']) == int(caps['union7_count'])
    row = dict(case_id=case, movable_count=int(early['movable_count']), ncap=len(successful),
               successful_caps=' '.join(successful))
    reference = {}
    for cap in successful:
        rec = record(ARCHIVE/'evaluation'/cap/case)
        assert rec['summary']['success']
        for pub, event in zip(rec['public_rows'][:-1], rec['events'], strict=True):
            reference.setdefault(pub['history_hash'], set()).add(key(event['requested']))
    for seed in SEEDS:
        name = f'single_s{seed}'
        prior = read(EARLY/'evaluation'/name/case/'summary.json')
        assert prior['complete'] and prior['success'] == bool(int(early[name]))
        path = OUT/'evaluation'/name/case
        run = record(path)
        s = read(path/'summary.json')
        assert s == run['summary'] and s['complete'] and s['error'] is None
        assert s['condition'] == name and s['case_id'] == case and s['epsilon'] == 0
        assert Path(run['checkpoint']) == study.CAPACITY/f'w64_s{seed}_u24000.pt'
        assert 0 < len(run['decisions']) == s['decisions'] <= 300 and 0 < s['ticks'] <= 1800
        assert len(run['public_rows']) == s['decisions'] + 1
        prefix = 0
        for i, (dec, event, pub) in enumerate(zip(run['decisions'], run['events'], run['public_rows'][:-1], strict=True)):
            assert dec['chosen'] == dec['greedy'] and dec['candidates'][dec['chosen']] == dec['action']
            assert key(dec['action']) == key(event['requested'])
            if i == 0 and successful:
                assert pub['history_hash'] in reference
            if i == prefix and key(dec['action']) in reference.get(pub['history_hash'], ()):
                prefix += 1
        if successful and prefix == s['decisions']:
            assert s['success']
        row.update({f'6k_s{seed}': int(prior['success']), f'24k_s{seed}': int(s['success']),
                    f'6k_s{seed}_prefix': int(early[name+'_prefix_length']), f'24k_s{seed}_prefix': prefix,
                    f'24k_s{seed}_ticks': s['ticks'], f'24k_s{seed}_stop': s['stop'],
                    f'24k_s{seed}_invalid': s['invalid_proposals'], f'24k_s{seed}_fallback': s['fallback_boundaries']})
    return row


def report():
    protocol = read(OUT/'protocol.json')
    assert study.sha(EARLY/'per_scene.csv') == protocol['early_per_scene_sha256']
    prior = read_rows(EARLY/'per_scene.csv')
    caps = read_rows(ARCHIVE/'per_scene.csv')
    cases = [c['case_id'] for c in read(ARCHIVE/'panel.json')['cases']]
    assert set(cases) == prior.keys() == caps.keys() and len(cases) == 200
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(inspect_case, [(c, prior[c], caps[c]) for c in cases]))
    write_csv(OUT/'per_scene.csv', rows)
    def column(k):
        return np.array([row[k] for row in rows])
    ncap, movable = column('ncap'), column('movable_count')
    rng = np.random.default_rng(20261119)
    strata = [np.flatnonzero(movable == n) for n in np.unique(movable)]
    draws = np.concatenate([rng.choice(idx, (10000, len(idx))) for idx in strata], axis=1)
    comparisons = {}
    for label, mask in [('all', ncap >= 0), ('union', ncap > 0), ('unique', ncap == 1), ('outside', ncap == 0)]:
        by_seed, diffs = {}, []
        for seed in SEEDS:
            old, new = column(f'6k_s{seed}'), column(f'24k_s{seed}')
            diffs.append(new - old)
            by_seed[str(seed)] = dict(old=int(old[mask].sum()), new=int(new[mask].sum()),
                kept=int((mask & (new == 1) & (old == 1)).sum()), lost=int((mask & (new < old)).sum()),
                gained=int((mask & (new > old)).sum()))
        delta = np.mean(diffs, axis=0)
        count = mask[draws].sum(1)
        estimate = 100 * (delta * mask)[draws].sum(1)[count > 0] / count[count > 0]
        comparisons[label] = dict(n=int(mask.sum()), seeds=by_seed, delta_pp=float(100*delta[mask].mean()),
                                  ci_pp=np.quantile(estimate, [.025, .975]).tolist())
    diagnostics = {}
    for seed in SEEDS:
        prefix = column(f'24k_s{seed}_prefix') > 0
        success = column(f'24k_s{seed}')
        diagnostics[str(seed)] = dict(first_supported=int(prefix.sum()), supported_failure=int((prefix & (success == 0)).sum()),
                                     invalid=int(column(f'24k_s{seed}_invalid').sum()), fallback=int(column(f'24k_s{seed}_fallback').sum()),
                                     stops=dict(Counter(r[f'24k_s{seed}_stop'] for r in rows)))
    singles = {cap: sum(int(row[cap]) for row in caps.values()) for cap in caps[cases[0]] if cap.startswith('CAP-RR-')}
    best = max(singles, key=singles.get)
    single_outcomes = np.array([int(caps[c][best]) for c in cases])
    delta_best = np.mean([column(f'24k_s{seed}') - single_outcomes for seed in SEEDS], axis=0)
    best_ci = np.quantile(100 * delta_best[draws].mean(1), [.025, .975]).tolist()
    save(OUT/'summary.json', dict(comparisons=comparisons, diagnostics=diagnostics, single_cap_success=singles,
                                  comparison_with_panel_best=dict(cap=best, delta_pp=float(100*delta_best.mean()),
                                                                 paired_ci_pp=best_ci)))
    lines = ['# 无 trace 的 64×64 GA 闭环对照', '',
             '单一优先教师监督训练，同一训练链的 6,000 与 24,000 步权重。两个固定种子各在同一批已查看的 200 个诊断场景贪心执行。每一步从七个 CAP 在当前公开历史上提出的合法候选中选分数最高者。新增 400 次完整执行，6,000 步结果来自同面板归档。', '',
             '| 范围 | 场景数 | seed17 成功 6k→24k | seed17 新增 / 失去 | seed29 成功 6k→24k | seed29 新增 / 失去 | 平均差及 95% 区间（百分点） |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for scope, desc in [('union','七 CAP 成功并集'), ('all','全部'), ('unique','恰一 CAP 成功'), ('outside','并集外')]:
        d = comparisons[scope]
        cells = []
        for s in d['seeds'].values():
            cells.extend([f"{s['old']} → {s['new']}", f"{s['gained']} / {s['lost']}"])
        lo, hi = d['ci_pp']
        lines.append('| ' + ' | '.join([desc, str(d['n']), *cells, f"{d['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"]) + ' |')
    lines += ['', '主指标为七 CAP 成功并集上的保留率。区间按物体数量分层，在场景层面配对重抽 10,000 次，每次保留两个种子的全部结果。', '',
              '24k 基于成功单 CAP 轨迹的首步诊断（完整动作含执行时长）', '']
    for seed, d in diagnostics.items():
        lines.append(f"- seed{seed} 在并集内有 {d['first_supported']} 例首步与至少一条成功轨迹一致，其中 {d['supported_failure']} 例最终失败。")
    lines += ['', f'同面板十个固定 CAP 中最高为 {best}，成功 {singles[best]}/200。24k GA 相对该固定 CAP 的平均差为 {100*delta_best.mean():+.2f} 个百分点，场景配对区间 [{best_ci[0]:+.2f}, {best_ci[1]:+.2f}]。此参照按本面板成绩选出，区间条件于该 CAP。', '',
              '## 配对回放', '', '按场景 ID 排序，各取一个并集内 seed17 成败变化的实例。', '']
    for old, new, desc in [(0, 1, '新增成功'), (1, 0, '失去成功')]:
        example = next((r for r in sorted(rows, key=lambda v:v['case_id']) if r['ncap'] and r['6k_s17']==old and r['24k_s17']==new), None)
        if example:
            case = example['case_id']
            dirname = Path('evaluation')/'single_s17'/case/'result.json.gz'
            lines.append(f'- {desc}，{case}。[6k 回放]({os.path.relpath(EARLY/dirname,OUT)}) · [24k 回放]({dirname})')
    lines += ['', '[逐场景成败与前缀](per_scene.csv) · [协议与权重摘要](protocol.json) · [离线拟合报告](../cap_actor_capacity_20261119_01/REPORT.md)']
    lines += ['', '## 单步拟合与闭环的关系', '',
              '从 6k 到 24k，平均训练教师动作一致率从 72.86% 升至 95.95%，留出教师前缀一致率从 71.28% 升至 75.48%。闭环表格衡量从初态执行到终止的任务结果。策略偏离教师动作后将访问新的公开历史；前缀诊断只能识别与归档成功轨迹一致的部分。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(OUT/'FINISHED.json', dict(complete=True, executions=len(rows)*len(SEEDS),
                                  files={name:study.sha(OUT/name) for name in ('protocol.json','per_scene.csv','summary.json','REPORT.md')}))


if __name__ == '__main__':
    report()
