"""Post-hoc paired analysis; never used for model fitting or selection."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs/semantic_rl/da_joint_20261117_01'
HOST = ROOT / 'outputs/semantic_rl_authors/da_joint_20261117_01'


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired_summary(left, right, strata, seed=1781):
    """Rows are independent cases; columns are shared training seeds.

    Average seeds within a case, then resample cases within object-count strata.
    The interval is conditional on the fitted models, not a training-seed CI.
    """
    left, right = np.asarray(left, float), np.asarray(right, float)
    assert left.shape == right.shape and left.ndim == 2
    strata = np.asarray(strata)
    assert len(strata) == len(left)
    diff = (left - right).mean(axis=1)
    rng = np.random.default_rng(seed)
    totals = np.zeros(10000)
    for group in sorted(set(strata.tolist())):
        d = diff[strata == group]
        totals += d[rng.integers(0, len(d), size=(len(totals), len(d)))].sum(axis=1)
    return dict(
        left_mean=float(left.mean()), right_mean=float(right.mean()),
        difference=float(diff.mean()),
        stratified_paired_case_bootstrap95=np.quantile(totals / len(diff), [.025, .975]).tolist(),
        per_seed=[dict(left=int(a.sum()), right=int(b.sum()),
                       both=int(((a == 1) & (b == 1)).sum()),
                       left_only=int(((a == 1) & (b == 0)).sum()),
                       right_only=int(((a == 0) & (b == 1)).sum()))
                  for a, b in zip(left.T, right.T)],
        cases=len(left), training_seeds=left.shape[1],
        interpretation='Exploratory, no multiple-comparison adjustment; conditional on these fitted models.')


def main():
    assert read(HOST / 'study.exit.json')['code'] == 0
    result = read(HOST / 'RESULT.json')
    rows = read(BASE / 'evaluation/result.json')['rows']
    jobs = read(BASE / 'evaluation/jobs.json')
    inputs = read(BASE / 'inputs.json')
    fresh = inputs['fresh']
    for path, sha in inputs['hashes'].items():
        assert digest(Path(path)) == sha, path
    assert digest(ROOT / 'docs/agent_exchange/tasks/DA_Joint_Representation_20261117.md') == inputs['protocol']
    metrics = result['metrics']
    names = sorted(metrics)
    for name in names:
        for split, expected in {'joint_fresh': 80, 'development_regression': 24,
                                'repair_train': 16, 'handoff_holdout': 8}.items():
            assert metrics[name][split]['n'] == expected, (name, split)
    key = lambda j: (j['condition'], j['evaluation'], j['case_id'])
    assert len(rows) == len(jobs) == 1536
    assert len({key(j) for j in jobs}) == len(jobs)
    assert {key(r['job']) for r in rows} == {key(j) for j in jobs}
    assert not any(r.get('error') or r['stop'] == 'wall_budget' for r in rows)
    for r in rows:
        if r['job']['evaluation'] == 'handoff':
            assert r['prefix_audit']['public_equal'] and r['prefix_audit']['physics_equal']
    for rel, sha in read(BASE / 'model_sources/manifest.json').items():
        assert digest(ROOT / rel) == sha, rel
    for name, sha in read(BASE / 'fit/seal.json').items():
        assert digest(BASE / 'fit' / name / 'model.pt') == sha, name
        complete = read(BASE / 'fit' / name / 'complete.json')
        assert complete['updates'] == 28000
        assert read(BASE / 'fit' / name / 'progress.json')['update'] == 28000
    assert result['physical_episodes'] == 1696
    assert result['physical_ticks'] + 28800 <= 3240000
    fresh = sorted(fresh, key=lambda r: r['id'])
    assert len({r['id'] for r in fresh}) == 80
    strata = [r['movable_count'] for r in fresh]

    def values(tier, mode):
        return [[int(case['id'] in metrics[f'{tier}_{mode}_s{seed}']['joint_fresh']['successful_cases'])
                 for seed in (17, 29)] for case in fresh]

    comparisons = {}
    for tier in ('base', 'repair'):
        for a, b in (('enriched', 'current'), ('joint', 'current'), ('joint', 'enriched')):
            comparisons[f'{tier}: {a} - {b}'] = paired_summary(values(tier, a), values(tier, b), strata)
    for mode in ('current', 'enriched', 'joint'):
        comparisons[f'{mode}: repair - base'] = paired_summary(values('repair', mode), values('base', mode), strata)
    seed_overlap = {}
    for tier in ('base', 'repair'):
        for mode in ('current', 'enriched', 'joint'):
            a, b = np.asarray(values(tier, mode), bool).T
            seed_overlap[f'{tier}_{mode}'] = dict(
                both_success=int((a & b).sum()), either_success=int((a | b).sum()),
                outcome_disagreements=int((a != b).sum()), cases=len(a),
                note='Descriptive comparison of only two fitted seeds; equal totals need not mean equal cases.')
    demonstrated = {r['case_id'] for r in read(BASE / 'data_repair/selection.json')}
    states = read(BASE / 'selection.json')['states']
    train_roots = {r['case_id'] for r in states if r['split'] == 'repair_train'}
    holdout_roots = {r['case_id'] for r in states if r['split'] == 'handoff_holdout'}
    assert len(train_roots) == 16 and len(holdout_roots) == 8
    assert not train_roots & holdout_roots and demonstrated <= train_roots
    suffix_reproduction = {}
    for name in names:
        if not name.startswith('repair_'):
            continue
        rr = [r for r in rows if r['job']['condition'] == name
              and r['job']['evaluation'] == 'handoff' and r['job']['case_id'] in demonstrated]
        assert len(rr) == len(demonstrated) == 10
        suffix_reproduction[name] = dict(success=sum(r['success'] for r in rr), n=len(rr),
                                         failed_cases=[r['job']['case_id'] for r in rr if not r['success']])
    summary = dict(comparisons=comparisons, seed_overlap=seed_overlap,
                   demonstrated_suffix_reproduction=suffix_reproduction, audit=dict(
        models=len(names), evaluation_episodes=len(rows), total_episodes=result['physical_episodes'],
        total_ticks=result['physical_ticks'], frozen_sources_and_models_verified=True,
        all_fits_28000=True, no_missing_or_duplicate_jobs=True,
        frozen_input_hashes_and_protocol_verified=True, all_288_handoff_prefixes_verified=True))
    save(BASE / 'analysis/paired_comparisons.json', summary)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    colors = ['#6a8caf', '#eea857', '#35a78b']
    for ax, tier in zip(axes, ('base', 'repair')):
        for x, (mode, color) in enumerate(zip(('current', 'enriched', 'joint'), colors)):
            v = np.asarray(values(tier, mode)).mean(axis=0) * 100
            ax.bar(x, v.mean(), color=color, alpha=.75)
            ax.scatter([x-.07, x+.07], v, c='black', s=22, zorder=3)
            ax.text(x, max(v)+2, '/'.join(str(int(round(z*.8))) for z in v), ha='center')
        ax.set_xticks(range(3), ['Current', 'Enriched', 'Joint'])
        ax.set_title('270 demonstrations' if tier == 'base' else '270 + 10 repair suffixes')
        ax.set_ylim(0, 100)
        ax.grid(axis='y', alpha=.15)
    axes[0].set_ylabel('Fresh-case success (%)')
    fig.suptitle('80 shared fresh cases; dots = training seeds 17 and 29; labels = successes / 80')
    fig.tight_layout()
    for suffix in ('png', 'pdf'):
        fig.savefig(BASE / f'analysis/fresh_comparison.{suffix}', dpi=180)
    plt.close(fig)

    report = ROOT / 'docs/agent_exchange/reviews/DA_Joint_Representation_20261117.md'
    marker = '\n## 配对比较与验收\n'
    text = report.read_text().split(marker)[0]
    lines = [marker, '', '按物体数分层重采样；先在同一场景内平均两训练seed，再进行场景配对bootstrap。区间仅反映这些固定模型下的场景差异；多项比较是探索性分析。', '',
             '| 对照（前者减后者） | 成功率差（百分点） | 95%区间（百分点） |', '|---|---:|---:|']
    for name, c in comparisons.items():
        lo, hi = c['stratified_paired_case_bootstrap95']
        lines.append(f"| {name} | {100*c['difference']:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |")
    lines += ['', '12模型均完成28000更新；1536评估无缺失、重复或基础设施异常，冻结源码与模型哈希一致。',
              f"正式执行共{result['physical_episodes']}次、{result['physical_ticks']} tick；首次作废分叉另计16次/至多28800 tick，工程验收另记。", '',
              '数据范围：原270示范9292边界中，直接选择非目标物体的动作是0。因此本轮不能声称验证主动清障子技能的整合。接管诊断的24个状态均在第8边界，留出8个仍属于开发诊断。', '',
              '真实前缀诊断：24状态原策略成功10，换一个动作后仍由原策略继续新增6；CAP持续接管新增4且没有额外救回。44个同首动作配对中，原策略继续独有成功12、CAP继续独有成功4、共同成功11、共同失败17。这是24根状态中的相关分叉，不是44个独立场景。', '',
              f"图表及逐对照证据：`{BASE.relative_to(ROOT)}/analysis/`。"]
    report.write_text(text + '\n'.join(lines) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
