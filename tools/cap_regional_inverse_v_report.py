"""Paired initial expert ranks, first proposals and closed-loop success for inverse-N V."""
import csv
import gzip
import json

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v, load_checkpoint
from tools import cap_regional_inverse_v as experiment
from tools import cap_regional_unique_v as diagnostic

study = experiment.study
PANELS = {'dev': '原开发', 'aug': '开发扰动', 'fresh': '留出房间', 'near': '困难族留出扰动'}


def initial_action(directory, panel, case, proposals):
    path = directory / 'evaluation' / panel / (case + '.json')
    result = study.read(path)
    with gzip.open(path.with_suffix('.json.gz'), 'rt') as f:
        trace = json.load(f)
    assert len(trace) == result['boundaries'] and result['case_id'] == case
    first = trace[0]
    sources = [i for i, action in enumerate(proposals) if action == first['action']]
    assert sources == first['sources'] and sources
    return int(result['success']), sources, trace


def first_disagreement(case, sids, winner, q_traces):
    cid, panel = case['case_id'], case['panel']
    experts = []
    for sid in sids:
        source = (study.ext.refs(sid, cid) if panel in ('dev', 'aug')
                  else study.OUT / 'evaluation' / sid)
        with gzip.open(source / (cid + '.json.gz'), 'rt') as f:
            experts.append(json.load(f))
    for i, choices in enumerate(zip(*experts)):
        actions = [r['a'] for r in choices]
        if any(action != actions[0] for action in actions):
            return i, [int(trace[i]['action'] == actions[winner]) for trace in q_traces]
        assert all(trace[i]['action'] == actions[0] for trace in q_traces)
    raise ValueError(f'No expert action disagreement in an exclusive success: {cid}')


@torch.inference_mode()
def records(fixtures, sids):
    value, ckpt = load_checkpoint(experiment.MODEL / 'V.pt')
    assert ckpt['kind'] == 'V' and ckpt['updates'] == 16000
    dq = study.dq
    baseline = {r['case_id']: r for r in experiment.rows_at(diagnostic.DOCS / 'per_scene.csv')}
    cases = experiment.rows_at(study.DOCS / 'per_scene.csv')
    assert len(fixtures) == len(baseline) == len(cases) == 505
    for case in cases:
        cid, panel = case['case_id'], case['panel']
        old = baseline[cid]
        labels = np.array([int(case[sid]) for sid in sids])
        n = int(labels.sum())
        assert n == int(old['n_success'])
        env = dq.fast_runtime.ScopedEnvironment(fixtures[cid], dq.study.CFG, fast_feedback=True,
            record_details=False, history_factory=dq.fast_runtime.AuditedCachedHistory)
        try:
            x, candidates, sources, valid, pool, proposals = dq.Proposals().panel(env.history)
        finally:
            env.close_reads()
            dq.fast_runtime.ACTIVE = None
        assert sources[valid].any(axis=0).all()
        scores = all_v(value, torch.from_numpy(x[None]))[0].numpy()
        assert np.isfinite(scores).all()
        order = np.argsort(-scores, kind='stable')
        top = order[0]
        before, orig_sources, old_trace = initial_action(experiment.BASE, panel, cid, proposals)
        after, new_sources, new_trace = initial_action(experiment.OUT, panel, cid, proposals)
        assert before == int(case['directq']) == int(old['directq'])
        row = dict(panel=panel, case_id=cid, parent=case['parent'], objects=int(case['objects']),
                   n_success=n, old_v=int(old['v_success']), new_v=int(labels[top]), old_q=before,
                   new_q=after, best=int(case['best']), old_top=old['top'], new_top=sids[top],
                   distinct_actions=int(valid.sum()), old_first_sources=' '.join(sids[i] for i in orig_sources),
                   new_first_sources=' '.join(sids[i] for i in new_sources), winner='', old_rank=None,
                   new_rank=None, old_match=None, new_match=None, first_fork=None, old_fork_match=None, new_fork_match=None, old_margin=None, new_margin=None)
        if n == 1:
            w = labels.argmax()
            assert sids[w] == old['winner']
            step, (old_match, new_match) = first_disagreement(case, sids, w, (old_trace, new_trace))
            row.update(first_fork=step, old_fork_match=old_match, new_fork_match=new_match)
            row.update(winner=sids[w], old_rank=int(old['rank']), new_rank=int(np.flatnonzero(order == w)[0]+1),
                       old_match=int(w in orig_sources), new_match=int(w in new_sources),
                       old_margin=float(old['margin']), new_margin=float(scores[w] - scores[labels == 0].max()))
        for sid, score in zip(sids, scores):
            row['v_'+sid] = float(score)
            row['success_'+sid] = int(case[sid])
        yield row


def paired(rows, before, after):
    return dict(n=len(rows), before=sum(r[before] for r in rows), after=sum(r[after] for r in rows),
                rescued=sum(r[before] < r[after] for r in rows), lost=sum(r[before] > r[after] for r in rows))


def summarize(rows):
    unique = [r for r in rows if r['n_success'] == 1]
    distinct = [r for r in unique if r['distinct_actions'] > 1]
    result = dict(q=paired(rows,'old_q','new_q'), unique_q=paired(unique,'old_q','new_q'),
                  unique_v=paired(unique,'old_v','new_v'), first=paired(distinct,'old_match','new_match'),
                  first_fork=paired(unique,'old_fork_match','new_fork_match'), all_v=paired(rows,'old_v','new_v'), best=sum(r['best'] for r in rows),
                  by_n={})
    for group, lo, hi in [('0',0,0),('1',1,1),('2-3',2,3),('4-10',4,10)]:
        result['by_n'][group] = paired([r for r in rows if lo <= r['n_success'] <= hi], 'old_q','new_q')
    for key in ('old','new'):
        result[key+'_top3'] = sum(r[key+'_rank'] <= 3 for r in unique)
        result[key+'_median_rank'] = float(np.median([r[key+'_rank'] for r in unique]))
        result[key+'_mean_margin'] = float(np.mean([r[key+'_margin'] for r in unique]))
    return result


def fraction_pair(m):
    return f"{m['before']}/{m['n']} → {m['after']}/{m['n']}"


def singleton_exposure(train):
    a = train.arrays
    starts = np.flatnonzero(a['boundary'] == 0)
    roots = a['root'][starts]
    unique_ids, inverse = np.unique(roots, return_inverse=True)
    singleton = unique_ids[np.bincount(inverse, weights=a['y'][starts]) == 1]
    positive = starts[np.isin(roots, singleton) & (a['y'][starts] == 1)]
    assert len(positive) == len(singleton) == 16
    # Paired offline fits use the exact same boundary index stream.
    rng = np.random.default_rng(17)
    draws = rng.integers(len(train), size=(16000, 64))
    hits = np.bincount(draws.ravel(), minlength=len(train))[positive]
    return dict(winner_starts=len(positive), draws=int(draws.size), hits=int(hits.sum()),
                min_per_room=int(hits.min()), max_per_room=int(hits.max()))


def report(fixtures):
    sids = study.read(study.OUT / 'frozen.json')['sids']
    study.init_models(sids)
    rows = list(records(fixtures, sids))
    assert len({r['case_id'] for r in rows}) == 505
    panels = {p:[r for r in rows if r['panel']==p] for p in PANELS}
    metrics = {p:summarize(group) for p,group in panels.items()}
    metrics['train'] = summarize(panels['dev'] + panels['aug'])
    assert metrics['fresh']['unique_v']['n'] == 18
    for p in ['dev','fresh','near']:
        metrics[p]['paired_q'] = study.compare(panels[p],'old_q','new_q')
    train = experiment.Store(experiment.BASE / 'arrays/train')
    nonterminal = ~train.arrays['done']
    q_targets = {}
    for tag, directory in [('old',experiment.BASE/'models/GA_s17'),('new',experiment.MODEL)]:
        targets = np.load(directory/'Q_init_targets.npy', mmap_mode='r')
        np.testing.assert_array_equal(targets[~nonterminal], train.arrays['reward'][~nonterminal])
        q_targets[tag] = float(targets[nonterminal].mean())
    protocol = study.read(experiment.OUT/'protocol.json')
    exposure = singleton_exposure(train)
    frozen_learner = experiment.OUT/'baseline_full_learner.py'
    old_sources = study.read(study.OUT/'method_sources.json')
    learner_name = 'experiments/planar_tasks/semantic_rl/program_trace_value_v1/full_learner.py'
    assert old_sources[learner_name] == study.sha(frozen_learner)
    source_names = [learner_name, 'tools/cap_regional_inverse_v.py', 'tools/cap_regional_inverse_v_report.py']
    hashes = {name:study.sha(study.ext.old.ROOT/name) for name in source_names}
    summary = dict(source_sha256=hashes, original_learner_sha256=old_sources[learner_name], exclusive_training_exposure=exposure, panels=metrics, nonterminal_q_target_mean=q_targets, protocol=protocol,
                   v_sha256=study.sha(experiment.MODEL/'V.pt'), q_sha256=study.sha(experiment.MODEL/'offline.pt'))
    out = experiment.DOCS
    out.mkdir(parents=True,exist_ok=True)
    with (out/'per_scene.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    study.save(out/'summary.json',summary)
    fresh = metrics['fresh']
    interval = fresh['paired_q']['ci']
    lines = ['# 区域十 CAP 的独胜收益对照', '',
        f"在 200 个留出房间中，18 个只有一个专家能成功。初态 V 选中唯一赢家的数量为 {fraction_pair(fresh['unique_v'])}，逐步使用 directQ 完成任务的数量为 {fraction_pair(fresh['unique_q'])}。全体留出房间的闭环结果为 {fraction_pair(fresh['q'])}。",
        '', '## 每个房间按原 V → 逆成功数 V 配对', '',
        '| 面板 | 独胜 V 排第一 | 独胜 V 前三（例） | 独胜 directQ 闭环 | 全体 directQ 闭环 | 全体救回 / 丢失 |',
        '|---|---:|---:|---:|---:|---:|']
    for p, label in PANELS.items():
        m = metrics[p]
        lines.append(f"| {label} | {fraction_pair(m['unique_v'])} | {m['old_top3']} → {m['new_top3']} | {fraction_pair(m['unique_q'])} | {fraction_pair(m['q'])} | {m['q']['rescued']} / {m['q']['lost']} |")
    first = fresh['first']
    lines += ['', f"18 个留出独胜房间中，唯一赢家的 V 排名中位数从 {fresh['old_median_rank']:g} 到 {fresh['new_median_rank']:g}。其中 {first['n']} 个初态有多个不同的合法提案，Q 首步选中赢家动作的数量为 {fraction_pair(first)}。多个 CAP 能提出同一动作，后续仍允许切换，所以首步一致与闭环成败分别计数。", '',
        f"沿所有专家共同执行的前缀，找到十个提案第一次实际分歧的位置。新旧 Q 在此前的每一步都沿共同动作执行。在这个可配对的分歧点上，Q 选中唯一赢家所提动作的数量为 {fraction_pair(fresh['first_fork'])}。该动作至少有一条已知成功后缀；其他提案在混合续跑下也可能成功。", '',
        f"全体 200 个留出房间上，闭环成功率相差 {fresh['paired_q']['pp']:+.1f} 个百分点，按物体数分层的同房间配对 bootstrap 95% 区间为 [{interval[0]:+.1f}, {interval[1]:+.1f}]。开发选定最佳单 CAP 在同一面板成功 {fresh['best']}/200。", '',
        '## 按原十 CAP 的成功数拆分留出闭环', '',
        '| 原成功 CAP 数 | 场景数 | 原 directQ | 新 directQ | 救回 / 丢失 |','|---|---:|---:|---:|---:|']
    for n,m in fresh['by_n'].items():
        lines.append(f"| {n} | {m['n']} | {m['before']} | {m['after']} | {m['rescued']} / {m['lost']} |")
    lines += ['', '## 训练与解释', '',
        '区域十 CAP 冻结。沿用 160 个开发初态和 55 个开发扰动、2,150 条成败轨迹和 144,099 个决策边界。16 个训练初态是独胜，54 个初态十个 CAP 都失败。V 在成功轨迹的每个边界回归 1/N，失败边界回归 0。N 由该开发初态的十次完整执行得出。部署仅用公开历史和 CAP 提案。留出面板的 N 只用于事后分组。',
        '', '保持按边界均匀抽样、无 trace 的 64×64 GA、相同归一化、seed 17、batch 64、Adam 学习率 0.0003。V 从头更新 16,000 次；Q 从该 V 对应参数初始化，更新 8,000 次。非终止动作仍回归实际后继最高合法 V，终止步仍回归实际成功 1 或失败 0。这一处目标尺度差异属于本次只改 V 标签的设计。',
        '', f"按两个拟合共用的随机索引重算，V 的 {exposure['draws']:,} 次边界抽取中，16 个独胜训练房间的赢家初始边界合计被抽中 {exposure['hits']} 次，每个房间 {exposure['min_per_room']}–{exposure['max_per_room']} 次。赢家的后续边界也参与训练。",
        '', f"非终止 Q 标签均值从 {q_targets['old']:.3f} 变为 {q_targets['new']:.3f}。新 V 的数值表达按初态难度缩放后的成功收益。独胜初态的赢家和输家训练目标在两组中均为 1 和 0，改动来自压低多解房间的成功目标。比较依据是专家排序和实际成功率。",
        '', '本次只有一个训练种子。两组共用冻结面板；留出结果在之前的诊断中已查看，90 个困难族扰动来自 45 个开发父房间。结论限于本次程序库和约定的预算。',
        '', '[逐场景排名、首步和闭环](per_scene.csv) · [指标与训练协议](summary.json)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    study.save(experiment.OUT/'FINISHED.json',dict(cases=len(rows),report=str(out),q_sha256=summary['q_sha256']))
    study.save(experiment.OUT/'stage.json',dict(stage='finished'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    report(study.read(study.OUT / study.ext.DEV) | study.read(study.OUT/'held.json'))
