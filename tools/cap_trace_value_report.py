"""Read-only, paired final-test analysis for the completed trace/value/RL study."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

ARMS = ('GA', 'LOCAL', 'GLOBAL')
SEEDS = (17, 29, 43)
CAPS = ['CAP-RR-'+s for s in ('009', '003', '001', '010', '006', '007', '002')]


def read(path):return json.loads(Path(path).read_text())


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def unpack(path):
    with gzip.open(path, 'rt') as f:return json.load(f)


def routing_record(path):
    result=unpack(path)
    common=set(range(7));segment=None;switches=0;fallback=False;counts=[0]*7
    for decision in result['decisions']:
        supporters=set(decision['sources'][decision['chosen']])
        for i in supporters:counts[i]+=1
        common &= supporters
        if not supporters:
            fallback=True
            continue
        if segment is None:segment=supporters
        elif segment & supporters:segment &= supporters
        else:
            switches+=1
            segment=supporters
    summary=result['summary']
    return dict(condition=summary['condition'],case_id=summary['case_id'],success=summary['success'],
                decisions=len(result['decisions']),support_counts=counts,
                compatible_single_caps=[CAPS[i] for i in sorted(common)],fallback=fallback,
                minimum_required_switches=None if fallback else switches)


def report(out):
    out = Path(out).resolve()
    final = read(out/'final_test.json')
    assert final['complete'] and len(final['results']) == 5600
    cases = read(out/'panels/final_test.json')['cases']
    case_ids = [c['case_id'] for c in cases]
    by_case = {c:i for i,c in enumerate(case_ids)}
    rows = final['results']
    conditions = sorted({r['condition'] for r in rows})
    table = {(r['condition'], r['case_id']):r for r in rows}
    assert len(table) == 5600 and len(conditions) == 28
    for condition in conditions:
        assert {c for method,c in table if method == condition} == set(case_ids)
    assert all(r['complete'] and not r.get('error') for r in rows)
    frozen = read(out/'ALL_MODELS_FROZEN.json')
    assert all(hashlib.sha256((out/p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items())
    scores = {name:np.array([table[name,c]['success'] for c in case_ids], dtype=float) for name in conditions}
    strata={n:np.array([c['movable_count']==n for c in cases]) for n in range(2,6)}
    by_movable_count={name:{str(n):dict(success=int(value[mask].sum()),total=int(mask.sum()))
                            for n,mask in strata.items()} for name,value in scores.items()}
    teacher_names = [f'CAP-RR-{i:03d}' for i in range(1,11)]
    best = min(teacher_names, key=lambda name:(-scores[name].sum(),name))
    union7 = np.max([scores[c] for c in CAPS], axis=0).astype(bool)
    union10 = np.max([scores[c] for c in teacher_names], axis=0).astype(bool)
    base = scores['CAP-RR-009'].astype(bool)
    rng = np.random.default_rng(2026111803)
    bootstrap = np.concatenate([rng.choice(np.flatnonzero(np.array([c['movable_count'] for c in cases])==n),
                              size=(10000,50), replace=True) for n in range(2,6)],axis=1)

    def compare(a,b):
        difference = a-b
        low,high = np.quantile(difference[bootstrap].mean(1), [.025,.975])
        return dict(difference_pp=float(100*difference.mean()), ci95_pp=[float(100*low),float(100*high)],
                    positive_scenes=int((difference>0).sum()), negative_scenes=int((difference<0).sum()))

    mean = {}
    groups = {}
    for arm in ARMS:
        groups[arm] = {}
        for stage in ('offline','online'):
            matrices = np.stack([scores[f'{arm}_s{s}_{stage}'] for s in SEEDS])
            mean[arm,stage] = matrices.mean(0)
            groups[arm][stage] = dict(seed_successes=matrices.sum(1).astype(int).tolist(), denominator=200,
                mean_success_rate=float(matrices.mean()), mean_successes=float(matrices.sum(1).mean()),
                versus_development_best=compare(matrices.mean(0),scores['CAP-RR-009']),
                versus_test_best_descriptive=compare(matrices.mean(0),scores[best]),
                retained_test_best_per_seed=matrices[:,scores[best].astype(bool)].sum(1).astype(int).tolist(),
                lost_test_best_per_seed=(1-matrices[:,scores[best].astype(bool)]).sum(1).astype(int).tolist(),
                gained_test_best_per_seed=matrices[:,~scores[best].astype(bool)].sum(1).astype(int).tolist(),
                retained_union7_per_seed=(matrices[:,union7]).sum(1).astype(int).tolist(),
                retained_union10_per_seed=(matrices[:,union10]).sum(1).astype(int).tolist(),
                new_outside_union7_per_seed=(matrices[:,~union7]).sum(1).astype(int).tolist(),
                new_outside_union10_per_seed=(matrices[:,~union10]).sum(1).astype(int).tolist(),
                recovered_complementarity_per_seed=matrices[:,union7 & ~base].sum(1).astype(int).tolist(),
                lost_development_best_per_seed=(1-matrices[:,base]).sum(1).astype(int).tolist())
    comparisons = dict(global_minus_ga=compare(mean['GLOBAL','online'],mean['GA','online']),
                       local_minus_ga_secondary=compare(mean['LOCAL','online'],mean['GA','online']),
                       global_minus_local=compare(mean['GLOBAL','online'],mean['LOCAL','online']),
                       offline_global_minus_ga=compare(mean['GLOBAL','offline'],mean['GA','offline']),
                       online_minus_offline={a:compare(mean[a,'online'],mean[a,'offline']) for a in ARMS})
    runtime = {}
    for condition in conditions:
        selected = [r for r in rows if r['condition']==condition]
        succeeded = [r for r in selected if r['success']]
        runtime[condition] = dict(mean_ticks=float(np.mean([r['ticks'] for r in selected])),
            successful_mean_ticks=None if not succeeded else float(np.mean([r['ticks'] for r in succeeded])),
            mean_decisions=float(np.mean([r['decisions'] for r in selected])),
            mean_worker_seconds=float(np.mean([r.get('seconds',r.get('wall_seconds')) for r in selected])),
            total_trace_seconds=float(sum(r.get('trace_seconds',0) for r in selected)),
            total_network_seconds=float(sum(r.get('network_seconds',0) for r in selected)),
            stops={s:sum(r['stop']==s for r in selected) for s in sorted({r['stop'] for r in selected})})
    monitor, online, fits, vmetrics = {}, {}, {}, {}
    all_physical = list(rows)
    for arm in ARMS:
        for seed in SEEDS:
            name = f'{arm}_s{seed}'
            model = out/'models'/name
            fits[name] = read(model/'offline_complete.json')
            vmetrics[name] = read(model/'V_metrics.json')
            assert fits[name]['v_updates']==16000 and fits[name]['q_updates']==8000
            assert read(model/'online_complete.json')['episodes']==200
            assert read(model/'online_r4.json')['updates']==6000
            monitor[name] = {}
            for phase in ('offline','online100','online200'):
                result = read(out/'monitor'/name/(phase+'.json'))
                assert result['complete'] and len(result['results'])==32
                monitor[name][phase] = sum(r['success'] for r in result['results'])
                all_physical.extend(result['results'])
            online[name] = []
            for round_number in range(1,5):
                result = read(out/'online'/name/f'r{round_number}.json')
                assert result['complete'] and len(result['results'])==50
                online[name].append(dict(round=round_number, success=sum(r['success'] for r in result['results']), total=50,
                                         gradient=read(model/f'online_r{round_number}.json')))
                all_physical.extend(result['results'])
    assert len(all_physical)==8264
    anomaly_names={'numeric_failure','penetration_limit','out_of_bounds'}
    anomalies=[dict(condition=r['condition'],case_id=r['case_id'],termination=r.get('termination'))
               for r in all_physical if r.get('termination') in anomaly_names]
    assert not anomalies, 'Physical anomalies require separate incomplete-study analysis: '+str(anomalies[:8])
    routing_path=out/'report/routing_per_case.json'
    if routing_path.exists():routing=read(routing_path)
    else:
        paths=[str(out/'evaluation'/r['condition']/r['case_id']/'result.json.gz') for r in rows
               if r['condition'].endswith('_online')]
        assert len(paths)==1800
        with ProcessPoolExecutor(max_workers=8) as pool:
            routing=list(pool.map(routing_record,paths,chunksize=4))
        save(routing_path,routing)
    assert len(routing)==1800
    routing_summary={}
    for arm in ARMS:
        records=[r for r in routing if r['condition'].startswith(arm+'_')]
        mixed=[r for r in records if not r['fallback'] and not r['compatible_single_caps']]
        ordinary=[r['minimum_required_switches'] for r in records if not r['fallback']]
        routing_summary[arm]=dict(episodes=len(records),require_multiple_caps=len(mixed),
            mixed_successes=sum(r['success'] for r in mixed),fallback_episodes=sum(r['fallback'] for r in records),
            mean_minimum_required_switches=float(np.mean(ordinary)) if ordinary else None,
            selected_action_support_counts={cap:sum(r['support_counts'][i] for r in records) for i,cap in enumerate(CAPS)})
    calibration = {}
    for arm in ARMS:
        for stage in ('offline','online'):
            selected = [r for r in rows if r['condition'].startswith(arm+'_') and r['condition'].endswith('_'+stage)]
            p = np.array([r['initial_value'] for r in selected])
            y = np.array([r['success'] for r in selected])
            bins = []
            for lo in np.arange(0,1,.2):
                mask=(p>=lo)&(p<lo+.2 if lo<.8 else p<=1)
                bins.append(dict(lower=float(lo), count=int(mask.sum()),
                                 predicted=None if not mask.any() else float(p[mask].mean()),
                                 observed=None if not mask.any() else float(y[mask].mean())))
            calibration[arm+'_'+stage] = dict(brier=float(((p-y)**2).mean()),
                mean_prediction=float(p.mean()), actual_success=float(y.mean()), bins=bins,
                note='Initial chosen-action Q versus this frozen selector rollout; 600 executions share 200 scenes.')
    initial_routing = {}
    for name, record in vmetrics.items():
        initial_routing[name] = {}
        for split in ('train', 'validation'):
            initial = record[split]['initial']
            initial_routing[name][split] = dict(
                selected_success=sum(r['chosen_success'] for r in initial),
                cap009_success=sum(r['outcomes'][0] for r in initial),
                teacher_union=sum(any(r['outcomes']) for r in initial),
                total=len(initial))
    candidate_counts = {}
    shards = read(out/'shards.json')
    training_roots=set(read(out/'split.json')['train'])
    cap_boundary_counts={cap:dict(train=0,validation=0,successful_train_boundaries=0) for cap in CAPS}
    for shard in shards:
        for k,n in shard['candidate_counts'].items():candidate_counts[k]=candidate_counts.get(k,0)+n
        split='train' if shard['case_id'] in training_roots else 'validation'
        cap_boundary_counts[shard['cap_id']][split]+=shard['boundaries']
        if split=='train' and shard['success']:
            cap_boundary_counts[shard['cap_id']]['successful_train_boundaries']+=shard['boundaries']
    summary = dict(complete=True, single_cap={c:int(scores[c].sum()) for c in teacher_names},
        development_selected_cap='CAP-RR-009', test_best_descriptive=best,
        by_movable_count=by_movable_count,
        union7=int(union7.sum()), union10=int(union10.sum()),
        complementarity_opportunities_vs_cap009=int((union7 & ~base).sum()), groups=groups,
        comparisons=comparisons, monitoring=monitor, online_collection=online, calibration=calibration,
        value_initial_routing=initial_routing, runtime=runtime, offline_fit=fits, routing=routing_summary,
        dataset=read(out/'dataset.json'), candidate_counts=candidate_counts, cap_boundary_counts=cap_boundary_counts,
        mean_distinct_candidates=sum(int(k)*n for k,n in candidate_counts.items())/sum(candidate_counts.values()),
        invalid_offline_proposals=sum(r['invalid_proposals'] for r in shards),
        fallback_offline_boundaries=sum(r['fallback_boundaries'] for r in shards),
        physical_accounting=dict(formal_episodes=len(all_physical), formal_ticks=sum(r['ticks'] for r in all_physical),
            formal_decisions=sum(r['decisions'] for r in all_physical), final_test_executions=5600,
            monitor_executions=864, online_executions=1800,
            final_invalid_proposals=sum(r.get('invalid_proposals',0) for r in rows),
            final_fallback_boundaries=sum(r.get('fallback_boundaries',0) for r in rows)),
        uncertainty='10000 stratified paired bootstrap resamples of 200 roots; conditional on these 3 trained seeds. Test-best CAP interval is descriptive, selection not corrected.',
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_manifest_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest())
    if (out/'cache_import.json').exists():
        cache = read(out/'cache_import.json')
        summary['feature_cache_reuse'] = cache
        if (out/'study.exit.json').exists():
            summary['wall_seconds_from_first_full_extraction'] = read(out/'study.exit.json')['finished']-cache['first_extract_epoch']
    save(out/'AUDITED_SUMMARY.json', summary)
    with (out/'per_case_results.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=['condition','case_id','success','ticks','decisions','stop','initial_value','seconds'])
        writer.writeheader()
        for r in sorted(rows,key=lambda r:(r['condition'],r['case_id'])):
            record={k:r.get(k) for k in writer.fieldnames}
            record['seconds']=r.get('seconds',r.get('wall_seconds'))
            writer.writerow(record)
    lines = ['# CAP trace＋RL完整评估报告', '',
        '本报告来自新冻结200场景的同机实验。所有模型固定后才执行最终测试；未按测试挑种子、checkpoint或表示组。', '',
        '命名说明：本轮日志中的GA表示“公开观测＋全部CAP动作建议＋当前候选动作及建议来源”的无trace基线。A不是旧全合法动作树实验中单独的CAP推荐偏好标志；所有候选均由CAP提出，那个布尔标志本身会退化。本轮与旧GA不是相同方法协议，不能直接合并结果。LOCAL/GLOBAL在这个共同基线上增加相应trace。', '',
        '## 主要结论', '']
    g = groups['GLOBAL']['online']
    best_count = int(scores[best].sum())
    if g['mean_successes'] > best_count:
        lines.append(f"GLOBAL＋RL平均成功{g['mean_successes']:.2f}/200，在这批测试的点估计上超过最好的单CAP {best}（{best_count}/200）。")
    else:
        lines.append(f"GLOBAL＋RL平均成功{g['mean_successes']:.2f}/200，没有超过本批最好的单CAP {best}（{best_count}/200）。")
    for title,result in [('GLOBAL＋RL−开发选定CAP009',g['versus_development_best']),
                         ('GLOBAL＋RL−测试后最强单CAP（描述性）',g['versus_test_best_descriptive']),
                         ('自动trace的主比较GLOBAL−GA',comparisons['global_minus_ga']),
                         ('次要比较LOCAL−GA',comparisons['local_minus_ga_secondary']),
                         ('跨CAP trace比较GLOBAL−LOCAL',comparisons['global_minus_local']),
                         ('GLOBAL在线−离线',comparisons['online_minus_offline']['GLOBAL'])]:
        lo,hi=result['ci95_pp']
        lines.append(f"{title}：{result['difference_pp']:+.2f}个百分点，配对场景bootstrap 95%区间[{lo:+.2f}, {hi:+.2f}]。" +
                     ('本批支持正向收益。' if lo>0 else '本批支持负向影响。' if hi<0 else '区间未支持稳定正向收益。'))
    lines += ['', '## 最终测试', '', '| 表示 | 离线Q：seed17/29/43 | 在线Q：seed17/29/43 | 在线均值 | 在线−离线 |',
              '|---|---|---|---:|---:|']
    for arm in ARMS:
        a,b=groups[arm]['offline'],groups[arm]['online']
        lines.append(f"| {arm} | {' / '.join(map(str,a['seed_successes']))} | {' / '.join(map(str,b['seed_successes']))} | {100*b['mean_success_rate']:.2f}% | {comparisons['online_minus_offline'][arm]['difference_pp']:+.2f} pp |")
    lines += ['', '每个数的分母200；三个种子共用同200初态，不是600个独立场景。', '',
              '![最终成功率与三个训练种子](report/success_comparison.png)', '',
              '| 单CAP | 成功/200 |', '|---|---:|']
    for cap in sorted(teacher_names,key=lambda c:(-scores[c].sum(),c)):
        lines.append(f'| {cap} | {int(scores[cap].sum())} |')
    lines += ['', f"开发阶段固定选出的CAP009为{int(scores['CAP-RR-009'].sum())}/200；{best}是测试后描述性最强单CAP，未用于调参或训练选择。", '',
              '| 在线策略 | 2物体 | 3物体 | 4物体 | 5物体 |',
              '|---|---:|---:|---:|---:|']
    for label,values in [(a,mean[a,'online']) for a in ARMS]+[(best,scores[best])]:
        lines.append('| '+label+' | '+' | '.join(f'{values[mask].sum():.2f}/50' for mask in strata.values())+' |')
    lines += ['', '神经策略分层数字为三个seed的平均成功数，各物体数共享50根；单CAP为实际成功数。', '',
              '## 是否整合了互补能力', '',
              '预先选择的7CAP在原开发集保留560/570个已知可解初态，因此这次筛选不是对10CAP的无损去重。最终测试同时给出7CAP与10CAP并集，区分候选库覆盖损失与学习选择器的损失。', '',
              f"选中7CAP从初态整段执行的成功并集为{int(union7.sum())}/200，原10CAP为{int(union10.sum())}/200。并集是逐初态回看成败的参照，不是可部署策略，也不是逐步混合策略的理论上界。", '',
              '| 在线组 | 7CAP并集内保留：3seed | CAP009失败而其他CAP可解的恢复数 | CAP009成功却丢失的数量 | 7CAP均失败时新增成功 |',
              '|---|---|---|---|---|']
    for arm in ARMS:
        r=groups[arm]['online']
        lines.append('| '+arm+' | '+' | '.join(' / '.join(map(str,r[k])) for k in (
            'retained_union7_per_seed','recovered_complementarity_per_seed','lost_development_best_per_seed','new_outside_union7_per_seed'))+' |')
    lines += ['', f"CAP009之外、7CAP并集内共有{int((union7 & ~base).sum())}个可恢复场景。应同时看恢复和丢失，不能仅展示成功切换的个例。", '',
              '| 在线组 | 无法由单个CAP全程建议解释的执行/600 | 这些混合执行中成功数 | 平均最少来源切换次数 |',
              '|---|---:|---:|---:|']
    for arm in ARMS:
        r=routing_summary[arm]
        minimum='NA' if r['mean_minimum_required_switches'] is None else f"{r['mean_minimum_required_switches']:.2f}"
        lines.append(f"| {arm} | {r['require_multiple_caps']} | {r['mixed_successes']} | {minimum} |")
    lines += ['', '来源统计读取实际所选动作的全部支持CAP，不为重复建议任意指定单一来源。若全轨迹支持集合的交集非空，至少有一个CAP在每个边界都推荐了实际动作；否则必须混合来源。最少切换数按连续区段能否由同一CAP支持计算，不是模型显式记录的路由身份。出现混合本身不证明有收益，应结合上表的能力恢复与丢失判断。600执行仍共享200初态。', '',
              '## 价值拟合与开发诊断', '',
              '| 模型 | V训练边界拟合率 | V训练Brier | V验证按根等权Brier | 32开发根：offline/online100/online200 |',
              '|---|---:|---:|---:|---|']
    for arm in ARMS:
        for seed in SEEDS:
            name=f'{arm}_s{seed}';v=vmetrics[name];m=monitor[name]
            lines.append(f"| {name} | {100*v['train']['boundary_accuracy']:.2f}% | {v['train']['brier']:.4f} | {v['validation']['root_equal_brier']:.4f} | {m['offline']} / {m['online100']} / {m['online200']} |")
    lines += ['', '![固定32开发场景上的学习曲线](report/learning_curves.png)', '',
              'V拟合率指以0.5阈值预测后缀成功，不是动作模仿准确率。开发监测未用于选checkpoint。Q初始化标签来自各组冻结V的估计，因此Q阶段比较的是完整表示流程，不是同伪标签输入消融。', '',
              '| V初态选CAP后整段续接 | 训练根成功/608 | 开发验证根成功/152 |',
              '|---|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            name=f'{arm}_s{seed}';r=initial_routing[name]
            lines.append(f"| {name} | {r['train']['selected_success']} | {r['validation']['selected_success']} |")
    lines += ['', '这一初态路由诊断使用原5320运行的已保存成败来核对V选出的CAP；没有重新执行物理，也不等同于逐步Q混合策略。开发划分的CAP选择曾用全部760根，因此152根不是整个方法流程的盲测。', '',
              '| 在线组 | 初态Q均值 | 实际成功率 | 初态Q Brier |',
              '|---|---:|---:|---:|']
    for arm in ARMS:
        c=calibration[arm+'_online']
        lines.append(f"| {arm} | {c['mean_prediction']:.4f} | {c['actual_success']:.4f} | {c['brier']:.4f} |")
    lines += ['', '此处校准对象是初态选中动作的Q与随后该固定选择器的完整执行结果；不是所有候选动作的反事实价值校准。', '',
              '| 策略 | 平均实际ticks | 平均决策数 | 每次执行平均worker秒 |',
              '|---|---:|---:|---:|']
    for label,names in [(a,[f'{a}_s{s}_online' for s in SEEDS]) for a in ARMS]+[(best,[best])]:
        values=[np.mean([runtime[n][k] for n in names]) for k in ('mean_ticks','mean_decisions','mean_worker_seconds')]
        lines.append(f'| {label} | {values[0]:.2f} | {values[1]:.2f} | {values[2]:.3f} |')
    lines += ['', '1 tick为0.05物理秒；表中时间是实际推进量，不含额外失败惩罚。worker墙钟用于运行诊断，不能当作充分优化后的部署耗时比较：本轮GA也计算全部trace后再屏蔽，且各worker的初始化计时口径并非严格的算法基准。', '',
              '## 输入、训练及成本', '',
              f"5320条原始成功/失败运行，共369726次决策；按初态分608训练/152开发验证，训练边界{summary['dataset']['train']['boundaries']}、验证边界{summary['dataset']['validation']['boundaries']}。均匀抽决策边界，长轨迹有更大有效权重；不是轨迹均衡采样。", '',
              'G为初态＋最近8边界的公开几何，原始维数11304；所有组另读全部7CAP建议602维及2个剩余预算。自动trace共2096槽并附缺失掩码；LOCAL仅开放候选建议来源CAP的trace，重复动作保留所有来源，GLOBAL开放全部。三组同架构、同参数量，Q为1808129参数。', '',
              f"原离线边界平均有{summary['mean_distinct_candidates']:.3f}个去重后的候选动作；候选数分布为{candidate_counts}。全部369726边界中非法CAP建议{summary['invalid_offline_proposals']}次，空候选回退{summary['fallback_offline_boundaries']}次。", '',
              '每模型16000次V更新、8000次Q初始化、200次在线完整任务和6000次Double-DQN更新。在线每50任务固定收集后更新1500次，离线/自身在线回放各50%，normalizer不更新。新在线200初态和最终测试200初态先于拟合冻结，与已知历史内容及彼此不重合；同一场景生成规则，不额外改变难度分布。', '',
              'V_i(h)用行为CAP i的完整成功/失败后缀监督；离线Q初始化目标为 r＋(1−done) max_i V_i(h_next)，只监督真实执行过的动作。在线目标为 r＋(1−done) Q_target(h_next, argmax_a Q_online(h_next,a))。因此最终Q的续接对象是共同动作选择器，不是承诺一直执行提出该动作的CAP；V与Q的含义不能混用。', '',
              '奖励为有限1800ticks/300决策内成功1、其余0，gamma=1。64条原decision_budget保留截断身份，但这个有限任务在决策预算耗尽时结束，不bootstrap。实际时长按执行periods计，不按命令ticks或原失败成本计。', '',
              f"正式物理执行{len(all_physical)}次、实际{summary['physical_accounting']['formal_ticks']}ticks。包含在线1800、开发监测864、最终模型3600和CAP基线2000。pilot另48执行，软件集成检查另1执行，均不进入最终测试成功率。最终非法建议计数{summary['physical_accounting']['final_invalid_proposals']}，空池回退边界{summary['physical_accounting']['final_fallback_boundaries']}。", '',
              '所有新闭环执行使用同一目标机、NumPy雷达后端。原离线数据来自源机原运行，不能把同机最终比较解释成跨机器逐轨迹复现。源码、特征schema、数据及模型哈希随输出保存。', '',
              '## 解释边界', '',
              'trace是有损程序计算特征：记录循环外标量/分支及返回路径，没有展开全部循环微状态，也没有证明它对动作价值充分。网络看到程序中间量不等于获得环境真值或准确反事实。初态外的其他CAP续接回报并未全部实测，V最大值初始化仍可能外推；在线RL用真实转移修正，不会自动消除覆盖和表示误差。', '',
              '按本实现，trace是公开历史与固定程序的确定性计算，不创造新的环境观测。相对只保留初态和最近8边界的G，它可能通过CAP内部记忆提供更长历史的摘要；即使有增益，也不能仅据本轮把原因归为局部表达式语义，而排除历史压缩的作用。GA也读取全部CAP建议，因此已经包含部分程序记忆信息。', '',
              '候选是7CAP各自提出的完整联合动作，精确去重后至多7个，不重组机器人子动作或额外枚举时长。trace来自已有CAP语义程序，部分变量也参与其原有描述特征计算；本轮没有单独证明自动trace优于手写D特征，也不检验对未见CAP程序的迁移。', '',
              '初态上的CAP成功并集说明存在可解该初态的整段程序，并不保证各程序的建议可以逐决策互换。混合执行会产生原CAP轨迹之外的前缀；所有CAP虽然都按真实前缀更新记忆，其后续建议仍可能落在训练覆盖之外。因此恢复了多少互补成功、同时丢失了多少原有成功，是本轮必须同时核对的指标。', '',
              '场景bootstrap区间仅反映这200根上的配对抽样不确定性，条件于这三个训练种子；测试后最优单CAP的区间未校正选择效应。不能从一次规模和三个种子推出普遍优势或样本效率规律。最终初态Q校准和逐场景得失见JSON/CSV，避免用critic拟合率替代任务成功。', '',
              '## 产物', '',
              '- `AUDITED_SUMMARY.json`：统计、配对区间、覆盖、校准和完整执行计账。',
              '- `per_case_results.csv`：5600次最终执行逐场景结果。',
              '- `models/`：V、离线Q、每轮在线Q、优化器/RNG及回放身份。',
              '- `evaluation/`、`online/`、`monitor/`：原始决策、公开历史与物理回放。',
              '- `report/success_comparison.png`、`report/learning_curves.png`、`report/replays.html`：最终结果图、开发学习曲线与精选配对回放。', '']
    (out/'REPORT.md').write_text('\n'.join(lines))
    (out/'report').mkdir(exist_ok=True)
    (out/'report/analysis.py').write_bytes(Path(__file__).read_bytes())
    plot(out, summary)
    replays(out, table, scores, case_ids)
    return summary


def plot(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    directory=out/'report';directory.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    names=list(summary['single_cap'])
    axes[0].bar([n[-3:] for n in names],[summary['single_cap'][n]/2 for n in names],color='#6a8299')
    axes[0].set(title='Frozen single CAPs',ylabel='Success (%)',ylim=(0,100))
    for i,arm in enumerate(ARMS):
        for offset,stage,color in [(-.18,'offline','#98b9d5'),(.18,'online','#2778ad')]:
            ys=np.array(summary['groups'][arm][stage]['seed_successes'])/2
            axes[1].bar(i+offset,ys.mean(),width=.32,color=color,label=stage if i==0 else None)
            axes[1].scatter(np.full(3,i+offset),ys,s=20,color='black',zorder=3)
    axes[1].axhline(max(summary['single_cap'].values())/2,color='#c34c39',linestyle='--',label='test-best CAP')
    axes[1].set(xticks=range(3),xticklabels=ARMS,ylabel='Success (%)',ylim=(0,100),title='Paired 200 scenes; dots = training seeds')
    axes[1].legend(fontsize=8)
    fig.savefig(directory/'success_comparison.png',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),sharey=True,constrained_layout=True)
    for ax,arm in zip(axes,ARMS):
        for seed in SEEDS:
            curve=summary['monitoring'][f'{arm}_s{seed}']
            ax.plot([0,100,200],[100*curve[p]/32 for p in ('offline','online100','online200')],
                    marker='o',label=f'seed {seed}')
        ax.set(title=arm,xlabel='Online training episodes',xticks=[0,100,200],ylim=(0,100))
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Success on fixed 32 development scenes (%)')
    axes[-1].legend(fontsize=8)
    fig.savefig(directory/'learning_curves.png',dpi=180)
    plt.close(fig)


def replays(out, table, scores, case_ids):
    name='GLOBAL_s17_online';baseline='CAP-RR-009'
    selected=[]
    for kind,predicate in [('recovered',lambda c:table[name,c]['success'] and not table[baseline,c]['success']),
                           ('lost',lambda c:not table[name,c]['success'] and table[baseline,c]['success'])]:
        selected += [(kind,c) for c in case_ids if predicate(c)][:2]
    failures=sorted((c for c in case_ids if not table[name,c]['success']),key=lambda c:-table[name,c]['initial_value'])
    selected += [('high_value_failure',c) for c in failures if c not in {c for _,c in selected}][:2]
    data=[]
    for kind,case in selected:
        pair=[]
        for condition in (baseline,name):
            result=unpack(out/'evaluation'/condition/case/'result.json.gz')
            replay=result['replay']
            replay['frames']=replay['frames'][::4]+[replay['frames'][-1]]
            decisions=[]
            for d in result['decisions']:
                supporters=([CAPS[i] for i in d['sources'][d['chosen']]] if 'sources' in d else [condition])
                decisions.append(dict(tick=d['event']['start_tick'],action=d['action'],
                                      score=d.get('chosen_value'),supporting_caps=supporters))
            pair.append(dict(condition=condition,summary=result['summary'],replay=replay,
                             decisions=decisions))
        prefix=0
        for a,b in zip(pair[0]['decisions'],pair[1]['decisions']):
            if a['action']!=b['action']:break
            prefix+=1
        data.append(dict(kind=kind,case_id=case,common_action_prefix=prefix,pair=pair))
    save(out/'report/replay_cases.json',[{k:v for k,v in r.items() if k!='pair'} for r in data])
    template='''<!doctype html><meta charset="utf-8"><title>CAP trace + RL paired replays</title>
<style>body{font:15px system-ui;margin:24px;background:#f5f7fa;color:#172b3a}canvas{background:white;border:1px solid #ccd4de;max-width:47vw}.row{display:flex;gap:20px}select,button,input{margin:8px}pre{white-space:pre-wrap;width:600px}</style>
<h2>CAP009 与 GLOBAL seed17 在线策略：配对回放</h2><p>仅供人工诊断的物理真值可视化，不是模型输入。成功恢复、能力丢失及高估失败样例。</p>
<select id="cases"></select><button id="play">播放 / 暂停</button><input id="time" type="range" min="0" max="1800" value="0" style="width:45vw"><span id="clock"></span>
<div class="row"><div><canvas id="a" width="600" height="480"></canvas><pre id="ta"></pre></div><div><canvas id="b" width="600" height="480"></canvas><pre id="tb"></pre></div></div>
<script>const DATA=__DATA__;let current=0,playing=false,tick=0,last=0;
const sel=document.getElementById('cases'),slider=document.getElementById('time');DATA.forEach((r,i)=>{let o=document.createElement('option');o.value=i;o.textContent=r.kind+' · '+r.case_id;sel.appendChild(o)});
sel.onchange=()=>{current=+sel.value;tick=0;draw()};slider.oninput=()=>{tick=+slider.value;draw()};document.getElementById('play').onclick=()=>playing=!playing;
function box(c,x,y,w,h,a,color){c.save();c.translate(x,y);c.rotate(a);c.fillStyle=color;c.fillRect(-w/2,-h/2,w,h);c.strokeStyle='#334';c.lineWidth=.015;c.strokeRect(-w/2,-h/2,w,h);c.restore()}
function panel(id,text,item){let c=document.getElementById(id).getContext('2d'),s=item.replay.scene,frames=item.replay.frames,f=frames[0];for(let q of frames){if(q.tick<=tick)f=q;else break}c.setTransform(1,0,0,1,0,0);c.clearRect(0,0,600,480);c.setTransform(100,0,0,-100,0,480);for(let o of s.obstacles)box(c,o.cx,o.cy,2*o.half_w,2*o.half_h,o.angle,'#969da5');c.beginPath();c.arc(s.goal.x,s.goal.y,s.goal.radius,0,7);c.fillStyle='#b8e9bd';c.fill();for(let [id,e] of Object.entries(f.entities)){let o=s.movable_objects.find(q=>q.label===id);box(c,e[0],e[1],o?o.width:.25,o?o.height:1,e[2],o?'#b7c6d4':'#efb44d')}f.robots.forEach((r,j)=>{c.beginPath();c.arc(r[0],r[1],s.robot_radius,0,7);c.fillStyle=j?'#dc6b72':'#3a94bc';c.fill();c.beginPath();c.moveTo(r[0],r[1]);c.lineTo(r[0]+.25*Math.cos(r[2]),r[1]+.25*Math.sin(r[2]));c.strokeStyle='#223';c.lineWidth=.025;c.stroke()});let d=item.decisions[0];for(let q of item.decisions){if(q.tick<=tick)d=q;else break}document.getElementById(text).textContent=item.condition+' · '+(item.summary.success?'成功':'未成功')+' · '+item.summary.ticks+' ticks\n'+JSON.stringify(d)}
function draw(){if(!DATA.length)return;slider.value=tick;document.getElementById('clock').textContent=(tick*.05).toFixed(1)+' s';panel('a','ta',DATA[current].pair[0]);panel('b','tb',DATA[current].pair[1])}
function animate(now){if(playing&&last){tick=Math.min(1800,tick+(now-last)*.16);if(tick>=1800)playing=false;draw()}last=now;requestAnimationFrame(animate)}draw();requestAnimationFrame(animate);</script>'''
    (out/'report/replays.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    result=report(args.out)
    print(json.dumps(dict(single_cap=result['single_cap'],groups=result['groups'],comparisons=result['comparisons']),ensure_ascii=False,indent=2))
