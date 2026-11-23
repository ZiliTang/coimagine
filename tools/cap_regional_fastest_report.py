"""Compare train-label fit, rollout deviations and coverage on the frozen ten-CAP training rooms."""
import csv
import gzip
import json

import numpy as np
import torch

from tools import cap_regional_fastest_actor as ex
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import load_checkpoint


def trajectory(directory, cid):
    with gzip.open(directory/(cid+'.json.gz'),'rt') as f:
        return json.load(f)


def initial_q_targets(teachers):
    a = ex.Store(ex.BASE/'arrays/train').arrays
    targets = np.load(ex.BASE/'models/GA_s17/Q_init_targets.npy',mmap_mode='r')
    starts = np.flatnonzero(a['boundary']==0)
    sids = ex.study.read(ex.study.OUT/'frozen.json')['sids']
    rows = {}
    for cid,t in teachers.items():
        indices = starts[a['root'][starts]==t['root']]
        indices = indices[np.argsort(a['cap'][indices])]
        assert len(indices)==len(sids)
        chosen = a['chosen'][indices]
        options = np.unique(chosen)
        assert np.array_equal(options,np.flatnonzero(a['valid'][indices[0]]))
        labels = {}
        for option in options:
            observed = targets[indices[chosen==option]]
            np.testing.assert_allclose(observed,observed[0],atol=1e-5)
            labels[option] = float(observed.mean())
        best = max(labels.values())
        winners = [j for j in labels if best-labels[j] <= 1e-5]
        teacher = chosen[sids.index(t['sid'])]
        q = trajectory(ex.BASE/'evaluation'/cid.split('_')[0],cid)[0]['chosen']
        rows[cid] = dict(initial_options=len(options), q_target_teacher_gap=best-labels[teacher],
                         teacher_max_target=int(teacher in winners), directq_max_target=int(q in winners),
                         init_teacher_q_agree=int(teacher==q), max_target_ties=len(winners))
    return rows


def known_successful_prefix(cid, sids, chosen_prefix):
    """A recorded complete success has the same public-action prefix through the departure."""
    for sid in sids:
        shard = ex.study.read(ex.BASE/'shards'/sid/(cid+'.json'))
        if shard['success']:
            trajectory_steps = trajectory(ex.study.ext.refs(sid,cid),cid)
            if [r['a'] for r in trajectory_steps[:len(chosen_prefix)]] == chosen_prefix:
                return True
    return False


def stats(rows):
    union = [r for r in rows if r['union']]
    summary = dict(n=len(rows),union=len(union))
    for key in ['directq','actor']:
        summary[key] = dict(overall=sum(r[key] for r in rows),retained=sum(r[key] for r in union),
              outside=sum(r[key] for r in rows if not r['union']),
              exact=sum(r[key+'_first_departure'] is None for r in union),
              after_departure=sum(r[key] for r in union if r[key+'_first_departure'] is not None),
              matched=sum(r[key+'_matched'] for r in union),
              informative_matched=sum(r[key+'_informative_matched'] for r in union))
    departed = [r for r in union if r['actor_first_departure'] is not None]
    known = [r for r in departed if r['actor_alternate_success_prefix']]
    summary.update(departed=len(departed), known_alternative=len(known),
        known_alternative_solved=sum(r['actor'] for r in known), lookup=sum(r['lookup'] for r in union), singleton=sum(r['ncap']==1 for r in union),
        singleton_actor=sum(r['actor'] for r in union if r['ncap']==1),
        singleton_directq=sum(r['directq'] for r in union if r['ncap']==1))
    return summary


def report():
    fixtures,teachers,data = ex.load_data()
    actor = ex.actor()
    actor.load_state_dict(torch.load(ex.OUT/'actor.pt',map_location='cpu',weights_only=True))
    actor.eval()
    fit,pred = ex.assess(actor,data)
    assert np.array_equal(pred,np.load(ex.OUT/'predictions.npy'))
    q,_ = load_checkpoint(ex.BASE/'models/GA_s17/offline.pt')
    qfit,qpred = ex.assess(q,data)
    labels = initial_q_targets(teachers)
    sids = ex.study.read(ex.study.OUT/'frozen.json')['sids']
    polish = ex.study.read(ex.OUT/'polish.json')
    polished_model = ex.actor()
    polished_model.load_state_dict(torch.load(ex.OUT/('actor.pt' if polish.get('skipped') else 'polish.pt'),map_location='cpu',weights_only=True))
    polished_fit,_ = ex.assess(polished_model.eval(),data)
    records=[]
    for cid in sorted(fixtures):
        panel = cid.split('_')[0]
        ref = ex.study.read(ex.BASE/'evaluation'/panel/(cid+'.json'))
        new = ex.study.read(ex.OUT/'actor/evaluation'/panel/(cid+'.json'))
        polish_dir = ex.OUT/('actor' if polish.get('skipped') else 'polish')/'evaluation'/panel
        polished = ex.study.read(polish_dir/(cid+'.json'))
        t = teachers.get(cid)
        row = dict(panel=panel,case_id=cid,union=int(t is not None),ncap=0 if not t else t['ncap'],
                   teacher=None if not t else t['sid'], directq=int(ref['success']), actor=int(new['success']), polished=int(polished['success']), polished_exact=None,
                   lookup=None,teacher_ticks=None,teacher_boundaries=None, informative=None)
        if t:
            expert = trajectory(ex.study.ext.refs(t['sid'],cid),cid)
            expected = [e['a'] for e in expert]
            ix = np.flatnonzero(data['root']==t['root'])
            assert len(ix)==len(expected)==t['boundaries']
            active = data['valid'][ix].sum(1)>1
            row.update(teacher_ticks=t['ticks'],teacher_boundaries=len(ix),informative=int(active.sum()),**labels[cid])
            lookup = ex.study.read(ex.OUT/'lookup/evaluation'/panel/(cid+'.json'))
            assert lookup['success'] and lookup['history']==t['history'] and lookup['ticks']==t['ticks']
            row['lookup'] = 1
            row['polished_exact'] = int(polished['history']==t['history'])
            if row['polished_exact']:
                assert polished['success'] and polished['ticks']==t['ticks']
            for key, directory, prediction, closed in [('actor',ex.OUT/'actor',pred,new),('directq',ex.BASE,qpred,ref)]:
                executed = trajectory(directory/'evaluation'/panel,cid)
                departure = next((i for i,(action,step) in enumerate(zip(expected,executed)) if action!=step['action']),None)
                if departure is None:
                    assert len(expected)==len(executed) and closed['history']==t['history'] and closed['success']
                if key == 'actor' and departure is not None:
                    row['actor_departure_tick'] = 0 if departure==0 else expert[departure-1]['tick']
                    row['teacher_departure_action'] = json.dumps(expected[departure],sort_keys=True)
                    row['actor_departure_action'] = json.dumps(executed[departure]['action'],sort_keys=True)
                    prefix = [step['action'] for step in executed[:departure+1]]
                    row['actor_alternate_success_prefix'] = int(known_successful_prefix(cid,sids,prefix))
                hit = prediction[ix]==data['chosen'][ix]
                mismatches = np.flatnonzero(~hit)
                open_first = int(mismatches[0]) if len(mismatches) else None
                row.update({key+'_first_departure':departure,key+'_teacher_forced_first_error':open_first,
                    key+'_matched':int(hit.sum()),key+'_informative_matched':int(hit[active].sum())})
        records.append(row)
    total = stats(records)
    polished_total = dict(overall=sum(r['polished'] for r in records),
        retained=sum(r['polished'] for r in records if r['union']),
        exact=sum(r['polished_exact'] for r in records if r['union']),
        singleton=sum(r['polished'] for r in records if r['ncap']==1), fit=polished_fit)
    groups = {p:stats([r for r in records if r['panel']==p]) for p in ('dev','aug')}
    initial = [r for r in records if r['union'] and r['initial_options']>1]
    target = dict(n=len(initial),teacher_optimal=sum(r['teacher_max_target'] for r in initial),
            q_optimal=sum(r['directq_max_target'] for r in initial),q_teacher=sum(r['init_teacher_q_agree'] for r in initial),
            tied=sum(r['max_target_ties']>1 for r in initial))
    output = ex.DOCS
    output.mkdir(parents=True,exist_ok=True)
    with (output/'per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records for k in r)))
        writer.writeheader()
        writer.writerows(records)
    history = ex.study.read(ex.OUT/'fit.json')
    summary = dict(polish_protocol=polish, polished=polished_total,panels=groups,train=total, actor_history=history, q_teacher_fit={k:v for k,v in qfit.items() if k!='ce'},
                   lookup=ex.study.read(ex.OUT/'lookup.json'),initial_targets=target,seed=17,learning_rate=3e-4,batch=64,
                   steps=ex.STEPS,teacher_tie='CAP ID ascending',sampling='uniform rooms, then uniform boundary',
                   actor_sha256=ex.study.sha(ex.OUT/'actor.pt'), sources={p.name:ex.study.sha(p) for p in
                       [ex.Path(ex.__file__),ex.Path(__file__),ex.Path(__file__).with_name('cap_regional_fastest_polish.py')]}, cap_ids=sids)
    ex.study.save(output/'metrics.json',summary)
    lines=['# 最快成功教师的训练集拟合与闭环', '',
           '冻结区域十 CAP 和 215 个训练初态，包括 160 个原开发房间和 55 个开发扰动。逐房间选成功运行中实际执行 ticks 最少的完整轨迹，同 ticks 按 CAP ID 升序。161 个房间有成功教师，54 个没有。最快指这十条已执行轨迹之间的比较。', '',
           '| 训练范围 | 房间 | 专家并集 | 原 directQ 保留并集 | 最快教师 actor 保留并集 | 公开输入查表保留并集 | actor 并集外新增 |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in [('原开发',groups['dev']),('开发扰动',groups['aug']),('合计',total)]:
        lines.append(f"| {name} | {m['n']} | {m['union']} | {m['directq']['retained']} | {m['actor']['retained']} | {m['lookup']} | {m['actor']['outside']} |")
    a=total['actor']
    lines += ['',f"原 directQ 全池成功 {total['directq']['overall']}/215，actor 成功 {a['overall']}/215。16 个独胜房间的闭环成功分别为 {total['singleton_directq']} 和 {total['singleton_actor']}。",'',
        '## 在已知成功路径上，动作能否学对', '',
        '无 trace GA 使用与当前 directQ 相同的公开几何历史、十个 CAP 的提案、来源与训练归一化。64×64 actor 从头拟合，对去重合法候选做交叉熵。沿用单教师 actor 的训练预算与抽样方式，seed 17，Adam 0.0003，每批 64，先均匀抽场景再抽该示范的一步。固定使用 24,000 步最终模型。', '',
        '| 更新 | 全部边界命中 / 6,150 | 多候选边界命中 / 3,890 | 全轨迹逐步全对 / 161 |','|---:|---:|---:|---:|']
    for h in history:
        lines.append(f"| {h['step']} | {h['matched']} | {h['informative_matched']} | {h['perfect']} |")
    lines += ['',f"真闭环中，actor 完整复现最快教师 {a['exact']} 条；发生偏离后仍成功 {a['after_departure']} 条。原 directQ 的对应数量为 {total['directq']['exact']} 和 {total['directq']['after_departure']}。首次偏离位置见逐例表，从 0 开始计数。",'',
        f"actor 发生首次偏离的 {total['departed']} 个训练房间中，有 {total['known_alternative']} 个能找到另一条成功专家轨迹，与 actor 直到偏离动作在内的全部动作一致。这些房间中 actor 最终成功 {total['known_alternative_solved']} 个。其余偏离动作的后缀可能性仍不确定。", '',
        '查表单独验证信息和执行链。按归一化、裁剪和缺失掩码后的公开 GA 输入、候选及来源匹配教师动作，6,150 个键互异，冲突为零。在全部 161 个有教师的初态从头运行，逐条核对终局历史哈希、耗时和成功，均与最快教师一致。查表只覆盖示范中出现过的输入。','',
        f"对 {target['n']} 个开局存在多个动作的有解训练房间，十个专家都从同一起点真实执行过，故可直接比较各动作原有的 Q 训练标签。最快教师动作属于最高标签的有 {target['teacher_optimal']} 个，directQ 实际首选属于最高标签的有 {target['q_optimal']} 个，二者首选动作一致的有 {target['q_teacher']} 个。标签相差不超过 0.00001 计并列，其中 {target['tied']} 个房间最高分并列。这里检验原目标与模仿目标是否一致，尚不能据此把整个闭环差额归因于 V。",'',
        '## 训练集能否完全拟合', '',
        f"固定预算之后的小学习率对照采用相同网络、数据、分层抽样。若 24,000 步仍有误差，则以其权重为起点重新初始化 Adam，学习率 0.00003，每 500 步检查一次，至多追加 12,000 步，只按训练集选择。选定检查点追加 {polish.get('selected_extra',0):,} 步，教师边界命中 {polished_fit['matched']}/6,150，多候选边界命中 {polished_fit['informative_matched']}/3,890。闭环保留并集 {polished_total['retained']}/161，复现相同终局历史 {polished_total['exact']}/161，独胜房间成功 {polished_total['singleton']}/16，全训练池成功 {polished_total['overall']}/215。", '',
        '本次只评估训练集。训练集上的可记忆性、给定神经网络与预算的拟合程度、对新初态的泛化是三个不同问题。','',
        '[逐场景表现与首次偏离](per_scene.csv) · [拟合曲线及配对统计](metrics.json)']
    failed = [r for r in records if r['union'] and not r['actor']]
    if len(failed) == 1:
        r = failed[0]
        lines[-1:-1] = [f"固定预算唯一未保留的房间为 {r['case_id']}，成功教师 {r['teacher']}。前 {r['actor_first_departure']} 步与最快教师相同，在 tick {r['actor_departure_tick']} 首次偏离。教师动作为 `{r['teacher_departure_action']}`，actor 动作为 `{r['actor_departure_action']}`。", '']
    (output/'README.md').write_text('\n'.join(lines)+'\n')
    ex.study.save(ex.OUT/'FINISHED.json',dict(report=str(output),train=total))


if __name__=='__main__':
    torch.set_num_threads(1)
    report()
