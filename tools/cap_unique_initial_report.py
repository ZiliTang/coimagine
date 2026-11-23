"""Summarize paired V estimates and closed-loop GA-Q outcomes on held rooms."""
import csv
import json
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2 import cap_unique_initial_weight as study


def paired(base, weighted, strata, seed):
    rng=np.random.default_rng(seed)
    delta=weighted-base
    draws=np.zeros(20000)
    for s in np.unique(strata):
        d=delta[strata==s]
        draws+=d[rng.integers(len(d),size=(len(draws),len(d)))].sum(axis=1)
    bounds=(100*np.quantile(draws,[.025,.975])/len(delta)).tolist()
    return dict(n=len(delta),baseline=int(base.sum()),weighted=int(weighted.sum()),
                gained=int((delta==1).sum()),lost=int((delta==-1).sum()),
                pp=float(100*delta.mean()),ci=bounds)


def metrics(v,y,legal):
    choice=np.where(legal,v,-1.).argmax(1)
    index=np.arange(len(v))
    sole=(y.sum(1)==1)
    rare=sole&(y[:,0]==0)
    main=sole&(y[:,0]==1)
    winner=y.argmax(1)
    return dict(rooms=len(v),legal=int(legal.sum()),picks=np.bincount(choice,minlength=7).tolist(),
                initial_success=int(y[index,choice].sum()),rare_hits=int((choice[rare]==winner[rare]).sum()),
                rare=int(rare.sum()),main_hits=int((choice[main]==0).sum()),main=int(main.sum()),
                rare_margin=float((v[index,winner]-v[:,0])[rare].mean()),
                brier=float(((v-y)**2)[legal].mean()))


def report():
    out=study.OUT
    docs=Path('docs/research/key_findings/2026-11-19_cap_unique_initial_weight')
    docs.mkdir(exist_ok=True,parents=True)
    panel=study.read(study.PANEL)
    with (study.scale.OUT/'per_scene.csv').open() as f:
        old={r['case_id']:r for r in csv.DictReader(f)}
    raw=[study.read(out/'cases'/(c['case_id']+'.json')) for c in panel]
    assert all(r['case_id']==c['case_id'] for r,c in zip(raw,panel,strict=True))
    legal=np.array([r['legal'] for r in raw],bool)
    assert legal.any(axis=1).all()
    base=np.array([int(old[r['case_id']]['q1001']) for r in raw])
    treat=np.array([r['success'] for r in raw])
    y=np.array([[int(old[r['case_id']][c]) for c in study.CAPS] for r in raw])
    scores={k:np.array([r[k] for r in raw]) for k in ['base_v','v']}
    assert set(np.unique(y)).issubset({0,1}) and all(np.isfinite(v).all() for v in scores.values())
    number=y.sum(1)
    shape=np.array([c['movable_count'] for c in panel])
    subsets=dict(all=np.ones(len(y),bool),unique_other=(number==1)&(y[:,0]==0),unique_009=(number==1)&(y[:,0]==1),
                 multiple=number>1,outside=number==0)
    closed={k:paired(base[m],treat[m],shape[m],20261119) for k,m in subsets.items()}
    rank={k:metrics(v,y,legal) for k,v in scores.items()}
    assert rank['base_v']['picks']==[600,0,0,0,0,0,0] and closed['all']['baseline']==323
    fit={}
    for name,folder in [('base',study.prior.BASE),('weighted',study.MODELS)]:
        fit[name]={}
        for split,data in study.read(folder/'V_metrics.json').items():
            entries=data['initial']
            v=np.array([r['scores'] for r in entries]); obs=np.array([r['outcomes'] for r in entries])
            fit[name][split]=metrics(v,obs,np.ones_like(obs,bool))
    assert rank['v']==study.read(out/'held_initial_ranking.json')
    result=dict(ranking=rank,closed=closed,fit=fit,targets=study.read(out/'target_shift.json'),
                singleton_scores=study.read(out/'train_singleton_scores.json'),input_check=study.read(out/'root_input_check.json'),baseline_check=study.read(out/'baseline_replay/audit.json'))
    study.save(docs/'summary.json',result)
    with (docs/'per_scene.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['case_id','movable_count','ncap','q_base','q_weighted',*study.CAPS,
                    *['base_v_'+c[-3:] for c in study.CAPS],*['v_'+c[-3:] for c in study.CAPS],*['legal_'+c[-3:] for c in study.CAPS]])
        for i,r in enumerate(raw):
            w.writerow([r['case_id'],shape[i],number[i],base[i],treat[i],*y[i],*scores['base_v'][i],*scores['v'][i],*legal[i]])
    for f in ['protocol.json','sampling.json','frozen.json']:
        study.save(docs/f,study.read(out/f))
    labels={'all':'全部','unique_other':'非 009 独胜','unique_009':'009 独胜','multiple':'多专家成功','outside':'七家均失败'}
    lines=['# 独胜初态七路成组加权对照','',
           '同一最高增广剂量、无 trace 的 GA-Q 和训练 seed17。使用原冻结基线，与加权重训的 V 和 Q 在已查看的同一 600 房间逐例配对。主指标为 72 个非 009 独胜房间中的贪心 Q 成功数。','',
           '| 房间 | 分母 | 原 GA-Q | 加权 GA-Q | 救回 / 丢失 | 成功率差及 95% 区间（百分点） |',
           '|---|---:|---:|---:|---:|---:|']
    for key,v in closed.items():
        lo,hi=v['ci']
        lines.append(f"| {labels[key]} | {v['n']} | {v['baseline']} | {v['weighted']} | {v['gained']} / {v['lost']} | {v['pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    vb,vw=rank['base_v'],rank['v']
    lines.extend(['','### 初态排序与后继评分','',
        f"原 V 在 600 个房间全部首选 009，加权后在 {vw['picks'][0]} 个房间首选 009。两种 V 在 72 个非 009 独胜房间均未选中正确专家，在 20 个 009 独胜房间均选对。按初态选一次并交给所选专家执行到底，两者各成功 {vw['initial_success']}/600。七路初态 Brier 为 {vb['brier']:.4f} → {vw['brier']:.4f}，非 009 独胜时真实赢家减去 009 的平均 V 差为 {vb['rare_margin']:.4f} → {vw['rare_margin']:.4f}。",'',
        '训练集中全部 230 个独胜房间通过七条起始记录的 GA 可见输入逐元素一致性检查。其中 178 个由非 009 独胜，两个 V 在这些训练初态都选对 0 次，真实赢家的名次中位数均为第四。单看被加权的 230 组，七路 Brier 从 0.1395 降到 0.1197，赢家平均预测从 0.198 降到 0.169，六个败者的均值从 0.191 降到 0.149。损失的下降与成功排序的改善可以分离。','',
        '从 565,458 条非终止训练转移中无放回抽 16,384 条，在实际后继重算最高合法专家。原模型由 009 取最大值 16,372 次，加权模型为 16,330 次。全部非终止训练标签均值从 0.1803 升至 0.2219，两套标签的平均绝对差 0.0696。即使最高专家身份大多相同，Q 收到的监督数值仍有变化。','',
        '### 采样、复现和适用范围','',
        '每批固定 64 条，处理组保留原随机流的前 50 个普通边界，将最后 14 条换为两个均匀、有放回抽取的独胜训练初态，每个初态使用七专家的实际 0/1 结果。所有独胜专家一视同仁。共 1,154 个训练初态、573,536 个边界，230 个独胜初态包含 52 个 009 独胜。V 从头更新 16,000 次，共额外抽取 32,000 组；Q 按原规则从本组 V 初始化并用新 V 的后继最大值监督，均匀抽普通决策边界更新 8,000 次。奖励、归一化、网络和优化器沿用原定义。因 V 权重同时用于 Q 初始化与制标签，闭环差值包含两条路径的共同作用。','',
        '基线使用冻结权重与原有 600 个结果；从四个可动物体数量层各取一例重跑，成功、ticks、决策数和最终公开历史哈希均复现。区间用同一房间配对、在对应子集内按可动物体数量分层的 20,000 次 bootstrap；单训练种子，此前看过的面板供诊断。运行中没有为中途的不同专家添补反事实失败标签。','',
        '[逐场景七路 V 和贪心 Q 成败](per_scene.csv) · [指标、分组与后继评分交叉表](summary.json) · [固定采样配方](protocol.json) · [模型 SHA256](frozen.json)','',
        '训练与评估入口为 `experiments.planar_tasks.authoring_v2.cap_unique_initial_weight` 的 `fit` 和 `evaluate`。训练的 staging 目录使用原 train 的非 x 数组（original 子目录），另从 E 盘缓存档 inverse_parts.tar.zst 的 parts/ 下恢复各剂量子目录的非 x 数组及 packed/；已有逐行无损压缩的特征供读取。后继检查和汇总入口分别为 `tools.cap_unique_target_check` 和 `tools.cap_unique_initial_report`。'])
    (docs/'README.md').write_text('\n'.join(lines)+'\n')
    study.save(out/'FINISHED.json',dict(complete=True,rooms=len(panel),seed=17,summary_sha=study.prior.sha(docs/'summary.json')))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    report()
