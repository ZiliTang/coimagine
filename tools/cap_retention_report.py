"""Paired scene-level retention, coverage and commitment diagnostics."""
from pathlib import Path
import shutil
import numpy as np
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS
from tools.cap_trace_value_long_report import write_csv


def report(out):
    from experiments.planar_tasks.authoring_v2.cap_retention_supervised import SEEDS,ARMS,MODES
    panel=read(out/'panel.json')['cases']
    ids=[c['case_id'] for c in panel]
    tables={}
    for file in (out/'evaluation').glob('*/results.json'):
        rows=read(file)
        table={r['case_id']:r for r in rows}
        assert len(rows)==200 and set(table)==set(ids) and all(r['complete'] for r in rows)
        tables[file.parent.name]=[table[i] for i in ids]
    assert len(tables)==30
    vectors={k:np.array([int(r['success']) for r in rows]) for k,rows in tables.items()}
    fixed={k:v for k,v in vectors.items() if k.startswith('CAP-RR')}
    best=min(fixed,key=lambda k:(-fixed[k].sum(),k))
    ncap=np.stack([fixed[k] for k in CAPS]).sum(0)
    union=ncap>0
    rng=np.random.default_rng(20261119)
    draws=np.concatenate([rng.choice([i for i,c in enumerate(panel) if c['movable_count']==n],(10000,50))
        for n in range(2,6)],axis=1)
    def compare(a,b,mask):
        delta=(a-b).mean(0)
        keep=mask[draws]
        estimates=(delta[draws]*keep).sum(1)/keep.sum(1)
        return dict(pp=100*float(delta[mask].mean()),ci95_pp=(100*np.quantile(estimates,[.025,.975])).tolist(),
            gains=((a[:,mask]>b[:,mask]).sum(1)).tolist(),losses=((a[:,mask]<b[:,mask]).sum(1)).tolist(),
            denominator=int(mask.sum()),per_seed_pp=(100*(a-b)[:,mask].mean(1)).tolist())
    per_scene=[dict(case_id=c['case_id'],movable_count=c['movable_count'],union7_count=int(ncap[i]),
                    **{k:int(v[i]) for k,v in vectors.items()}) for i,c in enumerate(panel)]
    write_csv(out/'per_scene.csv',per_scene)
    metrics=[];detail=[];groups={}
    for arm in (*ARMS,'PPO'):
        for mode in MODES if arm!='PPO' else ('greedy',):
            name=f'{arm}/{mode}'
            matrices=[]
            for seed in SEEDS:
                key=f'{arm}_s{seed}_{mode}' if arm!='PPO' else f'PPO_s{seed}'
                v=vectors[key];rows=tables[key];matrices.append(v)
                m=dict(arm=arm,mode=mode,seed=seed,total=int(v.sum()),retained7=int(v[union].sum()),
                    missed7=int((1-v[union]).sum()),outside7=int(v[~union].sum()),retain009=int(v[fixed[CAPS[0]]==1].sum()),
                    retain_panel_best=int(v[fixed[best]==1].sum()),retain_unique=int(v[ncap==1].sum()),
                    retain_multi=int(v[ncap>=2].sum()))
                if arm!='PPO':
                    once=vectors[f'{arm}_s{seed}_once']
                    init=[r['first_cap'] for r in rows]
                    assert all(init[i]==tables[f'{arm}_s{seed}_once'][i]['first_cap'] for i in range(200))
                    aligned=np.array([fixed[k][i] if k else -1 for i,k in enumerate(init)])
                    m.update(rescued_vs_once=int(((once==0)&(v==1)).sum()),lost_vs_once=int(((once==1)&(v==0)).sum()),
                        any_switch=sum(r['switches']>0 for r in rows),switches=sum(r['switches'] for r in rows),
                        forced=sum(r['forced'] for r in rows),once_disagreement_with_solo=int((aligned!=once).sum()))
                    for i,r in enumerate(rows):
                        detail.append(dict(arm=arm,seed=seed,mode=mode,case_id=ids[i],first_cap=init[i],
                            solo_success=int(aligned[i]),once=int(once[i]),success=int(v[i]),switches=r['switches'],
                            first_switch=r['first_switch'],forced=r['forced'],decisions=r['decisions']))
                metrics.append(m)
            groups[name]=np.stack(matrices)
    comparisons={}
    all_cases=np.ones(200,bool)
    def add(a,b):
        comparisons[a+' minus '+b]={m:compare(groups[a],groups[b],mask) for m,mask in
            [('union7',union),('all',all_cases),('cap009',fixed[CAPS[0]]==1)]}
    for mode in MODES:
        for a,b in [('paired','boundary'),('root','boundary'),('paired','root')]:add(a+'/'+mode,b+'/'+mode)
    for arm in ARMS:
        for a,b in [('free','once'),('guard','once'),('guard','free')]:add(arm+'/'+a,arm+'/'+b)
    write_csv(out/'metrics.csv',metrics)
    write_csv(out/'routing.csv',detail)
    cover=dict(union7=int(union.sum()),union10=int(np.stack(list(fixed.values())).max(0).sum()),unique=int((ncap==1).sum()),
               multi=int((ncap>=2).sum()),single={k:int(v.sum()) for k,v in fixed.items()},panel_best=best)
    save(out/'summary.json',dict(coverage=cover,metrics=metrics,comparisons=comparisons))
    primary=comparisons['paired/free minus boundary/free']['union7']
    lines=['# 能力保留：监督样本选择与切换对照','',
        f'新面板200个初态，七CAP并集 {cover["union7"]}，仅一个候选CAP能成功的场景 {cover["unique"]}。开发训练集最强CAP009成功 {cover["single"][CAPS[0]]}/200；新面板十CAP中最高 {best} 成功 {cover["single"][best]}/200。','',
        f'预定主比较（自由重选，配对样本相对边界均匀）在七CAP并集内的保留率差 {primary["pp"]:+.2f} 个百分点，按场景配对的95%区间 {primary["ci95_pp"]}。','',
        '|训练样本|执行规则|总成功 17/29|并集内保留 17/29|并集外成功 17/29|保留CAP009 17/29|相对选一次救回 17/29|相对选一次丢失 17/29|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for name in groups:
        arm,mode=name.split('/')
        selected=[m for m in metrics if m['arm']==arm and m['mode']==mode]
        fields=['total','retained7','outside7','retain009','rescued_vs_once','lost_vs_once']
        lines.append('|'+arm+'|'+mode+'|'+'|'.join(' / '.join(str(r[k]) for r in selected) if k in selected[0] else '—' for k in fields)+'|')
    validation=read(out/'validation_metrics.json')
    lines+=['','## 开发验证','',
        '|样本|seed|初态选定CAP成功/152|成败分歧前缀的排序正确率（按根等权）|','|---|---:|---:|---:|']
    for r in validation:
        lines.append(f'|{r["arm"]}|{r["seed"]}|{r["initial_success"]}|{100*r["mixed_prefix_root_equal_ranking"]:.2f}%|')
    inv=read(out/'training_inventory.json');val=read(out/'validation_inventory.json')
    lines+=['',f'608训练初态的4256条轨迹共有{inv["boundaries"]}个决策。发现并验证可成对监督的共同前缀 {inv["mixed_prefixes"]} 个，来自 {inv["eligible_roots"]} 个初态；其中非初态 {inv["noninitial_prefixes"]} 个。开发验证中来自 {val["eligible_roots"]} 根的 {val["mixed_prefixes"]} 个此类前缀。配对表示当前选择不同、分别持续执行各自CAP的已知成败，不把差异归因于孤立首动作。','',
        '三种模型共享无trace输入、网络、初始权重（seed内）、优化器和固定6000次监督更新。boundary均匀抽决策边界；root均匀抽初态、CAP、轨迹边界；paired一半按root、一半在共同前缀抽不同动作的一成一败两条监督记录。训练权重变化会影响分数校准。两种子固定，区间只重抽200个新场景。PPO是历史冻结GA策略的外部参照，训练方法及预算不同。','',
        'once在开局选择CAP后持续执行；free每步选择最高分的合法CAP，同动作维持原身份；guard需要同一挑战者连续两次领先至少0.15且提出不同动作。非法建议有紧急回退，次数见metrics.csv。门槛预先固定，不能解读为已校准的成功率差。分层、小组和其他切换比较均为探索性。','',
        '[逐场景成败](per_scene.csv) · [各seed指标](metrics.csv) · [逐场景切换记录](routing.csv) · [配对区间](summary.json) · [训练与评估协议](protocol.json)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    assets=Path('docs/agent_exchange/assets')/out.name
    assets.mkdir(exist_ok=True)
    for name in ('REPORT.md','per_scene.csv','metrics.csv','routing.csv','summary.json','protocol.json','FROZEN.json',
                 'training_inventory.json','validation_inventory.json','validation_metrics.json'):
        shutil.copyfile(out/name,assets/name)
    save(out/'FINISHED.json',dict(complete=True,physical_executions=200*len(tables),fits=6,updates_per_fit=6000,primary=primary))
