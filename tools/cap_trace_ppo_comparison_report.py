"""Paired confirmation panel summary, treating the scene as the sampling unit."""
import shutil
from pathlib import Path
import numpy as np
from tools.cap_trace_value_long_report import write_csv
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save


def report(out,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    seeds=(17,29,43)
    ids=[c['case_id'] for c in cases]
    records=[dict(case_id=c['case_id'],content_sha256=c['content_sha256'],movable_count=c['movable_count']) for c in cases]
    results={}
    for arm in ('GA','GLOBAL'):
        for stage in ('pre','post'):
            rows=[]
            for seed in seeds:
                name=f'{arm}_s{seed}_{stage}'
                source=read(out/'confirmation'/name/'results.json')
                by_id={r['case_id']:r for r in source}
                assert len(source)==200 and set(by_id)==set(ids) and all(r['complete'] for r in source)
                values=[int(by_id[i]['success']) for i in ids]
                rows.append(values)
                for record,v in zip(records,values):record[name]=v
            results[arm,stage]=np.array(rows)
    write_csv(out/'per_scene.csv',records)
    rng=np.random.default_rng(20261118)
    indices=np.concatenate([rng.choice([i for i,c in enumerate(cases) if c['movable_count']==n],
        size=(10000,50),replace=True) for n in range(2,6)],axis=1)
    def compare(delta):
        avg=delta.mean(0)
        return dict(pp=100*float(avg.mean()),ci95=(100*np.quantile(avg[indices].mean(1),[.025,.975])).tolist(),
            per_seed_pp=(100*delta.mean(1)).tolist(),net_per_seed=delta.sum(1).tolist())
    pre=results['GLOBAL','pre']-results['GA','pre']
    post=results['GLOBAL','post']-results['GA','post']
    contrasts={'trace_post':post,'trace_pre':pre,'GA_ppo':results['GA','post']-results['GA','pre'],
        'GLOBAL_ppo':results['GLOBAL','post']-results['GLOBAL','pre'],'interaction':post-pre}
    summary={k:compare(v) for k,v in contrasts.items()}
    for k,v in contrasts.items():
        if k!='interaction':
            summary[k].update(gains=(v>0).sum(1).tolist(),losses=(v<0).sum(1).tolist())
    counts={f'{a}_{s}':v.sum(1).tolist() for (a,s),v in results.items()}
    save(out/'summary.json',dict(counts=counts,comparisons=summary,seeds=seeds,scenes=200))
    first=summary['trace_post']
    lo,hi=first['ci95']
    conclusion=('固定这三个种子，确认集支持提供trace能提高最终策略成功率。' if lo>0 else
        '固定这三个种子，确认集显示提供trace后最终成功率下降。' if hi<0 else
        '配对区间包含零，这次确认未能确定trace带来净收益。')
    lines=['# 程序 trace 对 PPO 选择器的贡献','',conclusion,'',
        '|策略|seed17|seed29|seed43|平均成功率|','|---|---:|---:|---:|---:|']
    for name,values in counts.items():
        lines.append('|'+name+'|'+'|'.join(str(n)+'/200' for n in values)+f'|{np.mean(values)/2:.2f}%|')
    lines+=['','|配对比较|差值（百分点）|95% 场景 bootstrap 区间|','|---|---:|---:|']
    labels=['PPO后 GLOBAL − GA（主比较）','PPO前 GLOBAL − GA','GA微调前后','GLOBAL微调前后','两组微调收益之差']
    for label,key in zip(labels,contrasts):
        d=summary[key]; a,b=d['ci95']
        lines.append(f'|{label}|{d["pp"]:+.2f}|[{a:+.2f}, {b:+.2f}]|')
    lines+=['','六条训练链共享200个在线根，seed内共享采样顺序和动作随机种子。每条链有1800次新增PPO采集并固定取最后模型，其中 GLOBAL seed17 复用已有完整训练。新执行9000次训练采集、2400次确认评估。两组分别从各自的200 episode Q检查点开始，主比较衡量从预训练到PPO的整条流程获得trace的效果。GA的actor和critic均屏蔽trace。每个模型共享同一批200个新场景，全部预定检查点先冻结，再评估。区间按物体数分层，配对重抽200个场景，条件于三个训练种子。','','[逐场景结果](per_scene.csv) · [统计与配对得失](summary.json) · [曲线](curves.png) · [冻结权重哈希](FROZEN.json)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for arm,color in [('GA','#d97706'),('GLOBAL','#2563eb')]:
        for i,seed in enumerate(seeds):
            directory=Path('outputs/semantic_rl/cap_trace_value_ppo_20261118_01') if (arm,seed)==('GLOBAL',17) else out/f'{arm}_s{seed}'
            history=read(directory/'history.json')
            assert len(history)==36
            axes[0].plot([200+j*200 for j in range(1,10)],[sum(r['success'] for r in history[j:j+4])/200
                for j in range(0,36,4)],color=color,alpha=.6,label=arm if i==0 else None)
            axes[1].plot([0,1],[results[arm,'pre'][i].mean(),results[arm,'post'][i].mean()],
                marker='o',color=color,label=arm if i==0 else None)
    axes[0].set(xlabel='Cumulative online episodes',ylabel='Training success per 200 episodes')
    axes[1].set(xticks=[0,1],xticklabels=['Before PPO','After PPO'],ylabel='Greedy success on 200 confirmation scenes')
    for ax in axes:ax.legend()
    fig.savefig(out/'curves.png',dpi=140);plt.close(fig)
    assets=Path('docs/agent_exchange/assets')/out.name
    assets.mkdir(exist_ok=True)
    for name in ('REPORT.md','summary.json','per_scene.csv','curves.png','protocol.json','FROZEN.json'):
        shutil.copyfile(out/name,assets/name)
    save(out/'FINISHED.json',dict(complete=True,training_new=9000,confirmation=2400,**first))
    cur=Path('docs/agent_exchange/CURRENT.md')
    cur.write_text(cur.read_text().replace('<!-- TRACE_PPO_STATUS -->运行中','<!-- TRACE_PPO_STATUS -->已完成'))
