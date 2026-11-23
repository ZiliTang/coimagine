"""Summarize the predeclared width-by-budget teacher-fitting experiment."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from experiments.planar_tasks.authoring_v2 import cap_actor_capacity as study
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save


def draw_curves(out, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(9,6),sharex=True,layout='constrained')
    for width,color in [(64,'#245a8a'),(128,'#c45a21')]:
        rows=[curves[width,seed] for seed in study.SEEDS]
        steps=[r['step'] for r in rows[0]]
        for col,part in enumerate(['train','validation']):
            for ax,key,scale in [(axes[0,col],'top1',100),(axes[1,col],'nll',1)]:
                y=np.array([[r[part][key] for r in s] for s in rows])*scale
                ax.fill_between(steps,y.min(0),y.max(0),color=color,alpha=.15)
                ax.plot(steps,y.mean(0),color=color,marker='.',label=f'{width} x {width}')
                ax.grid(alpha=.15)
    for ax,title in zip(axes[0],['Training roots','Held-out roots']):
        ax.set_title(title)
    axes[0,0].legend(frameon=False)
    axes[0,0].set_ylabel('Teacher top-1 (%)')
    axes[1,0].set_ylabel('Teacher NLL (nats)')
    for ax in axes[1]:
        ax.set_xlabel('Optimizer updates')
    fig.savefig(out/'fit_curves.png',dpi=160)
    plt.close(fig)


def diagnose(out, frozen, curves):
    rank_order=read(study.PRIOR/'protocol.json')['ranking']
    ranking=np.array([study.distill.CAPS.index(cap) for cap in rank_order])
    rows=[]
    for part in ('train','validation'):
        data=study.TeacherData(part)
        supporters=data.sources
        ranked=supporters.any(1)[:,ranking]
        assert ranked.any(1).all()
        highest=ranking[ranked.argmax(1)]
        teacher=data.labels
        base=supporters[np.arange(len(teacher)),:,highest].argmax(1)
        correct=base==teacher
        nodes=[np.flatnonzero(data.roots==r) for r in data.root_ids]
        for width in study.WIDTHS:
            for seed in study.SEEDS:
                model=study.make_actor(width).eval()
                for step in (6000,study.END):
                    path=out/f'w{width}_s{seed}_u{step}.pt'
                    assert sha(path)==frozen['files'][path.name]
                    model.load_state_dict(torch.load(path,weights_only=True))
                    pred=[]
                    rare_teacher=[]
                    with torch.inference_mode():
                        for start in range(0,len(teacher),256):
                            x,c,s,valid,labels=data.tensors(slice(start,start+256))
                            scores=model.values(x,c,s).masked_fill(~valid,-torch.inf)
                            pred.append(scores.argmax(1).numpy())
                            logp=scores.log_softmax(1).gather(1,labels[:,None]).flatten()
                            rare_teacher.append((logp<np.log(.01)).numpy())
                    hit=np.concatenate(pred)==teacher
                    low=np.concatenate(rare_teacher)
                    observed=float(data.weights@hit)
                    saved=next(r[part]['top1'] for r in curves[width,seed] if r['step']==step)
                    assert abs(observed-saved)<1e-12, (width,seed,step,part)
                    for root,ix in zip(data.root_ids,nodes,strict=True):
                        rows.append(dict(split=part,width=width,seed=seed,updates=step,root=int(root),n=len(ix),
                            actor_accuracy=hit[ix].mean(),priority_accuracy=correct[ix].mean(),
                            corrected=(hit[ix]&~correct[ix]).mean(),lost=(~hit[ix]&correct[ix]).mean(),
                            teacher_probability_under_1pct=low[ix].mean()))
    with (out/'per_root.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(out):
    torch.set_num_threads(4)
    frozen=read(out/'FROZEN.json')
    curves={(w,s):read(out/f'w{w}_s{s}.json') for w in study.WIDTHS for s in study.SEEDS}
    assert all(tuple(v['step'] for v in series)==study.CHECKS for series in curves.values())
    assert len(frozen['files'])==8
    rows=diagnose(out,frozen,curves)
    baselines={}
    compares={}
    fits={}
    comparisons={'width_6k':((128,6000),(64,6000)), 'width_24k':((128,24000),(64,24000)),
                 'budget_64':((64,24000),(64,6000)), 'budget_128':((128,24000),(128,6000)),
                 'both':((128,24000),(64,6000))}
    for part in ('train','validation'):
        sets={}
        fits[part]={}
        for w in study.WIDTHS:
            for u in (6000,study.END):
                sample=[row for row in rows if (row['split'],row['width'],row['updates'])==(part,w,u)]
                root_ids=sorted({r['root'] for r in sample})
                lookup={(r['root'],r['seed']):r for r in sample}
                sets[w,u]=np.array([[lookup[r,s]['actor_accuracy'] for s in study.SEEDS] for r in root_ids]).mean(1)
                fits[part][f'{w}_{u}']={k:float(np.mean([r[k] for r in sample])) for k in ('actor_accuracy','priority_accuracy','corrected','lost','teacher_probability_under_1pct')}
                baselines[part]=fits[part][f'{w}_{u}']['priority_accuracy']
        compares[part]={}
        rng=np.random.default_rng(20261119)
        draw=rng.integers(len(root_ids),size=(10000,len(root_ids))) if part=='validation' else None
        for name,(a,b) in comparisons.items():
            diff=100*(sets[a]-sets[b])
            record=dict(pp=float(diff.mean()))
            if draw is not None:
                record['ci95']=np.quantile(diff[draw].mean(1),[.025,.975]).tolist()
            compares[part][name]=record
    exact={str(s):next(r.get('exact_previous_weights') for r in curves[64,s] if r['step']==6000) for s in study.SEEDS}
    save(out/'summary.json',dict(effects_pp=compares,baseline=baselines,fit=fits,previous_weights_match=exact))
    draw_curves(out,curves)
    lines=['# 单教师 GA actor 的网络宽度与更新预算','',
       '两层隐藏层使用相同宽度。下表按每例内决策平均，再对有成功教师的初态等权；每格按 seed17 / seed29 排列。四条训练链从头开始，每条链在固定步数保存权重。','',
       '| 隐藏层 | 更新数 | 训练教师 top-1 | 留出教师 top-1 | 训练交叉熵 | 留出交叉熵 |','|---|---:|---:|---:|---:|---:|']
    for w in study.WIDTHS:
        for u in (6000,study.END):
            selected=[next(row for row in curves[w,s] if row['step']==u) for s in study.SEEDS]
            cells=[' / '.join(f'{100*r[part]["top1"]:.2f}%' for r in selected) for part in ('train','validation')]
            cells+=[' / '.join(f'{r[part]["nll"]:.4f}' for r in selected) for part in ('train','validation')]
            lines.append(f'| {w}×{w} | {u:,} | '+' | '.join(cells)+' |')
    lines+=['','| 比较 | 训练变化（百分点） | 留出变化（百分点） | 留出 95% 配对区间 |','|---|---:|---:|---:|']
    titles={'width_6k':'6k 时加宽','width_24k':'24k 时加宽','budget_64':'64 宽度增加更新','budget_128':'128 宽度增加更新','both':'同时加宽和增加更新'}
    for k,title in titles.items():
        t=compares['train'][k];v=compares['validation'][k];lo,hi=v['ci95']
        lines.append(f"| {title} | {t['pp']:+.2f} | {v['pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines+=['','![训练和留出拟合曲线](fit_curves.png)','',
        '曲线为两个固定种子的均值，阴影为种子间范围。区间在 111 个有教师的留出初态上配对重抽 10,000 次，条件于这两个训练种子。多个次要比较呈描述性报告。','',
        f"直接执行训练覆盖排名最高的合法 CAP，在这些教师前缀上可得到训练 {100*baselines['train']:.2f}%、留出 {100*baselines['validation']:.2f}% 的动作一致率。该规则只读取训练排名和当前建议。逐初态文件同时报告网络纠正或破坏该规则选择的比例。",'',
        '训练涉及 449 个可解初态、18,643 个教师边界；留出为 111 个可解初态、4,858 个边界。沿用同一组公开输入、原训练集归一化、合法候选、单教师采样、Adam 学习率 0.0003、batch 64 和梯度裁剪 1。相同种子各宽度使用同一批次抽样序列。只测监督拟合，留出是历史开发验证面板；闭环成功率另需实测。','',
        f"64×64 在 6,000 步与上一轮权重逐张量一致检查为 {exact}。两个宽度的可训练参数量分别为 {frozen['parameters']['w64_s17']:,} 与 {frozen['parameters']['w128_s17']:,}。",'',
        '[实验协议](protocol.json) · [逐初态结果](per_root.csv) · [汇总](summary.json) · [冻结权重哈希](FROZEN.json)']
    low=[100*fits['validation'][k]['teacher_probability_under_1pct'] for k in ('64_6000','64_24000','128_24000')]
    lines+=['',f"在留出教师前缀中，模型赋予教师动作低于 1% 概率的比例，在 64×64／6,000、64×64／24,000、128×128／24,000 三个配置下分别为 {low[0]:.2f}%、{low[1]:.2f}% 和 {low[2]:.2f}%。各初态等权，两种子取平均。",'',
            '[逐种子训练面板](live/index.html)']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'FINISHED.json',dict(complete=True,models=4,updates_per_chain=study.END,closed_loop_episodes=0,
                                 report_sha256=sha(out/'REPORT.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
