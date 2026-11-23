"""Paired offline comparison of the GA and full-trace candidate actors."""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from experiments.planar_tasks.authoring_v2 import cap_actor_trace_ablation as study
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save


@torch.inference_mode()
def measure(model,data,probe):
    records=[]
    for start in range(0,len(data.roots),256):
        x,c,s,valid,label=data.tensors(slice(start,start+256))
        scores=model.values(x,c,s).masked_fill(~valid,-torch.inf)
        best=scores.argmax(1)
        nll=-scores.log_softmax(1).gather(1,label[:,None]).flatten()
        cols=[(best==label).float(),nll,(-nll).exp(),(nll>-np.log(.01)).float()]
        if probe:
            hidden=x.clone()
            hidden[:,study.GA_WIDTH:]=torch.nan
            perturbed=model.values(hidden,c,s).masked_fill(~valid,-torch.inf).argmax(1)
            cols.append((perturbed!=best).float())
        records.append(torch.stack(cols,dim=1).double().numpy())
    return np.concatenate(records)


def report(out):
    frozen=read(out/'FROZEN.json')
    protocol=read(out/'protocol.json')
    torch.set_num_threads(4)
    keys=['teacher_top1','nll','teacher_p','teacher_p_under_1pct','changed_if_trace_hidden']
    metrics={}
    diffs={}
    rows=[]
    trace_occupancy={}
    for part in ('train','validation'):
        data=study.teacher_data(part)
        finite=np.isfinite(data.features[:,study.GA_WIDTH:])
        trace_occupancy[part]=dict(ever_recorded=int(finite.any(0).sum()),average_present=float(data.weights@finite.sum(1)),width=finite.shape[1])
        groups=[np.flatnonzero(data.roots==r) for r in data.root_ids]
        collected={}
        for arm in ('ga','global'):
            for seed in study.base.SEEDS:
                model=study.base.make_actor(64) if arm=='ga' else study.trace_actor()
                curve=read(out/f'{arm}_s{seed}.json')
                for step in (6000,study.base.END):
                    if arm=='ga':
                        path=study.COMPARISON/f'w64_s{seed}_u{step}.pt'
                        assert sha(path)==protocol['base_model_hashes'][path.name]
                    else:
                        path=out/f'global_s{seed}_u{step}.pt'
                        assert sha(path)==frozen['files'][path.name]
                    model.load_state_dict(torch.load(path,weights_only=True))
                    model.eval()
                    obs=measure(model,data,arm=='global' and step==study.base.END and part=='validation')
                    root_means=np.stack([obs[ix].mean(0) for ix in groups])
                    mean=root_means.mean(0)
                    k=f'{part}/{arm}_s{seed}/{step}'
                    metrics[k]=dict(zip(keys,mean.tolist()))
                    for root,ix,m in zip(data.root_ids,groups,root_means,strict=True):
                        rows.append(dict(split=part,arm=arm,seed=seed,updates=step,root=int(root),n=len(ix),**dict(zip(keys,m))))
                    summary=next(r[part] for r in curve if r['step']==step)
                    assert abs(mean[0]-summary['top1'])<1e-12 and abs(mean[1]-summary['nll'])<1e-6
                    collected[arm,seed,step]=root_means[:,:4]
        rng=np.random.default_rng(20261119)
        draw=rng.integers(len(groups),size=(10000,len(groups)))
        for step in (6000,study.base.END):
            delta=np.stack([collected['global',s,step]-collected['ga',s,step] for s in study.base.SEEDS]).mean(0)
            k=f'{part}/{step}'
            diffs[k]=dict(top1_pp=100*delta[:,0].mean(),nll=float(delta[:,1].mean()))
            if part=='validation':
                diffs[k].update(top1_ci95_pp=(100*np.quantile(delta[:,0][draw].mean(1),[.025,.975])).tolist(),
                                nll_ci95=np.quantile(delta[:,1][draw].mean(1),[.025,.975]).tolist())
    with (out/'per_root.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['split','arm','seed','updates','root','n',*keys])
        writer.writeheader()
        writer.writerows(rows)
    save(out/'summary.json',dict(metrics=metrics,differences=diffs,trace_occupancy=trace_occupancy))
    effect=diffs['validation/24000']
    lo,hi=effect['top1_ci95_pp']
    lines=['# 64×64 单教师 actor 的 GLOBAL trace 对照','',
           f"固定 24,000 步，GLOBAL 相对 GA 的留出教师动作一致率平均差为 {effect['top1_pp']:+.2f} 个百分点，按初态配对的 95% 区间为 [{lo:+.2f}, {hi:+.2f}]。每个初态内先对教师边界平均，再对 111 个留出初态等权平均。训练集对应 449 个有示范初态。",'',
           '| 输入 | 更新数 | 训练教师 top-1 | 留出教师 top-1 | 留出交叉熵 | 留出教师动作概率 <1% 的比例 |','|---|---:|---:|---:|---:|---:|']
    for step in (6000,study.base.END):
        for arm in ('ga','global'):
            def pair(part,metric,scale,decimals):
                return ' / '.join(f'{scale*metrics[f"{part}/{arm}_s{s}/{step}"][metric]:.{decimals}f}' for s in study.base.SEEDS)
            lines.append(f"| {arm.upper()} | {step:,} | {pair('train','teacher_top1',100,2)}% | {pair('validation','teacher_top1',100,2)}% | {pair('validation','nll',1,3)} | {pair('validation','teacher_p_under_1pct',100,2)}% |")
    report_seed=[100*metrics[f'validation/global_s{s}/24000']['changed_if_trace_hidden'] for s in study.base.SEEDS]
    lines+=['','表中每格依次为 seed17 / seed29。两组共用 18,643 个训练决策和 4,858 个开发留出决策。留出面板已在前期诊断中查看，区间由 10,000 次配对初态重抽得到，条件于两个固定训练种子。','',
           '## 网络如何读取 trace','',
           '每一决策边界向七个 CAP 提供同一段实际公开历史。插桩为程序位置分配槽位，记录执行到的标量赋值、分支结果和返回位置，聚合同一步该位置的最后有限值。七个程序的 2,096 个槽按固定顺序连接。有限值用训练集统计量标准化并裁剪至 ±10；缺失槽填零，同时为每个槽附带是否有效的标记。因此额外送入第一层的是 4,192 个数。','',
           'GA 读取公开几何、七 CAP 建议、剩余预算，以及当前候选动作与推荐来源。GLOBAL 在所有候选的共同状态输入上加入全部七份 trace，不同动作保留各自的动作编码和推荐者掩码。在第一层，状态和动作的两个线性项相加，再经过两个 64 单元 ReLU 隐藏层及一个标量输出。只对不同的合法候选做 softmax，同一动作的多个推荐来源合并。历史 LOCAL 设计只向候选开放推荐该动作的 CAP 的 trace。','',
           f"在这些留出教师前缀中，每步平均有 {trace_occupancy['validation']['average_present']:.1f} 个槽记录有限值。固定最终 GLOBAL 权重，把所有 trace 槽同时设为缺失后，首选动作在 {report_seed[0]:.2f}% / {report_seed[1]:.2f}% 的留出前缀上改变（初态等权）。这个缺失干预测量对输入的敏感度；有无 trace 的比较使用分别训练的两组。",'',
           '## 实验约定','',
           '训练只用覆盖优先的成功教师整轨迹，动作交叉熵，按初态等权抽样。两组的归一化、样本顺序、Adam 学习率 0.0003、batch 64、梯度裁剪 1 和训练预算完全相同。GA 复用上一轮的冻结权重；两份 GLOBAL 从相同种子的初始化开始，共有输入和隐藏层权重逐项复制，新引入的 trace 与掩码输入列从零开始学习。全训练和留出样本在更新前校验候选评分的浮点差。','',
           'GA 有 1,539,841 个可训练参数，GLOBAL 有 1,808,129 个，差额来自新增输入列。比较衡量现有 GLOBAL 接法的增量效果。此次评估是教师前缀上的离线拟合，新增 0 次仿真和 0 次 PPO 更新。','',
           '[逐种子曲线](live/index.html) · [预先固定的协议](protocol.json) · [按初态配对的结果](per_root.csv) · [完整统计](summary.json) · [冻结权重与初始化核验](FROZEN.json)']
    assert all(r['parameters']==1808129 and r['trace_weight_norm']>0 for r in frozen['runs'].values())
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'FINISHED.json',dict(complete=True,new_fits=2,reference_models=2,closed_loop_episodes=0,report_sha256=sha(out/'REPORT.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    report(parser.parse_args().out.resolve())
