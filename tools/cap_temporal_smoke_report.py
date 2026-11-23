"""Publish the development-only pipeline smoke test and its fixed-probe curves."""
import shutil

import numpy as np
import torch

from experiments.planar_tasks.authoring_v2 import cap_temporal_attention as a
from experiments.planar_tasks.authoring_v2.cap_temporal_compare import graph
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.temporal_data import SequenceStore


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    folder=a.OUT/'smoke'
    r=read(folder/'result.json')
    seq=SequenceStore(folder/'validation')
    train=SequenceStore(folder/'train')
    assert not np.intersect1d(train.a['root'],seq.a['root']).size
    ids=np.random.default_rng(892).choice(np.flatnonzero(seq.a['boundary']>=7),128,replace=False)
    model=a.load_model(folder/'Q.pt')
    hist,initial,mask=seq.window(ids,model.f.frame)
    cand=torch.tensor(seq.a['candidates'][ids])
    support=torch.tensor(seq.a['sources'][ids])
    pred=model(hist,initial,mask,cand,support)
    ix=(torch.arange(len(ids)),torch.tensor(seq.a['chosen'][ids]))
    start=model.f.frame+model.layout['advice_width']+2
    differences={}
    for name,when in [('past_only',slice(None,-1)),('current_only',slice(-1,None))]:
        altered=hist.clone()
        altered[:,when,start:model.f.tail]=torch.nan
        delta=(pred-model(altered,initial,mask,cand,support))[ix].abs()
        differences[name]=dict(mean_absolute=float(delta.mean()),max_absolute=float(delta.max()))
        assert delta.max()>0
    # Padded rows in the shared MLP must remain missing even if callers provide arbitrary filler.
    starts=np.flatnonzero(seq.a['boundary']==0)
    h,initial,exists=seq.window(starts,model.f.frame)
    assert exists.sum(1).eq(1).all() and torch.isnan(h[~exists]).all()
    ytrain=train.a['y']
    fixed=np.sort(np.random.default_rng(741).choice(len(seq),1024,replace=False))
    v=a.load_model(folder/'V.pt')
    target=np.concatenate([a.bootstrap(v,seq,i) for i in np.array_split(fixed,16)])
    save(folder/'diagnostics.json',dict(masked_trace=differences,scored_boundaries=128,
         validation_V_constant_train_mean_mse=float(((seq.a['y'][fixed]-ytrain.mean())**2).mean()),
         validation_Q_target_std=float(target.std()),distinct_roots=True))
    graph(folder)
    live=a.scale.OUT/'live'
    shutil.copyfile(folder/'loss.png',live/'attention_smoke_loss.png')
    lines=['# GLOBAL attention 开发侧管线试跑','',
           f"从原训练集随机抽 12 个初态（84 条轨迹，{r['train_rows']} 个决策边界），从原验证集抽不重叠的 6 个初态（42 条轨迹，{r['validation_rows']} 个边界）。两组均含七个 CAP 的成功与失败运行。两组各固定 1,024 个边界画曲线，种子 17。",'',
           '| 阶段 | 更新次数 | 训练 MSE，初始 → 末轮 | 留出 MSE，初始 → 末轮 |','|---|---:|---:|---:|']
    for stage in ['V','Q']:
        p,q=r['curves'][stage][0],r['curves'][stage][-1]
        lines.append(f"| {stage} | {q['update']} | {p['train']['mse']:.5f} → {q['train']['mse']:.5f} | {p['validation']['mse']:.5f} → {q['validation']['mse']:.5f} |")
    lines.extend(['','V 学原 CAP 的终局成败，Q 由 V 初始化，并在实际执行动作上拟合一步自举目标。Q 的起始损失已较低，留出误差末轮略升。固定 V 生成的目标与真实闭环成功率是两项不同测量。','',
                  '![固定训练与验证样本上的拟合曲线](loss.png)','',
                  f"重放两个完整 episode 的 {r['parity']['boundaries']} 个决策边界，离线窗口重建原 GA 几何逐值一致，流式推理与训练评分的最大绝对差为 {r['parity']['max_absolute_score_difference']:.1f}。保存再加载模型的输出逐值一致。",
                  f"在 128 个有完整历史的留出边界，固定几何和全部动作建议，单独将过去七步 trace 设为缺失，实际动作 Q 平均绝对变化 {differences['past_only']['mean_absolute']:.5f}；单独屏蔽当前 trace 为 {differences['current_only']['mean_absolute']:.5f}。此测量检验输入接通，未测量其闭环贡献。",'',
                  '| 开发初态 | 决策 | 实际 tick | 成功 | 停止原因 |','|---|---:|---:|---:|---|'])
    for row in r['rollouts']:
        assert row['complete'] and not row['error']
        lines.append(f"| {row['case_id']} | {row['decisions']} | {row['ticks']} | {int(row['success'])} | {row['stop']} |")
    (folder/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    main()
