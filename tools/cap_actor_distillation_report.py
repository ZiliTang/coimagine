"""Paired reporting on the viewed retention panel and common held-out teacher prefixes."""
import argparse
import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2.cap_actor_distillation import ARMS, SEEDS, ARCHIVE, PARENT, CAPS
from experiments.planar_tasks.authoring_v2.cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_trace_value_long_report import write_csv


def record(path):
    with gzip.open(path/'result.json.gz','rt') as stream:
        return json.load(stream)


def key(action):
    return tuple(action[k] for k in ('target_id','a0','a1','ticks'))


def exposure(out):
    result={}
    for part in ('train','validation'):
        with np.load(out/f'{part}_prefixes.npz') as f:
            roots=f['root']
            on_teacher=f['priority_action']>=0
            target=f['successful_votes']>0
            valid=np.load(PARENT/'arrays'/part/'valid.npy',mmap_mode='r')[f['row']]
        many=target.sum(1)>1
        def mean(x):
            return float(np.mean([x[roots==r].mean() for r in np.unique(roots)]))
        result[part]=dict(root_uniform_multi_fraction=mean(many), root_uniform_priority_trajectory_fraction=mean(on_teacher),
                          zero_loss_multi_fraction=mean((target==valid).all(1)),root_uniform_multi_off_priority=mean(many & ~on_teacher))
    save(out/'target_exposure.json',result)
    return result


def vote_reference(out):
    """Align the vote baseline with the exact held-out prefixes and root weights."""
    archive=PARENT/'arrays/validation'
    with np.load(out/'validation_prefixes.npz') as stored:
        rows=stored['row']
        root=stored['root']
        priority=stored['priority_action']
        target=stored['successful_votes']>0
    sources=np.load(archive/'sources.npy',mmap_mode='r')[rows]
    votes=sources.sum(2)
    assert np.all(votes.sum(1)>0)
    probs=votes/votes.sum(1,keepdims=True)
    best=probs.argmax(1)
    def mean(x,mask):
        return float(np.mean([x[mask & (root==r)].mean() for r in np.unique(root[mask])]))
    i=np.arange(len(rows))
    teacher=priority>=0
    all_rows=i>=0
    return dict(priority_p=mean(probs[i,priority.clip(min=0)],teacher),priority_top1=mean(best==priority,teacher),
                set_p=mean((probs*target).sum(1),all_rows),set_top1=mean(target[i,best],all_rows),
                several_p=mean((probs*target).sum(1),target.sum(1)>1))


def compare_case(job):
    out, scene, old = job
    case=scene['case_id']
    reference={}
    successful_caps=[]
    for cap in CAPS:
        if int(old[cap]):
            successful_caps.append(cap)
            run=record(ARCHIVE/'evaluation'/cap/case)
            assert run['summary']['success'] and run['summary']['complete']
            for pub,event in zip(run['public_rows'][:-1],run['events'],strict=True):
                reference.setdefault(pub['history_hash'],set()).add(key(event['requested']))
    row=dict(case_id=case,movable_count=scene['movable_count'],successful_caps=' '.join(successful_caps),
             ncap=len(successful_caps))
    assert row['ncap']==int(old['union7_count'])
    for seed in SEEDS:
        row[f'PPO_s{seed}']=int(old[f'PPO_s{seed}'])
        for arm in ARMS:
            name=f'{arm}_s{seed}'
            path=out/'evaluation'/name/case
            run=record(path)
            summary=read(path/'summary.json')
            assert summary==run['summary'] and summary['complete'] and not summary['error']
            assert summary['case_id']==case and summary['condition']==name
            assert len(run['decisions'])==len(run['events'])==summary['decisions'] and len(run['public_rows'])==summary['decisions']+1
            assert 0<summary['ticks']<=1800 and 0<summary['decisions']<=300
            first_exit=None
            for i,(pub,decision) in enumerate(zip(run['public_rows'][:-1],run['decisions'],strict=True)):
                assert decision['greedy']==decision['chosen'] and decision['action']==decision['candidates'][decision['chosen']]
                if i==0 and successful_caps:
                    assert pub['history_hash'] in reference
                if first_exit is None and key(decision['action']) not in reference.get(pub['history_hash'],()):
                    first_exit=i
            if successful_caps and first_exit is None:
                assert summary['success']
            row[name]=int(summary['success'])
            row[name+'_prefix_length']=first_exit if first_exit is not None else len(run['decisions'])
            row[name+'_ticks']=summary['ticks']
            row[name+'_invalid']=summary['invalid_proposals']
            row[name+'_fallback']=summary['fallback_boundaries']
    return row


def figure(out, fit, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    colors={'single':'#24649c','multi':'#bc5922'}
    for ax,labels in zip(axes,[['Priority teacher','Known success set'],['All (200)','CAP union (146)','Unique (28)']],strict=True):
        for j,arm in enumerate(ARMS):
            marks=[]
            for seed in SEEDS:
                name=f'{arm}_s{seed}'
                v=fit[name]['validation']
                m=metrics[name]
                marks.append([v['priority']['all']['top1'],v['success_set']['all']['top1']] if len(labels)==2
                             else [m['total']/200,m['retained']/146,m['unique']/28])
            values=np.array(marks)*100
            x=np.arange(len(labels))+(j-.5)*.29
            ax.bar(x,values.mean(0),width=.27,color=colors[arm],alpha=.7,label=arm)
            for i in range(2):
                ax.scatter(x+(i-.5)*.07,values[i],color='black',s=16,marker=('o','x')[i],zorder=3)
        ax.set(xticks=np.arange(len(labels)),xticklabels=labels,ylim=(0,100))
        ax.grid(axis='y',alpha=.18)
    axes[0].set_title('Held-out prefix top-1 (%)')
    axes[1].set_title('Closed-loop success (%)')
    axes[0].legend(frameon=False)
    fig.savefig(out/'comparison.png',dpi=160)
    plt.close(fig)


def main(out):
    protocol=read(out/'protocol.json')
    assert sha(ARCHIVE/'per_scene.csv')==protocol['archived_reference_sha256']
    assert sha(ARCHIVE/'panel.json')==protocol['panel_sha256']
    assert all(sha(out/name)==h for name,h in read(out/'FROZEN.json').items())
    panel=read(ARCHIVE/'panel.json')['cases']
    with (ARCHIVE/'per_scene.csv').open() as f:
        old={r['case_id']:r for r in csv.DictReader(f)}
    with ProcessPoolExecutor(8) as pool:
        rows=list(pool.map(compare_case,[(out,c,old[c['case_id']]) for c in panel],chunksize=4))
    assert len(rows)==200
    write_csv(out/'per_scene.csv',rows)
    counts=np.array([r['ncap'] for r in rows])
    union=counts>0
    metrics={}
    groups={}
    for arm in (*ARMS,'PPO'):
        group=[]
        for seed in SEEDS:
            name=f'{arm}_s{seed}'
            success=np.array([r[name] for r in rows],int)
            group.append(success)
            stat=dict(total=int(success.sum()),retained=int(success[union].sum()),unique=int(success[counts==1].sum()),
                      outside=int(success[~union].sum()))
            if arm in ARMS:
                plen=np.array([r[name+'_prefix_length'] for r in rows])
                stat.update(start_success_set=int((union & (plen>0)).sum()),
                            failed_after_supported_start=int((union & (plen>0) & (success==0)).sum()),
                            invalid=sum(r[name+'_invalid'] for r in rows),fallback=sum(r[name+'_fallback'] for r in rows))
            metrics[name]=stat
        groups[arm]=np.stack(group)
    difference=groups['multi']-groups['single']
    rng=np.random.default_rng(20261119)
    draws=np.concatenate([rng.choice([i for i,c in enumerate(panel) if c['movable_count']==k],(10000,50)) for k in range(2,6)],axis=1)
    contrasts={}
    for tag,mask in [('union',union),('all',counts>=0),('unique',counts==1),('outside',~union)]:
        delta=difference.mean(0)
        bs=(delta[draws]*mask[draws]).sum(1)/mask[draws].sum(1)
        contrasts[tag]=dict(n=int(mask.sum()),pp=100*delta[mask].mean(),ci=(100*np.quantile(bs,[.025,.975])).tolist(),
                            gains=(difference[:,mask]>0).sum(1).tolist(),losses=(difference[:,mask]<0).sum(1).tolist())
    exp=exposure(out)['train']
    vote=vote_reference(out)
    save(out/'summary.json',dict(metrics=metrics,comparisons=contrasts,held_out_vote=vote))
    fit=read(out/'fit_metrics.json')
    inv={p:read(out/f'{p}_inventory.json') for p in ('train','validation')}
    main=contrasts['union']
    lines=['# GA actor 蒸馏：单一优先教师与成功动作集合','',
        f"冻结模型在已查看的 200 个诊断场景中各执行一次。多解组相对单教师组的七 CAP 并集保留率平均差为 {main['pp']:+.2f} 个百分点，按场景配对的 95% 区间为 [{main['ci'][0]:+.2f}, {main['ci'][1]:+.2f}]。两组均直接从头训练 actor，闭环每步选概率最高的合法候选。",'',
        '| 组别 | 种子 | 总成功 /200 | 并集内 /146 | 单 CAP 独有 /28 | 并集外 /54 | 首步有成功示范 /146 | 首步有示范仍失败 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in (*ARMS,'PPO'):
        for seed in SEEDS:
            m=metrics[f'{arm}_s{seed}']
            lines.append(f"| {arm} | {seed} | {m['total']} | {m['retained']} | {m['unique']} | {m['outside']} | {m.get('start_success_set','—')} | {m.get('failed_after_supported_start','—')} |")
    lines+=['',f"逐种子配对比较中，multi 在并集内相对 single 新增成功 {main['gains'][0]}/{main['gains'][1]} 例，同时失去 {main['losses'][0]}/{main['losses'][1]} 例。两种子和两组共用同一批 200 个初态。区间基于 10,000 次配对场景重抽，按物体数分层，训练种子固定。历史 GA＋PPO 仅为同面板外部参照。首步有示范表示至少一个最终成功的单 CAP 在相同公开初态上给出了相同完整动作；这并不保证混合策略续接成功。",'',
        '## 离线拟合','',
        '对两组使用相同的评估目标。优先教师指标在优先教师轨迹上测量；成功集合指标在所有成功轨迹的去重公开前缀上测量。先在每个初态内平均，再对有样本的初态等权。下表均为百分数；多解前缀只保留至少两个不同动作均有真实成功后续的前缀。','',
        '| 组别 | 种子 | 训练教师 top-1 | 留出教师 top-1 | 留出教师动作概率 | 留出已知成功集合概率 | 留出已知成功集合 top-1 | 留出多解前缀集合概率 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for seed in SEEDS:
            f=fit[f'{arm}_s{seed}'];t=f['train']['priority']['all'];v=f['validation'];p=v['priority']['all'];s=v['success_set']['all'];many=v['success_set']['several']
            values=[t['top1'],p['top1'],p['probability'],s['probability'],s['top1'],many['probability']]
            lines.append(f"| {arm} | {seed} | "+' | '.join(f'{100*x:.2f}' for x in values)+' |')
    lines.append('| 按 CAP 推荐数分配 | — | — | '+' | '.join(f'{100*vote[k]:.2f}' for k in ('priority_top1','priority_p','set_p','set_top1','several_p'))+' |')
    lines+=['','## 训练与解释边界','',
        '七 CAP 的全局优先级只按 608 个训练初态的成功数确定，同分按 CAP 名。每个有成功示范的初态选排名最高的成功 CAP，并保留其整条轨迹。多解组保留全部成功轨迹，对同初态、同完整公开历史哈希的前缀合并真实成功后续的动作，合并时逐行核对模型特征、候选、顺序和来源。多解损失为成功集合总概率的负对数，单标签时等价于交叉熵。每例不同 CAP 的未执行建议不自动获得成功标签。','',
        f"训练有 {inv['train']['roots']} 个可解初态和 {inv['train']['success_episodes']} 条成功轨迹。single 训练 {inv['train']['priority_boundaries']:,} 个决策边界，multi 训练 {inv['train']['unique_success_prefixes']:,} 个不同前缀，其中 {inv['train']['multi_action_prefixes']} 个有多个成功动作。留出集 152 个初态中有 {inv['validation']['roots']} 个可解，优先教师边界 {inv['validation']['priority_boundaries']:,} 个，成功去重前缀 {inv['validation']['unique_success_prefixes']:,} 个，其中 {inv['validation']['multi_action_prefixes']} 个有多个成功动作。两组差异同时包含轨迹覆盖、每例内的采样权重和目标函数，无法把差异单独归因于集合损失。",'',
        '两组共享 GA 的公开几何、七 CAP 建议、剩余预算、候选动作及其来源。相同训练集归一化、64/64 ReLU 候选评分网络、线性 logit，无 trace 输入。每模型 6,000 次 Adam 更新，学习率 0.0003，batch 64，梯度范数上限 1。相同 seed 共享初始权重及逐批初态抽样，组内按该例可用的不同前缀均匀抽取。固定取末轮模型后执行 800 个完整 episode，沿用原单 CAP 归档和同面板历史 GA＋PPO；四份模型均无 PPO 更新。32 个开发留出初态曾用于早期实验监测。','',
        '[逐初态结果](per_scene.csv) · [配对统计](summary.json) · [完整拟合指标](fit_metrics.json) · [冻结协议](protocol.json) · [逐例回放](evaluation/)']
    lines+=['',f"按实际的初态等权采样，多解组有 {100*exp['root_uniform_multi_fraction']:.2f}% 的样本拥有至少两个已知成功标签；{100*(1-exp['root_uniform_priority_trajectory_fraction']):.2f}% 的样本来自优先教师轨迹以外的前缀。这两个比例量化了本轮干预实际改变的监督信号。[样本暴露统计](target_exposure.json)"]
    lines+=['','## 配对回放','',
        '各方向取场景 ID 排序的第一例，优先取两个种子都观察到相同输赢方向的并集内场景。每个文件保留动作候选、模型分数、执行事件和物理帧。']
    for winner,other in [('single','multi'),('multi','single')]:
        candidates=sorted((r for r in rows if r['ncap'] and r[f'{winner}_s17'] and not r[f'{other}_s17']),
            key=lambda r:(not(r[f'{winner}_s29'] and not r[f'{other}_s29']),r['case_id']))
        if candidates:
            r=candidates[0]
            case=r['case_id']
            lines+=['',f"{case}，成功教师 {r['successful_caps']}。seed17 的 {winner} 成功、{other} 失败。"
                +f"[单教师回放](evaluation/single_s17/{case}/result.json.gz) · [多解回放](evaluation/multi_s17/{case}/result.json.gz)"]
    figure(out, fit, metrics)
    lines+=['','![两组在离线和闭环指标上的表现](comparison.png)','','柱为两个固定训练种子的均值，圆点和叉号分别为 seed17 和 seed29。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    save(out/'FINISHED.json',dict(complete=True,episodes=len(rows)*4,distinct_evaluation_scenes=len(rows),updates_per_model=6000,
                                 models=4,verified_replays=len(rows)*4,report=sha(out/'REPORT.md')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
