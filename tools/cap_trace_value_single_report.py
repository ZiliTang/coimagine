"""Single-seed screening report; keeps the new confirmation panel sealed."""
import json
import hashlib
from pathlib import Path
import shutil

import numpy as np

from tools import cap_trace_value_long_report as L

ROOT=Path(__file__).resolve().parents[1]
ASSETS=ROOT/'docs/agent_exchange/assets/cap_trace_value_single_20261118'
REVIEW=ROOT/'docs/agent_exchange/reviews/CAP_Trace_Value_Single_20261118.md'


def render_live(out):
    L.SEEDS=(17,)
    L.render_live(out)
    path=Path(out)/'report/dashboard.html'
    text=path.read_text().replace('三种子17/29/43','单种子17；29/43已暂停')
    text=text.replace('全新200确认集只在全部模型冻结后运行。','本轮仅做单种子诊断；全新200确认集保持封存，不执行。')
    text=text.replace('GLOBAL＋RL：200 → 2,000 episode 预算对照','GLOBAL＋RL：seed17 单种子长程筛查')
    path.write_text(text)


def final_report(out):
    out=Path(out).resolve()
    inputs=L.read(out/'inputs.json')
    assert L.read(out/'FINISHED.json')['complete']
    L.SEEDS=(17,)
    result,records=L.analyze_panel(out,'diagnostic')
    training,losses,evaluation=L.gather(out)
    assert len(training)==40 and max(r['updates'] for r in losses)==60000
    inherited=set(L.read(out/'inherited.json'))
    physical=[]
    for pattern in ('online/GLOBAL_s17/r*.json','monitor/*.json','diagnostic/*.json'):
        for path in out.glob(pattern):
            if path.name.endswith('_progress.json') or str(path.relative_to(out)) in inherited:continue
            data=L.read(path)
            if 'results' in data:
                assert data['complete']
                physical.extend(data['results'])
    assert len(physical)==2406 and all(r['complete'] and not r.get('error') for r in physical)
    frozen=L.read(out/'ALL_MODELS_FROZEN.json')
    assert all(hashlib.sha256((out/p).read_bytes()).hexdigest()==v for p,v in frozen['files'].items())
    perf=Path(inputs['performance'])
    speed=L.read(perf/'training_benchmark.json')
    physics=L.read(perf/'physical_benchmark.json')
    summary=dict(complete=True,seed=17,status='single-seed diagnostic screening; confirmation remains sealed',
        diagnostic=result,training_benchmark=speed,physical_benchmark=physics,
        replay_benchmark=L.read(perf/'replay_benchmark.json'),
        resume=L.read(out/'pause_resume.json'),new_physical_executions=len(physical),
        new_ticks=sum(r['ticks'] for r in physical),new_decisions=sum(r['decisions'] for r in physical),
        invalid_proposals=sum(r.get('invalid_proposals',0) for r in physical),
        fallback_boundaries=sum(r.get('fallback_boundaries',0) for r in physical),errors=0,
        confirmation_executions=0,budget=L.read(out/'FINISHED.json'),
        final_resident_memory=L.read(out/'memory.json'))
    L.save(out/'report/SUMMARY.json',summary)
    selected=[r for r in records if r['condition'].startswith(('CAP-RR-','GLOBAL_s17'))]
    L.write_csv(out/'report/per_case_results.csv',selected)
    render_live(out)
    counts={r['episodes']:r['success'] for r in evaluation if r['panel']=='diagnostic'}
    delta=result['long_minus_short'];low,high=delta['ci95_pp']
    verdict=('在这个seed和已查看诊断面板上，长预算有配对正收益，值得进一步确认。' if low>0 else
             '在这个seed和已查看诊断面板上，长预算出现配对负收益，不应仅凭训练loss下降继续加预算。' if high<0 else
             '这一条训练链未提供明确的诊断收益证据；区间跨零，不能据此排除其他seed上的收益。')
    mmap=next(r for r in speed['results'] if r['mode']=='mmap')
    fast=next(r for r in speed['results'] if r['mode']=='resident' and r['threads']==4)
    pold=sum(r['wall_seconds'] for r in physics['results'] if r['mode']=='original')
    pnew=sum(r['wall_seconds'] for r in physics['results'] if r['mode']=='fast')
    rows='\n'.join(f'|{n}|{counts[n]}/200|{counts[n]/2:.1f}%|' for n in (200,400,800,1200,2000))
    text=f'''# GLOBAL＋RL：性能优化与seed17长程筛查（2026-11-18）

用户要求暂停三seed训练、诊断和优化耗时，再推进预先固定的seed17。已完成。
旧200诊断场景上，原200预算 **{result['short']['successes'][0]}/200**，最终2000预算 **{result['long']['successes'][0]}/200**；
差 **{delta['difference_pp']:+.2f}pp**，配对场景bootstrap95%区间 **[{low:.2f},{high:.2f}]**。
{verdict}
本轮只有一个训练seed，而且使用已查看的诊断面板，**不是新的独立确认结论**。新确认200场景执行0次，继续封存。

## 固定预算曲线

|累计在线episode|成功数|成功率|
|---:|---:|---:|
{rows}

[实时/最终曲线面板](../assets/cap_trace_value_single_20261118/dashboard.html) ·
[PNG](../assets/cap_trace_value_single_20261118/learning_curves.png) ·
[训练reward/行为熵](../assets/cap_trace_value_single_20261118/training_returns.csv) ·
[loss/Q/梯度/抽样耗时](../assets/cap_trace_value_single_20261118/loss_metrics.csv) ·
[独立监测曲线](../assets/cap_trace_value_single_20261118/evaluation_curve.csv) ·
[逐场景结果](../assets/cap_trace_value_single_20261118/per_case_results.csv) ·
[完整统计](../assets/cap_trace_value_single_20261118/SUMMARY.json)。

固定最终2000episode/60000在线更新，不挑最好checkpoint；200、400预算结果复用原真实执行，800/1200/2000在同200根诊断。
seed29/43未继续训练，不按表现选择seed17。32开发根监测另报，不用于早停或选择。
原200在线初态重复采样，新增轨迹但没有新增独立训练根。网络、奖励、gamma=1、normalizer、Adam、target同步、梯度裁剪、回放32:32和边界均匀采样不变，epsilon=.05。
单种子配对区间只反映这200个根的抽样不确定性，不覆盖训练随机性，也不消除已查看面板的自适应偏差。

## 耗时诊断与优化

离线状态矩阵16.485GB超过本机可用内存，随机mmap读取产生大量磁盘IO。每行使用LZ4无损压缩，
全部294285行逐字节往返验收；压缩状态3.069GB，加候选/来源/标签等常驻总量3.847GB。
保持原float32字节、NaN掩码、signed zero、全局索引与下一状态关系；不是降精度或删减输入。
200批离线当前/下一状态读取：mmap13.459秒，驻内存压缩回放0.145秒；这是读取环节，不是端到端93倍提速。

相同真实模型与回放、相同100次梯度更新：原mmap单线程循环 **{mmap['loop_seconds']:.3f}秒**，
驻内存4线程 **{fast['loop_seconds']:.3f}秒**，约 **{mmap['loop_seconds']/fast['loop_seconds']:.2f}倍**。
1/2/4线程优化结果与原版的模型、Adam、目标网络、NumPy/Torch RNG逐值一致。实际长程耗时另以本轮日志为准。
采样worker仍12进程、各1线程；仅learner改4线程，保持同步采样/更新顺序，没有引入异步旧策略采样。

物理采样优化：在单episode的同步读取范围共享接触/传感结果；物理推进前后及切换目标时仍严格失效，
绝不把上一tick扫描用于当前控制。历史公开事件只解析一次，保留原JSON哈希协议和作者输入独立拷贝。
轨迹JSON先统一编码再写gzip，保留全部字段，减少大量零碎写入。
四种物体数量各一对完整轨迹：原版合计 **{pold:.3f}秒**，优化版 **{pnew:.3f}秒**，约 **{100*(1-pnew/pold):.1f}%** 更快。
该小样本测速存在缓存/调度波动，不当作普适提速保证；四对全部公开观测、哈希、动作、Q分数、事件、物理帧、训练数组字节一致。
接触查询约每tick4次降至1次；雷达仅消除同一物理状态的重复读取。7CAP各自语义/记忆计算仍独立，没有凭相似逻辑合并不同程序。

## 暂停与无损接续

原seed17在450次采集、12023次在线更新时按用户要求暂停。保存的中断检查点含完整Adam/target/RNG，
本轮先补齐该轮1477次更新，再继续后续31轮；没有丢弃已完成的23次更新，没有重采该轮50条轨迹。
这23次更新未形成100步日志窗口，明确保留指标缺口，不补造loss。原29/43检查点与三seed历史输出保持不变。
性能诊断另执行10条物理轨迹和400次临时梯度更新，不纳入正式训练结果。

本次正式新增物理执行2406＝在线1550＋开发监测256＋诊断600，全部完整、错误0。
累计seed17仍为2000在线episode/60000更新。新ticks {summary['new_ticks']}、新决策 {summary['new_decisions']}。
非法CAP建议{summary['invalid_proposals']}、fallback边界{summary['fallback_boundaries']}。
全新确认集执行0次，不自动补其他seed或扩大预算。本轮结果只能用于决定是否值得下一轮确认。
原始输出：`{out}`。性能诊断：`{perf}`。配置、源码、父检查点、缓存哈希和原始轨迹全部保留。
针对性8项软件测试通过，另完成上述真实轨迹/更新等价检查；未运行全仓库测试。
'''
    (out/'report/REPORT.md').write_text(text)
    REVIEW.write_text(text)
    ASSETS.mkdir(parents=True,exist_ok=True)
    for name in ('dashboard.html','learning_curves.png','training_returns.csv','loss_metrics.csv',
                 'evaluation_curve.csv','curves.json','SUMMARY.json','per_case_results.csv'):
        shutil.copyfile(out/'report'/name,ASSETS/name)
    for name in ('inputs.json','FINISHED.json','ALL_MODELS_FROZEN.json','pause_resume.json'):
        shutil.copyfile(out/name,ASSETS/name)
    L.save(ASSETS/'manifest.json',{p.name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)
            for p in ASSETS.iterdir() if p.name!='manifest.json' and p.is_file()})
    path=ROOT/'docs/agent_exchange/CURRENT.md'
    content=path.read_text();start,end='<!-- LONG_ONLINE_START -->','<!-- LONG_ONLINE_END -->'
    a,b=content.split(start,1);_,c=b.split(end,1)
    body=f'''\n**性能优化与seed17长程筛查已完成。** 原预算{result['short']['successes'][0]}/200，长预算{result['long']['successes'][0]}/200，诊断差{delta['difference_pp']:+.2f}pp。
[完整报告](reviews/CAP_Trace_Value_Single_20261118.md)；[曲线](assets/cap_trace_value_single_20261118/dashboard.html)。
单seed、已查看诊断面板，不能作为独立确认。新确认集执行0次，其他seed仍暂停。2406新增执行完成，停止，不自动扩预算。\n'''
    path.write_text(a+start+body+end+c)
    L.save(out/'REPORT_FINISHED.json',dict(complete=True,review=str(REVIEW),assets=str(ASSETS)))
    print('SINGLE_REPORT_COMPLETE',delta,flush=True)
