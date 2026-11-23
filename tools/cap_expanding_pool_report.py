"""Summarize measured simulation costs, branch yield and paired generalization."""
from collections import Counter
import csv
import json
from pathlib import Path

from tools import cap_expanding_pool as x
from tools.cap_option_handoff_opportunity_report import paired

OUT,DOC=x.OUT,x.DOC
read=x.e.read


def write_table(name,rows):
    with (DOC/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0])
        w.writeheader()
        w.writerows(rows)


def compare(rows):
    return paired([r['final'] for r in rows],[r['initial'] for r in rows])


def render(info,cohorts,scores):
    rounds=info['rounds']
    main=info['held']['all']
    lo,hi=[100*v for v in main['ci']]
    total=sum(r['total_ticks'] for r in rounds)
    signals=sum(r['informative'] for r in rounds)
    forks=sum(r['forks'] for r in rounds)
    lines=[
        '# 扩充母房间池、定向复访与分叉策略迭代',
        f"预先锁定的第 4 轮在 200 个全新自然房间成功 {main['first']} 个，初始蒸馏 actor 成功 {main['second']} 个。逐房间救回 {main['plus']} 个、丢失 {main['minus']} 个；差值的配对重采样 95% 区间为 [{lo:+.1f}, {hi:+.1f}] 个百分点。相同房间上单专家 A 成功 {sum(r['A'] for r in scores)} 个、B 成功 {sum(r['B'] for r in scores)} 个。诊断用的固定 64 间来自上一轮已查看的 200 个自然房间，与本轮训练房间和最终确认面板均无内容重合。五个 checkpoint 依次为 {'、'.join(map(str,info['diagnostic']))}/64；训练结束后才评测主面板。",
        '## 独立闭环结果',
        '| 专家从初态的结果 | 房间 | 初始成功 | 最终成功 | 救回 | 丢失 |\n|---|---:|---:|---:|---:|---:|\n'
        +'\n'.join(f"| {g} | {v['n']} | {v['second']} | {v['first']} | {v['plus']} | {v['minus']} |" for g,v in info['held'].items()),
        '## 每轮新房间与复访',
        '沿用固定 72 束、局部坐标中心、split_precision 执行器、seed17 初始蒸馏 actor、既有停滞触发和 300 步 Adam 配对偏好更新。每轮引入 200 个新几何房间，同一母房间的单专家筛查只做一次，取全部独有成功、20 个双胜、8 个双败。第二轮起在过去访问过的房间中额外选 32 个，优先复查观测到的成功丢失，其次为尚有候选点的独有成功难例。每次按新版 actor 的实际分歧排序分成至多三段，各抽一个，优先同段未测过的前缀与反事实命令。每轮只用本轮反事实标签拟合，沿用同轮全体可路由边界的软概率保持项。没有运行固定房间训练对照。',
        '| 轮次 | 首访 | 复访 | 实际分叉 | 有效偏好 | 其中改选可救回 | 提供信号的房间 | 每万训练 ticks 的信号 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
        +'\n'.join(f"| {r['round']} | {r['new']} | {r['revisits']} | {r['forks']} | {r['informative']} | {r['rescue']} | {r['signal_rooms']} | {r['signals_per_10k_ticks']:.3f} |" for r in rounds),
        f"共筛查 800 个不同训练房间，在其中 {info['distinct_train_rooms']} 间实际安排母轨迹采样。{forks} 次反事实执行提供 {signals} 条终局有差异的监督，来自 {info['distinct_signal_rooms']} 个不同房间。训练采样共 {total:,} ticks，计入全部单专家筛查、母轨迹、复查及每条分叉从初态开始的真实前缀重放。逐轮诊断与最终确认用量另计。",
        '| 采样来源 | 访问次数 | 分叉数 | 有效偏好 | 每万 ticks 信号 | 第 2 或 3 点才首次见信号的访问 |\n|---|---:|---:|---:|---:|---:|']
    for group in ('new','revisit'):
        selected=[r for r in cohorts if r['cohort']==group]
        s=lambda field:sum(r[field] for r in selected)
        lines[-1]+=f"\n| {group} | {s('visits')} | {s('forks')} | {s('informative')} | {10000*s('informative')/s('ticks'):.3f} | {s('first_signal_2')+s('first_signal_3')} |"
    lines+=['首访用量包含该轮全部 200 个房间的 A/B 筛查；复访用量包含成功保留检查。复访优先选择了疑难房间，两组的单位成本差异是本轮调度下的描述性结果。',
            '## 并行执行及多专家扩展',
            f"24 个单线程进程按房间与专家、或按母轨迹与反事实命令领取任务。1,600 次开发单专家筛查耗时 {info['screen_wall']/60:.2f} 分钟，进程 CPU 时间折合平均 {info['screen_avg_cores']:.1f} 核。实际分叉阶段耗时 {info['fork_wall']/60:.2f} 分钟，平均 {info['fork_avg_cores']:.1f} 核。全部仿真进程池阶段共 {info['simulation_pool_wall']/60:.1f} 分钟。",
            '单专家结果按实际源码哈希与内容房间哈希寻址，整个运行同时锁定感知与执行器版本。分叉任务在同一母前缀、同一后续 actor 下，按允许比较的邻接表展开，目标、两个动作编号和时长均相同的请求合并。使用五个模拟提案验证了星形邻接及去重；此次实测路由器和特征仍是二专家 A/B。不同后续策略的终局标签需要分别采样。当前仍需为每个替代动作重放物理前缀。',
            '[固定评估 64 间](diagnostic64.csv) · [逐轮成本](rounds.csv) · [首访与复访](cohorts.csv) · [数据分层](strata.csv) · [逐房间访问](visits.csv) · [独立面板结果](held.csv) · [采样器](../../../../tools/cap_expanding_pool.py)']
    (DOC/'README.md').write_text('\n\n'.join(lines)+'\n')


def main():
    lock=read(OUT/'lock.json')
    protocol=read(OUT/'protocol.json')
    assert lock['final']==x.d.source.graph.file_sha(OUT/'actor4.npz')
    DOC.mkdir(parents=True,exist_ok=True)
    diag=[]
    for stage in range(5):
        files=sorted((OUT/'diagnostic'/f's{stage}').glob('*.json'))
        assert len(files)==64
        diag.append(sum(read(f)['success'] for f in files))
    previous=x.d.rooms(x.trend.FRESH)
    names={x.e.s.phase.input_hash(f):label for label,f in previous.items()}
    ids=x.shuffled(names,'diagnostic64')[:64]
    assert set(ids).isdisjoint(k for keys in protocol['ids'].values() for k in keys)
    for stage in range(5):
        assert set(ids)=={f.stem for f in (OUT/'diagnostic'/f's{stage}').glob('*.json')}
    write_table('diagnostic64.csv',[dict(previous_room=names[k],room_hash=k,**{
        f'round{s}':read(OUT/'diagnostic'/f's{s}'/(k+'.json'))['success'] for s in range(5)}) for k in ids])
    rows=[];visits=[];groups=[];cohorts=[];strata=Counter()
    for stage in range(1,5):
        root=x.stage_dir(stage)
        select=read(root/'selection.json')
        seen=read(root/'visits.json')
        measured=read(OUT/f'fit{stage}.json')
        fresh=protocol['ids'][str(stage)]
        sc=[read(OUT/'screen'/protocol['profile'][arm]/(k+'.json')) for k in fresh for arm in x.ARMS]
        for k in fresh:
            a,b=[read(OUT/'screen'/protocol['profile'][arm]/(k+'.json'))['success'] for arm in x.ARMS]
            strata[('neither','B_only','A_only','both')[2*a+b]]+=1
        parents=[read(f) for f in (root/'parents').glob('*.json')]
        branches=[read(f) for f in (root/'branches').glob('*.json')]
        ticks=lambda records:sum(r['ticks'] for r in records)
        total=ticks(sc)+ticks(parents)+ticks(branches)
        pairs=sum(len(v['sampled']) for v in seen)
        assert pairs==len(branches)==measured['forks']
        rows.append(dict(round=stage,new=len(select['new']),revisits=len(select['repeat']),checked_successes=len(select['sentinels']),
            detected_loss=len(select['lost']),candidates=sum(v['candidates'] for v in seen),forks=pairs,informative=measured['informative'],
            rescue=measured['rescue'],protect=measured['protect'],ties=measured['ties'],signal_rooms=sum(bool(v['informative']) for v in seen),
            screen_ticks=ticks(sc),parent_ticks=ticks(parents),branch_ticks=ticks(branches),total_ticks=total,
            signals_per_10k_ticks=round(10000*measured['informative']/total,3),retested=sum(v['repeat_prefix'] for v in seen),
            diagnostic_success=diag[stage]))
        for label,fresh_visit in [('new',True),('revisit',False)]:
            selected=[v for v in seen if v['fresh']==fresh_visit]
            parent_ids={v['room'] for v in selected}
            if not fresh_visit:
                parent_ids.update(select['sentinels'])
            cohort_ticks=ticks(sc) if fresh_visit else 0
            cohort_ticks+=sum(read(root/'parents'/(k+'.json'))['ticks'] for k in parent_ids)
            first_hits=Counter()
            coverage=0
            for v in selected:
                k=v['room']
                with x.np.load(root/'fit_data'/(k+'.npz')) as data:
                    flipped=data['alternate']!=data['baseline']
                    if flipped.any():
                        first_hits[int(x.np.argmax(flipped))]+=1
                    cohort_ticks+=sum(read(root/'branches'/f'{k}_{int(i)}.json')['ticks'] for i in data['index'])
                    coverage+=len(flipped)
            signal=sum(v['informative'] for v in selected)
            cohorts.append(dict(round=stage,cohort=label,visits=len(selected),forks=coverage,informative=signal,
                                signal_rooms=sum(first_hits.values()),first_signal_1=first_hits[0],
                                first_signal_2=first_hits[1],first_signal_3=first_hits[2],ticks=cohort_ticks,
                                signals_per_10k=round(10000*signal/cohort_ticks,3) if cohort_ticks else 0))
        assert sum(r['ticks'] for r in cohorts[-2:])==total
        for v in seen:
            visits.append(dict(stage=stage,**{k:v[k] for k in ('room','stratum','fresh','success','candidates','remaining','repeat_prefix','informative','rescue')},forks=len(v['sampled'])))
    write_table('cohorts.csv',cohorts)
    write_table('rounds.csv',rows)
    write_table('visits.csv',visits)
    for group in ('A_only','B_only','both','neither'):
        v=[r for r in visits if r['stratum']==group]
        groups.append(dict(stratum=group,screened=strata[group],rooms=len({r['room'] for r in v}),visits=len(v),
            branches=sum(r['forks'] for r in v),signals=sum(r['informative'] for r in v),signal_rooms=len({r['room'] for r in v if r['informative']})))
    write_table('strata.csv',groups)
    scores=[]
    for k in lock['held']:
        a,b=[read(OUT/'screen'/protocol['profile'][arm]/(k+'.json'))['success'] for arm in x.ARMS]
        scores.append(dict(room=k,A=a,B=b,stratum=('neither','B_only','A_only','both')[2*a+b],
                           initial=read(OUT/'held/s0'/(k+'.json'))['success'],final=read(OUT/'held/s4'/(k+'.json'))['success']))
    assert len(scores)==200
    write_table('held.csv',scores)
    stats={g:compare([r for r in scores if g=='all' or r['stratum']==g]) for g in ('all','A_only','B_only','both','neither')}
    timing=[read(f) for f in (OUT/'timing').glob('*.json')]
    timescreen=read(OUT/'timing/screen.json')['wall']
    cpuscreen=sum(read(OUT/'screen'/protocol['profile'][a]/(k+'.json'))['cpu'] for s in range(1,5) for k in protocol['ids'][str(s)] for a in x.ARMS)
    cpu_fork=sum(read(f)['cpu'] for f in OUT.glob('round*/branches/*.json'))
    wall_fork=sum(read(OUT/f'timing/r{i}_fork.json')['wall'] for i in range(1,5))
    info=dict(held=stats,diagnostic=diag,rounds=rows,strata=groups,workers=timing[0]['workers'],screen_wall=timescreen,screen_avg_cores=cpuscreen/timescreen,
              fork_wall=wall_fork,fork_avg_cores=cpu_fork/wall_fork,simulation_pool_wall=sum(r['wall'] for r in timing),
              distinct_train_rooms=len({r['room'] for r in visits}),distinct_signal_rooms=len({r['room'] for r in visits if r['informative']}))
    render(info,cohorts,scores)
    check=OUT/'train_check.json'
    if check.is_file():
        info['training_endpoints']=read(check)
        t=info['training_endpoints']
        with (DOC/'README.md').open('a') as report:
            report.write(f"\n## 训练房间的首尾闭环\n\n在实际访问的同一批 {t['rooms']} 个训练母房间上，初始成功 {t['initial']} 个，最终成功 {t['final']} 个，救回 {t['gain']} 个，丢失 {t['loss']} 个。这里只补做闭环评估，没有训练固定房间对照模型。[逐房间结果](train_check.csv)\n")
    x.e.save(OUT/'FINISHED.json',info)
    print(json.dumps(info,indent=2))


if __name__=='__main__':
    main()
