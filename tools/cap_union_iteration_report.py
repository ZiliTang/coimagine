"""Report paired terminal rewards and retention inside and outside the CAP union."""
import csv
import json
from tools import cap_union_iteration as p

e=p.e


def compare(rows,left,right):
    return p.d.paired([r[left] for r in rows],[r[right] for r in rows])


def summary(rows):
    return dict(n=len(rows),counts={a:sum(r[a] for r in rows) for a in ('A','B','teacher','original','final')},
                paired=compare(rows,'final','original'))


def difference(q):
    d=(q['first']-q['second'])/q['n']*100
    return f"{d:+.1f} [{q['ci'][0]*100:+.1f}, {q['ci'][1]*100:+.1f}]"


def publish():
    ids=e.read(p.OUT/'fresh_ids.json')
    rows=[dict(room=k,sha=v,**e.read(p.OUT/'test'/(k+'.json'))) for k,v in sorted(ids.items())]
    assert len(rows)==len(set(ids.values()))==200
    groups={'全部':rows,'专家并集':[r for r in rows if r['A'] or r['B']],
            '并集外':[r for r in rows if not(r['A'] or r['B'])],
            '仅 A 成功':[r for r in rows if r['A']>r['B']],
            '仅 B 成功':[r for r in rows if r['B']>r['A']],
            '两者均成功':[r for r in rows if r['A'] and r['B']]}
    stats={k:summary(v) for k,v in groups.items()}
    with (p.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    union=e.read(p.OUT/'union.json')
    dev=[]
    import numpy as np
    for k in union:
        row={}
        for i in p.STAGES:
            with np.load(p.OUT/f'round{i}'/(k+'.npz')) as q:
                row['initial' if i==1 else 'round1']=int(q['baseline'])
        row['final']=e.read(p.OUT/'dev_eval'/(k+'.json'))
        dev.append(row)
    devsumm={a:sum(r[a] for r in dev) for a in ('initial','round1','final')}
    fits=[e.read(p.OUT/f'fit{i}.json') for i in p.STAGES]
    full=stats['全部']
    result=dict(groups=stats,fits=fits,development=devsumm,development_pairs=compare(dev,'final','initial'),
                vs_teacher=compare(rows,'final','teacher'),vs_A=compare(rows,'final','A'),vs_B=compare(rows,'final','B'),
                union_size=len(union),rooms=len(rows))
    e.save(p.OUT/'FINISHED.json',result)
    lines=[]
    for k,s in stats.items():
        c=s['counts'];q=s['paired']
        lines.append(f"| {k} | {s['n']} | {c['original']} | {c['final']} | {q['plus']} / {q['minus']} | {difference(q)} |")
    info='\n'.join(lines)
    forks='\n'.join(f"| {i} | {v['forks']} | {v['rescue']} | {v['protect']} | {v['ties']} | {v['rooms_with_rescue']} | {v['fit_correct']}/{v['informative']} |" for i,v in enumerate(fits,1))
    c=full['counts'];q=full['paired']
    report=f"""# 在两位专家的开发成功并集上做策略迭代

从留出分歧动作一致率 90.6% 的精简语义 actor 开始。在两个既有开发池的 400 个房间中，用冻结 A、B 独立执行的成败筛出并集 {len(union)} 个。两个更新轮次均使用整个开发并集。actor 特征、公开触发窗口、候选动作、执行器及两位专家维持原配置。训练时可以按房间结果选样，部署时看不到并集标签。

## 闭环结果

200 个全新自然房间，同场逐例执行。单 CAP A {c['A']}、B {c['B']}，冻结教师 {c['teacher']}，初始 actor {c['original']}，第二轮 actor {c['final']}。相对初始 actor 的双侧精确 McNemar p={q['mcnemar']:.3f}。各区间为按房间配对重采样 4,000 次的差值区间。主比较为全部房间第二轮对初始 actor，亚组为诊断。

| 测试范围 | 房间 | 初始成功 | 第二轮成功 | 救回 / 丢失 | 净差与 95% 区间（百分点） |
|---|---:|---:|---:|---:|---:|
{info}

同一开发并集上的闭环成功数从 {devsumm['initial']} 到 {devsumm['round1']} 再到 {devsumm['final']}，各自分母为 {len(union)}。每轮均从前一版 checkpoint 出发；第二轮预先指定为最终版本。

## 分叉与更新

| 轮次 | 同前缀配对 | 改选可救回 | 保留原选才成功 | 同结局 | 至少一处可救回的房间 | 更新后偏好拟合数 |
|---|---:|---:|---:|---:|---:|---:|
{forks}

每房间每轮沿当前 actor 采样，在真实有分歧的命令中按早、中、晚三段各抽至多一处，真实前缀、动作、输入特征逐项校验。原轨迹给出原选动作后继续执行当前 actor 的终局成败，另一条只在选定边界换一次动作，之后仍由同一当前 actor 贪心执行。两条分支的后续观测与动作可自然分化。同一房间可能贡献多个配对，每个配对均有独立的单步干预。

仅在终局成败不同时，以成功的 A/B 动作为二元监督。每轮从当前完整权重热启动，Adam 步长 0.0003，300 步，每步采样 64 个可分胜负的前缀，并在 128 个本轮并集轨迹的分歧状态上拟合更新前 actor 的软概率；两项二元交叉熵各占权重 1。奖励只用终局 0/1。未额外查询教师动作。两轮均使用 seed17 的同一蒸馏起点。开发成功数和分叉偏好拟合率都不能替代新自然房间的闭环结果。

[逐房间结果](rooms.csv)
"""
    (p.DOC/'README.md').write_text(report)
    registry=p.DOC.parents[1]/'experiment_journal/entries.json'
    entrylist=e.read(registry)
    entry=next(r for r in entrylist if r['id']=='union-iteration')
    entry.update(result=f"开发并集 {devsumm['initial']}→{devsumm['round1']}→{devsumm['final']}/{len(union)}。新自然房间 {c['original']}→{c['final']}/200。",analysis=f"对初始 actor 救回 {q['plus']} 个、丢失 {q['minus']} 个。")
    registry.write_text(json.dumps(entrylist,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    publish()
