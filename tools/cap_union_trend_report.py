"""Plot round-wise accumulation and retention, and report the fixed fresh comparison."""
import csv
import json
import numpy as np
from tools import cap_union_trend as t

e=t.e


def collect():
    rows=[]
    for k in e.read(t.PARENT/'union.json'):
        row=dict(panel='development',room=k,**e.read(t.PARENT/'screen'/(k+'.json')))
        for each in range(1,7):
            path=t.PARENT if each<=2 else t.OUT
            with np.load(path/f'round{each}'/(k+'.npz')) as data:
                row[f'r{each-1}']=int(data['baseline'])
        assert row['r2']==e.read(t.PARENT/'dev_eval'/(k+'.json'))
        row['r6']=e.read(t.OUT/'dev_eval'/(k+'.json'))
        rows.append(row)
    for panel,idpath in [('diagnostic',t.PARENT/'fresh_ids.json'),('confirmation',t.OUT/'confirmation_ids.json')]:
        for k in sorted(e.read(idpath)):
            rows.append(dict(panel=panel,room=k,**e.read(t.OUT/panel/(k+'.json'))))
    return rows


def by_group(rows,stages):
    groups={'all':rows,'A_only':[r for r in rows if r['A']>r['B']],
            'B_only':[r for r in rows if r['B']>r['A']],
            'both':[r for r in rows if r['A'] and r['B']],
            'neither':[r for r in rows if not(r['A'] or r['B'])]}
    return {g:dict(n=len(sub),wins=[sum(r[f'r{i}'] for r in sub) for i in stages]) for g,sub in groups.items()}


def retention(rows):
    result=[]
    for i in range(1,7):
        gain=[r for r in rows if r[f'r{i}'] and not r[f'r{i-1}']]
        result.append(dict(round=i,n=len(gain),retained=[sum(r[f'r{j}'] for r in gain) for j in range(i,7)],
                           continuous=[sum(all(r[f'r{k}'] for k in range(i,j+1)) for r in gain) for j in range(i,7)]))
    return result


def plot(groups,cohorts):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    for g,color in [('development','#156660'),('diagnostic','#a3572a')]:
        entry=groups[g]['all']
        axes[0,0].plot(range(7),np.array(entry['wins'])/entry['n']*100,'o-',label=f"{g} (n={entry['n']})",color=color)
    axes[0,0].set_title('Closed-loop success'); axes[0,0].legend()
    for ax,panel in [(axes[0,1],'development'),(axes[1,0],'diagnostic')]:
        for group in ('A_only','B_only'):
            entry=groups[panel][group]
            ax.plot(range(7),np.array(entry['wins'])/entry['n']*100,'o-',label=f"{group.replace('_',' ')} (n={entry['n']})")
        ax.set_title(panel.capitalize()+' exclusive successes');ax.legend()
    ax=axes[1,1]
    for item in cohorts['diagnostic'][:-1]:
        i=item['round']
        if item['n']:
            ax.plot(range(i,7),np.array(item['retained'])/item['n']*100,'o-',label=f"R{i} (n={item['n']})")
    ax.set_title('Diagnostic rooms newly rescued in a round');ax.legend(fontsize=8,ncol=2)
    for ax in axes.flat:
        ax.set_xlabel('Update round');ax.set_ylabel('Rooms successful (%)');ax.set_xticks(range(7));ax.grid(alpha=.2)
    fig.savefig(t.DOC/'curves.png',dpi=150)
    plt.close(fig)


def symmetry():
    raw=e.read(t.OUT/'symmetry/summary.json')
    transforms=raw['transforms']
    rooms=[e.read(t.OUT/'symmetry/rooms'/(k+'.json')) for k in raw['parents']]
    changed=sum(len({tuple(v['outcomes'][a] for a in ('A','B')) for v in r.values()})>1 for r in rooms)
    inverse=sum(r['original']['outcomes']['A']!=r['original']['outcomes']['B'] and
                r['halfturn']['outcomes']['A']==r['original']['outcomes']['B'] and
                r['halfturn']['outcomes']['B']==r['original']['outcomes']['A'] for r in rooms)
    lines=[]
    for name in ('halfturn','mirror_x','mirror_y'):
        v=transforms[name]
        lines.append(f"| {name} | {v['first_agreement']['A']}/32 | {v['A']['gained']} / {v['A']['lost']} | {v['first_agreement']['B']}/32 | {v['B']['gained']} / {v['B']['lost']} | {v['union']['gained']} / {v['union']['lost']} |")
    body='\n'.join(lines)
    (t.DOC/'symmetry.md').write_text(f"""# 同一房间的旋转与镜像

从原 400 个开发房间按预设哈希取 32 个母场景，覆盖成功与失败。分别旋转 180°，或沿房间中线左右、上下镜像。保持矩形房间宽高不变，变换两台机器人、目标物及可动物体、所有静态障碍、目标点的位置与朝向。重新仿真，CAP 读取新传感量并自由决策。原场景的 A、B 独立成败来自同一执行器的冻结开发记录。三组变体共享母房间。

| 变换 | A 首个完整提案映射相符 | A 成败 新增 / 丢失 | B 首个完整提案映射相符 | B 成败 新增 / 丢失 | 专家并集 新增 / 丢失 |
|---|---:|---:|---:|---:|---:|
{body}

halfturn 为绕房间中心旋转 180°，mirror_x 和 mirror_y 分别翻转横、纵坐标。初始动作比较同时核对目标、时长及两机器人方向编码（同义动作归一）。镜像交换顺、逆时针的旋转和绕行指令。首步一致只说明初始动作能映射，后续接触、选路和路由标签仍需验证。三种变换合计，有 {changed}/32 个母房间至少一次改变了 A、B 的成败组合。仅看 180° 旋转，就有 {inverse} 个房间从单独一位专家成功变成仅另一位专家成功。该诊断检验当前两个 CAP 的执行结果，不能单独归因于物理数值或 CAP 程序。建议以母房间为单位划分数据，增广房间用实际执行结果重新标注。

本轮 3–6 的训练数据全部来自原始 271 个房间，变体单独保存。
""")


def publish():
    t.DOC.mkdir(exist_ok=True,parents=True)
    rows=collect()
    with (t.DOC/'rooms.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['panel','room','A','B',*(f'r{i}' for i in range(7))]);w.writeheader();w.writerows(rows)
    sets={p:[r for r in rows if r['panel']==p] for p in ('development','diagnostic','confirmation')}
    assert [len(s) for s in sets.values()]==[271,200,200]
    groups={p:by_group(s,range(7) if p!='confirmation' else [0,2,6]) for p,s in sets.items()}
    cohorts={p:retention(s) for p,s in sets.items() if p!='confirmation'}
    fresh=sets['confirmation']
    def compare(sub,start):
        return t.d.paired([r['r6'] for r in sub],[r[f'r{start}'] for r in sub])
    pairs={k:compare(s,2) for k,s in sets.items()}
    pairs['new_vs_0']=compare(fresh,0)
    plot(groups,cohorts)
    symmetry()
    def nums(a):return ' → '.join(map(str,a))
    diag=groups['diagnostic']
    primary=pairs['confirmation']
    lo,hi=primary['ci']
    d2=cohorts['diagnostic'][1]
    table='\n'.join(f"| {g} | {s['n']} | "+' | '.join(map(str,s['wins']))+' |' for g,s in groups['confirmation'].items())
    report=f"""# 在相同开发并集上继续到第 6 轮

从冻结第 2 轮接续，训练房间、抽样规则、每轮 300 次更新、优化器、保留损失与仿真预算沿用此前定义。新增四轮预先固定，独立确认的主比较是第 6 轮相对第 2 轮。模型每轮只用当轮配对数据。从特定的一份 seed17 蒸馏模型开始。

| 面板 | 房间 | 第 0 至 6 轮成功数 |
|---|---:|---|
| 开发成功并集 | 271 | {nums(groups['development']['all']['wins'])} |
| 已查看的自然诊断房间 | 200 | {nums(diag['all']['wins'])} |
| 诊断中仅 A 成功 | {diag['A_only']['n']} | {nums(diag['A_only']['wins'])} |
| 诊断中仅 B 成功 | {diag['B_only']['n']} | {nums(diag['B_only']['wins'])} |

诊断面板第 2 轮相对第 1 轮新救回 {d2['n']} 个，之后在第 2 至 6 轮各有 {nums(d2['retained'])} 个成功。从第 2 轮开始一直到第 6 轮每一版都成功的有 {d2['continuous'][-1]} 个。图中按相邻轮的失败转成功划分集合，同一房间可在不同轮次再次被救回。

![各轮成功率、独有成功保留与逐轮救回集合表现](curves.png)

## 全新 200 房间

| 分组 | 房间 | 第 0 轮 | 第 2 轮 | 第 6 轮 |
|---|---:|---:|---:|---:|
{table}

第 6 轮相对第 2 轮救回 {primary['plus']} 个，丢失 {primary['minus']} 个；净差 {(primary['plus']-primary['minus'])/2:+.1f} 个百分点，配对 95% 区间 [{100*lo:+.1f}, {100*hi:+.1f}]，双侧 McNemar p={primary['mcnemar']:.3f}。区间按房间配对重采样 4,000 次。两位专家在同场从头独立执行，用于标记能力分组；预先冻结的三个 actor 均实际执行完整闭环。

[逐房间结果](rooms.csv) · [旋转与镜像诊断](symmetry.md) · [翻转机制与干预](mechanism.md)
"""
    (t.DOC/'README.md').write_text(report)
    e.save(t.OUT/'FINISHED.json',dict(groups=groups,cohorts=cohorts,pairs=pairs))
    registry=t.DOC.parents[1]/'experiment_journal/entries.json'
    entries=e.read(registry)
    item=next(x for x in entries if x['id']=='union-trend')
    item.update(result=f"开发 {nums(groups['development']['all']['wins'])}/271。独立面板第 0、2、6 轮 {nums(groups['confirmation']['all']['wins'])}/200。",analysis=f"相对第 2 轮独立确认救回 {primary['plus']} 个、丢失 {primary['minus']} 个。")
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':
    publish()
