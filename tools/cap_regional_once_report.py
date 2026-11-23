"""Pair final outcomes, distinguish sampled opportunities from deployable decisions."""
import csv
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools import cap_regional_once as exp


def write_rows(name, rows):
    with (exp.DOCS/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def describe(rows):
    ds=[b['delta'] for r in rows for b in r['branches']]
    return dict(rooms=len(rows),baseline=sum(r['baseline']['success'] for r in rows),
                eligible=sum(bool(r['branches']) for r in rows),branches=len(ds),wins=ds.count(1),
                losses=ds.count(-1),equal=ds.count(0),
                rescued_rooms=sum(any(b['delta']==1 for b in r['branches']) for r in rows),
                lost_rooms=sum(any(b['delta']==-1 for b in r['branches']) for r in rows))


def report(rows, acceptance, progress):
    read,save=exp.study.read,exp.study.save
    exp.DOCS.mkdir(parents=True,exist_ok=True)
    if progress.get('fit_curve'):
        save(exp.OUT/'fit_curve.json',progress['fit_curve'])
    elif (exp.OUT/'fit_curve.json').is_file():
        progress['fit_curve']=read(exp.OUT/'fit_curve.json')
    rooms=read(exp.OUT/'cases.json')
    frozen=read(exp.OUT/'frozen.json')
    protocol=read(exp.OUT/'protocol.json')
    assert len(rows)==200 and len(acceptance)==200 and len(rooms)==400
    assert len({r['cid'] for r in rows+acceptance})==400
    assert len({r['hash'] for r in rooms.values()})==400
    assert frozen['model_sha']==exp.study.sha(exp.OUT/'override.pt')
    forks=[];per_room=[]
    for r in rows:
        b=r['baseline']
        for f in r['branches']:
            assert f['delta']==f['outcome']['success']-b['success']
            assert f['outcome']['override']==f['step'] and f['base_action']!=f['action']
            forks.append(dict(case_id=r['cid'],split=r['split'],step=f['step'],prefix_sha256=f['prefix'],
                              original_action=f['base_action'],alternative=f['action'],before=b['success'],
                              after=f['outcome']['success'],delta=f['delta'],branch_history=f['outcome']['history']))
        per_room.append(dict(case_id=r['cid'],split=r['split'],objects=rooms[r['cid']]['objects'],
                             baseline=b['success'],branches=len(r['branches']),
                             rescuable=int(any(f['delta']>0 for f in r['branches'])),
                             vulnerable=int(any(f['delta']<0 for f in r['branches']))))
    write_rows('per_branch.csv',forks);write_rows('development_rooms.csv',per_room)
    groups={s:describe([r for r in rows if r['split']==s]) for s in ('fit','tune')}
    groups['all']=describe(rows)
    pairs=[]
    for r in acceptance:
        b,c=r['baseline'],r['candidate']
        if c['override'] is None:assert b==c
        pairs.append(dict(case_id=r['cid'],objects=rooms[r['cid']]['objects'],baseline=b['success'],
                          candidate=c['success'],delta=c['success']-b['success'],step=c['override'],
                          original_ticks=b['ticks'],candidate_ticks=c['ticks'],
                          original_history=b['history'],candidate_history=c['history']))
    write_rows('acceptance_pairs.csv',pairs)
    n=len(pairs);ds=[r['delta'] for r in pairs]
    wins,losses=ds.count(1),ds.count(-1)
    lower=exp.lower_bound(wins,losses,n)
    before=sum(r['baseline'] for r in pairs)
    after=sum(r['candidate'] for r in pairs)
    overrides=sum(r['step'] is not None for r in pairs)
    result=dict(n=n,baseline=before,candidate=after,rescued=wins,lost=losses,overrides=overrides,
                lower=lower,net=(after-before)/n,accepted=lower>0,
                deployed='override' if lower>0 else 'frozen_supervised_actor')
    assert result['rescued']==progress['accept']['wins'] and result['lost']==progress['accept']['losses']
    assert sum(r['branches'] for r in per_room)==progress['collect']['branches']
    # Equal room weighting matches the loss used in fitting. A constant prior
    # quantifies holdout cross entropy without using any acceptance labels.
    def class_frequencies(group):
        subset=[r for r in rows if r['split']==group and r['branches']]
        return [sum(sum(b['delta']==label for b in r['branches'])/len(r['branches']) for r in subset)/len(subset)
                for label in (-1,0,1)]
    train_prior=class_frequencies('fit');valid_prior=class_frequencies('tune')
    constant_ce=-sum(p*math.log(q) for p,q in zip(valid_prior,train_prior) if p>0)
    tune=frozen['tuning']
    stats=dict(result=result,opportunities=groups,fit_curve=progress.get('fit_curve',[]),
               protocol=protocol,frozen=frozen,tune_constant_cross_entropy=constant_ce)
    save(exp.DOCS/'results.json',stats)
    fig, axes=plt.subplots(1,3,figsize=(12,3.6),layout='constrained')
    for ax,key,labels in zip(axes,['collect_curve','fit_curve','accept_curve'],
            [('Completed development rooms','Alternate suffixes','Rescued','Lost'),
             ('Training updates','Cross entropy','Fit','Tuning'),
             ('Completed acceptance pairs','Rooms','Rescued','Lost')]):
        curve=progress.get(key,[])
        if curve:
            x,y,z=zip(*curve)
            ax.plot(x,y,label=labels[2],color='#007e74')
            ax.plot(x,z,label=labels[3],color='#aa6138')
        ax.set(xlabel=labels[0],ylabel=labels[1]);ax.grid(alpha=.2);ax.legend()
    fig.savefig(exp.DOCS/'progress.png',dpi=145);plt.close(fig)
    g=groups['all']
    lines=['# 十维监督 actor 的单次改选实验','',
           f"冻结的改选候选在 {n} 个独立新房间中成功 {after} 次，原 actor 成功 {before} 次。实际触发 {overrides} 次，救回 {wins} 局，丢失 {losses} 局。成功率净差 {100*result['net']:+.1f} 个百分点，单侧 95% 保守下界 {100*lower:+.2f} 个百分点。按事先冻结的规则，{'接受改选模块' if result['accepted'] else '保留原监督 actor'}。",'',
           '## 分叉机会','',
           '从 seed17、24,000 步的原监督 actor 采集前缀。每个房间在第 0–5、6–23、24–79 步各取第一个存在不同合法动作的边界（从 0 编号）。按 actor 的 CAP 排名取最多两个不同于首选的物理动作。重放相同前缀并比对公开历史哈希，在该处改一步，再由原 actor 贪心执行到底。未触发分叉的房间保留在分母中。','',
           '| 用途 | 房间 | 含分叉 | 后缀 | 救回 / 破坏 / 同结局后缀 | 可救回的原失败房间 | 可破坏的原成功房间 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for k,title in [('fit','训练'),('tune','选阈值'),('all','合计')]:
        v=groups[k]
        lines.append(f"| {title} | {v['rooms']} | {v['eligible']} | {v['branches']} | {v['wins']} / {v['losses']} / {v['equal']} | {v['rescued_rooms']}/{v['rooms']-v['baseline']} | {v['lost_rooms']}/{v['baseline']} |")
    lines+=['',f"在测过的分叉中，事后知道结局再决定是否改选，可把开发池成功数从 {g['baseline']}/{g['rooms']} 提高到 {g['baseline']+g['rescued_rooms']}/{g['rooms']}。这个数刻画本次采样发现的机会，部署时无法读取分叉终局。",'',
            '## 如何选出一次改选','',
            '十个 C++ CAP 与原 actor 全部冻结。小网络读取公开历史经原 actor 编码后的隐藏向量、原动作与替代动作的显式参数和几何特征、提案来源、logit 差及决策步数。三分类标签为破坏、同结局、救回。每个有分叉的训练房间权重相同。训练 120 步，用预测救回概率减去预测破坏概率作为分数。在每个符合条件的边界取最高分，超过阈值即执行一次，之后全部交回原 actor。', '',
            '| 阈值 | 调参房间触发 | 救回 | 丢失 | 净增 |','|---:|---:|---:|---:|---:|']
    for s in tune:lines.append(f"| {s['threshold']:.2f} | {s['overrides']} | {s['rescued']} | {s['lost']} | {s['net']:+d} |")
    lines+=['',f"在 40 个调参房间中，取净增最多者，平局优先改选次数少、阈值高者。选定 {frozen['threshold']:.2f}。同一前缀已经枚举部署规则允许的替代动作，调参按最先触发的实测后缀结算。验收使用候选与原 actor 各自的完整物理闭环。",'',
            (f"末次训练交叉熵与调参交叉熵为 {progress['fit_curve'][-1][1]:.3f} 和 {progress['fit_curve'][-1][2]:.3f}。" if progress.get('fit_curve') else '') + f"在同一批调参后缀上，总是输出训练房间的类别频率会得到 {constant_ce:.3f} 的交叉熵。损失按房间等权。",'',
            '![采样、拟合与验收过程](progress.png)','',
            '## 验收范围','',
            '训练、调参、验收按房间内容去重。单独抽取的验收房间按 2 至 5 个物体等概率混合生成，未用结局筛选。一次验收一个冻结候选。以结局不同的概率乘以其中救回相对于丢失的净比例计算净差；对前者构造精确二项 97.5% 双侧区间，对条件救回率构造精确 97.5% 单侧下界，使用 Bonferroni 合并。只在净差下界严格为正时采用候选。这个界限针对同分布的平均成功率。', '',
            '[逐分叉记录](per_branch.csv) · [开发房间](development_rooms.csv) · [验收逐场景](acceptance_pairs.csv) · [统计与冻结协议](results.json)']
    (exp.DOCS/'README.md').write_text('\n'.join(lines)+'\n')
    save(exp.OUT/'FINISHED.json',result)
    return result
