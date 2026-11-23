"""Pair the ten-logit actor with frozen hard action-actor and single-CAP runs."""
import csv
from collections import Counter


from tools import cap_regional_id_actor as run

study=run.study


def read_rows(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def records(sids):
    hard=run.hard
    baseline={r['case_id']:r for r in read_rows(hard.DOCS/'per_scene.csv')+read_rows(hard.DOCS/'held_per_scene.csv')}
    teachers=study.read(hard.OUT/'teachers.json')
    for r in read_rows(study.DOCS/'per_scene.csv'):
        cid,panel=r['case_id'],r['panel']
        ref=baseline[cid]
        train=panel in ('dev','aug')
        n=sum(int(r[s]) for s in sids)
        assert n==int(ref['ncap'])
        teacher=teachers.get(cid,{}).get('sid','') if train else ref['fastest']
        row=dict(case_id=cid,panel=panel,parent=r['parent'],objects=int(r['objects']),ncap=n,
                 fastest=teacher,best=int(r['best']),directq=int(r['directq']))
        for key,hardkey in [('fixed','actor' if train else 'fixed'),('continued','polished' if train else 'polish')]:
            result=study.read(run.OUT/key/'evaluation'/panel/(cid+'.json'))
            assert result['case_id']==cid and result['checkpoint']==key and 0<result['ticks']<=1800 and 0<result['boundaries']<=300
            sid=sids[result['initial_cap']]
            row['action_'+key]=int(ref[hardkey])
            row['id_'+key]=int(result['success'])
            row['initial_'+key]=sid
            row['commit_'+key]=int(r[sid])
            row['switches_'+key]=result['switches']
            row['distinct_switches_'+key]=result['distinct_switches']
            row['length_'+key]=result['boundaries']
            row['fallback_'+key]=result['fallback']
            row['history_'+key]=result['history']
            if not result['switches'] and not result['fallback']:
                assert int(result['success'])==int(r[sid])
            if teacher and train:
                row['exact_'+key]=int(result['history']==teachers[cid]['history'])
        yield row


def count(group,key):
    return dict(success=sum(r[key] for r in group), retained=sum(r[key] for r in group if r['ncap']),
                single=sum(r[key] for r in group if r['ncap']==1),
                outside=sum(r[key] for r in group if not r['ncap']))


def summarize(group):
    stats=dict(n=len(group),union=sum(bool(r['ncap']) for r in group),single=sum(r['ncap']==1 for r in group))
    for key in ['fixed','continued']:
        for method in ('action','id','commit'):
            name=method+'_'+key
            stats[name]=count(group,name)
        commit='commit_'+key
        actor='id_'+key
        stats['routing_'+key]=dict(episodes_with_switch=sum(r['switches_'+key]>0 for r in group),
            episodes_with_distinct_switch=sum(r['distinct_switches_'+key]>0 for r in group),
            switches=sum(r['switches_'+key] for r in group), distinct_switches=sum(r['distinct_switches_'+key] for r in group),
            boundaries=sum(r['length_'+key] for r in group),fall_back_steps=sum(r['fallback_'+key] for r in group),
            initial_experts=dict(Counter(r['initial_'+key] for r in group)),
            initial_fastest=sum(r['fastest']==r['initial_'+key] for r in group),
            rescued=sum(r[actor]>r[commit] for r in group),lost=sum(r[actor]<r[commit] for r in group))
    stats['directq']=count(group,'directq');stats['best']=count(group,'best')
    return stats


def report():
    frozen=study.read(run.OUT/'protocol.json')
    rows=list(records(frozen['sids']))
    assert len(rows)==len({r['case_id'] for r in rows})==505
    groups={p:[r for r in rows if r['panel']==p] for p in run.PANELS}
    assert {p:len(g) for p,g in groups.items()}==run.PANELS
    groups['train']=groups['dev']+groups['aug']
    metrics={p:summarize(g) for p,g in groups.items()}
    for p in ('fresh','near'):
        for key in ('fixed','continued'):
            for old in ('action_'+key,'commit_'+key,'best','directq'):
                metrics[p][old+'_vs_id_'+key]=study.compare(groups[p],old,'id_'+key)
    replay=study.read(run.OUT/'held_teacher_replay.json')
    assert {r['case_id'] for r in replay}=={r['case_id'] for r in groups['fresh']+groups['near'] if r['ncap']}
    replay_stats={}
    for panel in ('fresh','near'):
        part=[r for r in replay if r['panel']==panel]
        replay_stats[panel]={k:sum(r[k] for r in part) for k in part[0] if k not in ('panel','case_id','teacher')}
    fit=study.read(run.OUT/'fit.json')
    docs=run.DOCS
    docs.mkdir(exist_ok=True)
    for fn,records_ in [('per_scene.csv',rows),('held_teacher_paths.csv',replay)]:
        with (docs/fn).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in records_ for k in r)))
            writer.writeheader();writer.writerows(records_)
    stats=dict(panels=metrics,protocol=frozen,training=fit,replay=replay_stats,
               sha256={k:study.sha(run.OUT/(k+'.pt')) for k in ('fixed','continued')})
    study.save(docs/'results.json',stats)
    lines=['# 十维 CAP 身份 actor 与合并动作 actor','',
        '同一 seed 17、同一批 161 条最快成功示范，共 6,150 个边界。目标为该条轨迹的教师 CAP 编号，每次决策输出十个独立编号的 logits。训练对合法编号做交叉熵，闭环取最大的合法编号，执行该 CAP 当步提出的动作。每次执行后重新选择。不同编号可提出完全相同的动作。', '',
        f"网络输入沿用归一化后的公开几何历史、十 CAP 固定顺序的提案及预算。隐藏层为 64×64，身份网络 {frozen['parameters']:,} 个参数，动作网络 {frozen['hard_parameters']:,} 个参数。十维输出层同时改变了监督类别和打分结构，原 actor 对每个不同候选显式输入其动作与来源。",
        '', '主对照为两模型均训练 24,000 步；次对照均追加 4,500 步，身份模型的续训长度在评估前确定。两者均用 batch 64、先均匀抽房间再抽边界、Adam 初始学习率 0.0003、续训重置 Adam 至 0.00003。', '',
        '| 更新 | 面板 | 全体成功（动作 → 编号） | 并集内成功 | 独胜成功 | 并集外成功 |',
        '|---|---|---:|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('train','训练'),('fresh','自然留出'),('near','困难族扰动')]:
            m=metrics[p];a=m['action_'+key];b=m['id_'+key]
            cols=[f"{a[k]} → {b[k]} / {n}" for k,n in [('success',m['n']),('retained',m['union']),('single',m['single']),('outside',m['n']-m['union'])]]
            lines.append(f"| {steps} | {title} | "+' | '.join(cols)+' |')
    lines+=['','| 更新 | 留出面板 | 编号 actor 救回 / 丢失 | 净差（百分点） | 配对 95% 区间 |',
        '|---|---|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('fresh','自然留出'),('near','困难族扰动')]:
            m=metrics[p]['action_'+key+'_vs_id_'+key]
            lines.append(f"| {steps} | {title} | {m['rescued']} / {m['lost']} | {m['pp']:+.1f} | [{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] |")
    fixed_fit=next(r for r in fit if r['checkpoint']=='fixed' and r['step']==24000)
    natural=metrics['fresh']
    lines+=['',f"自然留出 200 个房间中，原 directQ 成功 {natural['directq']['success']} 个，开发集选定最佳单 CAP 成功 {natural['best']['success']} 个。训练集中 6,150 个教师边界有 {frozen['shared_action']:,} 个至少两个专家提出教师动作；硬标签只奖励最快教师的编号。固定 24,000 步的身份网络在这些示范上选对编号 {fixed_fit['cap_correct']:,}/6,150 次，实际动作命中 {fixed_fit['action_correct']:,}/6,150 次。", '',
        '## 留出成功示范上的拟合','',
        '在同一条最快成功教师轨迹的真实历史上分别给两个模型打分。215 条重放逐步核对实际动作和结束 tick，并核对最终成功与公开历史哈希。每个边界可以选对实际动作而选了另一个编号。下表动作命中只统计至少有两个不同动作的边界。','',
        '| 更新 | 留出面板 | 编号 actor 教师身份命中 / 所有边界 | 动作 actor 教师动作命中 / 多动作边界 | 编号 actor 教师动作命中 / 多动作边界 |',
        '|---|---|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('fresh','自然留出'),('near','困难族扰动')]:
            m=replay_stats[p]
            lines.append(f"| {steps} | {title} | {m[key+'_id_hit']} / {m['boundaries']} | {m[key+'_distinct_hard_action']} / {m['distinct']} | {m[key+'_distinct_id_action']} / {m['distinct']} |")
    lines+=['','## 开局选择与后续切换','',
        '用身份 actor 实际首选编号，在同一房间已有的完整单 CAP 执行中读出一直跟随它的成败，作为开局选一次的参照。另计有过至少一次编号改变且所选动作与前一 CAP 当前提案不同（或前一 CAP 已失效）的房间。', '',
        '| 更新 | 面板 | 开局选一次 | 每步重选 | 至少一次换编号 / 其中有动作差异的房间 | 相对开局固定救回 / 丢失 |',
        '|---|---|---:|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('fresh','自然留出'),('near','困难族扰动')]:
            m=metrics[p];r=m['routing_'+key]
            lines.append(f"| {steps} | {title} | {m['commit_'+key]['success']}/{m['n']} | {m['id_'+key]['success']}/{m['n']} | {r['episodes_with_switch']} / {r['episodes_with_distinct_switch']} | {r['rescued']} / {r['lost']} |")
    lines+=['','两套身份检查点各完成 505 个场景的独立闭环，另做 215 条共用教师重放。留出面板此前已查看。配对 bootstrap 按物体数分层重抽 10,000 次；困难扰动以 45 个父房间为单位成对重抽。区间仅反映该种子与该程序库下的场景差异。', '',
        '[逐场景结果](per_scene.csv) · [逐轨迹教师拟合](held_teacher_paths.csv) · [协议、训练曲线和统计](results.json)']
    (docs/'README.md').write_text('\n'.join(lines)+'\n')
    study.save(run.OUT/'FINISHED.json',dict(episodes=2*len(rows),teacher_replay=len(replay),report=str(docs/'README.md')))


if __name__=='__main__':
    report()
