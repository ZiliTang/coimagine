"""Compare the two budget-matched pairs with the frozen hard actor results."""
import csv
import gzip
import json

import numpy as np
import torch

from tools import cap_regional_soft_actor as run

study = run.study


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def records():
    hard = run.hard
    baseline = {r['case_id']:r for r in read_csv(hard.DOCS/'per_scene.csv')+read_csv(hard.DOCS/'held_per_scene.csv')}
    library = read_csv(study.DOCS/'per_scene.csv')
    frozen = study.read(study.OUT/'frozen.json')
    assert len(library)==len(baseline)==505
    for r in library:
        cid,panel = r['case_id'],r['panel']
        ref=baseline[cid]
        train=panel in ('dev','aug')
        n=sum(int(r[sid]) for sid in frozen['sids'])
        assert n==int(ref['ncap']) and panel==ref['panel']
        row=dict(case_id=cid,panel=panel,parent=r['parent'],objects=int(r['objects']),
                 ncap=n,directq=int(r['directq']),best=int(r['best']))
        for key,hardcolumn in [('fixed','actor' if train else 'fixed'),('continued','polished' if train else 'polish')]:
            hardfolder=('actor' if key=='fixed' else 'polish') if train else ('fixed' if key=='fixed' else 'polish')
            hardout=hard.OUT if train else hard.OUT/'held'
            res=study.read(run.OUT/key/'evaluation'/panel/(cid+'.json'))
            old=study.read(hardout/hardfolder/'evaluation'/panel/(cid+'.json'))
            assert res['case_id']==old['case_id']==cid and res['cap_id']==key and int(old['success'])==int(ref[hardcolumn])
            assert 0<res['ticks']<=1800 and 0<res['boundaries']<=300
            row['hard_'+key]=int(old['success'])
            row['soft_'+key]=int(res['success'])
            row['hard_'+key+'_ticks']=old['ticks']
            row['soft_'+key+'_ticks']=res['ticks']
            row[key+'_same_history']=int(old['history']==res['history'])
        yield row


def training_departures(rows):
    """Check whether the first departure took an action with a recorded successful suffix."""
    teachers=study.read(run.hard.OUT/'teachers.json')
    with np.load(run.OUT/'targets.npz') as stored:
        roots,prob,boundary=[stored[k] for k in ('root','prob','boundary')]
    summary={}
    for name,arm,path in [('hard_fixed','actor',run.hard.OUT),('soft_fixed','fixed',run.OUT),
                          ('hard_continued','polish',run.hard.OUT),('soft_continued','continued',run.OUT)]:
        stats=dict(exact=0,known=0,known_solved=0,other=0,other_solved=0)
        for r in rows:
            cid=r['case_id']
            if cid not in teachers:
                continue
            t=teachers[cid]
            example=run.trace(t['sid'],cid)
            indices=np.flatnonzero(roots==t['root'])
            with gzip.open(path/arm/'evaluation'/r['panel']/(cid+'.json.gz'),'rt') as f:
                executed=json.load(f)
            first=next((i for i,(step,ref) in enumerate(zip(executed,example)) if step['action']!=ref['a']),None)
            if first is None:
                assert len(executed)==len(example) and r[name]
                kind='exact'
            else:
                assert boundary[indices[first]]==first
                kind='known' if prob[indices[first],executed[first]['chosen']]>0 else 'other'
                stats[kind+'_solved']+=r[name]
            stats[kind]+=1
            r[name+'_departure']=kind
        assert stats['exact']+stats['known']+stats['other']==len(teachers)
        summary[name]=stats
    return summary


def results(rows):
    methods=['hard_fixed','soft_fixed','hard_continued','soft_continued','directq','best']
    metrics = dict(n=len(rows),union=sum(r['ncap']>0 for r in rows),singleton=sum(r['ncap']==1 for r in rows))
    for name in methods:
        metrics[name]=dict(success=sum(r[name] for r in rows),
            retained=sum(r[name] for r in rows if r['ncap']),
            unique=sum(r[name] for r in rows if r['ncap']==1),
            outside=sum(r[name] for r in rows if not r['ncap']))
    return metrics


def compare_votes():
    """Recover equal-time votes independently from matching recorded action prefixes."""
    frozen=study.read(study.OUT/'frozen.json')
    teachers=study.read(run.hard.OUT/'teachers.json')
    with np.load(run.OUT/'targets.npz') as saved:
        prob=saved['prob']
        vote=np.zeros(prob.shape,dtype=int)
        for cid,t in teachers.items():
            ix=np.flatnonzero(saved['root']==t['root'])
            teacher=run.trace(t['sid'],cid)
            for sid in frozen['sids']:
                result=study.read(run.hard.BASE/'shards'/sid/(cid+'.json'))
                if not result['success']:
                    continue
                path=run.trace(sid,cid)
                with np.load(run.hard.BASE/'shards'/sid/(cid+'.npz')) as shard:
                    chosen=shard['chosen']
                for idx,record,ref,action in zip(ix,path,teacher,chosen):
                    vote[idx,action]+=1
                    if record['a']!=ref['a']:
                        break
        np.testing.assert_array_equal(vote.sum(1),saved['contributors'])
        np.testing.assert_array_equal(vote>0,prob>0)
    mixed=(vote>0).sum(1)>1
    v=vote[mixed]
    p=prob[mixed]
    tied=(v==v.max(axis=1,keepdims=True)).sum(1)>1
    choice=p.argmax(1)
    lower=v[np.arange(len(v)),choice]<v.max(1)
    tv=np.abs(p-v/v.sum(1,keepdims=True)).sum(1)/2
    return dict(mixed=int(mixed.sum()),vote_ties=int(tied.sum()),weighted_choice_below_highest_vote=int(lower.sum()),
                mean_total_variation=float(tv.mean()))


def training_fit():
    _,_,data=run.hard.load_data()
    probs=np.load(run.OUT/'targets.npz')['prob']
    metrics={}
    for key,checkpoint in [('hard_fixed',run.hard.OUT/'actor.pt'),('hard_continued',run.hard.OUT/'polish.pt'),
                           ('soft_fixed',run.OUT/'fixed.pt'),('soft_continued',run.OUT/'continued.pt')]:
        model=run.hard.actor()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True))
        metrics[key]=run.fit_metrics(model.eval(),data,probs)
    return metrics


def report():
    docs=run.DOCS
    rows=list(records())
    assert len({r['case_id'] for r in rows})==len(rows)==505
    departures=training_departures(rows)
    groups={p:[r for r in rows if r['panel']==p] for p in run.PANELS}
    assert {k:len(g) for k,g in groups.items()}==run.PANELS
    groups['train']=groups['dev']+groups['aug']
    metrics={p:results(g) for p,g in groups.items()}
    for p in ('fresh','near'):
        m=metrics[p];g=groups[p]
        for key in ['fixed','continued']:
            comparison=study.compare(g,'hard_'+key,'soft_'+key)
            common=[r['soft_'+key+'_ticks']-r['hard_'+key+'_ticks'] for r in g if r['hard_'+key] and r['soft_'+key]]
            comparison['common_success_n']=len(common)
            comparison['common_success_median_delta_ticks']=float(np.median(common)) if common else None
            m['paired_'+key]=comparison
    docs.mkdir(exist_ok=True)
    with (docs/'per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    label=study.read(run.OUT/'label_stats.json')
    fits=training_fit()
    votes=compare_votes()
    summary=dict(training_departures=departures,panels=metrics,fit=fits,labels=label,vote_diagnostic=votes,protocol=study.read(run.OUT/'protocol.json'),
                 hashes={key:study.sha(run.OUT/(key+'.pt')) for key in ('fixed','continued')})
    study.save(docs/'results.json',summary)
    lines=['# 区域十 CAP 的剩余时间软标签 actor', '',
           '沿用最快成功教师的 161 条示范和 6,150 个训练边界。同一初态内，逐步比对实际动作、结束 tick、公开特征和有序候选。一个成功专家与教师有相同历史时，其当前动作获得 1 / 剩余实际 ticks 的权重；从首次不同动作之后停止使用该专家的后缀。相同动作的权重相加，再在合法候选内归一化。失败轨迹贡献零。训练和部署均使用无 trace 的 GA 输入，部署逐步贪心选动作。', '',
           f"983 条成功运行参与构造。3,344 个教师边界有多个成功专家共享前缀，其中 {label['mixed']} 个边界有多种成功动作，占全部边界的 {100*label['mixed']/label['boundaries']:.1f}%，按训练采样分布为 {100*label['sampler_mixed']:.1f}%。它们涉及 {label['rooms_mixed']} 个房间，包括 {label['starts_mixed']} 个初始边界；其中 {label['soft_mode_beats_teacher']} 处另一动作的权重严格高于最快教师动作。16 个独胜训练房间的动作目标都与硬标签一致。", '',
           f"在 {votes['mixed']} 个有多种成功动作的边界上，若每个成功专家投等权票，{votes['vote_ties']} 处会出现最高票并列。按时间加权后的第一动作在 {votes['weighted_choice_below_highest_vote']} 处票数严格低于等权最高票。这是对目标分布的核对，未另训等权投票模型。",'',
           '## 固定预算的配对闭环','',
           '表内为硬标签 → 软标签。主比较为同一 seed 17 的 24,000 步模型在自然留出面板上的成败。另在相同输入、归一化、初始化和抽样下，以 Adam 0.00003 从 24,000 步各自的权重再训练 4,500 步。软标签续训长度事先按历史硬标签检查点对齐；未按留出成绩选模型。','',
           '| 预算 | 面板 | 全体成功 | 并集内成功 | 独胜成功 | 并集外成功 |',
           '|---|---|---:|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('train','训练'),('fresh','自然留出'),('near','困难族扰动')]:
            m=metrics[p];h=m['hard_'+key];s=m['soft_'+key]
            cols=[f"{h[k]} → {s[k]} / {n}" for k,n in [('success',m['n']),('retained',m['union']),('unique',m['singleton']),('outside',m['n']-m['union'])]]
            lines.append(f"| {steps} | {title} | "+' | '.join(cols)+' |')
    lines+=['','| 预算 | 留出面板 | 软标签救回 / 丢失 | 净差（百分点） | 配对 95% 区间 |',
            '|---|---|---:|---:|---:|']
    for key,steps in [('fixed','24,000'),('continued','28,500')]:
        for p,title in [('fresh','自然留出'),('near','困难族扰动')]:
            pair=metrics[p]['paired_'+key]
            lines.append(f"| {steps} | {title} | {pair['rescued']} / {pair['lost']} | {pair['pp']:+.1f} | [{pair['ci'][0]:+.1f}, {pair['ci'][1]:+.1f}] |")
    primary=metrics['fresh']['paired_fixed']
    qualified=('该区间包含零。' if primary['ci'][0]<=0<=primary['ci'][1] else '该区间未跨零。')
    lines+=['',f"主比较的自然留出净变化为 {primary['pp']:+.1f} 个百分点，{qualified} 同一面板上，开发选定单 CAP 成功 {metrics['fresh']['best']['success']}/200，原 directQ 成功 {metrics['fresh']['directq']['success']}/200。",
        '',f"在 {label['mixed']} 个具有多个成功动作的训练边界上，24,000 步硬标签与软标签模型选中最高软权重动作分别为 {fits['hard_fixed']['mixed_top']} 与 {fits['soft_fixed']['mixed_top']} 次。沿最快教师输入逐步全对的房间分别为 {fits['hard_fixed']['perfect']}/161、{fits['soft_fixed']['perfect']}/161。软标签允许多条成功后缀，任务最终通过实际闭环判断。", '',
        f"在 161 个可解训练房间的实际闭环中，固定预算软标签 actor 有 {departures['soft_fixed']['exact']} 个完整跟随最快示范。首次偏离选择了另一条已知成功后缀首动作的有 {departures['soft_fixed']['known']} 个，其中最终成功 {departures['soft_fixed']['known_solved']} 个。模型只在原教师边界接受监督，从该分歧点往后的另一条完整示范没有加入训练样本。", '',
        '两个软标签检查点各执行 215 个训练初态、200 个自然留出初态与 90 个困难族扰动，共 1,010 次。硬标签结果复用同房间已归档闭环。自然面板按房间配对，扰动面板按 45 个父房间配对重抽，每个父房间有两个扰动。区间按物体数分层 bootstrap 10,000 次，条件于本次库和 seed 17。留出面板此前已查看。', '',
        '[逐场景配对](per_scene.csv) · [训练及闭环统计](results.json) · [硬标签留出报告](../2026-11-20_regional_fastest_actor/held.md)']
    (docs/'README.md').write_text('\n'.join(lines)+'\n')
    study.save(run.OUT/'FINISHED.json',dict(episodes=2*len(rows),report=str(docs/'README.md')))


if __name__=='__main__':
    torch.set_num_threads(1)
    report()
