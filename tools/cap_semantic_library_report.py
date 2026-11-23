"""Paired interaction and specialization on the frozen common test panel."""
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from tools import cap_semantic_library_eval as experiment
from tools.cap_regional_token_meter import tail_usage

OUT,DOC,WORK,ARMS,read,save = (getattr(experiment,k) for k in ('OUT','DOC','WORK','ARMS','read','save'))


def boot(values, counts):
    v=np.asarray(values);rng=np.random.default_rng(17);result=np.zeros(10000)
    for n in sorted(set(counts)):
        g=v[counts==n]
        result+=rng.choice(g,(len(result),len(g))).sum(1)
    return (100*np.percentile(result/len(v),[2.5,97.5])).tolist()


def paired(before,after,counts):
    d=after-before
    return dict(first=int(before.sum()),second=int(after.sum()),rescued=int(sum(d>0)),lost=int(sum(d<0)),pp=float(100*d.mean()),ci=boot(d,counts))


def author_usage(arm):
    probes=[read(p) for p in (OUT/arm).glob('probe_m*.json')]
    state=read(OUT/arm/'campaign.json')
    result=dict(probes=len(probes),episodes=sum(p['total'] for p in probes),ticks=state['used']['execution_ticks'])
    path=Path.home()/'.codex/sessions/2026/11/23'
    for p in path.glob('*.jsonl'):
        with p.open() as f:meta=json.loads(f.readline())['payload']
        if meta.get('agent_path')=='/root/'+{'usual':'usual23','specialized':'special23'}[arm]:
            result['tokens']=tail_usage(p)
            from datetime import datetime
            result['wall_to_final_s']=round((WORK/arm/'FINAL.json').stat().st_mtime-datetime.fromisoformat(meta['timestamp'].replace('Z','+00:00')).timestamp(),1)
    return result


def report():
    dev=read(OUT/'development.json');test=read(OUT/'test.json');rules=read(OUT/'conditions.json')
    ids=sorted(test);counts=np.array([len(test[k]['movable_objects'])+1 for k in ids])
    geom={}
    for k,fixture in test.items():
        s=experiment.corridor_semantics(experiment.base.public(fixture));s['observed']=s['kind']
        geom[k]=s
    scores={};out=dict(arms={})
    rows={k:dict(room=k,objects=int(n),conditions=';'.join(r['name'] for r in rules if experiment.study.matches(r,geom[k]))) for k,n in zip(ids,counts)}
    for arm in ARMS:
        members=read(OUT/f'{arm}_library.json')
        def matrix(keys):
            return np.array([[int(read(OUT/'singles'/m['sha']/(k+'.json'))['success']) for m in members] for k in keys])
        dy=matrix(sorted(dev)); y=matrix(ids); best=int(np.argmax(dy.sum(0)))
        table=dict(members=members,dev_single=dy.sum(0).tolist(),held_single=y.sum(0).tolist(),dev_union=int(dy.any(1).sum()),
                   held_union=int(y.any(1).sum()),dev_best=best,author=author_usage(arm),fit={},conditions=[])
        for rule in rules:
            dmask=np.array([experiment.study.matches(rule,read_initial) for read_initial in (INITIAL[k] for k in sorted(dev))])
            mask=np.array([experiment.study.matches(rule,geom[k]) for k in ids])
            table['conditions'].append(dict(name=rule['name'],dev_n=int(dmask.sum()),dev_single=dy[dmask].sum(0).tolist(),
                test_n=int(mask.sum()),held_single=y[mask].sum(0).tolist(),dev_best=int(np.argmax(dy[dmask].sum(0))),
                held_rescues=(y[mask] & (1-y[mask,0,None])).sum(0).tolist(),held_loss=((1-y[mask])*y[mask,0,None]).sum(0).tolist()))
        # Deterministic fallback for overlapping regions, using each membership pattern rather than overriding rules in sequence.
        def pattern(obs):
            return tuple(experiment.study.matches(r,obs) for r in rules)
        dev_keys=sorted(dev);by={}
        for pat in {pattern(s) for s in INITIAL.values()}:
            msk=[pattern(INITIAL[k])==pat for k in dev_keys]
            if sum(msk)>=12:by[pat]=int(np.argmax(dy[msk].sum(0)))
        tagscore=np.array([y[j,by.get(pattern(geom[k]),best)] for j,k in enumerate(ids)])
        table['initial_tag_choice']=paired(y[:,best],tagscore,counts)
        for mode in ('basic','semantic'):
            v=np.array([int(read(OUT/'evaluation'/arm/mode/(k+'.json'))['success']) for k in ids])
            scores[arm,mode]=v;table['fit'][mode]=read(OUT/f'{arm}_{mode}_fit.json')
            table[mode+'_vs_dev_single']=paired(y[:,best],v,counts)
        table['semantic_vs_basic']=paired(scores[arm,'basic'],scores[arm,'semantic'],counts)
        out['arms'][arm]=table
        for j,k in enumerate(ids):
            rows[k].update({arm+f'_member{i}':int(w) for i,w in enumerate(y[j])})
            rows[k].update({arm+'_'+mode:int(scores[arm,mode][j]) for mode in ('basic','semantic')})
            rows[k][arm+'_initial_tag']=int(tagscore[j])
    usual,specialized=ARMS
    d=(scores[specialized,'semantic']-scores[specialized,'basic'])-(scores[usual,'semantic']-scores[usual,'basic'])
    out['interaction']=dict(pp=float(100*d.mean()),ci=boot(d,counts),room_differences=dict(Counter(map(str,d))))
    out['between_libraries']={mode:paired(scores[usual,mode],scores[specialized,mode],counts) for mode in ('basic','semantic')}
    DOC.mkdir(exist_ok=True,parents=True)
    with (DOC/'rooms.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[ids[0]]));w.writeheader();w.writerows(rows.values())
    save(DOC/'results.json',out)
    markdown(out)
    save(OUT/'FINISHED.json',out)


def markdown(out):
    names={'usual':'常规增量库','specialized':'半开放语义专精库'}
    def interval(ci):
        return f'[{ci[0]:+.1f}, {ci[1]:+.1f}]'
    inter=out['interaction']
    lines=['# 半开放语义专精能否生成更容易混合的 CAP 库','',
        f'同一张新面板共 200 个房间。语义输入对两库的增益之差为 {inter["pp"]:+.1f} 个百分点，房间配对重抽的 95% 区间为 {interval(inter["ci"])}。主比较以每房间四个逐边界路由结果计算。','',
        '| 专家库 | 最佳开发单 CAP 在留出的成功 | 留出专家并集 | 基础评分 | 加三个语义通道 | 救回／丢失 | 语义增益（百分点，95% 区间） |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,t in out['arms'].items():
        d=t['semantic_vs_basic']
        lines.append(f'| {names[arm]} | {t["held_single"][t["dev_best"]]} | {t["held_union"]} | {d["first"]} | {d["second"]} | {d["rescued"]}／{d["lost"]} | {d["pp"]:+.1f} {interval(d["ci"])} |')
    lines+=['','## 作者与训练','']
    for arm,t in out['arms'].items():
        a=t['author'];fit=t['fit']
        tokens=(a.get('tokens') or {}).get('total_tokens')
        lines.append(f'- {names[arm]}。开发试跑 {a["probes"]} 批、{a["episodes"]} 局、实际 {a["ticks"]:,} ticks。作者会话到提交约 {a.get("wall_to_final_s",0)/60:.1f} 分钟，累计 tokens（含缓存输入）{tokens if tokens is not None else "未取得"}。开发单成员成功数（母本在首位）{t["dev_single"]}，并集 {t["dev_union"]}/160；新面板单成员 {t["held_single"]}。学习用 {fit["basic"]["rooms"]} 条教师轨迹，{fit["basic"]["boundaries"]:,} 个多候选边界，基础／语义教师动作拟合率 {fit["basic"]["accuracy"]:.1%}／{fit["semantic"]["accuracy"]:.1%}。')
    lines+=['','## 专精成员是否在声明条件内体现优势','','以下按开局公开观测分组，条件允许重叠。各成员仍在所有房间整局执行。组内救回、丢失均相对共同母本。','',
            '| 专精条件 | 闭环设计 | 满足条件的留出房间 | 母本成功 | 对应成员成功 | 救回／丢失 |',
            '|---|---|---:|---:|---:|---:|']
    t=out['arms']['specialized']
    for i,m in enumerate(t['members'][1:],1):
        c=next((c for c in t['conditions'] if c['name']==m.get('condition')),None)
        if c is not None:
            lines.append(f'| {c["name"]} | 第 {i} 位新增成员 | {c["test_n"]} | {c["held_single"][0]} | {c["held_single"][i]} | {c["held_rescues"][i]}／{c["held_loss"][i]} |')
    lines+=['','另按三个初始谓词的真假组合在开发集选一个固定专家，少于 12 例的组合回退至开发集最强单成员。两库对应的留出成功数分别为 '+ '、'.join(str(out['arms'][arm]['initial_tag_choice']['second']) for arm in ARMS)+'/200。该结果与上表的逐边界混合单独解读。','',
        '只有一对作者、一个共享母本和一个拟合种子。这轮可以比较四个已训练策略在同一批房间上的表现；推广到其他作者运行需要独立重复。所有运行沿用同一版本的物理、option、机器人半径与避障裕度。','',
        '[作者信息、选类规则与训练协议](protocol.md) · [每房间全部单 CAP 和混合结果](rooms.csv) · [数值、权重来源与配对统计](results.json) · [后续 option 卡滞待办](../../future_directions/2026-11-23_option_contract_stalls.md)']
    interesting = next((i for i,m in enumerate(t['members']) if m.get('condition')=='near-wall destination'),None)
    if interesting is not None:
        c=next(c for c in t['conditions'] if c['name']=='near-wall destination')
        base_out=t['held_single'][0]-c['held_single'][0];member_out=t['held_single'][interesting]-c['held_single'][interesting]
        lines[-1:-1]=[f'靠墙成员在条件外 {200-c["test_n"]} 间的成功数为 {member_out}，母本为 {base_out}。此次净优势集中在作者事先声明的区域，该局部观察来自三个条件中的一个，仍需用不同作者或新面板复验。','']
    (DOC/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    INITIAL=read(OUT/'initial.json')
    report()
