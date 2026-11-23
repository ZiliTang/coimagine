"""Report room-paired retention, recovery and overall results with shared-room bootstrap."""
import csv
import json
from pathlib import Path

import numpy as np
from tools import cap_ungated_replay as r


def comparison(mask,actor,mixed,draws):
    denom=mask.sum(axis=1)
    a=(mask*actor).sum(axis=1)
    b=(mask*mixed).sum(axis=1)
    den=denom[draws].sum(axis=1)
    ok=den>0
    diffs=100*(b-a)[draws].sum(axis=1)[ok]/den[ok]
    return dict(rooms=int((denom>0).sum()),n=int(denom.sum()),actor=int(a.sum()),mixed=int(b.sum()),
                gained=int((mask & mixed & ~actor).sum()),lost=int((mask & actor & ~mixed).sum()),
                delta_pp=float(100*(b-a).sum()/denom.sum()) if denom.sum() else None,
                ci95=np.quantile(diffs,[.025,.975]).tolist() if ok.any() else None,
                by_seed=[dict(n=int(m.sum()),actor=int((m*x).sum()),mixed=int((m*y).sum()))
                         for m,x,y in zip(mask.T,actor.T,mixed.T,strict=True)])


def read_panel(panel,keys,draws):
    room_rows=[]
    for key in keys:
        def read(name):
            return r.e.read(r.OUT/panel/name/(key+".json"))
        a,b=[read(who)["success"] for who in ("A","B")]
        for seed in r.SEEDS:
            outcomes={mode:read(f"s{seed}_{mode}") for mode in ("parent","actor","mixed")}
            room_rows.append(dict(panel=panel,room=key,seed=seed,A=a,B=b,
                **{mode:row["success"] for mode,row in outcomes.items()},
                **{mode+"_early_B":row["pre_stall_B"] for mode,row in outcomes.items()}))
    def arr(field):
        return np.array([v[field] for v in room_rows],bool).reshape(-1,len(r.SEEDS))
    a,b,p,x,y=map(arr,("A","B","parent","actor","mixed"))
    union=a|b
    masks=dict(all=np.ones_like(x),parent_success=p,parent_fail=~p,
               expert_recovery=union&~p,A_recovery=a&~b&~p,B_recovery=b&~a&~p,both_recovery=a&b&~p,
               actor_unique=~union&p,outside_recovery=~union&~p)
    totals={n:v.sum(axis=0).tolist() for n,v in [("parent",p),("actor",x),("mixed",y)]}
    res=dict(singles=dict(A=int(a[:,0].sum()),B=int(b[:,0].sum()),union=int(union[:,0].sum())),
             totals=totals,cells={name:comparison(m,x,y,draws) for name,m in masks.items()},
             vs_parent={n:comparison(masks["all"],p,v,draws) for n,v in [("actor",x),("mixed",y)]},
             vs_single={arm:{name:comparison(masks["all"],z,v,draws) for name,z in (("A",a),("B",b))}
                        for arm,v in (("parent",p),("actor",x),("mixed",y))},
             early_B_rooms={n:arr(n+"_early_B").sum(axis=0).tolist() for n in totals})
    return room_rows,res


def prefix_choices(plan):
    from tools.cap_ungated_fit import load_rows
    records={}
    for seed in r.SEEDS:
        x,b,_,ids,y,pools,_=load_rows(seed,"mixed",plan)
        metrics={}
        for arm in ("parent","actor","mixed"):
            path=r.prior.checkpoint(seed,4) if arm=="parent" else r.OUT/"models"/f"s{seed}_{arm}.npz"
            r.d.MODELS["history"]=r.study.cold.weights(path)
            choice=r.d.logits(x,"history")>0
            by_pool={}
            for name,rooms in pools.items():
                indices=np.concatenate(rooms)
                by_pool[name]=dict(correct=int((choice[indices]==b[indices]).sum()),total=len(indices))
            by_pool["strict"]=dict(correct=int((choice[ids]==y).sum()),total=len(ids))
            metrics[arm]=by_pool
        records[str(seed)]=metrics
    return records


def main():
    plan=r.e.read(r.OUT/"protocol.json")
    r.DOC.mkdir(parents=True,exist_ok=True)
    results={}
    rows=[]
    for panel in ("train","fresh"):
        n=len(plan[panel])
        draws=np.random.default_rng(71117).integers(n,size=(10000,n))
        items,results[panel]=read_panel(panel,plan[panel],draws)
        rows.extend(items)
    with (r.DOC/"rooms.csv").open("w") as f:
        writer=csv.DictWriter(f,rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fits={s.stem:r.e.read(s) for s in sorted((r.OUT/"fit").glob("*.json"))}
    results["fit"]=fits
    agreement=results["prefix_agreement"]=prefix_choices(plan)
    assert len(fits)==6
    r.e.save(r.DOC/"results.json",results)
    fresh=results["fresh"]
    cells=fresh["cells"]
    overall=cells["all"]
    def diff(row):
        if row["ci95"] is None:
            return "无符合条件的房间"
        low,high=row["ci95"]
        return f'{row["delta_pp"]:+.2f} [{low:+.2f}, {high:+.2f}]'
    def cell(name,label):
        v=cells[name]
        return f'| {label} | {v["rooms"]} | {v["actor"]}/{v["n"]} | {v["mixed"]}/{v["n"]} | {diff(v)} |'
    table="\n".join(cell(n,l) for n,l in [("all","全部"),("parent_success","父 actor 成功，更新后保住"),
           ("parent_fail","父 actor 失败，更新后救回"),("expert_recovery","至少一个专家成功，父 actor 失败"),
           ("A_recovery","仅 A 成功，父 actor 失败"),("B_recovery","仅 B 成功，父 actor 失败"),
           ("both_recovery","两专家均成功，父 actor 失败"),("actor_unique","两专家均失败，父 actor 成功")])
    totals=fresh["totals"]
    by_seed="\n".join(f'| {s} | {totals["parent"][i]} | {totals["actor"][i]} | {totals["mixed"][i]} |'
                      for i,s in enumerate(r.SEEDS))
    bas=fresh["singles"]
    result=f'新 200 间上，混合余量相对全 actor 余量的成功率差为 {diff(overall)} 个百分点。'
    expert=cells["expert_recovery"]
    uniq=cells["actor_unique"]
    analysis=f'父 actor 遗漏的专家成功共 {expert["n"]} 次，两组分别救回 {expert["actor"]} 和 {expert["mixed"]} 次。双专家都失败的父 actor 成功共 {uniq["n"]} 次，两组保住 {uniq["actor"]} 和 {uniq["mixed"]} 次。'
    strict=[fits[f"s{s}_actor"]["strict"] for s in r.SEEDS]
    forks=r.e.read(r.OUT/"collection_done.json")["total"]
    train=results["train"]["totals"]
    early=fresh["early_B_rooms"]["parent"]
    def agreement_line(source,label):
        fields=[]
        for arm in ("parent","actor","mixed"):
            rows=[v[arm][source] for v in agreement.values()]
            fields.append(f'{sum(v["correct"] for v in rows)}/{sum(v["total"] for v in rows)}')
        return "| "+" | ".join([label,*fields])+" |"
    agreement_text="\n".join(agreement_line(s,l) for s,l in
                    [("own","父 actor 成功路线"),("A","纯 A 成功路线"),("B","纯 B 成功路线"),("strict","成败翻转的分叉")])
    detail=f"""# 开放全程路由后，纯专家成功示范的增量价值

{result}{analysis}

两组对父 actor 失败房间的救回总数相同。混合组在三个种子上的总成功数均较低，合并统计时旧成功保留率差为 {diff(cells['parent_success'])} 个百分点。单独按同一批开发前缀检查时，加入专家示范提高了 A、B 两类动作的拟合率。把纯专家的首步交给混合 actor 续跑仍涉及后续动作选择，单步模仿更准没有带来相应的闭环收益。现有证据支持在当前 A/B 组合保留以 actor 自身成功为主的回放基线；本轮仅检验了把一半余量换为纯专家的配置。

## 同一批 200 个新房间上的闭环

纯 A 成功 {bas['A']}/200，纯 B 成功 {bas['B']}/200，二者成功并集 {bas['union']}/200。每个模型都在相同房间执行到底。

| 种子 | 未更新父 actor | 全 actor 余量 | 半数余量换成纯专家 |
|---|---:|---:|---:|
{by_seed}

| 条件 | 涉及的不同房间 | 全 actor 余量 | 半数换成纯专家 | 配对差与 95% 区间（百分点） |
|---|---:|---:|---:|---:|
{table}

每个分母统计符合条件的房间与种子组合。95% 区间对房间做 10,000 次配对重抽，每次同时保留该房间的三个种子，条件比例随重抽重新计算。区间描述这三个已训练种子的房间抽样不确定性。全 actor 组相对各自未更新的父模型，成功率变化为 {diff(fresh["vs_parent"]["actor"])} 个百分点，混合组为 {diff(fresh["vs_parent"]["mixed"])} 个百分点。两种更新直接对比，混合组独赢 {overall['gained']} 次，全 actor 组独赢 {overall['lost']} 次。

## 实验条件

每个种子从同一个已训练四轮、此前带有停滞门控的父模型分出两个训练臂。本轮采样、分叉续跑、评估均从第一次提案分歧起允许选 A 或 B。两者提案相同则执行共同命令，记规范编号 A。公开停滞特征保留，教师附加的九个语义维度屏蔽。三个父 actor 在新面板分别有 {early[0]}、{early[1]}、{early[2]} 间在停滞触发前执行过有别于 A 的 B 提案。全部 250 间的固定 A/B 与各自原生专家逐条核对了完整历史哈希、终止和成本记录，共 500 对。

50 个相同开发房间，每个种子独立按父 actor 采样，每间最多三个按时间分层抽取的分叉。每次仅改选一步，两支均由该种子的同一父 actor 继续。共 {forks} 个配对，三个种子分别获得 {strict[0]}、{strict[1]}、{strict[2]} 个二元成败不同的强标签。平局不提供方向。弱示范限于实际提案不同的成功轨迹边界，先均匀抽成功轨迹，再均匀抽边界。同一可见输入已获强标签时移除其弱监督，各臂自身样本若对同一可见输入支持相反选择也移除该点。分来源的可用轨迹、边界和冲突数见 results.json。

两臂共用强标签、父模型概率和相同的训练步数与强标签抽样。每步 64 个强标签，128 个父模型软保持，64 个成功余量。全 actor 组余量全来自成功父 actor；混合组抽取 32 个父 actor、16 个纯 A、16 个纯 B。弱示范采用加权铰链损失，logit 余量 0.2、权重 0.25。网络 64×64，每种子每臂 300 次 Adam 更新，学习率 0.0003。开发房间上父 actor、全 actor 余量、混合余量逐种子成功数分别为 {train['parent']}、{train['actor']}、{train['mixed']}，分母均为 50。

## 固定开发前缀上的选择

在混合组可用的同一批输入上，逐步比较各模型是否选择示范动作或实测胜者。这里沿真实历史输入模型，分母是决策边界，按三个种子汇总。混合组在三个种子分别发现 {fits['s17_mixed']['ambiguous_signatures']}、{fits['s29_mixed']['ambiguous_signatures']}、{fits['s43_mixed']['ambiguous_signatures']} 个有相反成功选择的相同输入，移除后可抽的父 actor 轨迹数也相应变化。该对照检验整套回放构造，包括对重复输入的处理。

| 输入来自 | 未更新父 actor | 全 actor 余量 | 半数换成纯专家 |
|---|---:|---:|---:|
{agreement_text}

估计量是开放门控后替换一半余量来源的单轮增量。父模型来自既有带门控训练，未在本轮估计门控本身的因果效应。新测试房间在拟合前按场景内容哈希冻结，六份模型锁定后才评估。纯专家示范提供一条可行后缀，单个边界的反事实优劣仅用配对分叉判断。

[逐房间成败](rooms.csv) · [完整统计](results.json)
"""
    (r.DOC/"README.md").write_text(detail)
    p=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(p.read_text())
    next(v for v in entries if v["id"]=="ungated-expert-replay").update(result=result,analysis=analysis)
    p.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    r.e.save(r.OUT/"FINISHED.json",dict(overall=overall,eval_rooms=len(plan["fresh"]),
            results_sha=r.d.source.graph.file_sha(r.DOC/"results.json")))
    print(result,analysis)


if __name__=="__main__":
    main()
