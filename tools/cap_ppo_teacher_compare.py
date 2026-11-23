"""Replay the original authored router against the two locked PPO room lists."""
import csv
from pathlib import Path

import numpy as np
import torch
from tools import cap_ungated_ppo as t
from tools import cap_authored_router as author

r=t.r
OUT=t.OUT.parent/"cap_ppo_teacher_compare_20261123_01"
DOC=t.DOC
PANELS=("diagnostic","confirm")


def episode(job):
    panel,key=job
    row=author.play(r.study.ROOMS[key],stall_gate=True)
    result=row["result"]
    src=(r.OUT/"fresh") if panel=="diagnostic" else (t.OUT/"confirm")
    if not row["different"]:
        assert result["history"]==r.e.read(src/"A"/(key+".json"))["history"]
    r.e.save(OUT/panel/(key+".json"),dict(success=result["success"],ticks=result["ticks"],history=result["history"],
        triggered=row["hit"] is not None,chose_distinct_B=row["different"]>0))


def report(plan):
    data=[]
    results={}
    for panel in PANELS:
        keys=plan[panel]
        src=r.OUT/"fresh" if panel=="diagnostic" else t.OUT/"confirm"
        table=[]
        for k in keys:
            expert=[r.e.read(src/a/(k+".json"))["success"] for a in ("A","B")]
            row=r.e.read(OUT/panel/(k+".json"))
            ppo=[r.e.read(t.OUT/panel/f"s{s}_r40"/(k+".json"))["success"] for s in r.SEEDS]
            table.append([*expert,row["success"],*ppo])
            data.append([panel,k,*table[-1],row["triggered"],row["chose_distinct_B"]])
        a,b,router,*ppo=np.array(table,dtype=int).T
        ppo=np.array(ppo).T
        deltas=ppo-router[:,None]
        gap=deltas.mean(1)
        draws=np.random.default_rng(71117).integers(len(keys),size=(10000,len(keys)))
        masks={"both":a*b==1,"only_A":a>b,"only_B":b>a,"neither":a+b==0}
        def single(ref):
            return dict(count=int(sum(ref)),gained=int(sum(router>ref)),lost=int(sum(router<ref)))
        results[panel]=dict(A=single(a),B=single(b),router=int(router.sum()),ppo=ppo.sum(0).tolist(),
            ppo_only=(deltas>0).sum(0).tolist(),router_only=(deltas<0).sum(0).tolist(),
            ppo_minus_router_pp=float(100*gap.mean()),ci95=(100*np.quantile(gap[draws].mean(1),[.025,.975])).tolist(),
            cells={k:dict(n=int(mask.sum()),router=int(router[mask].sum()),ppo=ppo[mask].sum(0).tolist()) for k,mask in masks.items()})
    with (DOC/"teacher_ppo_rooms.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["panel","room","A","B","router",*[f"ppo_s{s}" for s in r.SEEDS],"triggered","chose_distinct_B"])
        w.writerows(data)
    r.e.save(DOC/"teacher_ppo.json",results)
    lines=["# 冻结程序 router 与 PPO 的同房间比较","",
           "直接执行最初冻结的 v3 路由，触发公开停滞前由 A 控制。两个面板的场景内容哈希和两位专家在 fixed72-local-v1 下的代码指纹均匹配 PPO 协议。"
           "原 118/200 来自另一批房间，采用 fixed72-world-v0。对照使用最终第 40 轮贪心 PPO 的已存逐房间结果。","",
           "| 面板 | A | B | 程序 router | PPO 17/29/43 | PPO 平均减 router（百分点，95% 区间） |",
           "|---|---:|---:|---:|---|---|"]
    for p,name in zip(PANELS,("诊断 200 间","PPO 确认 200 间"),strict=True):
        q=results[p];lo,hi=q["ci95"]
        lines.append(f'| {name} | {q["A"]["count"]} | {q["B"]["count"]} | {q["router"]} | {" / ".join(map(str,q["ppo"]))} | {q["ppo_minus_router_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}] |')
    lines+=["","按房间配对重抽 10,000 次，保留同一房间的三个 PPO 种子与一份确定性程序结果。区间只反映这批训练种子下的房间不确定性。"
            "确认面板最初为 PPO 预留，此处为 PPO 结果公布后补做的冻结程序比较。","",
            "两者具有相同房间、专家提案来源、option 执行器与终止预算。程序路由由作者利用开发记录与同一 24 间上的 72 次试跑选版，PPO 从分叉训练的第四轮父模型继续，"
            "每种子 2,000 次在线采样。PPO 从第一次分歧起能自由选择，两者的公开特征、记忆和开发成本有别。","",
            "[逐房间配对](teacher_ppo_rooms.csv) · [按专家覆盖分层](teacher_ppo.json)"]
    (DOC/"teacher_ppo.md").write_text("\n".join(lines)+"\n")
    return results


def main():
    torch.set_num_threads(1)
    plan=r.e.read(t.OUT/"protocol.json")
    assert r.d.init()==plan["profile"]
    author.ROUTE=r.d.TEACHER
    frozen=r.e.read(author.OUT/"frozen.json")
    assert frozen["selection"]["revision"]==3
    for panel,seed in zip(PANELS,(r.FRESH_SEED,t.CONFIRM),strict=True):
        rooms=r.prior.fixtures(seed)
        assert sorted(rooms)==plan[panel] and not set(rooms)&set(r.e.read(author.OUT/"confirmation_ids.json").values())
        r.study.ROOMS.update(rooms)
    protocol=dict(router=frozen["router_sha"],ppo_protocol=t.sha(t.OUT/"protocol.json"),profile=plan["profile"],
                  gate="original first eligible stall; deterministic frozen program",seeds=list(r.SEEDS))
    if (OUT/"protocol.json").is_file():assert r.e.read(OUT/"protocol.json")==protocol
    r.e.save(OUT/"protocol.json",protocol)
    r.exp.OUT=OUT
    r.exp.WORKERS=r.prior.first.WORKERS
    jobs=[(panel,k) for panel in ("confirm","diagnostic") for k in plan[panel]
          if not (OUT/panel/(k+".json")).exists()]
    r.exp.timed_pool("router",episode,jobs)
    report(plan)


if __name__=="__main__":
    main()
