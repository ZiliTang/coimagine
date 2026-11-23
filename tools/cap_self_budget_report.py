"""Report paired endpoints and round-wise diagnostic curves for the budget study."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tools import cap_self_budget as run
from tools.cap_self_replication_report import comparison, delta_text

e, out, doc = run.e, run.OUT, run.DOC


def load_panel(panel, keys):
    stages=range(4,9) if panel=="trend" else (4,8)
    models=["B",*(f"s{s}_r{r}" for s in run.SEEDS for r in stages)]
    data={}
    for m in models:
        root=out/panel/"test"/m
        if panel=="trend" and (m=="B" or m.endswith("_r4")):
            root=run.first.OUT/"fresh"/"test"/m
        assert {f.stem for f in root.glob("*.json")}==set(keys)
        x=np.array([e.read(root/(k+".json"))["success"] for k in keys],dtype=int)
        assert np.isin(x,[0,1]).all()
        data[m]=x
    return data


def summarize(data, bootstrap):
    before=np.stack([data[f"s{s}_r4"] for s in run.SEEDS])
    after=np.stack([data[f"s{s}_r8"] for s in run.SEEDS])
    b=data["B"]
    metrics={}
    for seed,x,y in zip(run.SEEDS,before,after,strict=True):
        metrics[seed]=dict(r4=int(x.sum()),r8=int(y.sum()),change=comparison(y,x,bootstrap),
                           r4_vs_B=comparison(x,b,bootstrap),r8_vs_B=comparison(y,b,bootstrap),
                           retained=int(sum((x==1)&(y==1))))
    mean={}
    for label,diffs in [("change",after-before),("r4_vs_B",before-b),("r8_vs_B",after-b)]:
        m=comparison(diffs.mean(axis=0),np.zeros(len(b)),bootstrap)
        m.update(rescued=float(np.maximum(diffs,0).sum(1).mean()),lost=float(np.maximum(-diffs,0).sum(1).mean()))
        mean[label]=m
    return dict(B=int(b.sum()),seed=metrics,mean=mean,
                r4_mean=float(before.sum(1).mean()),r8_mean=float(after.sum(1).mean()))


def plot_trend(totals):
    fig,ax=plt.subplots(figsize=(7.6,4.5),layout="constrained")
    stages=[r for r in range(4,9) if f"s{run.SEEDS[0]}_r{r}" in totals]
    for s,color,marker in zip(run.SEEDS,("#136d71","#ad5323","#5c4c95"),("o","s","^"),strict=True):
        ax.plot(stages,[totals[f"s{s}_r{r}"] for r in stages],marker=marker,color=color,label=f"seed {s}")
    ax.axhline(totals["B"],color="#555",ls="--",linewidth=1.4,label="single CAP B")
    ax.set(xticks=list(range(4,9)), xlim=(3.9,8.1),xlabel="Cumulative training rounds",ylabel="Successful rooms (out of 200)",
           title="Greedy closed-loop performance on the previously viewed panel")
    ax.grid(alpha=.19)
    ax.legend(ncol=4,loc="best",fontsize="small")
    pending=doc/"trend.pending.png"
    fig.savefig(pending,dpi=160)
    pending.replace(doc/"trend.png")
    plt.close(fig)


def main():
    plan=e.read(out/"protocol.json")
    doc.mkdir(exist_ok=True,parents=True)
    bootstrap=np.random.default_rng(172943).integers(200,size=(10000,200))
    stats={}
    with (doc/"rooms.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["panel","room_content_sha256","model","success"])
        for panel in ("confirm","trend"):
            keys=plan[panel]
            assert len(set(keys))==len(keys)==200
            data=load_panel(panel,keys)
            for m,x in data.items():
                writer.writerows((panel,k,m,int(v)) for k,v in zip(keys,x,strict=True))
            stats[panel]=summarize(data,bootstrap)
            stats[panel]["totals"]={m:int(x.sum()) for m,x in data.items()}
    assert set(plan["confirm"]).isdisjoint(plan["train"]+plan["trend"])
    t=stats["trend"]["totals"]
    plot_trend(t)
    main_result=stats["confirm"]
    change=main_result["mean"]["change"]
    c8=main_result["mean"]["r8_vs_B"]
    positive=sum(r["r8"]>r["r4"] for r in main_result["seed"].values())
    summary=(f'新确认面板单 CAP B 成功 {main_result["B"]}/200。三种子平均从 {main_result["r4_mean"]:.1f} 到 '
             f'{main_result["r8_mean"]:.1f}/200，第 8 轮相对第 4 轮 {delta_text(change)} 个百分点。')
    conclusion = ("本次加倍预算未在确认面板上显示稳定的泛化提升。区间仍允许小幅改善或退化，无法据此确定第 4 轮已达上限。"
                  if change["ci_pp"][0]<=0<=change["ci_pp"][1] else
                  ("新增预算带来了可测的正向净收益。" if change["delta_pp"]>0 else "新增预算带来了可测的负向净变化。"))
    lines=["# 无教师路由的第 4 轮与第 8 轮对照","",summary,"",conclusion,"",
           "新确认面板在训练前冻结。每个种子从上一轮锁定的末轮权重接续，使用相同顺序的 200 个新开发房间，分四轮采样，每轮 50 间、300 次 Adam 更新。第 8 轮累计见过 400 个开发房间、更新 2,400 次。自采前缀、单步分叉、自身续跑、保持损失、公开停滞门控和九个语义特征的屏蔽方式沿用上一轮。沿用原实现，每轮重置优化器，采样小批次的随机种子为训练种子加绝对轮数。",
           "","## 新确认面板的成对比较","",
           "| 种子 | 第 4 轮 | 第 8 轮 | 相对第 4 轮救回/丢失 | 保留第 4 轮成功 | 净变化及 95% 区间（百分点） | 第 8 轮相对 B 救回/丢失 |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    for s,v in main_result["seed"].items():
        a,b=v["change"],v["r8_vs_B"]
        lines.append(f'| {s} | {v["r4"]} | {v["r8"]} | {a["rescued"]:.0f}/{a["lost"]:.0f} | {v["retained"]}/{v["r4"]} | {delta_text(a)} | {b["rescued"]:.0f}/{b["lost"]:.0f} |')
    lines+=["",f'三个种子中有 {positive} 个在第 8 轮多解了房间。平均救回 {change["rescued"]:.1f} 间、丢失 {change["lost"]:.1f} 间。相对固定单 CAP B 的净差从第 4 轮 {delta_text(main_result["mean"]["r4_vs_B"])} 到第 8 轮 {delta_text(c8)} 个百分点。',
            "", f'从 B 单独执行的结果看，第 4 轮平均救回 {main_result["mean"]["r4_vs_B"]["rescued"]:.1f} 间、丢失 {main_result["mean"]["r4_vs_B"]["lost"]:.1f} 间。到第 8 轮分别为 {c8["rescued"]:.1f} 和 {c8["lost"]:.1f} 间。新接手带来的收益需要与原有成功的保留一起衡量。',
            "", "## 已查看面板的逐轮诊断","","![三个训练种子的逐轮闭环曲线](trend.png)","",
            "| 种子 | 第 4 轮 | 第 5 轮 | 第 6 轮 | 第 7 轮 | 第 8 轮 |","|---|---:|---:|---:|---:|---:|"]
    for s in run.SEEDS:
        lines.append(f"| {s} | "+" | ".join(str(t[f"s{s}_r{r}"]) for r in range(4,9))+" |")
    lines+=["",f'该面板固定 B 为 {t["B"]}/200。第 4 轮与 B 的结果沿用上轮相同房间、相同权重的已完成闭环。第 5 至 8 轮全部重新执行。按预定方案，确认面板只比较两端和 B。',"","## 新增训练数据","",
        "| 种子 | 新增分叉 | 产生硬偏好的分叉 | 第 5 至 8 轮各轮硬偏好 |","|---|---:|---:|---|"]
    fit={}
    for s in run.SEEDS:
        root=out/f"s{s}"/"self"
        rows=[e.read(root/f"fit{i}.json") for i in range(5,9)]
        fit[s]=rows
        lines.append(f'| {s} | {sum(x["forks"] for x in rows)} | {sum(x["informative"] for x in rows)} | '+", ".join(str(x["informative"]) for x in rows)+" |")
    lines+=["","新增训练房间和确认房间按去除 scene_id 后的内容哈希去重。三个种子共用训练房间和评估房间。区间按房间成对重采样 10,000 次，每次保留同一房间的三个种子，描述这三个固定模型在新房间上的不确定性。该对照同时增加训练房间和参数更新次数，可判断两者合计的收益。","",
            "[逐房间结果](rooms.csv) · [配对统计](results.json)"]
    e.save(doc/"results.json",dict(panels=stats,fit=fit))
    (doc/"README.md").write_text("\n".join(lines)+"\n")
    registry=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(registry.read_text())
    item=next(row for row in entries if row["id"]=="self-budget")
    item["result"]=summary
    item["analysis"]=f'第 8 轮相对 B {delta_text(c8)} 个百分点。相对第 4 轮，每个种子平均救回 {change["rescued"]:.1f} 间、丢失 {change["lost"]:.1f} 间。'
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    e.save(out/"FINISHED.json",dict(primary=main_result["mean"],counts=main_result["totals"],
           result_sha=run.d.source.graph.file_sha(doc/"results.json")))


if __name__=="__main__":
    main()
