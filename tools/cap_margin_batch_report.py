"""Paired results for replacing all soft retention samples with successful routes."""
import csv
import json
from pathlib import Path
import numpy as np
from tools import cap_margin_batch as run
from tools import cap_success_replay_report as old_report

out,doc,e=run.OUT,run.DOC,run.e
names={"parent":"第 4 轮","vanilla":"仅软保持 128","success":"软保持 128 ＋成功余量 64","only64":"仅成功余量 64","only192":"仅成功余量 192"}


def fmt(row):
    a,b=row["ci_pp"]
    return f'{row["delta_pp"]:+.2f} [{a:+.2f}, {b:+.2f}]'


def successful_ties(keys):
    totals=dict(n=0,parent_B=0,only192_B=0,changed=0)
    for seed in run.SEEDS:
        root=run.old.prior.OUT/f"s{seed}"/"self"
        run.old.d.MODELS["history"]=run.old.study.cold.weights(out/"models"/f"s{seed}_only192.npz")
        for key in keys:
            indices=[int(f.stem.split("_")[-1]) for f in (root/"forks").glob(key+"_*.json")
                     if e.read(f)["outcomes"]==[1,1]]
            if not indices:
                continue
            with np.load(root/"parents"/(key+".npz")) as data:
                parent=data["b"][indices]
                updated=run.old.d.logits(data["x"][indices],"history")>0
            totals["n"]+=len(indices)
            totals["parent_B"]+=int(parent.sum())
            totals["only192_B"]+=int(updated.sum())
            totals["changed"]+=int((updated!=parent).sum())
    return totals


def main():
    plan=e.read(out/"protocol.json")
    source=e.read(run.BASE/"protocol.json")
    assert plan["diagnostic"]==source["fresh"] and plan["train"]==source["train"]
    roots={m:(out if m in run.ARMS else run.BASE) for m in names}
    comparisons=[("only192","success"),("only64","success"),("only192","only64"),("only192","vanilla"),("only192","parent")]
    with (doc/"rooms.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["panel","room_content_sha256","seed","arm","success"])
        panels={p:old_report.closed_loop(p,source[p],writer,roots=roots,comparisons=comparisons) for p in ("fresh","train")}
    old_report.curves(doc,tuple(run.ARMS),("Margin only: 64 samples","Margin only: 192 samples"))
    grad=e.read(doc/"gradient_summary.json")
    old_grad=e.read(run.old.DOC/"gradient_summary.json")
    v=panels["fresh"]["contrasts"]["only192_vs_success"]
    msg=f'同一 200 间上，仅成功余量 192 对比软保持 128 ＋成功余量 64，闭环净差 {fmt(v)} 个百分点。'
    lines=["# 去掉全局软保持并增大成功余量批次","",msg,"",
           "三个种子各自从既有第 4 轮开始，以同一 50 个开发房间的第 5 轮分叉做一次 300 步更新。每次抽 64 个硬偏好。关闭软保持的两组分别另抽 64 和 192 个成功路线边界，先均匀抽成功轨迹，再均匀抽轨迹中的可用边界，均有放回。192 组的前 64 个与小批组及原有双保持组逐步相同，剩余 128 个使用独立随机数流。每批余量损失仍对所有余量样本取平均，权重 0.25，logit 余量 0.2，阈值对应旧动作概率约 55%。模型、优化器、分叉偏好小批次和训练次数一致。","",
           "每个可路由边界的旧动作标签跟随旧 actor 的实际选择。两条首步在旧 actor 续跑下都成功时保留该选择，余量满足后梯度为零。两专家从同一初态分别成功不会触发整局的专门选路规则。",""]
    for p,title in [("fresh","此前已查看的 200 个自然房间"),("train","用于本次更新的 50 个开发房间")]:
        res=panels[p]
        lines+=["## "+title,"","| 更新方案 | seed17 | seed29 | seed43 | 平均救回 | 平均丢失 | 旧成功保留率 | 旧失败救回率 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
        for m,r in res["arms"].items():
            lines.append("| "+names[m]+" | "+" | ".join(map(str,r["success"]))+f' | {np.mean(r["gain"]):.1f} | {np.mean(r["loss"]):.1f} | {r["retention"]:.1%} | {r["rescue"]:.1%} |')
        lines+=["","| 配对比较 | 成功率差 | 保留率差 | 救回率差 |","|---|---:|---:|---:|"]
        for a,b in comparisons:
            c=res["contrasts"][a+"_vs_"+b]
            lines.append(f'| {names[a]} 对 {names[b]} | {fmt(c)} | {fmt(c["retention_difference"])} | {fmt(c["rescue_difference"])} |')
        lines.append("")
    lines+=["## 固定开发边界与梯度","",
            "| 组别 | 成功边界改选 | 分叉拟合命中 | 有效余量比例 | 全程 Adam 方向份额（偏好 / 软保持 / 余量） |","|---|---:|---:|---:|---:|"]
    for m in ("success","only64","only192"):
        fits=[e.read(roots[m]/"fit"/f"s{s}_{m}.json") for s in run.SEEDS]
        z=(grad if m in run.ARMS else old_grad)[m]["all"]
        shares=" / ".join(f'{z[k]["share"]:.1%}' for k in ("pref","soft","margin"))
        lines.append(f'| {names[m]} | {sum(f["changed_on_success"] for f in fits)}/{sum(f["successful_anchors"] for f in fits)} | {sum(f["hard_correct"] for f in fits)}/{sum(f["hard"] for f in fits)} | {z["margin_active"]:.1%} | {shares} |')
    lines+=["","![梯度和 Adam 方向份额](gradients.png)","",
            "三个种子使用同一评估面板，所有差值单位为百分点，95% 区间按房间配对重采样 10,000 次，保留每个房间的三个模型。训练数据上的区间仅为描述。该面板已在之前的实验中查看，单轮结果不能证明多轮稳定性。旧模型结果来自同房间的历史实测，新组全部从模拟器闭环执行；梯度回放的权重与部署权重逐值一致。","",
            "[逐房间结果](rooms.csv) · [统计](results.json) · [逐步梯度](gradients.csv)"]
    ties=successful_ties(plan["train"])
    fits=[e.read(out/"fit"/f"s{s}_only192.json") for s in run.SEEDS]
    ties["eligible_routes"]=sum(r["pool_rooms"] for r in fits)
    ties["eligible_boundaries"]=sum(r["pool_boundaries"] for r in fits)
    ties["untested"]=ties["eligible_boundaries"]-ties["n"]
    panels["successful_forks"]=ties
    lines+=["","## 两条首步都成功的训练分叉","",
            f'在 {ties["n"]} 个这样的固定前缀上，旧 actor 选 B {ties["parent_B"]} 次，仅余量 192 更新后选 B {ties["only192_B"]} 次，总共改选 {ties["changed"]} 次。三个种子按记录计数。成功余量池合计 {ties["eligible_routes"]} 条可抽轨迹、{ties["eligible_boundaries"]} 个边界，其中 {ties["untested"]} 个没有实测另一首步。']
    e.save(doc/"results.json",panels)
    (doc/"README.md").write_text("\n".join(lines)+"\n")
    p=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(p.read_text())
    j=next(x for x in entries if x["id"]=="margin-batch")
    j["result"]=msg
    j["analysis"]=f'关闭软保持后 192 对 64 的成功率净差为 {fmt(panels["fresh"]["contrasts"]["only192_vs_only64"])} 个百分点。'
    p.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    e.save(out/"FINISHED.json",dict(primary=v,results_sha=run.old.d.source.graph.file_sha(doc/"results.json")))


if __name__=="__main__":
    main()
