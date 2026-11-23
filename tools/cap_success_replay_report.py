"""Closed-loop retention and optimizer-component results for the single update."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tools import cap_success_replay as run

OUT,DOC,e=run.OUT,run.DOC,run.e
ARM_NAMES={"parent":"第 4 轮起点","vanilla":"原更新","all":"全部轨迹加 64","success":"成功轨迹加 64"}


def paired_interval(matrix,indices):
    per_room=matrix.mean(axis=0)
    resampled=per_room[indices].mean(axis=1)*100
    return [float(x) for x in np.quantile(resampled,[.025,.975])]


def conditional_difference(delta,mask,indices):
    numerator=(delta*mask).sum(axis=0)
    denom=mask.sum(axis=0)
    draws=numerator[indices].sum(axis=1)/denom[indices].sum(axis=1)*100
    return dict(delta_pp=float(numerator.sum()/denom.sum()*100),ci_pp=np.quantile(draws,[.025,.975]).tolist())


def closed_loop(panel,keys,writer,*,roots=None,comparisons=None):
    if roots is None:
        roots={arm:OUT for arm in run.MODES}
    if comparisons is None:
        comparisons=[("success","all"),("all","vanilla"),("success","vanilla"),("success","parent")]
    matrices={}
    rows={}
    for arm,directory in roots.items():
        values=[]
        for s in run.SEEDS:
            root=directory/panel/f"s{s}_{arm}"
            assert {p.stem for p in root.glob("*.json")}==set(keys)
            y=np.array([e.read(root/(k+".json"))["success"] for k in keys])
            assert np.isin(y,[0,1]).all()
            writer.writerows((panel,k,s,arm,int(v)) for k,v in zip(keys,y,strict=True))
            values.append(y)
        matrices[arm]=np.stack(values)
    p=matrices["parent"]
    for arm,v in matrices.items():
        retained=(p*v).sum(1); lost=p.sum(1)-retained
        gain=((1-p)*v).sum(1)
        rows[arm]=dict(success=v.sum(1).tolist(),parent_success=p.sum(1).tolist(),
            retained=retained.tolist(),loss=lost.tolist(),gain=gain.tolist(),
            retention=float(retained.sum()/p.sum()),rescue=float(gain.sum()/(1-p).sum()))
    idx=np.random.default_rng(170529).integers(len(keys),size=(10000,len(keys)))
    pairs={}
    for arm,ref in comparisons:
        v,b=matrices[arm],matrices[ref]
        delta=v-b
        pairs[arm+"_vs_"+ref]=dict(delta_pp=float(delta.mean()*100),ci_pp=paired_interval(delta,idx),
             retention_difference=conditional_difference(delta,p,idx),
             rescue_difference=conditional_difference(delta,1-p,idx),
             wins=(delta>0).sum(1).tolist(),losses=(delta<0).sum(1).tolist())
    return dict(rooms=len(keys),arms=rows,contrasts=pairs)


def curves(doc=DOC,arms=("all","success"),titles=("All routes for the extra 64","Successful routes for the extra 64")):
    with (doc/"gradients.csv").open() as f:
        rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(2,2,figsize=(10,6.6),layout="constrained",sharex=True)
    for j,arm in enumerate(arms):
        selected=[r for r in rows if r["arm"]==arm]
        blocks=[[r for r in selected if k<int(r["step"])<=k+25] for k in range(0,300,25)]
        x=np.arange(13,301,25)
        for name,title,color in [("pref","Fork preference","#205ea6"),("soft","Soft retention","#ce6417"),("margin","Margin retention","#208264")]:
            axes[0,j].plot(x,[np.mean([float(r[name+"_norm"]) for r in b]) for b in blocks],label=title,color=color)
            axes[1,j].plot(x,[100*sum(float(r[name+"_step_dot"]) for r in b)/sum(float(r["step_sq"]) for r in b) for b in blocks],color=color)
        axes[0,j].set_title(titles[j])
        axes[1,j].set_xlabel("Adam update (25-step windows)")
    axes[0,0].set_ylabel("Weighted gradient L2 norm")
    axes[1,0].set_ylabel("Projection onto Adam step (%)")
    axes[0,1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    fig.savefig(doc/"gradients.png",dpi=160)
    plt.close(fig)


def main():
    plan=e.read(OUT/"protocol.json")
    assert set(plan["fresh"]).isdisjoint(plan["train"])
    DOC.mkdir(exist_ok=True,parents=True)
    with (DOC/"rooms.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["panel","room_content_sha256","seed","arm","success"])
        panels={p:closed_loop(p,plan[p],writer) for p in ("fresh","train")}
    stats={}
    for arm in run.MODES[1:]:
        rows=[e.read(OUT/"fit"/f"s{s}_{arm}.json") for s in run.SEEDS]
        stats[arm]={k:sum(r[k] for r in rows) for k in ["hard","hard_correct","changed_on_success","successful_anchors","chose_different_expert","parent_switches","anchors"]}
    grads=e.read(DOC/"gradient_summary.json")
    curves()
    fresh=panels["fresh"]
    contrast=fresh["contrasts"]["success_vs_all"]
    def interval(v):
        lo,hi=v["ci_pp"]
        return f'{v["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
    result=(f'新 200 间上，成功路线组相对等预算全部轨迹组的净差为 {interval(contrast)} 个百分点。'
            f'旧成功的保留率差为 {interval(contrast["retention_difference"])} 个百分点，旧失败的救回率差为 {interval(contrast["rescue_difference"])} 个百分点。')
    lines=["# 成功路线额外保留样本的单轮配对实验","",result,"",
      "三种子各自从锁定的第 4 轮权重出发，读取原第 5 轮同一批 50 个开发房间的前缀和分叉。固定 64 个偏好样本和 128 个软保持样本的小批抽样序列，原更新复现的最终权重逐值等于此前第 5 轮。两组新方法都加 64 个额外边界，用同一个独立随机数流先均匀抽符合条件的轨迹，再均匀抽边界。两组额外池均排除已知有严格偏好的边界，后者继续接受原硬监督。成功组只改变可抽取轨迹的成败条件。两组的网络、九个语义特征的屏蔽、Adam、300 次更新保持一致。","",
      "余量项为 0.25 × mean(max(0, 0.2 − s × logit))，原选择 B 时 s=+1，原选择 A 时 s=−1。0.2 logit 对应原选择概率约 0.55。三个损失分别取均值后相加。权重和余量在看结果之前固定。",""]
    for panel,title in [("fresh","新冻结的 200 个自然房间"),("train","用于此次更新的 50 个开发房间")]:
        r=panels[panel]
        lines += [f"## {title}","",
            "| 组别 | seed17 成功 | seed29 成功 | seed43 成功 | 平均保住 | 平均丢失 | 平均救回 | 旧成功保留率 | 旧失败救回率 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm,a in r["arms"].items():
            lines.append("| "+ARM_NAMES[arm]+" | "+" | ".join(map(str,a["success"]))+f' | {np.mean(a["retained"]):.1f} | {np.mean(a["loss"]):.1f} | {np.mean(a["gain"]):.1f} | {a["retention"]:.1%} | {a["rescue"]:.1%} |')
        lines+=["","均以各自种子的第 4 轮成败区分旧成功和旧失败。以下差值按房间配对，95% 区间使用 10,000 次房间重采样，每次保留三个固定种子的整组记录。保留率和救回率的分母分别是各次重采样中的旧成功、旧失败数。训练房间上的区间仅作描述。",""]
        for k in ["success_vs_all","success_vs_vanilla"]:
            v=r["contrasts"][k]
            lines.append(f'- {ARM_NAMES["success"]} 相对 {ARM_NAMES[k.split("_vs_")[1]]}，闭环净差 {interval(v)}，保留率差 {interval(v["retention_difference"])}，救回率差 {interval(v["rescue_difference"])} 个百分点。')
        lines.append("")
    lines+=["## 固定成功前缀上的改选","",
            "| 组别 | 旧成功边界发生改选 | 有效分叉偏好命中 |","|---|---:|---:|"]
    for arm,r in stats.items():
        lines.append(f'| {ARM_NAMES[arm]} | {r["changed_on_success"]}/{r["successful_anchors"]} | {r["hard_correct"]}/{r["hard"]} |')
    lines+=["","## 梯度与参数更新","",
      "每一步在同一参数和同一批次，分别计算三部分加权损失的全参数梯度。方向余弦为负表示两项在一阶上拉向相反方向。将各项梯度分别维护 Adam 一阶矩，使用原更新的共同二阶矩分母，可将实际步长分解为三项。表中份额为各分量与总步长的内积之和除以总步长平方范数之和，按步长能量加权。这里使用实际共同的 Adam 缩放，各份额之和为 100%；它衡量这次更新方向的投影，不能直接归因任务成功率。","",
      "| 额外池 | 更新区间 | 梯度范数（偏好 / 软保持 / 余量） | Adam 方向份额（同顺序） | 偏好与软保持余弦 | 偏好与余量余弦 |",
      "|---|---|---|---|---:|---:|"]
    for arm in ("vanilla","all","success"):
        for k,label in [("early","1–50"),("all","1–300"),("late","251–300")]:
            row=grads[arm][k]
            norms=" / ".join(f'{row[n]["mean_norm"]:.3f}' for n in ("pref","soft","margin"))
            shares=" / ".join(f'{row[n]["share"]*100:.1f}%' for n in ("pref","soft","margin"))
            margin_cos=row["pref_margin_cos"]["mean"]
            lines.append(f'| {ARM_NAMES[arm]} | {label} | {norms} | {shares} | {row["pref_soft_cos"]["mean"]:.2f} | '+("—" if margin_cos is None else f"{margin_cos:.2f}")+" |")
    lines+=["",f'成功组额外抽取的边界中，全程平均 {grads["success"]["all"]["margin_active"]:.1%} 低于余量、会触发梯度，最后 50 步仅 {grads["success"]["late"]["margin_active"]:.1%}。全部轨迹组对应为 {grads["all"]["all"]["margin_active"]:.1%} 和 {grads["all"]["late"]["margin_active"]:.1%}。', "", "![逐步梯度范数和 Adam 方向份额](gradients.png)","",
      "九份优化回放的最终权重均与闭环所用模型逐值一致，方向余弦仅汇总范数乘积超过 1e-12 的项。该实验只更新一次，没有测试多轮积累或其他余量权重。","",
      "[逐房间数据](rooms.csv) · [配对统计](results.json) · [每步梯度](gradients.csv) · [分时段和分种子梯度](gradient_summary.json)"]
    e.save(DOC/"results.json",dict(panels=panels,training=stats))
    (DOC/"README.md").write_text("\n".join(lines)+"\n")
    registry=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(registry.read_text())
    j=next(x for x in entries if x["id"]=="success-replay")
    j["result"]=result
    j["analysis"]="成功组前 50 步偏好损失主导 Adam 更新方向，最后 50 步软保持项占约 76%。"
    if ["gradients.png","梯度与参数更新曲线"] not in j["assets"]:
        j["assets"].append(["gradients.png","梯度与参数更新曲线"])
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    e.save(OUT/"FINISHED.json",dict(primary=contrast,results_sha=run.d.source.graph.file_sha(DOC/"results.json")))


if __name__=="__main__":
    main()
