"""Aggregate paired room outcomes without treating training seeds as extra rooms."""
import csv
import json
from pathlib import Path
import numpy as np
from tools import cap_self_replication as run

e, out, doc = run.e, run.OUT, run.DOC


def outcomes(panel, rooms):
    data={}
    for name in ("A","B",*[f"s{s}_r{r}" for s in run.SEEDS for r in (0,4)]):
        root=out/panel/"test"/name
        if panel=="diagnostic" and name in ("A","B","s17_r0","s17_r4"):
            old_name={"s17_r0":"self0","s17_r4":"self4"}.get(name,name)
            root=run.PREVIOUS/"test"/old_name
        assert {f.stem for f in root.glob("*.json")}==set(rooms)
        values=np.array([e.read(root/(k+".json"))["success"] for k in rooms],int)
        assert np.isin(values,[0,1]).all()
        data[name]=values
    return data


def comparison(x,y,indices):
    delta=x-y
    lo,hi=np.quantile(delta[indices].mean(axis=1)*100,[.025,.975])
    return dict(delta_pp=float(delta.mean()*100),ci_pp=[float(lo),float(hi)],
                rescued=float(np.maximum(delta,0).sum()),lost=float(np.maximum(-delta,0).sum()))


def describe(values,indices):
    def comp(x,y):
        return comparison(x,y,indices)
    a,b=values["A"],values["B"]
    groups={"A_only":(a==1)&(b==0),"B_only":(a==0)&(b==1),"both":(a==1)&(b==1),"neither":(a==0)&(b==0)}
    metrics={}
    for seed in run.SEEDS:
        x,y=(values[f"s{seed}_r{r}"] for r in (4,0))
        metrics[seed]=dict(initial=int(y.sum()),final=int(x.sum()),vs_initial=comp(x,y),vs_A=comp(x,a),vs_B=comp(x,b),
             strata={g:int(x[mask].sum()) for g,mask in groups.items()})
    endpoints=np.stack([values[f"s{s}_r4"] for s in run.SEEDS])
    starts=np.stack([values[f"s{s}_r0"] for s in run.SEEDS])
    comparisons={}
    for label,baseline in [("initial",starts),("A",a),("B",b)]:
        diff=endpoints-baseline
        gain=np.maximum(diff,0).sum(1).mean()
        loss=np.maximum(-diff,0).sum(1).mean()
        # First average each room across the three seeds, then resample the room vector.
        result=comp(diff.mean(0),np.zeros(len(a)))
        result.update(rescued=float(gain),lost=float(loss))
        comparisons[label]=result
    return dict(A=int(a.sum()),B=int(b.sum()),union=int(np.maximum(a,b).sum()),counts={g:int(mask.sum()) for g,mask in groups.items()},
                seed=metrics,mean_final=float(endpoints.sum(1).mean()),mean_initial=float(starts.sum(1).mean()),
                mean_vs=comparisons,seed_agreement_rooms=int(np.all(endpoints==endpoints[0],axis=0).sum()))


def train_stats(keys):
    stats={}
    for seed in run.SEEDS:
        root=run.model_dir(seed)
        rows=[]
        for step,group in enumerate(np.array_split(keys,4),1):
            a=b=branch_a=branch_b=room_count=0
            fit=e.read(root/f"fit{step}.json")
            for k in group:
                with np.load(root/"parents"/(k+".npz")) as z:
                    b+=int(z["b"].sum())
                    a+=len(z["b"])-int(z["b"].sum())
                labels=[e.read(path)["outcomes"] for path in (root/"forks").glob(k+"_*.json")]
                branch_a+=labels.count([1,0])
                branch_b+=labels.count([0,1])
                room_count+=int(any(x!=y for x,y in labels))
            assert a+b==fit["anchors"] and branch_a+branch_b==fit["informative"]
            rows.append({**fit,"chose_B":b,"A_wins":branch_a,"B_wins":branch_b,"informative_rooms":room_count})
        stats[seed]=rows
    return stats


def delta_text(metric):
    lo,hi=metric["ci_pp"]
    return f'{metric["delta_pp"]:+.1f} [{lo:+.1f}, {hi:+.1f}]'


def main():
    protocol=e.read(out/"protocol.json")
    doc.mkdir(exist_ok=True,parents=True)
    assert set(protocol["fresh"]).isdisjoint(protocol["diagnostic"]+protocol["train"])
    indices=np.random.default_rng(172943).integers(200,size=(10000,200))
    panels={}
    with (doc/"rooms.csv").open("w",newline="") as f:
        writer=csv.writer(f)
        for panel in ("fresh","diagnostic"):
            rooms=protocol[panel]
            assert len(rooms)==len(set(rooms))==200
            values=outcomes(panel,rooms)
            if panel=="fresh":
                writer.writerow(["panel","content_sha256",*values])
            writer.writerows([panel,k,*[v[i] for v in values.values()]] for i,k in enumerate(rooms))
            panels[panel]=describe(values,indices)
    train=train_stats(protocol["train"])
    primary=panels["fresh"]
    baseline=max(primary["A"],primary["B"])
    beats=sum(row["final"]>baseline for row in primary["seed"].values())
    improved=sum(row["final"]>row["initial"] for row in primary["seed"].values())
    summary=f'新面板 A 成功 {primary["A"]}/200，B 成功 {primary["B"]}/200。seed17、29、43 的最终成功数分别为 '+ '、'.join(str(primary["seed"][s]["final"]) for s in run.SEEDS)+f'。三个种子中有 {improved} 个超过自身起点，{beats} 个超过同面板最佳单 CAP。'
    mean=primary["mean_vs"]
    lines=["# 无教师路由的自采自续能否跨种子复现","",summary,"",
           f'主评估按种子等权，最终平均成功 {primary["mean_final"]:.1f}/200。相对先验比较对象 B 的成功率差为 {delta_text(mean["B"])} 个百分点，相对各自起点为 {delta_text(mean["initial"])} 个百分点。区间按房间配对重采样，条件于这三个已训练模型。',"",
           "## 冻结方案","",
           "沿用首轮协议和 200 个开发房间的顺序，四轮各使用 50 间。seed17 复用已锁定的初始和末轮权重；seed29、43 各自从头采样和拟合。每个种子控制 PyTorch 随机初始化和每轮小批抽样，选取分叉的时间分层与哈希规则相同。每个边界先由 A、B 各提一个动作，在共享停滞门控开放且提案不同时由 actor 选择。每间至多抽三个分叉，在同一前缀比较两条首步，均由该轮冻结的 actor 续跑。四轮各 300 次 Adam 更新。训练和部署均屏蔽九个显式教师语义特征，保留几何、动作和短历史。","",
           "拟合前冻结新的 200 个自然房间，按去除 scene_id 的内容哈希核对与开发房间、上一轮面板和历史登记集无重合。各种子共享房间。原先得到 114/200 的面板作为诊断，与新面板分开报告。仅评估起点和预先指定的第四轮，未根据评测选种子或轮次。",""]
    outside=[primary["seed"][s]["strata"]["neither"] for s in run.SEEDS]
    both_lost=[primary["counts"]["both"]-primary["seed"][s]["strata"]["both"] for s in run.SEEDS]
    lines+=["## 结果解读","",
            f'在两位专家分别执行都失败的 {primary["counts"]["neither"]} 间中，三个模型分别救回 {outside[0]}、{outside[1]}、{outside[2]} 间。在两位专家都成功的 {primary["counts"]["both"]} 间中，三个模型也分别丢失 {both_lost[0]}、{both_lost[1]}、{both_lost[2]} 间。能力组合可以跨种子观察到，稳定的整局净收益在新面板上仍缺乏证据。',
            "",f'相对 B，每个种子平均救回 {mean["B"]["rescued"]:.1f} 间，同时丢掉 {mean["B"]["lost"]:.1f} 间。保留原有成功与发现新解对最终收益都很重要。',""]
    for panel,title in (("fresh","新冻结的 200 间"),("diagnostic","此前已查看的 200 间")):
        r=panels[panel]
        lines += [f"## {title}","",f'单 CAP A 为 {r["A"]}，B 为 {r["B"]}，两者整局成功并集为 {r["union"]}。表中救回/丢失均以同一房间配对。区间为 10,000 次重采样得到的 95% 区间，差值单位为百分点。',"",
                "| 种子 | 起点 | 第四轮 | 对起点救回/丢失 | 对 A 救回/丢失 | 对 B 救回/丢失 | 对 B 净差及区间 |","|---|---:|---:|---:|---:|---:|---:|"]
        def movement(v):
            return f'{v["rescued"]:.0f}/{v["lost"]:.0f}'
        for seed,v in r["seed"].items():
            lines.append(f'| {seed} | {v["initial"]} | {v["final"]} | {movement(v["vs_initial"])} | {movement(v["vs_A"])} | {movement(v["vs_B"])} | {delta_text(v["vs_B"])} |')
        lines += ["",f'按种子等权，相对起点、A、B 的百分点差及区间分别为 {delta_text(r["mean_vs"]["initial"])}、{delta_text(r["mean_vs"]["A"])}、{delta_text(r["mean_vs"]["B"])}。三个末轮模型在 {r["seed_agreement_rooms"]}/200 间上成败一致。',"",
                  "| 层别 | 房间数 | seed17 | seed29 | seed43 |","|---|---:|---:|---:|---:|"]
        for g,label in [("A_only","仅 A 成功"),("B_only","仅 B 成功"),("both","两者成功"),("neither","两者失败")]:
            lines.append(f'| {label} | {r["counts"][g]} | '+' | '.join(str(r["seed"][s]["strata"][g]) for s in run.SEEDS)+' |')
        lines.append("")
    lines+=["## 分叉监督","", "| 种子 | 采样分叉 | 成败不同 | 仅 A 首步胜 | 仅 B 首步胜 | 有偏好的房间 | 首轮采样选 B 的边界比例 |","|---|---:|---:|---:|---:|---:|---:|"]
    for seed,rows in train.items():
        totals=[sum(r[field] for r in rows) for field in ("forks","informative","A_wins","B_wins","informative_rooms")]
        lines.append(f"| {seed} | "+" | ".join(map(str,totals))+f' | {rows[0]["chose_B"]/rows[0]["anchors"]:.1%} |')
    lines+=["","首次启动的两个新增目录因房间生成种子覆盖了训练种子，在测试前隔离并重新从头训练。纳入报告的三份初始权重均从其声明种子独立重建逐值核对。","","同一房间的三个种子保留在同一重采样单位内。均值区间描述这三个固定模型在新房间上的不确定性；训练重复仅三次，跨随机初始化的变异用逐种子结果呈现。所有种子共用训练房间，尚未检验更换开发池后的变异。","",
            "[逐房间成败](rooms.csv) · [配对统计与逐轮监督量](results.json)"]
    e.save(doc/"results.json",dict(panels=panels,training=train))
    (doc/"README.md").write_text("\n".join(lines)+"\n")
    registry=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(registry.read_text())
    entry=next(x for x in entries if x["id"]=="self-replication")
    entry["result"]=summary
    entry["analysis"]=f'按种子等权，相对自身起点 {delta_text(mean["initial"])}，相对 B {delta_text(mean["B"])} 个百分点。'
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    e.save(out/"FINISHED.json",dict(results=run.d.source.graph.file_sha(doc/"results.json"),fresh=primary["mean_final"],seeds=run.SEEDS))



if __name__=="__main__":
    main()
