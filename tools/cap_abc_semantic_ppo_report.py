"""Report paired effects of author features before and after ABC PPO."""
import csv
import json

import numpy as np

from tools import cap_abc_semantic_ppo as run


def write_csv(name, rows):
    with (run.DOC/name).open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def effect(left, right, draws):
    delta = left - right
    rooms = delta.mean(axis=1)
    return dict(pp=100*float(rooms.mean()), ci=(100*np.quantile(rooms[draws].mean(1),[.025,.975])).tolist(),
                gained=(delta>0).sum(0).tolist(), lost=(delta<0).sum(0).tolist())


def main():
    plan = run.read(run.OUT/"protocol.json")
    assert run.ppo.sha(run.DOC/"features.py")==plan["feature_sha"]
    tags = {(arm,stage):[f"{arm}_s{s}_r{stage}" for s in run.SEEDS]
            for arm in run.ARMS for stage in run.EVAL_STAGES}
    rooms, stats = [], []
    for key in plan["evaluation"]:
        row = dict(room=key)
        for model in (*run.EXPERTS, *(t for models in tags.values() for t in models)):
            data = run.read(run.OUT/"evaluation"/model/(key+".json"))
            row[model] = data["success"]
            row[model+"_ticks"] = data["ticks"]
        rooms.append(row)
    assert len(rooms)==200 and len({r["room"] for r in rooms})==200
    write_csv("rooms.csv",rooms)
    for models in tags.values():
        if not models[0].endswith("r4"):
            continue
        for tag in models:
            name = tag.rsplit("_",1)[0]
            for stage in range(1,5):
                stats.append(dict(tag=name, **run.read(run.OUT/"metrics"/f"{name}_r{stage}.json")))
    assert sum(v["episodes"] for v in stats)==1200
    write_csv("training.csv",stats)
    values = {a:np.asarray([[r[a]] for r in rooms],int) for a in run.EXPERTS}
    for (arm,stage), models in tags.items():
        values[f"{arm}{stage}"] = np.asarray([[r[t] for t in models] for r in rooms],int)
    masks = {a:np.logical_and.reduce([(values[k][:,0] == (k==a)) for k in run.EXPERTS]) for a in run.EXPERTS}
    masks["none"] = np.logical_and.reduce([values[k][:,0]==0 for k in run.EXPERTS])
    masks["AB_neither"] = (values["A"][:,0]==0) & (values["B"][:,0]==0)
    draws = np.random.default_rng(154991).integers(len(rooms),size=(10000,len(rooms)))
    pairs = {
        "S4_minus_B4": ("semantic4","basic4"),
        "S0_minus_B0": ("semantic0","basic0"),
        "S4_minus_S0": ("semantic4","semantic0"),
        "B4_minus_B0": ("basic4","basic0"),
        "S0_minus_expertB": ("semantic0","B"),
        "B0_minus_expertB": ("basic0","B"),
        "S4_minus_expertB": ("semantic4","B"),
        "B4_minus_expertB": ("basic4","B"),
    }
    effects = {key:effect(values[x],values[y],draws) for key,(x,y) in pairs.items()}
    effects["increment_difference"] = effect(values["semantic4"]-values["semantic0"],
                                              values["basic4"]-values["basic0"],draws)
    counts = {key:arr.sum(0).tolist() for key,arr in values.items()}
    strata = {key:dict(n=int(mask.sum()),counts={a:v[mask].sum(0).tolist() for a,v in values.items()})
              for key,mask in masks.items()}
    demo = run.read(run.OUT/"demo_summary.json")
    supervision = {p.stem:run.read(p) for p in sorted((run.OUT/"supervision").glob("*.json"))}
    first = {a:[v["entropy"] for v in stats if v["stage"]==1 and v["tag"].startswith(a+"_")] for a in run.ARMS}
    feature_count = len(plan["feature_names"])
    variance_count = np.zeros(feature_count,int)
    for path in (run.OUT/"demonstration").glob("*/*.npz"):
        with np.load(path) as sample:
            x = sample["x"][...,run.BASE_SIZE:run.BASE_SIZE+feature_count]
            variance_count += (np.ptp(x,axis=1)>1e-9).sum(0)
    feature_variation = dict(zip(plan["feature_names"],variance_count.tolist(),strict=True))
    out = dict(counts=counts, effects=effects, strata=strata, demonstration=demo,
               feature_distinct_candidates=feature_variation,
               supervised_fit=supervision, first_round_entropy=first, cases=len(rooms))
    run.save(run.DOC/"results.json",out)

    def interval(key):
        q = effects[key]
        lo,hi = q["ci"]
        return f'{q["pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'

    lines = ["# A/B/C 路由的作者语义输入对照", "",
             f"主比较为 200 局 PPO 后语义组减基础组，**{interval('S4_minus_B4')} 个百分点**。",
             f"两组在监督起点的差为 {interval('S0_minus_B0')}，PPO 增幅之差为 {interval('increment_difference')}。", "",
             "| 策略 | 成功 / 200，seed 17 / 29 / 43 | 平均成功率 |",
             "|---|---:|---:|"]
    labels = dict(A="纯 A", B="纯 B", C="纯 C", basic0="基础输入暖启动",
                  semantic0="加入作者语义的暖启动", basic4="基础输入＋PPO",
                  semantic4="加入作者语义＋PPO")
    for key,n in labels.items():
        vals = counts[key]
        lines.append(f'| {n} | {" / ".join(map(str,vals))} | {np.mean(vals)/2:.2f}% |')
    lines += ["", "| 闭环配对比较 | 成功率差，百分点和 95% 区间 | 各种子救回 / 丢失 |",
              "|---|---:|---:|"]
    for key,n in [("S4_minus_B4","PPO 后语义减基础"),("S0_minus_B0","暖启动语义减基础"),
                  ("B4_minus_B0","基础 PPO 后减起点"),("S4_minus_S0","语义 PPO 后减起点"),
                  ("B4_minus_expertB","基础 PPO 后减纯 B"),("S4_minus_expertB","语义 PPO 后减纯 B")]:
        q = effects[key]
        pairs = "、".join(f"{w}/{l}" for w,l in zip(q["gained"],q["lost"],strict=True))
        lines.append(f"| {n} | {interval(key)} | {pairs} |")
    lines += ["", "区间按新评测面板的房间配对重抽 10,000 次，每次保留同一房间的三个种子。初始化和更新使用配对种子。区间以这三份初始化为条件。",
              "", "## 能力保留", "",
              "| 纯专家成败分层 | 房间数 | 基础起点 | 语义起点 | 基础 PPO 后 | 语义 PPO 后 |",
              "|---|---:|---:|---:|---:|---:|"]
    for key, n in [("A","只有 A 成功"),("B","只有 B 成功"),("C","只有 C 成功"),
                   ("AB_neither","A、B 都失败"),("none","A、B、C 都失败")]:
        q = strata[key]
        cells = ' | '.join('/'.join(map(str,q["counts"][a])) for a in ("basic0","semantic0","basic4","semantic4"))
        lines.append(f'| {n} | {q["n"]} | {cells} |')
    lines += ["", "A、B 都失败一行包含 C 独胜和三个专家全败的房间。这里的成败按同场实际纯专家完整执行定义。","",
              f"原有 200 个开发房间的纯专家成功数为 A {demo['by_expert']['A']}、B {demo['by_expert']['B']}、C {demo['by_expert']['C']}。"
              f"示范覆盖并集 {demo['union']}，有效分歧边界 {demo['boundaries']:,}。两组共用全部样本与优化预算。"
              f"首轮基础组与语义组的平均动作熵分别为 {np.mean(first['basic']):.3f} 和 {np.mean(first['semantic']):.3f} nats。","",
              "四个冻结语义量为保持有效接触推动、保持有效接触准备、改动反复停滞的动作意图、双机器人推动方向对抗。"
              "这些量由相同公开输入与任一候选命令计算，critic 在两组均采用基础信息。"
              f"四个量在成功示范的候选之间分别有 {', '.join(map(str,variance_count))} / {demo['boundaries']} 个边界出现差异。"
              "较早的 A/B 或 A/B/C 面板未用于给本轮选择 PPO 权重。","",
              "[协议](plan.md) · [作者笔记](author_notes.md) · [冻结函数](features.py) · [逐房间结果](rooms.csv) · [训练记录](training.csv) · [配对与分层统计](results.json)"]
    (run.DOC/"README.md").write_text("\n".join(lines)+"\n")
    registry = run.DOC.parent.parent/"experiment_journal/entries.json"
    items = run.read(registry)
    item = next(v for v in items if v["id"]=="abc-semantic-ppo")
    item["result"] = f"语义组相对基础组在 warm start 时为 {interval('S0_minus_B0')}，PPO 后为 {interval('S4_minus_B4')} 个百分点。"
    item["analysis"] = f"200 个新房间，三种子配对。PPO 增幅之差为 {interval('increment_difference')} 个百分点。"
    registry.write_text(json.dumps(items,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
