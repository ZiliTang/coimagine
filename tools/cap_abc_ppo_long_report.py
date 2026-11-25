"""Summarize the fixed-seed ABC continuation and its independent confirmation."""
import csv
import json

import numpy as np

from tools import cap_abc_ppo_long as run
from tools.cap_option_handoff_opportunity_report import paired

ARMS = ("A","B","C","r0","r4","r40")


def write_csv(name, rows):
    with (run.DOC/name).open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    plan=run.read(run.OUT/"protocol.json")
    assert run.read(run.OUT/"progress/confirm.json")["done"]==1200
    rooms=[]
    for key in plan["confirm"]:
        row=dict(room=key)
        for name in ARMS:
            result=run.read(run.OUT/"evaluation"/name/(key+".json"))
            row[name]=result["success"]
            row[name+"_ticks"]=result["ticks"]
        rooms.append(row)
    assert len(rooms)==len({q["room"] for q in rooms})==200
    write_csv("rooms.csv",rooms)

    training=[]
    for r in range(1,run.END+1):
        source=(run.PARENT/"metrics"/f"basic_s17_r{r}.json" if r<=run.START
                else run.OUT/"metrics"/f"r{r:02d}.json")
        row=run.read(source)
        row.pop("actor_sha",None)
        row.pop("state_sha",None)
        training.append(dict(cumulative_episodes=r*50,**row))
    assert sum(q["episodes"] for q in training)==2000
    write_csv("training.csv",training)

    raw={arm:np.array([r[arm] for r in rooms],int) for arm in ARMS}
    values={a:int(v.sum()) for a,v in raw.items()}
    rates={"r40_vs_r4":paired(raw["r40"],raw["r4"]),"r40_vs_r0":paired(raw["r40"],raw["r0"])}
    for arm in ("A","B","C"):
        rates["r40_vs_"+arm]=paired(raw["r40"],raw[arm])
    strata={}
    for arm in ("A","B","C","none"):
        mask=np.logical_and.reduce([raw[a]==int(a==arm) for a in ("A","B","C")])
        strata[arm]=dict(n=int(mask.sum()),counts={a:int(v[mask].sum()) for a,v in raw.items()})
    diag=[run.read(run.monitor_result(r)) for r in run.CHECKS]
    union=int(np.maximum.reduce([raw[a] for a in ("A","B","C")]).sum())
    run.save(run.DOC/"results.json",dict(confirmation=values,paired=rates,strata=strata,
                                      diagnostic=diag,union=union))

    def change(key):
        x=rates[key]
        lo,hi=(100*c for c in x["ci"])
        return f"{(x['first']-x['second'])/2:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    def rolling(batch):
        return sum(r["success"] for r in batch) / sum(r["episodes"] for r in batch)

    last=training[-5:]
    earlier=training[:5]
    s0,s4,s40=(values[k] for k in ("r0","r4","r40"))
    lines=["# A/B/C 路由的长程 PPO 续训", "",
           f"预先固定 seed 17 和第 40 轮。在新 200 间中，监督起点、200 局和 2,000 局分别成功 **{s0}、{s4}、{s40} 间**。"
           f"追加 1,800 局相对父模型差为 **{change('r40_vs_r4')} 个百分点**。",
           f"A/B/C 同房间纯执行为 {values['A']} / {values['B']} / {values['C']}，三者事后覆盖并集 {union}/200。",
           "", "![训练成功率、固定房间贪心成功率、熵和损失](learning_curves.png)", "",
           "## 固定 64 间上的学习曲线", "",
           "| 累计 PPO 采样 | 成功 / 64 | 相对监督起点的救回 / 丢失 |",
           "|---:|---:|---:|"]
    for row in diag:
        lines.append(f"| {row['round']*50:,} | {row['success']} | {row['gained']} / {row['lost']} |")
    lines += ["", f"首五批与末五批训练采样成功率分别为 {rolling(earlier):.1%}、{rolling(last):.1%}。"
              f"这两段访问不同房间。相同固定诊断房间的起点、200 局和末轮分别为 {diag[0]['success']}、{diag[1]['success']}、{diag[-1]['success']}/64。",
              "曲线上的检查点均预登记。64 间子集此前已查看，确认结论来自下表的新房间。", "",
              "## 新 200 间的配对闭环", "",
              "| 2,000 局末轮相对 | 救回 | 丢失 | 成功率差（百分点，配对 95% 区间） |",
              "|---|---:|---:|---:|"]
    for arm,label in (("r4","200 局父模型"),("r0","监督起点"),("A","纯 A"),("B","纯 B"),("C","纯 C")):
        key="r40_vs_"+arm
        q=rates[key]
        lines.append(f"| {label} | {q['plus']} | {q['minus']} | {change(key)} |")
    lines += ["", "| 纯专家成败分层 | 房间 | 监督起点 | 200 局 | 2,000 局 |",
              "|---|---:|---:|---:|---:|"]
    for arm,label in (("A","仅 A 成功"),("B","仅 B 成功"),("C","仅 C 成功"),("none","三位专家均失败")):
        q=strata[arm]
        lines.append(f"| {label} | {q['n']} | {q['counts']['r0']} | {q['counts']['r4']} | {q['counts']['r40']} |")
    lines+=["","恢复父模型的 actor、critic 与 Adam 状态，走同样的公开提案评分、终局奖励和温度 1。"
            "追加阶段每轮加入 50 个新房间。配对区间按房间重抽，适用于这一训练种子的成品策略。"
            "本次变化同时包括在线局数与不同训练房间数。","",
            "[协议](plan.md) · [逐房间成败](rooms.csv) · [逐轮训练](training.csv) · [配对区间](results.json)"]
    (run.DOC/"README.md").write_text("\n".join(lines)+"\n")
    reg=run.DOC.parent.parent/"experiment_journal/entries.json"
    items=run.read(reg)
    item=next(x for x in items if x["id"]=="abc-ppo-long")
    item["result"]=f"新 200 间：监督起点 {s0}，200 局 {s4}，2,000 局 {s40}。末轮减父模型 {change('r40_vs_r4')} 个百分点。"
    item["analysis"]=f"固定诊断 64 间从 {diag[1]['success']} 到 {diag[-1]['success']}。独立确认报告分开列出能力救回与丢失。"
    reg.write_text(json.dumps(items,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
