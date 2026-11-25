"""Report the 20,000-episode ABC continuation on the locked confirmation panel."""
import csv
import json

import numpy as np

from tools import cap_abc_ppo_extended as run
from tools.cap_option_handoff_opportunity_report import paired


def csv_rows(name, rows):
    with (run.DOC/name).open("w") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    plan=run.read(run.OUT/"protocol.json")
    tags=(*run.base.EXPERTS,*(f"r{s}" for s in run.TEST_STAGES))
    cases=[]
    for k in plan["confirm"]:
        rec=dict(room=k)
        for tag in tags:
            row=run.read(run.OUT/"evaluation"/tag/(k+".json"))
            rec[tag]=row["success"]
            rec[tag+"_ticks"]=row["ticks"]
        cases.append(rec)
    assert len(cases)==len({q["room"] for q in cases})==200
    csv_rows("rooms.csv",cases)
    new_train=[run.read(run.OUT/"metrics"/f"r{r:03d}.json") for r in range(run.START+1,run.END+1)]
    training_rooms=[key for row in new_train for key in row["rooms"]]
    assert len(training_rooms)==len(set(training_rooms))==18000
    assert set(training_rooms).isdisjoint(plan["confirm"]+plan["diagnostic"])
    csv_rows("training.csv",new_train)
    y={tag:np.array([r[tag] for r in cases],int) for tag in tags}
    endpoint=f"r{run.END}"
    risks={tag:paired(y[endpoint],y[tag]) for tag in tags if tag!=endpoint}
    scores={tag:int(v.sum()) for tag,v in y.items()}
    df=[run.read(run.OUT/"diagnostic"/f"r{r:03d}.json") for r in run.DIAG_CHECKS]
    start_diag=run.read(run.PARENT/"diagnostic"/f"r{run.START:02d}.json")
    diag=list([start_diag,*df])
    strata={}
    for tag in (*run.base.EXPERTS,"none"):
        m=np.logical_and.reduce([y[k]==int(k==tag) for k in run.base.EXPERTS])
        strata[tag]=dict(n=int(m.sum()),scores={k:int(v[m].sum()) for k,v in y.items()})
    windows=[]
    for i in range(0,len(new_train),20):
        rows=new_train[i:i+20]
        windows.append(dict(at=rows[-1]["stage"]*50,sampled_success=100*sum(x["success"] for x in rows)/1000,
                            entropy=float(np.mean([x["entropy"] for x in rows])),
                            signed_progress_reward=float(np.mean([x["mean_dense_reward"] for x in rows]))))
    run.save(run.DOC/"results.json",dict(scores=scores,paired=risks,diagnostic=diag,
                                         training_windows=windows,strata=strata))

    def pct(q):
        lo,hi=(100*r for r in q["ci"])
        return f"{(q['first']-q['second'])/2:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    primary=risks["r40"]
    max_diag=max(x["success"] for x in diag)
    lines=["# A/B/C PPO 的 20,000 局新房间续训","",
           f"冻结的新 200 间上，2,000 局父模型成功 **{scores['r40']}/200**，20,000 局末轮成功 **{scores[endpoint]}/200**。"
           f"末轮救回 {primary['plus']} 间、丢失 {primary['minus']} 间，配对差为 **{pct(primary)} 个百分点**。","",
           "![累计训练成功率、固定房间贪心曲线、动作熵与 PPO 损失](learning_curves.png)","",
           "| 模型 | 成功 / 200 |","|---|---:|"]
    labels={"A":"纯 A","B":"纯 B","C":"纯 C","r0":"监督起点","r4":"PPO 200 局","r40":"PPO 2,000 局（续训起点）",
            endpoint:"PPO 20,000 局（预定末轮）"}
    for tag,name in labels.items():
        lines.append(f"| {name} | {scores[tag]} |")
    lines+=["",f"固定诊断 64 间的续训起点与末轮分别为 {start_diag['success']} 和 {diag[-1]['success']}，中间及末轮最高为 {max_diag}。"
            f"追加阶段首个与最后一个完整 1,000 局训练窗口的随机策略成功率为 "
            f"{windows[0]['sampled_success']:.1f}% 和 {windows[-1]['sampled_success']:.1f}%。"
            "训练窗口用不同房间，固定诊断房间用于核对部署曲线，独立主比较使用预登记的末轮。","",
            "| 独立面板专家成败分层 | 房间 | 2,000 局 | 20,000 局 |","|---|---:|---:|---:|"]
    for tag in ("A","B","C","none"):
        row=strata[tag]
        label=f"仅 {tag} 成功" if tag!="none" else "三位专家均失败"
        lines.append(f"| {label} | {row['n']} | {row['scores']['r40']} | {row['scores'][endpoint]} |")
    lines+=["","从固定 seed 17 的正向进展父模型恢复 actor、critic 与 Adam。每轮以新种子生成并选 50 个新房间。"
            "追加 18,000 个哈希唯一的房间，与固定诊断和末轮确认房间均无重合。奖励是每米 0.10 的有符号距离缩短，另给终端成功 1。"
            "配对区间按房间重抽。本次同时延长训练、扩大场景覆盖和加入退步惩罚，结论以这一训练设置和种子为条件。","",
            "[协议](plan.md) · [逐房间结果](rooms.csv) · [逐轮记录](training.csv) · [配对统计](results.json)"]
    (run.DOC/"README.md").write_text("\n".join(lines)+"\n")

    p=run.DOC.parent.parent/"experiment_journal/entries.json"
    entries=run.read(p)
    entry=next(e for e in entries if e["id"]=="abc-ppo-extended")
    entry["result"]=f"独立新 200 间，2,000→20,000 局为 {scores['r40']}→{scores[endpoint]}。配对差 {pct(primary)} 个百分点。"
    entry["analysis"]=f"同一固定 64 间的起点→末轮为 {start_diag['success']}→{diag[-1]['success']}，新增 18,000 间均未重复。"
    tmp=p.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    tmp.replace(p)


if __name__=="__main__":
    main()
