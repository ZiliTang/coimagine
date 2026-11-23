"""Room-paired results for removing the success replay loss."""
import csv
import json
from pathlib import Path

import numpy as np
from tools import cap_ungated_no_replay as trial
from tools.cap_ungated_replay_report import comparison

run=trial.run


def paired(panel,keys):
    rows=[]
    for key in keys:
        old_base=run.OUT/panel
        experts=[run.e.read(old_base/m/(key+".json"))["success"] for m in ("A","B")]
        for seed in run.SEEDS:
            def outcome(directory,arm):
                return run.e.read(directory/panel/f"s{seed}_{arm}"/(key+".json"))["success"]
            rows.append(dict(panel=panel,room=key,seed=seed,A=experts[0],B=experts[1],
                             parent=outcome(run.OUT,"parent"),replay=outcome(run.OUT,"actor"),
                             no_replay=outcome(trial.OUT,"no_replay")))
    def matrix(name):
        return np.asarray([v[name] for v in rows],bool).reshape(len(keys),len(run.SEEDS))
    a,b,p,x,y=map(matrix,["A","B","parent","replay","no_replay"])
    masks=dict(overall=np.ones_like(p),retained=p,rescued=~p,
               expert_recovery=(a|b)&~p,actor_unique=(~a&~b)&p)
    draws=np.random.default_rng(71117).integers(len(keys),size=(10000,len(keys)))
    cells={}
    for name,mask in masks.items():
        v=comparison(mask,x,y,draws)
        for group in [v,*v["by_seed"]]:
            group["replay"]=group.pop("actor")
            group["no_replay"]=group.pop("mixed")
        cells[name]=v
    return rows,dict(totals={k:matrix(k).sum(axis=0).tolist() for k in ("parent","replay","no_replay")},cells=cells)


def main():
    plan=run.e.read(run.OUT/"protocol.json")
    trial.DOC.mkdir(exist_ok=True,parents=True)
    dataset=[]
    report={}
    for panel in ("train","fresh"):
        rows,report[panel]=paired(panel,plan[panel])
        dataset.extend(rows)
    with (trial.DOC/"rooms.csv").open("w") as f:
        wr=csv.DictWriter(f,dataset[0])
        wr.writeheader()
        wr.writerows(dataset)
    audits=[run.e.read(trial.OUT/"audit"/f"s{s}.json") for s in run.SEEDS]
    fit={arm:{field:sum(v[arm][field] for v in audits) for field in audits[0][arm]}
         for arm in ("replay","no_replay")}
    report["fit"]=fit
    run.e.save(trial.DOC/"results.json",report)
    diag=report["fresh"]
    counts=diag["totals"]
    cells=diag["cells"]
    overall=cells["overall"]
    def effect(v):
        lo,hi=v["ci95"]
        return f'{v["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
    sumline=f'关闭成功回放后，200 间诊断面板的成功率变化为 {effect(overall)} 个百分点。'
    analysis=f'旧成功保留率变化 {effect(cells["retained"])} 个百分点，旧失败救回率变化 {effect(cells["rescued"])} 个百分点。'
    seedrows=[f'| {s} | {counts["parent"][i]} | {counts["replay"][i]} | {counts["no_replay"][i]} |'
              for i,s in enumerate(run.SEEDS)]
    tablerows=[]
    for key,label in [("overall","全部"),("retained","父 actor 成功，更新后保住"),("rescued","父 actor 失败，更新后救回"),
                      ("expert_recovery","至少一位专家成功，但父 actor 失败"),("actor_unique","两位专家均失败，但父 actor 成功")]:
        v=cells[key]
        tablerows.append(f'| {label} | {v["replay"]}/{v["n"]} | {v["no_replay"]}/{v["n"]} | {effect(v)} |')
    f0,f1=fit["replay"],fit["no_replay"]
    train=report["train"]["totals"]
    content=["# 无门控路由中的成功回放消融","",sumline+analysis,"",
             "本面板中，成功回放合计多成功 8 次，其中 7 次净差来自救回父 actor 的失败，旧成功的净保留只多 1 次。"
             "三个种子的成功数差（加回放减去回放）分别为 +9、+1、−2。回放明显减少了固定训练前缀的改选，"
             "闭环差值的区间仍跨零。当前证据适合将其列作可选的训练项，未确立其为整局成功所必需。","","## 配对闭环","",
             "| 种子 | 父 actor | 加成功回放 | 去掉成功回放 |","|---|---:|---:|---:|",*seedrows,"",
             "| 房间条件 | 加成功回放 | 去掉成功回放 | 去掉后的差值与 95% 区间（百分点） |",
             "|---|---:|---:|---:|",*tablerows,"",
             f'两种更新直接比较，去掉回放独赢 {overall["gained"]} 次，加回放独赢 {overall["lost"]} 次。'
             '三个种子共用同一批 200 间。95% 区间按房间配对自助抽样 10,000 次，每次把三个种子一起抽取，条件指标同步重算分母。'
             '该面板此前已查看，区间只覆盖这三份模型的房间抽样不确定性。',"","## 训练数据上的效果","",
             f'固定父 actor 的成功前缀，加回放后改选 {f0["changed"]}/{f0["n"]} 个边界，去掉后改选 {f1["changed"]}/{f1["n"]} 个。'
             f'两组在有实测胜者的训练分叉分别选对 {f0["hard_correct"]}/{f0["hard_n"]} 和 {f1["hard_correct"]}/{f1["hard_n"]} 个。'
             f'50 间开发闭环中，父 actor、加回放和去回放的逐种子成功数分别为 {train["parent"]}、{train["replay"]}、{train["no_replay"]}。',"",
             "## 实验条件","",
             "每个种子从原已训练四轮的同一父权重接续一轮。全程开放路由，共用此前在 50 间上采到的 419 个分叉，"
             "其中 37 个同前缀分叉具有唯一胜者。每步 64 个分叉偏好、128 个全局软保持，均取相同小批和固定父模型概率。"
             "加回放臂额外使用 64 个成功前缀的余量损失，权重 0.25、logit 余量 0.2。去掉该项后直接对其余两项求和，"
             "保留其原权重。每种子均为 300 次 Adam 更新，学习率 0.0003。带回放组重新拟合的权重与原锁定模型全部逐值一致，复用其既有闭环结果。"
             "去回放组三份模型锁定后，重新执行 600 次诊断面板闭环与 150 次开发闭环。",
             "","[逐房间数据](rooms.csv) · [统计结果](results.json)"]
    (trial.DOC/"README.md").write_text("\n".join(content)+"\n")
    path=Path("docs/research/experiment_journal/entries.json")
    registry=json.loads(path.read_text())
    next(v for v in registry if v["id"]=="ungated-no-replay").update(result=sumline,analysis=analysis)
    path.write_text(json.dumps(registry,ensure_ascii=False,indent=2)+"\n")
    run.e.save(trial.OUT/"FINISHED.json",dict(primary=overall,results_sha=run.d.source.graph.file_sha(trial.DOC/"results.json")))
    print(sumline,analysis)


if __name__=="__main__":
    main()
