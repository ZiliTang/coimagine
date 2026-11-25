"""Summarize paired binary success, terminal distance, and learning curves."""
import csv
import json

import numpy as np

from tools import cap_abc_ppo_distance as run
from tools.cap_option_handoff_opportunity_report import paired


def main():
    control = run.read(run.CONTROL/"protocol.json")
    assert run.read(run.OUT/"control_replay.json")["model_parameters_identical"]
    assert run.read(run.OUT/"progress/same_panel.json")["done"]==400
    rooms=[]
    for key in control["confirm"]:
        row=dict(room=key)
        for kind in ("binary","shaped"):
            q=run.read(run.OUT/"evaluation"/f"{kind}_r{run.END}"/(key+".json"))
            row[kind]=q["success"]
            row[kind+"_distance_m"]=q["distance_m"]
            row[kind+"_ticks"]=q["ticks"]
        row["parent"]=run.read(run.CONTROL/"evaluation/r4"/(key+".json"))["success"]
        rooms.append(row)
    assert len(rooms)==len({r["room"] for r in rooms})==200
    with (run.DOC/"rooms.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rooms[0]))
        writer.writeheader()
        writer.writerows(rooms)

    y={a:np.array([q[a] for q in rooms],int) for a in ("binary","shaped","parent")}
    dist={a:np.array([q[a+"_distance_m"] for q in rooms],float) for a in ("binary","shaped")}
    growth={"reward_vs_binary":paired(y["shaped"],y["binary"]),
            "reward_vs_parent":paired(y["shaped"],y["parent"])}
    both_failed=(y["binary"]==0)&(y["shaped"]==0)
    rng=np.random.default_rng(94117)

    def paired_dist(mask):
        b,d=dist["binary"][mask],dist["shaped"][mask]
        delta=d-b
        if not len(delta):
            return None
        ids=rng.integers(len(delta),size=(10000,len(delta)))
        return dict(n=len(delta),binary_median=float(np.median(b)),shaped_median=float(np.median(d)),
                    mean_delta=float(delta.mean()),
                    ci=np.quantile(delta[ids].mean(1),[.025,.975]).tolist(),
                    closer=int(sum(delta<0)),farther=int(sum(delta>0)))

    distances=dict(all=paired_dist(np.ones(200,bool)),same_failures=paired_dist(both_failed))
    curve=[]
    for stage in run.binary.CHECKS:
        stem=f"r{stage:02d}.json"
        b=run.read(run.CONTROL/"diagnostic"/stem)
        d=run.read(run.OUT/"diagnostic"/stem) if stage>run.START else b
        assert d["n"]==b["n"]==64
        curve.append(dict(episodes=stage*50,binary=b["success"],shaped=d["success"]))
    trains={a:[run.read(f) for f in sorted((root/"metrics").glob("r*.json"))]
            for a,root in (("binary",run.CONTROL),("shaped",run.OUT))}
    assert all(sum(q["episodes"] for q in tr)==1800 for tr in trains.values())
    counts={k:int(v.sum()) for k,v in y.items()}
    stats=dict(counts=counts,paired_success=growth,paired_distance_m=distances,curve=curve)
    run.save(run.DOC/"results.json",stats)

    def contrast(q):
        lo,hi=(100*v for v in q["ci"])
        return f"{(q['first']-q['second'])/2:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    lines=["# A/B/C PPO 的终端距离奖励对照","",
           f"共同的 200 局父模型成功 {counts['parent']}/200。两个 2,000 局末轮分别为二元奖励 **{counts['binary']}**、距离奖励 **{counts['shaped']}**。"
           f"距离组相对二元组的闭环差为 **{contrast(growth['reward_vs_binary'])} 个百分点**。","",
           "![二元成功率、固定房间闭环、策略熵与训练失败终距](learning_curves.png)","",
           "| 累计在线局 | 二元奖励贪心成功 /64 | 距离奖励贪心成功 /64 |",
           "|---:|---:|---:|"]
    for q in curve:
        lines.append(f"| {q['episodes']:,} | {q['binary']} | {q['shaped']} |")

    lines += ["","| 末轮距离奖励组相对 | 救回 | 丢失 | 成功率差与 95% 区间（百分点） |",
              "|---|---:|---:|---:|"]
    for name,label in (("reward_vs_binary","二元奖励末轮"),("reward_vs_parent","共同 200 局父模型")):
        q=growth[name]
        lines.append(f"| {label} | {q['plus']} | {q['minus']} | {contrast(q)} |")
    lines += ["","| 终距比较 | 房间 | 二元奖励中位数（m） | 距离奖励中位数（m） | 平均配对差与 95% 区间（m） |",
              "|---|---:|---:|---:|---:|"]
    for key,label in (("all","全部房间"),("same_failures","两组均失败")):
        q=distances[key]
        if q is not None:
            lo,hi=q["ci"]
            lines.append(f"| {label} | {q['n']} | {q['binary_median']:.3f} | {q['shaped_median']:.3f} | {q['mean_delta']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines += ["","两个末轮为同一 seed 17。均使用同一评测房间列表，房间此前已查看。区间按房间配对重抽，以这一次训练为条件。",
              "父模型、执行器、感知、采样温度、房间次序和 PPO 预算相同。训练中成功仍得 1，失败奖励为 0.25 × exp(−终距 / 1 m)，只有末端给奖励。",
              "用距离组首轮的同一批轨迹计算二元更新，所得 actor 和 critic 与二元基线的对应检查点逐值一致。"
              "二元末轮重跑的全部 200 间也核对了成功、ticks 和历史哈希。","",
              "[实验协议](plan.md) · [逐房间配对](rooms.csv) · [成功与终距统计](results.json)"]
    (run.DOC/"README.md").write_text("\n".join(lines)+"\n")
    registry=run.DOC.parent.parent/"experiment_journal/entries.json"
    entries=run.read(registry)
    entry=next(e for e in entries if e["id"]=="abc-ppo-distance")
    entry["result"]=f"同 200 间末轮二元 {counts['binary']}，距离奖励 {counts['shaped']}，配对差 {contrast(growth['reward_vs_binary'])} 个百分点。"
    entry["analysis"]=f"同一父模型和训练预算。固定 64 间的最终贪心成功分别为 {curve[-1]['binary']}、{curve[-1]['shaped']}。"
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
