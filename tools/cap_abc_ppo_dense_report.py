"""Report task success, terminal distance, and reward use for the dense arm."""
import csv
import json

import numpy as np

from tools import cap_abc_ppo_dense as run
from tools.cap_option_handoff_opportunity_report import paired

ARMS=("binary","terminal","dense")


def main():
    control=run.read(run.CONTROL/"protocol.json")
    assert run.read(run.OUT/"control_replay.json")["model_parameters_identical_with_dense_disabled"]
    assert run.read(run.OUT/"progress/same_panel.json")["done"]==200
    rows=[]
    for key in control["confirm"]:
        row=dict(room=key,parent=run.read(run.CONTROL/"evaluation/r4"/(key+".json"))["success"])
        for arm,folder,tag in (("binary",run.PREVIOUS,"binary_r40"),
                               ("terminal",run.PREVIOUS,"shaped_r40"),("dense",run.OUT,"shaped_r40")):
            result=run.read(folder/"evaluation"/tag/(key+".json"))
            row.update({arm:result["success"],arm+"_distance_m":result["distance_m"],arm+"_ticks":result["ticks"]})
        rows.append(row)
    assert len(rows)==len({r["room"] for r in rows})==200
    with (run.DOC/"rooms.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    y={arm:np.array([r[arm] for r in rows]) for arm in (*ARMS,"parent")}
    paired_success={a:paired(y["dense"],y[a]) for a in ("binary","terminal","parent")}
    distance={a:np.array([r[a+"_distance_m"] for r in rows]) for a in ARMS}
    rng=np.random.default_rng(174801)

    def compare(a,mask):
        delta=distance["dense"][mask]-distance[a][mask]
        if len(delta)==0:
            return None
        samples=rng.integers(len(delta),size=(10000,len(delta)))
        return dict(n=len(delta),mean=float(delta.mean()),ci=np.quantile(delta[samples].mean(1),[.025,.975]).tolist(),
                    dense_median=float(np.median(distance["dense"][mask])),
                    ref_median=float(np.median(distance[a][mask])))

    paired_distance={a:dict(all=compare(a,np.ones(len(rows),bool)),
                            same_failures=compare(a,(y["dense"]==0)&(y[a]==0)))
                     for a in ("binary","terminal")}
    curve=[]
    for stage in run.binary.CHECKS:
        name=f"r{stage:02d}.json"
        folders=[run.CONTROL]*3 if stage<=run.START else (run.CONTROL,run.PREVIOUS,run.OUT)
        d={a:run.read(p/"diagnostic"/name)["success"] for a,p in zip(ARMS,folders,strict=True)}
        curve.append(dict(cumulative_episodes=stage*50,**d))
    metrics=[run.read(f) for f in sorted((run.OUT/"metrics").glob("r*.json"))]
    episodes=sum(r["episodes"] for r in metrics)
    assert episodes==1800
    bonus_at_least_one=sum(m["bonus_at_least_one"] for m in metrics)
    maximum_bonus=max(m["max_dense_reward"] for m in metrics)
    average_bonus=sum(m["mean_dense_reward"]*m["episodes"] for m in metrics)/episodes
    reference_metrics=[run.read(p) for p in sorted((run.CONTROL/"metrics").glob("r*.json"))]
    assert [m["stage"] for m in metrics]==[m["stage"] for m in reference_metrics]
    counts_correlation=float(np.corrcoef([m["success"] for m in metrics],
                                        [m["success"] for m in reference_metrics])[0,1])
    def actor_weights(folder, stage):
        with np.load(run.binary.model_path(folder,stage)) as f:
            return np.concatenate([f[k].astype(np.float64).ravel() for k in f.files])

    old_b = old_d = actor_weights(run.CONTROL,run.START)
    updates=[]
    for stage in range(run.START+1,run.END+1):
        new_b,new_d=actor_weights(run.CONTROL,stage),actor_weights(run.OUT,stage)
        dx,dy=new_b-old_b,new_d-old_d
        norm_x,norm_y=np.linalg.norm(dx),np.linalg.norm(dy)
        updates.append(dict(stage=stage,
                            cosine=float(dx@dy/(norm_x*norm_y)) if norm_x and norm_y else None,
                            norm_ratio=float(norm_y/norm_x) if norm_x else None))
        old_b,old_d=new_b,new_d
    first_update=updates[0]
    update_cosine_median=float(np.median([q["cosine"] for q in updates if q["cosine"] is not None]))
    counts={arm:int(arr.sum()) for arm,arr in y.items()}
    outcome=dict(counts=counts,paired_success=paired_success,paired_distance_m=paired_distance,curve=curve,
                 dense_train=dict(episodes=episodes,bonus_at_least_one=bonus_at_least_one,mean_dense_reward=average_bonus,max_dense_reward=maximum_bonus,
                                  binary_batch_success_correlation=counts_correlation,
                                  actor_update_cosine_median=update_cosine_median,actor_updates=updates))
    run.save(run.DOC/"results.json",outcome)

    def pp(a):
        q=paired_success[a]
        lo,hi=[100*x for x in q["ci"]]
        return f"{(q['first']-q['second'])/2:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    lines=["# A/B/C PPO 的逐 option 进展奖励","",
           f"共同的 200 局父模型成功 {counts['parent']}/200。同房间的 2,000 局末轮为二元奖励 **{counts['binary']}**、"
           f"终端距离 **{counts['terminal']}**、逐 option 进展 **{counts['dense']}**。"
           f"进展组相对二元组成功率差为 **{pp('binary')} 个百分点**。","",
           "![三组训练成功率、固定面板成功率、动作熵与训练失败终距](learning_curves.png)","",
           "| 累计 PPO 局 | 固定 64 间 二元 | 固定 64 间 终端距离 | 固定 64 间 逐步进展 |",
           "|---:|---:|---:|---:|"]
    for d in curve:
        lines.append(f"| {d['cumulative_episodes']:,} | {d['binary']} | {d['terminal']} | {d['dense']} |")
    lines+=["","| 逐步进展末轮相对 | 救回 | 丢失 | 成功率差及房间配对 95% 区间（百分点） |",
            "|---|---:|---:|---:|"]
    for a,name in (("binary","二元奖励末轮"),("terminal","终端距离末轮"),("parent","共同 200 局父模型")):
        q=paired_success[a]
        lines.append(f"| {name} | {q['plus']} | {q['minus']} | {pp(a)} |")
    lines+=["","| 终距比较 | 房间数 | 参照中位数（米） | 逐步组中位数（米） | 平均配对差及 95% 区间（米） |",
            "|---|---:|---:|---:|---:|"]
    for a,label in (("binary","二元奖励"),("terminal","终端距离")):
        for group,part in (("all","全部房间"),("same_failures","两组均失败")):
            q=paired_distance[a][group]
            if q:
                lo,hi=q["ci"]
                lines.append(f"| 对{label}，{part} | {q['n']} | {q['ref_median']:.3f} | {q['dense_median']:.3f} | {q['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    lines+=["",
            f"每靠近 1 米给 0.10，后退给 0。进展分逐次累加。"
            f"追加的 {episodes:,} 局中，进展奖励平均 {average_bonus:.3f}，最大 {maximum_bonus:.3f}，有 {bonus_at_least_one} 局达到 1。成功额外给 1。",
            f"逐步与二元组在 36 个相同批次的成功数相关为 {counts_correlation:.3f}。"
            f"两组首批共享行为轨迹时，actor 参数更新方向余弦为 {first_update['cosine']:.3f}，更新幅度比为 {first_update['norm_ratio']:.3f}。"
            f"36 次更新的余弦中位数为 {update_cosine_median:.3f}，后期策略访问的前缀可能已经不同。"
            "这里比较实际 Adam 参数步。"
            "每 200 个房间按生成种子产生并洗牌，连续 50 间为一批。每 200 间内 2–5 个可移动物体各 50 间，各批的物体数构成和难度可以浮动。",
            "首批 50 局关闭进展奖励后，所得 actor 和 critic 与旧二元基线逐值一致。三个分支沿用同一批训练房间、父模型、采样与选模规则。"
            "两批评测房间此前已查看，区间按房间配对重抽，以固定 seed 17 和奖励参数为条件。",
            "", "[实验协议](plan.md) · [逐房间结果](rooms.csv) · [配对统计](results.json)"]
    (run.DOC/"README.md").write_text("\n".join(lines)+"\n")

    registry=run.DOC.parent.parent/"experiment_journal/entries.json"
    items=run.read(registry)
    entry=next(x for x in items if x["id"]=="abc-ppo-dense")
    entry["result"]=f"同 200 间末轮二元、终端距离、逐步进展为 {counts['binary']}、{counts['terminal']}、{counts['dense']}。逐步相对二元 {pp('binary')} 个百分点。"
    entry["analysis"]=f"固定 64 间末轮三组分别为 {curve[-1]['binary']}、{curve[-1]['terminal']}、{curve[-1]['dense']}。"
    registry.write_text(json.dumps(items,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
