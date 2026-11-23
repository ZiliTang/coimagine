"""Summarize frozen paired initialization and PPO endpoints on one room panel."""
import csv

import numpy as np

from tools import cap_distilled_ppo as run
from tools import cap_ppo_teacher_compare as teacher

r, doc = run.r, run.DOC


def csvfile(name, rows):
    with (doc / name).open("w") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(left, right, draws):
    delta = left - right
    gaps = delta.mean(1)
    return dict(pp=float(100*gaps.mean()),
                ci=(100*np.quantile(gaps[draws].mean(1), [.025,.975])).tolist(),
                gained=(delta>0).sum(0).tolist(), lost=(delta<0).sum(0).tolist())


def main():
    plan = r.e.read(run.OUT / "protocol.json")
    assert teacher.t.sha(teacher.t.OUT / "protocol.json") == r.e.read(teacher.OUT / "protocol.json")["ppo_protocol"]
    assert plan["panel"] == r.e.read(teacher.t.OUT / "protocol.json")["confirm"]
    for seed in run.SEEDS:
        original = r.study.cold.weights(r.d.OUT / "fit/semantic.npz")
        initial = r.study.cold.weights(run.checkpoint("distilled", seed, 0))
        assert all(np.array_equal(x, y) for x, y in zip(initial, original, strict=True))
    histories_changed = {arm: [0]*len(run.SEEDS) for arm in run.ARMS}
    rows = []
    for key in plan["panel"]:
        row = dict(room=key)
        for who in ("A", "B"):
            row[who] = int(r.e.read(run.p.OUT / "confirm" / who / (key+".json"))["success"])
        row["teacher"] = int(r.e.read(teacher.OUT / "confirm" / (key+".json"))["success"])
        for arm in run.ARMS:
            starts = {}
            for stage in (0,4):
                for seed in run.SEEDS:
                    source = run.SEEDS[0] if arm == "distilled" and stage == 0 else seed
                    tag = f"{arm}_s{source}_r{stage}"
                    played = r.e.read(run.OUT / "evaluation" / tag / (key+".json"))
                    row[tag] = int(played["success"])
                    if stage == 0:
                        starts[seed] = played["history"]
                    else:
                        histories_changed[arm][run.SEEDS.index(seed)] += int(played["history"] != starts[seed])
        rows.append(row)
    assert len({q["room"] for q in rows}) == 200
    doc.mkdir(exist_ok=True, parents=True)
    csvfile("rooms.csv", rows)
    metrics = [r.e.read(run.OUT/"metrics"/f"{arm}_s{seed}_r{stage}.json") | dict(arm=arm)
               for arm in run.ARMS for seed in run.SEEDS for stage in range(1,5)]
    assert sum(q["episodes"] for q in metrics) == 1200
    csvfile("training.csv", metrics)
    arrays = {}
    for arm in run.ARMS:
        for stage in (0,4):
            seeds = [17]*3 if arm == "distilled" and stage == 0 else run.SEEDS
            arrays[f"{arm}{stage}"] = np.array([[q[f"{arm}_s{s}_r{stage}"] for s in seeds] for q in rows])
    draws = np.random.default_rng(823116).integers(200,size=(10000,200))
    router = np.array([q["teacher"] for q in rows])[:,None]
    counts = {k:v.sum(0).tolist() for k,v in arrays.items()}
    pairs = [("distilled4","cold4"),("distilled0","cold0"),("distilled4","distilled0"),("cold4","cold0"),
             ("distilled0","teacher"),("distilled4","teacher"),("cold4","teacher")]
    arrays["teacher"] = router
    differences = {x+"-"+y:compare(arrays[x],arrays[y],draws) for x,y in pairs}
    interaction = compare(arrays["distilled4"]-arrays["distilled0"], arrays["cold4"]-arrays["cold0"],draws)
    masks = {(a,b):[i for i,q in enumerate(rows) if q["A"]==a and q["B"]==b] for a in (0,1) for b in (0,1)}
    strata = {f"A{a}B{b}":dict(n=len(ix),counts={k:v[ix].sum(0).tolist() for k,v in arrays.items()})
              for (a,b),ix in masks.items()}
    result = dict(counts=counts,router=int(router.sum()),comparisons=differences,
                  change_difference=interaction,strata=strata,changed_room_histories=histories_changed)
    r.e.save(doc/"results.json",result)

    def interval(stat):
        lo, hi = stat["ci"]
        return f'{stat["pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
    lines = ["# 蒸馏语义 actor 与随机初始化的短程 PPO","",
             "同一套 A/B 冻结专家，两组只改变 actor 初始权重。确认面板已在前序研究中查看，本次作为配对诊断。",
             "", "| 初始权重 | 贪心评估 | seed 17 / 29 / 43，各 200 间 | 平均成功率 |",
             "|---|---|---:|---:|"]
    names = {"distilled":"程序路由蒸馏","cold":"随机初始化"}
    for arm in run.ARMS:
        for stage, label in ((0,"更新前"),(4,"200 局后")):
            c = counts[f"{arm}{stage}"]
            lines.append(f'| {names[arm]} | {label} | {" / ".join(map(str,c))} | {np.mean(c)/2:.2f}% |')
    lines += ["",f'同房间单 CAP A、B 和原程序教师分别成功 {sum(q["A"] for q in rows)}、{sum(q["B"] for q in rows)} 和 {router.sum()}/200。蒸馏组共享同一个已冻结的起点，零轮只执行一次。',"",
              "| 配对比较 | 差值和 95% 区间（百分点） |","|---|---:|"]
    for pair, label in zip(pairs, ("更新后蒸馏减冷启动（主比较）","更新前蒸馏减冷启动","蒸馏组更新后减更新前","冷启动组更新后减更新前",
                                  "冻结蒸馏 actor 减程序教师","蒸馏加 PPO 减程序教师","随机加 PPO 减程序教师"),strict=True):
        lines.append(f'| {label} | {interval(differences["-".join(pair)])} |')
    lines += [f"| 两组增幅之差（蒸馏减冷启动） | {interval(interaction)} |","",
              "区间按同一房间重抽 10,000 次，每次保留所有方法和种子。区间以这三次 PPO 初始化及固定蒸馏检查点为条件。",
              "", "每种子使用相同顺序的 4×50 个不同新房间，两组配对 rollout 随机数和 critic 初始化。策略每轮各自在环境中采样，蒸馏标签只用于既有的监督预训练。两组都采用原教师的公开停滞门控，门控前执行 A，相同提案统一执行 A。输入均为完整公开几何、动作参数、历史和 9 个提案语义量。终局成功为 1，其余为 0。沿用现有 PPO-Clip、独立 critic、温度 1 和贪心评估。预先固定第 4 轮，无种子或检查点择优。",
              "", "蒸馏检查点源自 400 间房的 6,898 条教师标签，在当时的留出教师边界上一致率为 90.6%。本轮训练和评测使用当前 fixed72-local-v1 感知。蒸馏的前期成本另计。此前的无门控 PPO 使用分叉预训练父模型，输入屏蔽教师语义量，不能用作本次冷启动对照。","",
              "[逐房间结果](rooms.csv) · [逐轮训练指标](training.csv) · [配对区间与专家成败分层](results.json)"]
    lines += ["", "## 能力保留", "",
              "蒸馏组更新前成功 111/200，与程序教师各自独赢 11 间。短程 PPO 后相对蒸馏起点，三个种子分别改变了 20、9、25 间的执行历史，救回 0、0、0 间，丢失 0、0、2 间。随机组的三个种子分别改变 0、0、20 间的执行历史。",
              "", "79 个纯 A 和纯 B 都失败的房间里，冻结蒸馏 actor 成功 10 间，PPO 后为 10、10、9 间，随机组更新前后均为 0。19 个仅 B 成功的房间中，随机组各保住 0、18、18 间；20 个仅 A 成功的房间中，各保住 20、0、0 间。蒸馏组在两类中分别保住 9 和 13 间。对这三次初始化和这批房间，最终成功率差主要已在 PPO 开始前形成。"]
    (doc/"README.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
