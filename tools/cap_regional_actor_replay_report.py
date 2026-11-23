"""Persist the two predeclared paired comparisons and per-case evidence."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from tools import cap_regional_actor_replay as replay
from tools import cap_regional_expert_suffix as diag

study = replay.study
docs = replay.DOCS
REFERENCE = replay.engine.ppo.OUT.parent / "cap_regional_actor_iteration_20261120_01"


def csvfile(name, rows):
    with (docs / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def suffix_report():
    docs.mkdir(parents=True, exist_ok=True)
    rows = replay.records(diag.OUT / "rooms")
    s = study.read(diag.OUT / "result.json")
    assert len(rows) == 96 and diag.summarize(rows) == s
    actions = []
    for r in rows:
        if not r["available"]:
            continue
        for side in ("base", "alt"):
            a = r["actor_" + side]
            e = r["expert_" + side]
            actions.append(
                dict(
                    room=r["id"],
                    step=r["step"],
                    prefix=r["prefix"],
                    side=side,
                    action=r[side],
                    expert=replay.engine.cpp.SID_ORDER[e["expert"]],
                    supporters=e["supporters"],
                    complete=e["complete"],
                    actor=a["success"],
                    expert_result=e.get("success"),
                    actor_history=a["history"],
                    expert_history=e.get("history"),
                    invalid_at=e.get("invalid_at"),
                )
            )
    csvfile("suffix_actions.csv", actions)
    study.save(docs / "suffix.json", s)
    d = s["deltas"]
    text = [
        "# 同一分叉动作由混合 actor 或提案专家接手",
        "",
        f"在 {s['pure']} 个分叉点的 {s['pure'] * 2} 个首步动作上，仅 actor 续跑成功 {s['actor_only']} 个，仅提案专家续跑成功 {s['expert_only']} 个，两者均成功 {s['both']} 个，均失败 {s['neither']} 个。每个房间贡献一对动作，动作数包含原选与替代。",
        "",
        "| actor 下替代相对原选 | 专家下破坏 | 专家下同结局 | 专家下救回 |",
        "|---|---:|---:|---:|",
    ]
    for label, vals in zip(["破坏", "同结局", "救回"], d):
        text.append(f"| {label} | {vals[0]} | {vals[1]} | {vals[2]} |")
    text += [
        "",
        f"只有 actor 后缀能产生救回标签的分叉点有 {s['actor_only_rescue']} 个，只有专家后缀有 {s['expert_only_rescue']} 个，两者均产生救回标签有 {s['both_rescue']} 个。",
        "",
        "使用原实验第 1、2 轮各 48 个房间。各房间按预定顺序交替选取优先替代或探索替代名额。当时的采样 actor 在两个实验臂中完全相同。actor 两条分支复用已记录的真实终局；专家两条分支从同一初态重放，逐次核对分叉点公开历史哈希、原选动作和候选的合法性。分叉后始终使用提出首步动作的一个 CAP，其记忆在分叉前也随真实历史推进。多个 CAP 提出相同动作时取采样 actor 打分最高者，同分取最小编号。",
        "",
        f"共盘点 {s['done']} 个房间，其中 {s['available']} 个有分叉，{s['pure']} 个具有两条完整纯专家后缀。可比较首步中有 {s['multi']} 个由多个 CAP 提出，诊断只覆盖所选提案者。这里评价的是具体开发前缀的后缀差异。专家后缀标签仅用于诊断。",
        "",
        "[逐动作证据](suffix_actions.csv) · [汇总](suffix.json)",
    ]
    text += [
        "",
        "| 首步 | 仅 actor 成功 | 仅提案专家成功 | 两者成功 | 两者失败 |",
        "|---|---:|---:|---:|---:|",
    ]
    for side, title in (("base", "原选"), ("alt", "替代")):
        selected = [a for a in actions if a["complete"] and a["side"] == side]
        totals = [
            sum((a["actor"], a["expert_result"]) == v for a in selected)
            for v in ((1, 0), (0, 1), (1, 1), (0, 0))
        ]
        text.append(f"| {title} | " + " | ".join(map(str, totals)) + " |")
    (docs / "suffix.md").write_text("\n".join(text) + "\n")
    return s


def report(hist, evals):
    from tools import cap_regional_replay_relabel as relabel

    expert = suffix_report()
    replay.engine.publish(stage="relabel", round=24)
    drift = relabel.run(REFERENCE, replay.OUT)
    assert len(hist) == 24 and len(evals) == 200
    assert sum(r["steps"] for r in hist) == study.read(REFERENCE / "FINISHED.json")["updates"]
    assert all(r["history_draws"] == r["current_draws"] == 384 for r in hist[1:])
    old = {
        name: replay.records(REFERENCE / "final" / arm)
        for name, arm in [("initial", "initial"), ("current", "iterative"), ("pooled", "pooled")]
    }
    assert all(len(g) == 200 for g in old.values())
    stats = {name: replay.engine.paired(g, evals) for name, g in old.items()}
    result = dict(
        n=200,
        replay=sum(r["baseline"]["success"] for r in evals),
        comparisons=stats,
        **{name: m["first"] for name, m in stats.items()},
        updates=sum(r["steps"] for r in hist),
    )
    rooms = {}
    for name, rows in {**old, "replay": evals}.items():
        for r in rows:
            rooms.setdefault(r["id"], dict(room=r["id"], objects=r["objects"])).update(
                {
                    name: r["baseline"]["success"],
                    name + "_ticks": r["baseline"]["ticks"],
                    name + "_history": r["baseline"]["history"],
                }
            )
    csvfile("final_rooms.csv", [rooms[k] for k in sorted(rooms)])
    round_rows = []
    for h in hist:
        m = h["vs_current"]
        t = h["train_vs_current"]
        round_rows.append(
            dict(
                round=h["round"],
                actor_sha=h["actor_sha"],
                collection_actor=h["collector_sha"],
                fresh_pairs=h["pairs"],
                historical_pairs=h["old_pairs"],
                new_draws=h["current_draws"],
                old_draws=h["history_draws"],
                fresh_before=h["before"],
                fresh_after=h["after"],
                old_before=h["history_before"],
                old_after=h["history_after"],
                monitor_before=m["first"],
                monitor_after=m["second"],
                rescued=m["rescued"],
                lost=m["lost"],
                train_current=t["first"],
                train_replay=t["second"],
                train_rescued=t["rescued"],
                train_lost=t["lost"],
            )
        )
    csvfile("rounds.csv", round_rows)
    # Both arms are scored on identical new frozen-policy forks, before seeing those rooms.
    # Stored labels are explicitly conditional on the original actor as the continuation.
    common = {"n": 0, "current": 0, "replay": 0}
    with torch.inference_mode():
        for r in range(3, 25):
            data = replay.engine.datasets([REFERENCE / f"r{r:02d}" / "frozen"])
            keep = data["d"] != 0
            if not keep.any():
                continue
            data = {k: v[keep] for k, v in data.items()}
            common["n"] += len(data["d"])
            for name, path in [("current", REFERENCE), ("replay", replay.OUT)]:
                model = replay.engine.actor_load(path / f"r{r - 1:02d}" / "checkpoint/actor.pt")
                common[name] += round(replay.acc(model, data) * len(data["d"]))
    retention = {}
    for arm, root in [("current", REFERENCE), ("replay", replay.OUT)]:
        directories = [
            (REFERENCE if arm == "replay" and r <= 2 else root) / f"r{r:02d}" / "iterative"
            for r in range(1, 24)
        ]
        old_data, _ = replay.load_pairs(directories)
        n = len(old_data["d"])
        model = replay.engine.actor_load(root / "r24/checkpoint/actor.pt")
        if arm == "replay":
            as_learned = sum(round(z["after"] * z["pairs"]) for z in hist[:-1])
        else:
            fits = [study.read(root / f"r{r:02d}" / "checkpoint/fit.json") for r in range(1, 24)]
            as_learned = sum(round(z["after"] * z["pairs"]) for z in fits)
        retention[arm] = dict(n=n, as_learned=as_learned, final=round(n * replay.acc(model, old_data)))
    preupdate = {
        arm: sum(r["train_vs_current"][key] for r in hist[2:])
        for arm, key in [("current", "first"), ("replay", "second")]
    }
    preupdate["n"] = sum(r["train_vs_current"]["n"] for r in hist[2:])
    study.save(
        docs / "results.json",
        dict(
            summary=result,
            common_new_pairs=common,
            retention=retention,
            preupdate=preupdate,
            drift=drift,
            expert=expert,
            history=hist,
            protocol=study.read(replay.OUT / "protocol.json"),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    x = [h["round"] for h in hist]
    for name, key in [("Replay", "second"), ("Current only", "first")]:
        axes[0].plot(x, [h["vs_current"][key] for h in hist], label=name)
    axes[0].set_ylabel("Greedy successes / 48")
    for label, k in [("Current pairs", "after"), ("Historical pairs", "history_after")]:
        axes[1].plot(x, [h[k] if h[k] is not None else np.nan for h in hist], label=label)
    axes[1].set_ylabel("Training pair accuracy after update")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("Round")
        ax.grid(alpha=0.2)
    fig.savefig(docs / "rounds.png", dpi=150)
    plt.close(fig)
    text = [
        "# 当前 actor 采样与历史动作对回放",
        "",
        f"固定第 24 轮 actor 在 200 个房间中成功 {result['replay']} 次。仅当轮更新为 {result['current']} 次，监督起点为 {result['initial']} 次，冻结采样集中训练为 {result['pooled']} 次。",
        "",
        "| 参照 | 回放救回 / 丢失 | 净差（百分点） | 场景配对 95% 区间 |",
        "|---|---:|---:|---:|",
    ]
    for name, label in [
        ("current", "仅当轮（主对照）"),
        ("initial", "监督起点"),
        ("pooled", "集中拟合"),
    ]:
        m = stats[name]
        text.append(
            f"| {label} | {m['rescued']} / {m['lost']} | {m['pp']:+.1f} | [{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] |"
        )
    text += [
        "",
        "24 轮，每轮 48 个随机房间，每个房间两个分叉名额。所有标签是实测的终局成败差，按采样 actor 的 SHA256 记录后缀版本。第一轮相同，直接复用采集、模型和优化器；第二轮复用相同 parent 在同批房间上的采集。从第二轮起每步抽 16 条当轮和 16 条历史有效偏好，历史部分先等概率抽有有效偏好的房间，再抽该房间内的偏好。KL 约束继续取当轮 32 条边界。两组各更新 576 步，batch 32。",
        "",
        "固定房间相同，随着模型更新，真实走到的状态和实际成败标签可以不同。第 24 轮预先确定。最终 200 个面板房间此前已查看，按物体数分层、场景配对 bootstrap 10,000 次，区间只描述同一训练种子的场景差异。",
        "",
        f"另在第 3–24 轮共同的新房间分叉上，用进入当轮前的两个模型排序，后缀固定为初始 actor；在 {common['n']} 个实测成败不同的动作对中，回放排对 {common['replay']}，仅当轮排对 {common['current']}。",
        "",
        f"每个新训练房间在当轮更新前已有完整基线执行。第 3–24 轮累计 {preupdate['n']} 个这样的房间上，回放成功 {preupdate['replay']}，仅当轮成功 {preupdate['current']}；这里汇总的是一系列检查点。",
        "",
        "| 复查各自前 23 轮实测偏好 | 各批刚学完 | 第 24 轮重测 |",
        "|---|---:|---:|",
        *[
            f"| {title} | {q['as_learned']}/{q['n']} | {q['final']}/{q['n']} |"
            for title, q in [("仅当轮", retention["current"]), ("回放", retention["replay"])]
        ],
        "",
        "复查沿用采集时的旧标签。两组从各自到达的状态采样，此表用于跟踪各组自身的保留情况。",
        "",
        "![逐轮曲线](rounds.png)",
        "",
        "[后缀诊断](suffix.md) · [逐房间结果](final_rooms.csv) · [逐轮统计](rounds.csv) · [协议与配对统计](results.json)",
    ]
    text.extend(
        [
            "",
            "## 同一旧前缀更换 actor 后缀",
            "",
            f"另取第一轮事先选入后缀诊断的 {drift['n']} 个分叉点。固定原历史和两个首步，从下一步开始改由第 24 轮回放 actor 执行。原来 {drift['original_informative']} 对有明确偏好，其中 {drift['strict_flips']} 对排序严格反转、{drift['preference_to_tie']} 对变成同结局；原先同结局的动作对中 {drift['tie_to_preference']} 对出现新偏好。这部分只检验标签随接手策略变化的程度。",
        ]
    )
    study.save(docs / "relabel.json", drift)
    (docs / "README.md").write_text("\n".join(text) + "\n")
    study.save(replay.OUT / "FINISHED.json", result)
    return result
