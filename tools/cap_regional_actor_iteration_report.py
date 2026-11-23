"""Fixed final-checkpoint comparisons for the ten-CAP rollout actor experiment."""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tools import cap_regional_actor_iteration as run


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(hist, results, pooled_fit):
    docs = run.DOCS
    docs.mkdir(parents=True, exist_ok=True)
    assert len(hist) == run.ROUNDS and [r["round"] for r in hist] == list(range(1, run.ROUNDS + 1))
    assert pooled_fit["steps"] == sum(r["updates"] for r in hist)
    groups = {arm: {r["id"]: r for r in rows} for arm, rows in results.items()}
    keys = set(groups["initial"])
    assert len(keys) == 200 and all(set(g) == keys for g in groups.values())
    rows = []
    for cid in sorted(keys):
        row = dict(case_id=cid, objects=groups["initial"][cid]["objects"])
        for arm, g in groups.items():
            b = g[cid]["baseline"]
            row[arm] = b["success"]
            row[arm + "_ticks"] = b["ticks"]
            row[arm + "_history"] = b["history"]
        rows.append(row)
    write_csv(docs / "final_rooms.csv", rows)
    series = []
    for h in hist:
        record = dict(
            round=h["round"],
            updates=h["updates"],
            ordering_before=h["before"],
            ordering_after=h["after"],
            monitor=h["monitor"]["second"],
            rescued=h["monitor"]["rescued"],
            lost=h["monitor"]["lost"],
            previous_rescued=h["previous"]["rescued"],
            previous_lost=h["previous"]["lost"],
        )
        for arm in ("iterative", "frozen"):
            assert sum(h[arm].values()) == 2 * run.ROOMS
            record.update({arm + "_" + k: v for k, v in h[arm].items()})
        series.append(record)
    write_csv(docs / "rounds.csv", series)
    summary = {arm: sum(r[arm] for r in rows) for arm in groups}
    comparisons = {
        a + "_to_" + b: run.paired(results[a], results[b])
        for a, b in [("initial", "iterative"), ("initial", "pooled"), ("pooled", "iterative")]
    }
    summary.update(comparisons=comparisons, updates=pooled_fit["steps"], n=len(keys))
    run.study.save(
        docs / "results.json",
        dict(
            summary=summary,
            protocol=run.study.read(run.OUT / "protocol.json"),
            history=hist,
            pooled_fit=pooled_fit,
        ),
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    x = [r["round"] for r in series]
    axes[0].plot([0, *x], [hist[0]["monitor"]["first"], *[r["monitor"] for r in series]], marker=".")
    axes[0].axhline(hist[0]["monitor"]["first"], color="gray", linestyle="--")
    axes[0].set_ylabel("Greedy successes (48 rooms)")
    for arm in ("iterative", "frozen"):
        axes[1].plot(x, [r[arm + "_wins"] + r[arm + "_losses"] for r in series], label=arm)
    axes[1].set_ylabel("Outcome-changing pairs")
    axes[1].legend()
    for key in ("ordering_before", "ordering_after"):
        axes[2].plot(x, [r[key] for r in series], label=key.split("_")[1])
    axes[2].set_ylabel("Training pair ordering accuracy")
    axes[2].legend()
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlabel("Round")
    fig.savefig(docs / "rounds.png", dpi=150)
    plt.close(fig)
    lines = [
        "# 分叉动作排序直接更新十维 actor",
        "",
        f"预定第 {run.ROUNDS} 轮的迭代模型在 200 个新房间成功 {summary['iterative']} 次，冻结策略采样后集中拟合成功 {summary['pooled']} 次，初始监督模型成功 {summary['initial']} 次。",
        "",
        "| 参照 → 候选 | 救回 / 丢失 | 净差（百分点） | 配对 95% 区间 |",
        "|---|---:|---:|---:|",
    ]
    labels = ["监督起点 → 迭代", "监督起点 → 集中拟合", "集中拟合 → 迭代（主对照）"]
    for label, m in zip(labels, comparisons.values()):
        lines.append(
            f"| {label} | {m['rescued']} / {m['lost']} | {m['pp']:+.1f} | [{m['ci'][0]:+.1f}, {m['ci'][1]:+.1f}] |"
        )
    lines += [
        "",
        "两组从同一 seed17 十维监督模型开始，每轮 48 个同批随机房间。各组每房间执行基线和两个分叉名额。第一轮采样完全相同，共用轨迹。决策边界从实际走过的多动作边界的前、后两半各抽一个，随机分配优先替代与探索替代。有低排名动作时探索名额在它们中均匀抽取，无可用的不重复分叉时重跑原策略校验。后缀均由各组的采样 actor 接手。训练房间按独立轮次种子生成，不做历史几何去重。",
        "",
        f"成败不同的动作对用于排序，给出同一个物理动作的 CAP 取最高合法 logit。全部 actor 权重可学，保留初始策略的 KL 约束。迭代组仅用当轮新样本更新，冻结采样组在最后集中训练，两组各 {pooled_fit['steps']} 次 Adam 更新，batch 32。主对照同时涵盖后续采样分布、后缀与更新日程三者的差异。",
        "",
        "每轮监测使用固定 48 个诊断房间，最终 200 个同场景配对房间采用单独种子。最终评测使用预定末轮，两组均为 seed17。配对区间在物体数分层内按场景重抽 10,000 次，只反映场景差异。",
        "",
        "![逐轮诊断、成败分叉与排序拟合](rounds.png)",
        "",
        "[每轮数据](rounds.csv) · [最终逐房间配对](final_rooms.csv) · [配置和统计](results.json)",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")
    index = docs.parent / "README.md"
    marker = "|---|---|---|\n"
    entry = f"| 2026-11-20 | [分叉动作排序直接更新十维 actor]({docs.name}/README.md) | {run.ROUNDS} 轮，新 200 房间起点 {summary['initial']}、迭代 {summary['iterative']}、冻结采样集中拟合 {summary['pooled']} |\n"
    content = index.read_text()
    if entry not in content:
        index.write_text(content.replace(marker, marker + entry, 1))
    current = run.study.ext.old.ROOT / "docs/agent_exchange/CURRENT.md"
    content = current.read_text()
    if "<!-- ACTOR_ITERATION -->" in content:
        lines = content.splitlines()
        lines = [
            f"<!-- ACTOR_ITERATION -->十维 actor 的分叉迭代已完成 {run.ROUNDS} 轮。最终 200 房间中监督起点 {summary['initial']} 次成功、迭代 {summary['iterative']} 次、冻结采样集中拟合 {summary['pooled']} 次。[报告](../research/key_findings/{docs.name}/README.md)"
            if l.startswith("<!-- ACTOR_ITERATION -->")
            else l
            for l in lines
        ]
        current.write_text("\n".join(lines) + "\n")
    run.study.save(run.OUT / "FINISHED.json", summary)
    return summary
