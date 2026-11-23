"""Summarize the predeclared final checkpoint, paired by scene and perturbation family."""

import csv
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tools import cap_regional_id_ppo as run
from tools.cap_regional_id_report import count, read_rows

study = run.study


def report():
    out, docs = run.OUT, run.DOCS
    docs.mkdir(parents=True, exist_ok=True)
    protocol = study.read(out / "protocol.json")
    history = study.read(out / "history.json")
    schedule = study.read(out / "schedule.json")
    backend = study.read(out / "cpp_backend.json")
    assert len(history) == len(schedule) == 40
    for i, h in enumerate(history, 1):
        cases = study.read(out / f"round_{i:02d}.json")
        assert [r["case_id"] for r in cases] == [cid for cid, _ in schedule[i - 1]]
        assert len(cases) == 50 and sum(r["boundaries"] for r in cases) == h["boundaries"]
        assert sum(r["success"] for r in cases) == h["success"] and h["episodes"] == i * 50
    sids = study.read(study.OUT / "frozen.json")["sids"]
    baseline = {r["case_id"]: r for r in read_rows(run.base.DOCS / "per_scene.csv")}
    matrix = {r["case_id"]: r for r in read_rows(study.DOCS / "per_scene.csv")}
    rows = []
    panels = {}
    for panel, n in run.base.PANELS.items():
        episodes = study.read(out / "evaluation" / panel / "results.json")
        assert len(episodes) == n
        for r in episodes:
            cid = r["case_id"]
            b = baseline[cid]
            m = matrix[cid]
            assert (
                b["panel"] == m["panel"] == panel
                and 0 < r["ticks"] <= 1800
                and r["boundaries"] == len(r["selected_caps"])
            )
            sid = sids[r["initial_cap"]]
            rows.append(
                dict(
                    case_id=cid,
                    panel=panel,
                    parent=m["parent"],
                    objects=int(m["objects"]),
                    ncap=int(b["ncap"]),
                    before=int(b["id_fixed"]),
                    after=r["success"],
                    best=int(m["best"]),
                    directq=int(m["directq"]),
                    initial_before=b["initial_fixed"],
                    initial_after=sid,
                    commit_before=int(b["commit_fixed"]),
                    commit_after=int(m[sid]),
                    switches_before=int(b["switches_fixed"]),
                    switches_after=r["switches"],
                    distinct_before=int(b["distinct_switches_fixed"]),
                    distinct_after=r["distinct_switches"],
                    ticks=r["ticks"],
                    boundaries=r["boundaries"],
                    fallback=r["fallback"],
                    history=r["history"],
                    same_path=int(r["history"] == b["history_fixed"]),
                )
            )
            if not r["switches"] and not r["fallback"]:
                assert r["success"] == int(m[sid])
    assert len({r["case_id"] for r in rows}) == len(rows) == 505
    for panel in [*run.base.PANELS, "train"]:
        g = [
            r for r in rows if (r["panel"] in ("dev", "aug") if panel == "train" else r["panel"] == panel)
        ]
        stats = dict(
            n=len(g), union=sum(bool(r["ncap"]) for r in g), single=sum(r["ncap"] == 1 for r in g)
        )
        for k in ("before", "after", "directq", "best", "commit_before", "commit_after"):
            stats[k] = count(g, k)
        stats["rescued"] = sum(r["after"] > r["before"] for r in g)
        stats["lost"] = sum(r["after"] < r["before"] for r in g)
        stats["same_path"] = sum(r["same_path"] for r in g)
        stats["same_initial"] = sum(r["initial_before"] == r["initial_after"] for r in g)
        stats["switching"] = {}
        for method in ("before", "after"):
            stats["switching"][method] = dict(
                distinct_episodes=sum(r["distinct_" + method] > 0 for r in g),
                rescued=sum(r[method] > r["commit_" + method] for r in g),
                lost=sum(r[method] < r["commit_" + method] for r in g),
                initial=dict(Counter(r["initial_" + method] for r in g)),
            )
        if panel in ("near", "fresh"):
            stats["comparisons"] = {
                k: study.compare(g, k, "after") for k in ("before", "best", "directq", "commit_after")
            }
        panels[panel] = stats
    monitor = []
    for i in range(0, 41, 10):
        results = study.read(out / "monitor" / f"r{i:02d}" / "results.json")
        assert [r["case_id"] for r in results] == study.read(out / "monitor_ids.json")
        monitor.append(sum(r["success"] for r in results))
    with (docs / "per_scene.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with (docs / "training.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(history[0]))
        wr.writeheader()
        wr.writerows(history)
    study.save(
        docs / "results.json",
        dict(
            panels=panels,
            monitor=monitor,
            protocol=protocol,
            backend=backend,
            final_actor_sha256=study.sha(out / "final_actor.pt"),
        ),
    )
    x = [h["episodes"] for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    ax = axes[0, 0]
    ax.plot(x, [h["success"] / 50 for h in history], label="Sampled / 50")
    ax.plot(range(0, 2001, 500), np.array(monitor) / 40, "o-", label="Greedy / 40 fixed online rooms")
    ax.set_ylabel("Success rate")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    for ax, key, label in [
        (axes[0, 1], "entropy", "CAP identity entropy (nats)"),
        (axes[1, 0], "actor_loss", "Clipped actor loss"),
        (axes[1, 1], "critic_loss", "Critic MSE"),
    ]:
        ax.plot(x, [h[key] for h in history])
        ax.set_ylabel(label)
    for ax in axes.flat:
        ax.set_xlabel("Online training episodes")
        ax.grid(alpha=0.2)
    fig.savefig(docs / "learning_curves.png", dpi=150)
    plt.close(fig)
    f = panels["fresh"]
    primary = f["comparisons"]["before"]
    best = f["comparisons"]["best"]
    c = protocol["temperature_calibration"]
    lines = [
        "# 十维 CAP 编号 actor 的 PPO 微调",
        "",
        f"固定最终模型在 200 个自然留出房间成功 {f['after']['success']} 例，监督起点成功 {f['before']['success']} 例。配对净差 {primary['pp']:+.1f} 个百分点，95% 区间 [{primary['ci'][0]:+.1f}, {primary['ci'][1]:+.1f}]。开发集选定的最强单 CAP 成功 {best['first']} 例，PPO 相对它救回 {best['rescued']} 例、丢失 {best['lost']} 例，区间 [{best['ci'][0]:+.1f}, {best['ci'][1]:+.1f}]。",
        "",
        "| 面板 | 监督起点 | PPO 最终 | 救回 / 丢失 | 专家并集内保留（前 → 后） | 独胜保留（前 → 后） | 并集外成功（前 → 后） |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in [
        ("dev", "原始开发"),
        ("aug", "开发扰动"),
        ("fresh", "自然留出"),
        ("near", "困难族扰动"),
    ]:
        p = panels[key]
        a = p["before"]
        b = p["after"]
        lines.append(
            f"| {label} | {a['success']}/{p['n']} | {b['success']}/{p['n']} | {p['rescued']} / {p['lost']} | {a['retained']} → {b['retained']} / {p['union']} | {a['single']} → {b['single']} / {p['single']} | {a['outside']} → {b['outside']} / {p['n'] - p['union']} |"
        )
    lines += [
        "",
        "## 开局选择与后续切换",
        "",
        "按同一房间的单 CAP 实测结果，计算如果始终跟随 actor 首选的 CAP 会成功几次。每步重选另统计相对该参照救回及丢失的场景。",
        "",
        "| 面板 | 模型 | 固定首选 | 每步重选 | 救回 / 丢失 | 发生过不同动作切换的房间 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, label in [("fresh", "自然留出"), ("near", "困难族扰动")]:
        p = panels[key]
        for method, name in [("before", "监督起点"), ("after", "PPO")]:
            s = p["switching"][method]
            lines.append(
                f"| {label} | {name} | {p['commit_' + method]['success']}/{p['n']} | {p[method]['success']}/{p['n']} | {s['rescued']} / {s['lost']} | {s['distinct_episodes']}/{p['n']} |"
            )
    lines += [
        "",
        f"200 个自然房间中，两模型有 {f['same_initial']} 个首选同一 CAP，{f['same_path']} 个完整公开历史哈希相同。困难族扰动中对应为 {panels['near']['same_initial']}/90、{panels['near']['same_path']}/90。历史哈希覆盖实际执行的动作和观测。",
        "",
        "## 训练与采样",
        "",
        "从 seed 17、24,000 次监督更新的固定模型起步，保持 64×64 十维 actor 和动作掩码。使用与监督和评估均按内容去重的 200 个在线训练房间，每轮均匀无放回抽 50 个，共 40 轮。保留单独的 critic，成功时末端奖励 1，其余为 0，预算截断按终止处理。γ=1、GAE λ=0.95、clip=0.2、batch 128，actor/critic 学习率分别为 3×10⁻⁵、3×10⁻⁴。无熵奖励。按预定第 40 轮评测。",
        "",
        f"监督 logits 很尖锐。在训练池 {c['multi_action_starts']} 个至少有两种实际动作的初态上，固定温度 {c['temperature']:.3f} 把采样到与贪心不同的物理动作的平均概率从 {c['raw_alternative']:.1%} 调至 {c['scaled_alternative']:.1%}，PPO 新旧概率均用此温度。实际采样共有 {sum(h['boundaries'] for h in history):,} 个边界，存在多种动作的边界中 {sum(h['alternate'] for h in history):,}/{sum(h['eligible'] for h in history):,} 次采到了其他动作。",
        "",
        f"固定 40 个在线训练房间的贪心成功数（每 500 个 episode）依次为 {'、'.join(map(str, monitor))}。主结果来自此前已查看的自然留出面板。场景配对 bootstrap 按物体数量分层，困难扰动以 45 个父房间重抽。单训练种子的区间只刻画场景抽样差异。",
        "",
        "![训练回报、监测成功率、熵和损失](learning_curves.png)",
        "",
        "## C++ 后端",
        "",
        f"第 5 轮后核对十个策略的 C++ 转写。{backend['paired_episodes']} 个在线房间的 {backend['paired_boundaries']:,} 个边界逐项比较十路动作、合法候选、几何特征及策略记忆。另做 {backend['ppo_paired_episodes']} 对同模型同种子的完整随机 PPO 采样，核对状态张量、选择、概率、优势、回报与最终历史。使用 16 个工作进程，以 Cython→C++、C++→Cython 两种顺序计时，包含进程池、仿真、网络推理与分片写入。",
        "",
        "十个 CAP 共做 6,500 次合成边界更新，对原始程序逐项检查完整记忆和上下文，数值完全一致。真实边界核对要求动作、候选和输入张量完全一致，内部浮点记忆容差为绝对 2×10⁻¹²、相对 10⁻¹²。每轮计时前已完成原 Cython 和新 C++ 模块的编译。每种后端累计计时 64 个 episode，来自 32 个房间和两组采样种子。",
        "",
        "| 顺序组 | 后端 | episode | 墙钟秒数 | episode/s |",
        "|---|---|---:|---:|---:|",
    ]
    for m in backend["measurements"]:
        lines.append(
            f"| {m['block'] + 1} | {m['backend']} | {m['n']} | {m['seconds']:.2f} | {m['n'] / m['seconds']:.2f} |"
        )
    lines += [
        "",
        f"总时长比为 {backend['speedup']:.2f}×。后 35 轮使用 {'C++' if backend['use_cpp'] else 'Cython'} 后端。",
        "",
        "[逐场景结果](per_scene.csv) · [逐轮训练](training.csv) · [协议、配对统计和吞吐](results.json)",
    ]
    (docs / "README.md").write_text("\n".join(lines) + "\n")
    study.save(
        out / "FINISHED.json",
        dict(episodes=2000, monitor_episodes=200, final_evaluations=505, report=str(docs / "README.md")),
    )
    study.save(out / "stage.json", dict(stage="finished", round=40))


if __name__ == "__main__":
    report()
