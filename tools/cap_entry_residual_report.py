"""Room-paired report for the frozen single-opportunity CAP deferral experiment."""
import argparse
import csv
import io
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/semantic_rl/cap_entry_residual_20261124_01"
DOC = ROOT / "docs/research/key_findings/2026-11-24_entry_residual"
REGISTRY = ROOT / "docs/research/experiment_journal/entries.json"
ID = "entry-residual"
ARMS = {"parent": "母策略", "zero": "原程序（零残差）", "learned": "拟合的暂缓规则",
        "always_defer": "每逢接手都暂缓"}
COMPARISONS = [("zero", "parent"), ("learned", "zero"), ("learned", "parent"),
               ("learned", "always_defer"), ("always_defer", "parent")]


def read(path):
    return json.loads(path.read_text())


def atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as f:
        f.write(content)
        temp = Path(f.name)
    temp.replace(path)


def as_json(data):
    return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def update_registry(**changes):
    text = REGISTRY.read_text()
    start = text.rindex("\n  {", 0, text.index(f'"id": "{ID}"')) + len("\n  ")
    entry, length = json.JSONDecoder().raw_decode(text[start:])
    if entry["id"] != ID:
        raise ValueError("unexpected journal entry")
    entry.update(changes)
    body = json.dumps(entry, ensure_ascii=False, indent=2).replace("\n", "\n  ")
    updated = text[:start] + body + text[start + length:]
    json.loads(updated)
    atomic(REGISTRY, updated)


def progress(sizes):
    stages = [dict(label="冻结面板与协议", file="protocol.json", total=1),
              dict(label="零残差与同前缀复现", file="parity.json", total=1)]
    for split, title in (("train", "训练房间"), ("validation", "选阈值房间")):
        stages.append(dict(label=f"{title}原程序扫描", glob=f"runs/{split}/zero/*.json", total=sizes[split]))
    stages.append(dict(label="成对分叉标签与拟合", file="fit.json", total=1))
    for arm, name in ARMS.items():
        stages.append(dict(label=f"新房间 · {name}", glob=f"runs/held/{arm}/*.json", total=sizes["held"]))
    stages.append(dict(label="冻结后新房间分叉诊断", file="held_diagnostic.json", total=1))
    stages.append(dict(label="逐房间配对与报告", file="finished.json", total=1))
    update_registry(progress=stages)


def paired(rows, candidate, reference):
    pairs = [(r[candidate], r[reference]) for r in rows]
    n = len(pairs)
    plus = sum(a > b for a, b in pairs)
    minus = sum(a < b for a, b in pairs)
    # Multinomial draw of discordant and tied pairs equals paired room bootstrap.
    sampled = np.random.default_rng(17).multinomial(
        n, [plus/n, minus/n, (n-plus-minus)/n], size=20000)
    interval = (100 * np.quantile((sampled[:, 0]-sampled[:, 1])/n, [.025, .975])).tolist()
    return dict(candidate=candidate, reference=reference, n=n, gains=plus, losses=minus,
                delta_pp=100*(plus-minus)/n, ci95_pp=interval)


def decomposition(rows):
    groups = {}
    for parent, zero, name in ((0, 1, "original_rescues"), (1, 0, "original_losses"),
                               (1, 1, "common_successes"), (0, 0, "common_failures")):
        subset = [r for r in rows if r["parent"] == parent and r["zero"] == zero]
        wins = sum(r["learned"] for r in subset)
        groups[name] = dict(rooms=len(subset), learned_successes=wins, learned_failures=len(subset)-wins)
    return groups


def held_rows(out, rooms):
    groups = {}
    for arm in ARMS:
        files = {p.stem: p for p in (out/"runs"/"held"/arm).glob("*.json")}
        if set(files) != set(rooms):
            raise ValueError(f"{arm}: missing or unexpected held rooms")
        groups[arm] = files
    rows = []
    for room in sorted(rooms):
        row = {"room": room}
        for arm, files in groups.items():
            record = read(files[room])
            result = record["result"]
            if result["success"] not in (0, 1) or not result["ticks"] > 0 or not result["history"]:
                raise ValueError(f"{arm}/{room}: incomplete rollout")
            entries = record["entries"]
            row.update({arm: int(result["success"]), arm+"_ticks": result["ticks"],
                        arm+"_history": result["history"], arm+"_entries": len(entries),
                        arm+"_deferred": sum(e["deferred"] for e in entries)})
        rows.append(row)
    return rows


def collect(out):
    protocol, fit = read(out/"protocol.json"), read(out/"fit.json")
    held = read(out/"fixtures/held.json")
    if len(held) != protocol["sizes"]["held"] or not held:
        raise ValueError("incomplete held fixture manifest")
    rows = held_rows(out, held)
    if not (out/"held_diagnostic.json").is_file():
        from tools.cap_entry_residual_study import held_diagnostic
        held_diagnostic(out)
    dev = {}
    for split in ["train", "validation", "held"]:
        fixtures = read(out/"fixtures"/f"{split}.json")
        if set(fixtures) != {p.stem for p in (out/"runs"/split/"zero").glob("*.json")}:
            raise ValueError(f"incomplete {split} scan")
        supplies = {}
        for path in (out/"runs"/split/"zero").glob("*.json"):
            entries = read(path)["entries"]
            if entries:
                supplies[path.stem] = min(protocol["entry_cap"], len(entries))
        seen = set()
        counts = dict(fail_fail=0, both_win=0, defer_wins=0, take_wins=0)
        by_event = {}
        for p in sorted((out/"labels"/split).glob("*.json")):
            batch = read(p)
            if p.stem != batch["room"] or len(batch["rows"]) != supplies.get(p.stem):
                raise ValueError(f"unexpected labels: {p}")
            for item in batch["rows"]:
                if item["room"] != p.stem:
                    raise ValueError(f"wrong room in label file: {p}")
                k = {(0, 0):"fail_fail", (1, 1):"both_win", (0, 1):"defer_wins", (1, 0):"take_wins"}
                outcome = k[item["take"], item["defer"]]
                counts[outcome] += 1
                group = by_event.setdefault(item["anchor"]["kind"], dict(fail_fail=0, both_win=0, defer_wins=0, take_wins=0))
                group[outcome] += 1
            seen.add(p.stem)
        if set(supplies) != seen:
            raise ValueError(f"missing paired labels in {split}")
        dev[split] = dict(room_count=len(fixtures), eligible_rooms=len(seen), by_event=by_event, **counts)
    stats = {}
    for arm in ARMS:
        stats[arm] = dict(success=sum(r[arm] for r in rows),
                          offers=sum(r[arm+"_entries"] for r in rows),
                          deferrals=sum(r[arm+"_deferred"] for r in rows),
                          rooms_with_offers=sum(bool(r[arm+"_entries"]) for r in rows),
                          rooms_with_deferrals=sum(bool(r[arm+"_deferred"]) for r in rows))
    held_diagnostic = read(out/"held_diagnostic.json")
    comparisons = {a+"_vs_"+b: paired(rows, a, b) for a, b in COMPARISONS}
    results = dict(n=len(rows), sizes=protocol["sizes"], entry_cap=protocol["entry_cap"],
                   parity=read(out/"parity.json"), scans_and_fork_labels=dev, held_diagnostic=held_diagnostic,
                   fit=fit, held=stats, paired=comparisons, decomposition=decomposition(rows),
                   always_defer_parent_hash_matches=sum(r["always_defer_history"]==r["parent_history"] for r in rows),
                   learned_zero_hash_matches=sum(r["learned_history"]==r["zero_history"] for r in rows),
                   changed_with_no_learned_deferral=sum(
                       r["learned_history"]!=r["zero_history"] and not r["learned_deferred"] for r in rows),
                   ci_method="empirical paired room percentile bootstrap, 20000 draws, seed 17; single fitted gate")
    if results["changed_with_no_learned_deferral"]:
        raise ValueError("learned trajectory changed without any recorded deferral")
    if (out/"room_selector.json").is_file():
        from tools.cap_entry_room_yield import evaluate
        results["room_selection_yield"] = evaluate(out)
    return rows, results


def csv_text(rows):
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def percent(value):
    return "—" if value is None else f"{value*100:.1f}%"


def render(d):
    n, held, fit = d["n"], d["held"], d["fit"]
    learned, zero, parent = (held[a]["success"] for a in ("learned", "zero", "parent"))
    q = d["paired"]["learned_vs_zero"]
    dec = d["decomposition"]
    rescued, lost = dec["original_rescues"], dec["original_losses"]
    summary = (f"{n} 个新房间中，母策略成功 {parent} 间，原恢复程序 {zero} 间，拟合暂缓规则 {learned} 间。"
               f"相对原恢复程序救回 {q['gains']} 间、丢失 {q['losses']} 间。")
    partition = (f"原补丁独自救回的 {rescued['rooms']} 间中保住 {rescued['learned_successes']} 间；"
                 f"原补丁丢失的 {lost['rooms']} 间中避免 {lost['learned_successes']} 间。")
    lines = ["# 只校准一次接手是否暂缓", "", summary, "", partition, "",
             "## 新房间从头运行", "",
             "| 冻结策略 | 成功 / 全部 | 出现接手提议的房间 | 实际暂缓的房间 | 暂缓次数 / 提议次数 |",
             "|---|---:|---:|---:|---:|"]
    for arm, title in ARMS.items():
        s = held[arm]
        col = [s["rooms_with_offers"], s["rooms_with_deferrals"], f"{s['deferrals']} / {s['offers']}"]
        if arm == "parent":
            col = ["—"]*3
        lines.append(f"| {title} | {s['success']} / {n} | "+" | ".join(map(str, col))+" |")
    lines.extend(["", "| 同房间配对 | 救回 | 丢失 | 净差（百分点） | 95% 配对区间（百分点） |",
                  "|---|---:|---:|---:|---:|"])
    for a, b in COMPARISONS:
        val = d["paired"][a+"_vs_"+b]
        lo, hi = val["ci95_pp"]
        lines.append(f"| {ARMS[a]} 对 {ARMS[b]} | {val['gains']} | {val['losses']} | "
                     f"{val['delta_pp']:+.2f} | [{lo:+.2f}, {hi:+.2f}] |")
    lines.extend(["", f"按房间成对重抽 20,000 次，区间反映在本次冻结策略下的房间差异。四份运行共享 {n} 个初态。"
                  "若没有观察到成败翻转，自助区间会缩为零，有限样本仍不能排除其他新房间中的差异。", "",
                  "## 已有路线的保留", "",
                  "| 同一个新房间的原始结果 | 房间 | 拟合后成功 | 拟合后失败 |",
                  "|---|---:|---:|---:|"])
    for key, title in [("original_rescues", "母策略失败、原补丁成功"), ("original_losses", "母策略成功、原补丁失败"),
                       ("common_successes", "两者都成功"), ("common_failures", "两者都失败")]:
        r = dec[key]
        lines.append(f"| {title} | {r['rooms']} | {r['learned_successes']} | {r['learned_failures']} |")
    lines.extend(["", f"每次暂缓与单独执行母策略的终点公开历史在 {d['always_defer_parent_hash_matches']}/{n} 间相同。"
                  f"拟合规则与零残差的终点公开历史在 {d['learned_zero_hash_matches']}/{n} 间相同。",
                  "", "## 训练和选阈值的同前缀分叉", "",
                  "冻结 trial09 的母策略、trial06 的恢复动作与交还规则。"
                  "在原程序提出可执行恢复动作的机会，从同一段已执行历史比较立即接手与暂缓这一次。"
                  "暂缓时执行母策略一步，之后重新开放原入口。训练标签为两支最终成功值之差。", "",
                  f"每个开发房间最多取 {d['entry_cap']} 个机会，超过上限时取首、中、末机会。"
                  "一间房的多次分叉共享环境，拟合和离线评价按房间等权，房间内部机会等权。", "",
                  "| 分区 | 扫描房间 | 有分叉的房间 | 配对机会 | 立即接手独胜 | 暂缓独胜 | 两支都胜 | 两支都败 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for split, title in (("train","训练"), ("validation","选阈值")):
        r = d["scans_and_fork_labels"][split]
        num = sum(r[k] for k in ["take_wins", "defer_wins", "both_win", "fail_fail"])
        lines.append(f"| {title} | {r['room_count']} | {r['eligible_rooms']} | {num} | {r['take_wins']} | "
                     f"{r['defer_wins']} | {r['both_win']} | {r['fail_fail']} |")
    r = d["scans_and_fork_labels"]["held"]
    num = sum(r[k] for k in ("take_wins", "defer_wins", "both_win", "fail_fail"))
    lines.append(f"| 新房间（冻结后诊断） | {r['room_count']} | {r['eligible_rooms']} | {num} | {r['take_wins']} | "
                 f"{r['defer_wins']} | {r['both_win']} | {r['fail_fail']} |")
    lines.extend(["", "## 接手决定出现在哪种边界", "",
                  "普通选动作时可处理新接手提议。单推手 PUSH 驱动期间，有效接触连续 3 tick 或接触中推手机器人沿指令方向累计推进 0.20 m，也会复核。一次暂缓只影响眼前这次机会。", "",
                  "| 新房间配对机会 | 采样数 | 立即接手独胜 | 暂缓一次独胜 |",
                  "|---|---:|---:|---:|"])
    for kind, title in (("boundary","普通决策"),("contact","接触复核"),("progress","推进复核")):
        group = r["by_event"].get(kind, {})
        lines.append(f"| {title} | {sum(group.values())} | {group.get('take_wins',0)} | {group.get('defer_wins',0)} |")
    lines.extend(["", "## 哪些接手值得推迟？", "",
                  f"以 {len(fit['gate']['features'])} 个公开特征拟合线性岭回归，正则系数 "
                  f"{fit['fit_protocol']['ridge']:g}，在独立开发房间选定暂缓阈值 {fit['gate']['threshold']:g}。"
                  "当预测的暂缓优势严格超过阈值时，母策略先执行一步。整局评估始终实跑这份拟合规则。", "",
                  "| 同前缀分叉 | 全部接手 | 全部暂缓一次 | 拟合规则 | 逐机会事后取较好分支 | 排序 AUC |",
                  "|---|---:|---:|---:|---:|---:|"])
    for split,title in (("train","训练"), ("validation","选阈值"), ("held","新房间")):
        s = d["held_diagnostic"] if split == "held" else fit["offline"][split]
        rates = [percent(s[k]["room_mean_chosen_success"]) for k in ("zero", "always_defer", "learned")]
        rates.append(percent(s["zero"]["room_mean_best_observed_single_fork"]))
        auc = s["discordant_room_weighted_auc"]
        rates.append(f"{auc:.3f}" if auc is not None else "—")
        lines.append(f"| {title} | "+" | ".join(rates)+" |")
    lines.extend(["", "此表仅含有合格接手提议的房间。各列从原程序实际到达的时刻选一次分支，下一边界起沿用原程序。"
                  "最右侧的排序指标只比较立即接手独胜与暂缓独胜的机会。事后较好分支一列给出看过两支结局后逐机会择优的房间平均成功率。整局部署反复作决定，会走到其他状态。", ""])
    learned_forks = d["held_diagnostic"]["learned"]
    preserved = learned_forks["better_takes_retained"]
    found = learned_forks["gained_against_take"]
    lines.extend([f"在独立房间分叉中，拟合规则保留了 {preserved}/{learned_forks['take_only']} 次立即接手独胜，"
                  f"找到 {found}/{learned_forks['defer_only']} 次暂缓独胜。"
                  f"实际部署暂缓的房间数是 {held['learned']['rooms_with_deferrals']}/{n}。", "",
                  "## 解释", ""])
    supported = fit["validation_supports_selective_gate"]
    selection = ("选阈值开发集的房间加权结果超过了两个固定选择，开发规则选用了拟合门。"
                 if supported else "预先约定的开发选择保留原程序，拟合门在选阈值房间没有同时超过两个固定选择。")
    lines.append(selection)
    analysis = partition
    offers, veto = held['learned']['offers'], held['learned']['deferrals']
    if offers and veto == 0:
        analysis += "拟合规则在新房间没有行使暂缓。"
    elif offers and veto == offers:
        analysis += "拟合规则在新房间暂缓了每一次合格提议。"
    else:
        analysis += f"拟合规则在新房间暂缓了 {held['learned']['deferrals']}/{held['learned']['offers']} 次合格提议。"
    lines.extend(["", analysis, ""])
    if q["ci95_pp"][0] <= 0 <= q["ci95_pp"][1]:
        lines.extend(["相对原程序的整局成功率差值区间包含零。本轮尚不足以断定在该场景分布中校准入口能稳定提高成功率。", ""])
    lines.extend(["只评估原恢复程序已经给出合法改选的边界；在原本沉默的状态提前提出恢复动作，以及校准交还时机，仍待检验。", "",
                  "[研究日志](http://localhost:8775/research-journal#entry-residual) · "
                  "[逐房间记录](http://localhost:8775/journal-file/entry-residual/rooms.csv) · "
                  "[完整统计](http://localhost:8775/journal-file/entry-residual/results.json)", ""])
    if "room_selection_yield" in d:
        lines += ["## 按布局优先选择可能出现接手提议的新房间", "",
                  "训练阶段另做五折检查，从同分布候选的初始布局中取预测触发率最高的 20%。只读取静态与可动障碍物的平面坐标，以训练房间中 64 个几何近邻的触发率加权打分。在读取新面板原补丁扫描结果之前，冻结该面板入选的房间编号。此项只改变统计中挑选的房间，完整闭环评测仍覆盖自然分布。", "",
                  "| 面板 | 所有房间有提议 | 入选房间有提议 | 所有房间测到终局翻转 | 入选房间测到终局翻转 |",
                  "|---|---:|---:|---:|---:|"]
        for split,title in [("training","训练池五折"),("held","新房间冻结筛选")]:
            a=d["room_selection_yield"][split]
            lines.append(f"| {title} | {a['triggered']}/{a['all']} | {a['selected_triggered']}/{a['selected']} | {a['discordant']}/{a['all']} | {a['selected_discordant']}/{a['selected']} |")
        lines.extend(["", "翻转沿用每间最多三个同前缀分叉的口径。命中率衡量模拟一间的样本产出，未实测只执行入选房间时节约的墙钟时间。", ""])
    return "\n".join(lines), summary, analysis


def build(out=OUT):
    out = Path(out)
    rows, d = collect(out)
    report, headline, analysis = render(d)
    atomic(DOC/"rooms.csv", csv_text(rows))
    atomic(DOC/"README.md", report)
    atomic(DOC/"results.json", as_json(d))
    update_registry(result=headline, analysis=analysis)
    atomic(out/"finished.json", as_json(dict(rooms=d["n"], summary=headline, analysis=analysis)))
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["progress", "build"])
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    if args.stage == "progress":
        progress(read(args.out/"protocol.json")["sizes"])
    else:
        build(args.out)
