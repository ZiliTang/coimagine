"""Publish held results of the locked fault-curriculum comparison after all rounds."""
import csv
from io import StringIO
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from tools.cap_regional_evaluate import paired

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs/research/key_findings/2026-11-24_fault_curriculum"
REGISTRY = ROOT / "docs/research/experiment_journal/entries.json"
ARMS = ("random", "graph")
NAMES = {"random": "随机前缀", "graph": "图选前缀"}
STRATA = {"clear": "无阻挡", "static": "静态障碍", "movable": "可动障碍", "both": "两者均阻挡"}
START = "<!-- fault-curriculum-results -->"
END = "<!-- /fault-curriculum-results -->"


def write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as f:
        tmp = Path(f.name)
        f.write(text)
    tmp.replace(path)


def update_registry(result, analysis):
    # Modify this experiment's entry, preserving unrelated hand-edited formatting.
    raw = REGISTRY.read_text()
    id_index = raw.index('"id": "fault-curriculum"')
    start = raw.rindex("\n  {", 0, id_index) + len("\n  ")
    entry, length = json.JSONDecoder().raw_decode(raw[start:])
    if entry["id"] != "fault-curriculum":
        raise ValueError("wrong journal entry")
    entry.update(result=result, analysis=analysis)
    entry.setdefault("assets", [])
    if ["rooms.csv", "留出逐房间结果"] not in entry["assets"]:
        entry["assets"].append(["rooms.csv", "留出逐房间结果"])
    item = json.dumps(entry, ensure_ascii=False, indent=2).replace("\n", "\n  ")
    new = raw[:start] + item + raw[start+length:]
    json.loads(new)
    write_atomic(REGISTRY, new)


def pp_interval(stat):
    lo, hi = stat["ci"]
    return f"{stat['pp']:+.1f} [{lo:+.1f}, {hi:+.1f}]"


def development_usage(directory):
    requests = [json.loads(p.read_text()) for p in directory.glob("r*/call_*/request.json")]
    accounts = [json.loads(p.read_text()) for p in directory.glob("r*/call_*/accounting.json")]
    trials = list(directory.glob("r*/candidate*/packet_runs.json")) + list(directory.glob("r*/winner_dev.json"))
    runs = ticks = 0
    for path in trials:
        rows = json.loads(path.read_text())
        runs += len(rows)
        ticks += sum(r["result"]["ticks"] for r in rows.values())
    return dict(requests=len(requests), responses=len(accounts),
                prompt_bytes=sum(r["prompt_bytes"] for r in requests),
                response_wait_seconds=round(sum(r["elapsed"] for r in accounts), 2),
                complete_logged_dev_runs=runs, complete_logged_dev_ticks=ticks,
                reported_usage=[r.get("usage") for r in accounts])


def build(out, mother, histories):
    frozen = json.loads((out / "rooms.json").read_text())
    keys = sorted(frozen["rooms"]["held"])
    strata = np.array([frozen["records"][k]["stratum"] for k in keys])
    if set(histories) != set(ARMS) or any(len(r) != 3 for r in histories.values()):
        raise ValueError("three locked rounds required for each arm")
    if set(strata) != set(STRATA):
        raise ValueError("expected all four geometric strata")
    trajectories = {"mother": mother}
    selections = {}
    for arm in ARMS:
        selections[arm] = []
        for r, rows in enumerate(histories[arm], 1):
            trajectories[f"{arm}_{r}"] = rows
            info = json.loads((out / arm / f"r{r}" / "selected.json").read_text())
            selections[arm].append({k: info[k] for k in ("dev_before", "candidate_dev", "dev_after", "admitted")})

    table = StringIO()
    writer = csv.writer(table, lineterminator="\n")
    writer.writerow(["room", "content_sha256", "initial_straight_path_obstacle",
                     *(tag+"_"+k for tag in trajectories for k in ("success", "ticks", "termination", "history"))])
    for tag, rows in trajectories.items():
        if set(rows) != set(keys):
            raise ValueError(f"incomplete or different held room set for {tag}")
    for k, st in zip(keys, strata):
        cells = [k, frozen["records"][k]["content_sha256"], st]
        for rows in trajectories.values():
            result = rows[k]["result"]
            termination = result["termination"]
            if termination is None:
                if result["ticks"] >= 1800:
                    termination = "tick_budget"
                elif result["decisions"] >= 300:
                    termination = "decision_budget"
                else:
                    raise ValueError(f"episode did not finish for {k}")
            if result["success"] not in (0, 1) or not result["history"]:
                raise ValueError(f"missing outcome or history for {k}")
            cells.extend((result["success"], result["ticks"], termination, result["history"]))
        writer.writerow(cells)
    outcomes = {tag: np.array([rows[k]["result"]["success"] for k in keys])
                for tag, rows in trajectories.items()}
    by_arm, paired_rounds = {}, []
    for arm in ARMS:
        stages = [outcomes[f"{arm}_{r}"] for r in (1, 2, 3)]
        by_arm[arm] = []
        for r, current in enumerate(stages):
            prior = stages[r-1] if r else outcomes["mother"]
            by_arm[arm].append(dict(selection=selections[arm][r],
                                    versus_mother=paired(outcomes["mother"], current, strata),
                                    versus_previous=paired(prior, current, strata)))
    for r in (1, 2, 3):
        paired_rounds.append(paired(outcomes[f"random_{r}"], outcomes[f"graph_{r}"], strata))
    strata_stats = {}
    for s in STRATA:
        ok = strata == s
        strata_stats[s] = dict(n=int(ok.sum()), successes={name: int(y[ok].sum()) for name, y in outcomes.items()},
                               graph_minus_random=paired(outcomes["random_3"][ok], outcomes["graph_3"][ok], strata[ok]))
    return dict(rooms=len(keys), mother=int(outcomes["mother"].sum()), rounds=by_arm, graph_minus_random=paired_rounds,
                by_stratum=strata_stats, usage={arm: development_usage(out/arm) for arm in ARMS}, paired_seed=17, paired_stratified_bootstrap=10000,
                witnessed_in_seven=int(np.logical_or.reduce(list(outcomes.values())).sum())), table.getvalue()


def publish(out, mother_rows, histories):
    out = Path(out).resolve()
    summary, table = build(out, mother_rows, histories)
    chief = summary["graph_minus_random"][-1]
    n = chief["n"]
    result = (f"共同母程序 {summary['mother']}/{n}，第 3 轮图选前缀 {chief['second']}/{n}，"
              f"随机前缀 {chief['first']}/{n}。图组相对随机组净差 {pp_interval(chief)} 个百分点。")
    if chief["ci"][0] <= 0 <= chief["ci"][1]:
        evidence = "房间配对区间跨过或触及零，当前面板未分辨出稳定的组间净提升。"
    else:
        higher = NAMES["graph" if chief["pp"] > 0 else "random"]
        evidence = f"本次开发产生的{higher}程序在该布局分布中有更高的配对成功率。"
    analysis = (f"终点仅图组成功 {chief['rescued']} 间，仅随机组成功 {chief['lost']} 间。"
                + evidence + "每组只有一次独立作者开发，区间只描述房间抽样的不确定性。")
    body = [START, "## 冻结留出的闭环结果", "", result, "", analysis, "",
            "| 组与轮次 | 开发前成功 | 选中候选开发成功 | 纳入后开发成功 | 纳入 | 留出成功 | 相对母程序救回 / 丢失 | 相对母程序净差，百分点 [95%] | 相对上轮救回 / 丢失 |",
            "|---|---:|---:|---:|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        for i, row in enumerate(summary["rounds"][arm], 1):
            a, b, c = row["selection"], row["versus_mother"], row["versus_previous"]
            body.append(f"| {NAMES[arm]} {i} | {a['dev_before']} | {a['candidate_dev'] if a['candidate_dev'] is not None else '未完成扫描'} | {a['dev_after']} | {'是' if a['admitted'] else '否'} | {b['second']} | {b['rescued']} / {b['lost']} | {pp_interval(b)} | {c['rescued']} / {c['lost']} |")
    body += ["", "| 轮次 | 图组成功 / 随机组成功 | 仅图组成功 / 仅随机组成功 | 图减随机，百分点 [95%] |",
             "|---:|---:|---:|---:|"]
    for r, stat in enumerate(summary["graph_minus_random"], 1):
        body.append(f"| {r} | {stat['second']} / {stat['first']} | {stat['rescued']} / {stat['lost']} | {pp_interval(stat)} |")
    body += ["", "## 初始直线输运的几何分层", "", "| 初始阻挡 | 房间数 | 母程序 | 随机第 3 轮 | 图选第 3 轮 | 仅图组成功 / 仅随机组成功 | 图减随机，百分点 [95%] |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for k, group in summary["by_stratum"].items():
        stat, wins = group["graph_minus_random"], group["successes"]
        body.append(f"| {STRATA[k]} | {group['n']} | {wins['mother']} | {wins['random_3']} | {wins['graph_3']} | {stat['rescued']} / {stat['lost']} | {pp_interval(stat)} |")
    body += ["", "## 作者预算和已记录开发执行", "",
             "| 组 | 发起 / 已返回 | 提示词字节 | 回应等待合计（秒） | 已存完整开发执行 | 实际 ticks |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm, usage in summary["usage"].items():
        body.append(f"| {NAMES[arm]} | {usage['requests']} / {usage['responses']} | {usage['prompt_bytes']:,} | {usage['response_wait_seconds']:.1f} | {usage['complete_logged_dev_runs']:,} | {usage['complete_logged_dev_ticks']:,} |")
    body += ["", "开发执行涵盖候选的小组试跑和选中候选的全开发池扫描。两组共用的母程序、最终留出评估与未完成的扫描另计。当前作者传输通道报告调用与等待时间，精确模型 token 消耗未提供。"]
    stats_link = os.path.relpath(out/"finished.json", DOC)
    body += ["", f"母程序与两组三轮检查点的实测成功并集为 {summary['witnessed_in_seven']}/{n}，其中每间都有一次满足同样单局预算的执行见证。尚无成功见证的房间仍计入分母。总体配对区间按四层的冻结比例，在每层独立对房间有放回重抽 10,000 次。分层区间和逐轮区间均为描述性结果，未作多重比较校正。各轮共用房间及累计开发过程。", "",
             f"[逐房间成败、时长及历史哈希](rooms.csv) · [完整配对统计]({stats_link})", END]
    readme = DOC / "README.md"
    intro = readme.read_text()
    if START in intro:
        if END not in intro:
            raise ValueError("existing report has an incomplete generated section")
        left, remainder = intro.split(START, 1)
        _, right = remainder.split(END, 1)
        text = left.rstrip() + "\n\n" + "\n".join(body) + "\n" + right.lstrip("\n")
    else:
        text = intro.rstrip() + "\n\n" + "\n".join(body) + "\n"
    write_atomic(DOC / "rooms.csv", table)
    write_atomic(readme, text)
    update_registry(result, analysis)
    write_atomic(out / "finished.json", json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
