"""Report marginal coverage and same-prefix recovery of the frozen third CAP."""
import csv
import json

from tools import cap_authored_third as study
from tools.cap_option_handoff_opportunity_report import paired

ARMS = ("A", "B", "C", "AC", "BC")


def main():
    study.DOC.mkdir(exist_ok=True, parents=True)
    plan = study.read(study.OUT / "protocol.json")
    member = study.read(study.OUT / "member.json")
    rows = []
    for key in plan["held_content"]:
        games = study.read(study.OUT/"held"/(key+".json"))
        rows.append(dict(room=key, **{a: games[a]["result"]["success"] for a in ARMS},
                         **{a+"_has_hit": bool(games[a]["hit"]) for a in study.ANCHORS}))
    assert len(rows) == 200
    with (study.DOC / "rooms.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {a: sum(r[a] for r in rows) for a in ARMS}
    ab = sum(bool(r["A"] or r["B"]) for r in rows)
    covered = [r for r in rows if not (r["A"] or r["B"])]
    unique = {a: sum(r[a] for r in covered) for a in ("C", "AC", "BC")}
    all_new = sum(bool(r["C"] or r["AC"] or r["BC"]) for r in covered)
    paired_switches = {}
    for a in study.ANCHORS:
        active = [r for r in rows if r[a+"_has_hit"]]
        paired_switches[a+"C"] = dict(
            all=paired([r[a+"C"] for r in rows], [r[a] for r in rows]),
            eligible=paired([r[a+"C"] for r in active], [r[a] for r in active]),
            triggered=len(active),
        )
    result = dict(counts=counts, union_AB=ab, union_ABC=ab+unique["C"],
                  combined_observed_union=ab+all_new, joint_failure_200=len(covered),
                  joint_failure_rescues=unique, paired=paired_switches)
    study.save(study.DOC/"results.json", result)
    probes = [study.read(p) for p in sorted(study.OUT.glob("probe[1-3].json"))]
    notes = study.WORK/"author_notes.md"
    if notes.exists():
        (study.DOC/"author_notes.md").write_text(notes.read_text())
    lines = ["# 利用 A、B 的共同失败开发第三位专家 C", "",
             f"新 200 个同房间对照中，A、B、C 各自执行成功 {counts['A']}、{counts['B']}、{counts['C']} 间。"
             f"纯专家成功并集从 A/B 的 **{ab}/200** 到 A/B/C 的 **{ab+unique['C']}/200**。"
             f"C 从初态净新增 **{unique['C']}** 间，允许在既有 A/B 前缀接手又提供可单独计量的机会。", "",
             "| 路线 | 全部成功 / 200 | A/B 纯执行都失败中的成功 |",
             "|---|---:|---:|"]
    names = {"A":"全 A", "B":"全 B", "C":"全 C", "AC":"A 首次触发后交给 C", "BC":"B 首次触发后交给 C"}
    for a in ARMS:
        lines.append(f"| {names[a]} | {counts[a]} | {unique.get(a,0)} / {len(covered)} |")
    lines += ["",f"把两条固定接手与三条纯路线一起事后去重，覆盖 {ab+all_new}/200。每条路线都有真实执行，"
              f"其中 {all_new-unique['C']} 个新增房间只靠固定接手路线才解出。实际部署的路由仍需在线决定何时接手。","",
              "| 同一前缀对照 | 触发房间 | 救回 | 丢失 | 全部房间百分点差、配对 95% 区间 |",
              "|---|---:|---:|---:|---:|"]
    for a, values in paired_switches.items():
        q = values["all"]
        lo, hi = [100*x for x in q["ci"]]
        lines.append(f"| {a[0]}→C 对全 {a[0]} | {values['triggered']} | {q['plus']} | {q['minus']} | {(q['first']-q['second'])/2:+.2f} [{lo:+.2f}, {hi:+.2f}] |")
    lines += ["", "## 开发和选择", "",
              f"冻结 A、B 后扫描原有 200 间开发房间，得到 {plan['jointly_failed']} 间双败。"
              f"触发家族为 `{plan['family']}`。固定 24 个房间中的 16 间为双败，另 8 间会由首发专家自行恢复。"
              "每层按 A、B 前缀等额取样。作者使用 gpt-6.2-sol，可读公开前缀和两份代码。"
              f"共完成 {sum(q['cases'] for q in probes)} 局试跑，{sum(q['ticks'] for q in probes):,} ticks，包括重放前缀。"
              f"冻结第 {member['revision']} 版。", "",
              "| 版本 | 救回 / 16 | 原成功丢失 / 8 |",
              "|---|---:|---:|"]
    for q in probes:
        lines.append(f"| {q['revision']} | {q['rescued']} | {q['lost']} |")
    lines += ["",f"作者选版理由。{member['reason']}","",
              "独立面板在开发前从自然房间生成器冻结，按内容哈希核对已知面板。A/B/C 运行均使用固定 72 束和局部中心估计，"
              "接手时重放真实公开历史，保持原剩余预算。区间按房间配对重抽。","",
              "[协议](plan.md) · [逐房间结果](rooms.csv) · [配对指标](results.json)"]
    if notes.exists():
        lines[-1] += " · [作者开发笔记](author_notes.md)"
    (study.DOC/"README.md").write_text("\n".join(lines)+"\n")

    registry = study.DOC.parent.parent / "experiment_journal/entries.json"
    entries = study.read(registry)
    item = next(x for x in entries if x["id"] == "authored-third")
    item["result"] = f"新 200 间 A/B/C 各为 {counts['A']} / {counts['B']} / {counts['C']}，纯策略并集 {ab}→{ab+unique['C']}。"
    item["analysis"] = f"C 从初态解出 {unique['C']} 间 A/B 双败房间，A→C 和 B→C 分别解出 {unique['AC']} 和 {unique['BC']} 间。"
    registry.write_text(json.dumps(entries, ensure_ascii=False, indent=2)+"\n")


if __name__ == "__main__":
    main()
