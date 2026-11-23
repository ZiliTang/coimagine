"""Report the paired endpoints and label changes from the prefix discovery study."""
from collections import Counter
import csv
import json
from pathlib import Path
import numpy as np
from tools import cap_fork_discovery as study
from tools.cap_option_handoff_opportunity_report import paired

out, doc, e = study.OUT, study.DOC, study.e
simple = study.SIMPLE
full = "semantic"
masked = "history"


def fork_stats(collector, label_source, selection):
    by_room = {}
    for room,row in selection.items():
        rr=[]
        for i in row[collector]:
            v=e.read(out/label_source/"forks"/f"{room}_{i}.json")
            rr.append(tuple(v["outcomes"]))
        by_room[room]=rr
    counts=Counter(pair for rr in by_room.values() for pair in rr)
    return dict(pairs=sum(counts.values()),A_only=counts[1,0],B_only=counts[0,1],both=counts[1,1],neither=counts[0,0],
                informative_rooms=sum(any(a!=b for a,b in rr) for rr in by_room.values()))


def main():
    ids=e.read(out/"protocol.json")
    rooms=ids["held"]
    selection=e.read(out/"selections.json")
    direct=e.read(out/"rule_selection.json")
    extra=e.read(out/"rule_done.json")
    a_match="matched_A" if "matched_A/history" in extra["models"] else simple
    direct_a=f"{a_match}/history"
    direct_t="rule_teacher/history"
    model_names={
        "A":"单 CAP A","B":"单 CAP B","rule0":"直接执行原教师","distilled0":"直接执行蒸馏 actor",
        "expert_A/history":"A 采样，蒸馏续跑，屏蔽语义","expert_A/semantic":"A 采样，蒸馏续跑，完整输入",
        "distilled/history":"蒸馏采样，蒸馏续跑，屏蔽语义","distilled/semantic":"蒸馏采样，蒸馏续跑，完整输入",
        simple+"/history":"A 采样，A 续跑，屏蔽语义", direct_t:"原教师采样，A 续跑，屏蔽语义"}
    model_names.update(self0="从随机权重直接执行",self4="冷启动 actor 自采自续四轮，屏蔽语义")
    assert (out/"self"/"done.json").is_file()
    if a_match=="matched_A":
        model_names[direct_a]="与原教师逐房间匹配分叉数的 A 采样，A 续跑，屏蔽语义"
    values={m:np.asarray([e.read(out/"test"/m/(k+".json"))["success"] for k in rooms],int) for m in model_names}
    with (doc/"rooms.csv").open("w") as f:
        w=csv.writer(f)
        w.writerow(["room",*model_names])
        w.writerows([k,*[values[m][i] for m in model_names]] for i,k in enumerate(rooms))
    by_pair={f"{a} vs {b}":paired(x,values[b]) for a,x in values.items() for b in values if b!=a}
    sources={k:fork_stats(c,s,g) for k,c,s,g in (
      ("A, distilled","expert_A","expert_A",selection),("distilled, distilled","distilled","distilled",selection),
      ("A, A","expert_A",simple,selection),("matched A, A","expert_A",simple,direct),
      ("teacher, A","rule_teacher","rule_teacher",direct))}
    self_results=[e.read(f) for f in (out/"self"/"forks").glob("*.json")]
    self_by_pair=Counter(tuple(r["outcomes"]) for r in self_results)
    informative_rooms={f.name.split("_")[0] for f in (out/"self"/"forks").glob("*.json") if len(set(e.read(f)["outcomes"]))==2}
    sources["self, self"]=dict(pairs=len(self_results),A_only=self_by_pair[1,0],B_only=self_by_pair[0,1],
            both=self_by_pair[1,1],neither=self_by_pair[0,0],informative_rooms=len(informative_rooms))
    changes=Counter()
    for k,row in selection.items():
        for i in row["expert_A"]:
            vals=[]
            for source in [simple,"expert_A"]:
                r=e.read(out/source/"forks"/f"{k}_{i}.json")
                a,b=r["outcomes"]
                vals.append(b-a)
            changes[tuple(vals)]+=1
    overlap=Counter()
    for k,row in direct.items():
        by_prefix=[]
        for collector,source in [("expert_A",simple),("rule_teacher","rule_teacher")]:
            rr=[e.read(out/source/"forks"/f"{k}_{i}.json") for i in row[collector]]
            by_prefix.append({r["prefix"]:tuple(r["outcomes"]) for r in rr})
        common=by_prefix[0].keys() & by_prefix[1].keys()
        assert all(by_prefix[0][h]==by_prefix[1][h] for h in common)
        overlap["shared"]+=len(common)
        overlap["shared_informative"]+=sum(by_prefix[0][h][0]!=by_prefix[0][h][1] for h in common)
        for arm,rr in zip(("A","teacher"),by_prefix):
            overlap[arm+"_only_informative"]+=sum(a!=b for h,(a,b) in rr.items() if h not in common)
    groups=2*values["A"]+values["B"]
    by_group={m:[int(x[groups==i].sum()) for i in (2,1,3,0)] for m,x in values.items()}
    group_n=[int(sum(groups==i)) for i in (2,1,3,0)]
    totals={m:int(x.sum()) for m,x in values.items()}
    interaction=values["expert_A/semantic"]-values["expert_A/history"]-values["distilled/semantic"]+values["distilled/history"]
    rng=np.random.default_rng(17)
    ci_interaction=np.percentile(rng.choice(interaction,(10000,len(rooms))).mean(1)*100,[2.5,97.5]).tolist()
    e.save(doc/"results.json",dict(success=totals,pairs=by_pair,sources=sources,strata=by_group,
                 semantic_interaction_pp=[float(interaction.mean()*100),*ci_interaction],identical_sampled_prefixes=dict(overlap),n_Aonly_Bonly_both_neither=group_n,A_suffix_to_distilled_suffix_label_changes={str(k):v for k,v in changes.items()}))
    def contrast(a,b):
        r=by_pair[f"{a} vs {b}"]
        lo,hi=100*np.array(r["ci"])
        return f'{r["plus"]} / {r["minus"]}，净差 {100*(r["first"]-r["second"])/len(rooms):+.1f} 个百分点，95% 区间 [{lo:+.1f}, {hi:+.1f}]'
    lines=["# 谁发现分叉点，谁负责续跑","",
           "200 个新的开发房间全部用于采样，另冻结 200 个自然房间。按去除 scene_id 后的房间内容哈希查重。所有冷启动组从同一份 seed17 随机权重开始，四批各 50 间，每批 300 次更新。屏蔽组在训练和部署均将 9 个教师启发的显式语义输入置零，保留公开几何、两位专家动作及短历史。所有学习路由器沿用相同停滞门控。","",
           "## 直接使用规则教师发现分叉点","",
           f'在同样的 {extra["paired_forks"]} 对分叉预算下，两组首步之后都由 A 执行到底。原教师采样的学生成功 {totals[direct_t]}/200，A 采样的学生成功 {totals[direct_a]}/200。两者逐房间独胜 / 独败为 {contrast(direct_t,direct_a)}。',
           "", "## 同场闭环","", "| 执行策略 | 成功 / 200 | 仅 A | 仅 B | 两者成功 | 两者失败 |",
           "|---|---:|---:|---:|---:|---:|",
           "| 各层房间数 | 200 | "+" | ".join(map(str,group_n))+" |"]
    for m,name in model_names.items():
        lines.append(f"| {name} | {totals[m]} | "+" | ".join(map(str,by_group[m]))+" |")
    lines+=["","## 采样信息量","","同前缀分别执行 A 或 B 的第一条命令，然后使用指定的同一续跑者。只有两条后缀终局成败不同才产生硬偏好。每房间至多三个分叉，按可路由边界的前中后三段抽取。对原教师重新逐房间匹配双方可用的分叉数，表中的匹配 A 行使用该子集。","",
            "| 采样者，续跑者 | 动作对 | 仅 A 首步胜 | 仅 B 首步胜 | 均成功 | 均失败 | 提供偏好的房间 |","|---|---:|---:|---:|---:|---:|---:|"]
    for name,stats in sources.items():
        lines.append("| "+name+" | "+" | ".join(str(stats[k]) for k in ("pairs","A_only","B_only","both","neither","informative_rooms"))+" |")
    lines+=["",f'匹配采样中两组抽中了 {overlap["shared"]} 对相同的公开前缀（含路由历史及提案），其中 {overlap["shared_informative"]} 对有成败偏好。其余采样前缀中，原教师组得到 {overlap["teacher_only_informative"]} 个有效偏好，A 组 {overlap["A_only_informative"]} 个。未抽中同一前缀也可能来自同一路线上的不同抽样时刻。']
    lines+=["","## 配对差异","","下面各项按同房间统计前者独胜 / 独败，区间为 4,000 次房间配对重采样，均为单个比较的区间。",""]
    for a,b in [("self4","self0"),("self4","A"),("self4","B"),("self4",simple+"/history"),(direct_t,direct_a),(direct_t,"rule0"),(simple+"/history","A"),(simple+"/history","B"),("expert_A/history",simple+"/history"),
                ("distilled/history","expert_A/history"),("distilled/semantic","expert_A/semantic"),
                ("expert_A/semantic","expert_A/history"),("distilled/semantic","distilled/history"),("rule0","distilled0")]:
        lines.append(f"- {model_names[a]} 对 {model_names[b]}，{contrast(a,b)}。")
    lines+=["",f"显式语义在 A 采样和蒸馏采样两组中的收益差为 {100*interaction.mean():+.1f} 个百分点，逐房间 bootstrap 区间 [{ci_interaction[0]:+.1f}, {ci_interaction[1]:+.1f}]。这是单种子附加分析。"]
    lines+=["",f"在同一批 A 采样前缀上把后续控制者从 A 改为蒸馏 actor，原有偏好变平局 {changes[-1,0]+changes[1,0]} 对，平局变偏好 {changes[0,-1]+changes[0,1]} 对，偏好直接反向 {changes[-1,1]+changes[1,-1]} 对。",
            "", "固定采样器的直接对照逐房间匹配分叉数。自采自续使用相同房间和每房间三处的上限，实际得到 490 对，固定采样器各 492 对。各组实际物理 ticks 也可因终止时间而不同。采样器也决定用于保持项的边界分布。单个训练种子无法刻画初始化方差；本次屏蔽仅检验九个显式派生特征，动作与历史仍然携带可重建这些特征的信息。","",
            "[逐房间结果](rooms.csv) · [配对统计与标签变化](results.json)"]
    lines+=["","## 完全跳过教师路由的自采自续","",
            "从 seed17 随机初始化开始，按同一顺序每轮访问 50 个新开发房间，用当轮冻结的自身 actor 发现分叉、比较两条命令后各自交还自身的完整后缀。每房间至多三处，平局仅保留边界。冻结的专家 A、B 和公开停滞门控仍然提供基础能力。","",
            f'在 200 个新自然房间中，随机起点成功 {totals["self0"]}，第四轮 {totals["self4"]}，单 CAP A、B 分别为 {totals["A"]}、{totals["B"]}。相对 A 的独胜 / 独败为 {contrast("self4","A")}；相对 B 为 {contrast("self4","B")}。',
            "", "| 轮 | 房间 | 采样记录与 A 完全匹配 | 分叉 | 硬偏好 | 当轮拟合正确 |","|---|---:|---:|---:|---:|---:|"]
    for i in range(1,5):
        r=e.read(out/"self"/f"fit{i}.json")
        keys=ids["train"][(i-1)*50:i*50]
        exact=sum(e.read(out/"self"/"parents"/(k+".json"))==e.read(out/"expert_A"/"parents"/(k+".json")) for k in keys)
        lines.append(f'| {i} | {r["rooms"]} | {exact} | {r["forks"]} | {r["informative"]} | {r["fit_correct"]} |')
    (doc/"README.md").write_text("\n".join(lines)+"\n")
    entry_path=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(entry_path.read_text())
    entry=next(v for v in entries if v["id"]=="fork-discovery")
    entry["result"]=f'同为 A 续跑、无显式语义，规则教师采样 {totals[direct_t]}/200，匹配 A 采样 {totals[direct_a]}/200。原教师直接执行 {totals["rule0"]}/200。'
    entry["result"]+=f' 自采自续冷启动 actor {totals["self0"]}→{totals["self4"]}，单 CAP A/B 为 {totals["A"]}/{totals["B"]}。'
    entry["analysis"]=f'原教师采样相对 A 采样的配对独胜 / 独败为 {contrast(direct_t,direct_a)}。'
    entry["progress"][-2]["total"]=len(model_names)*200
    entry_path.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    e.save(out/"FINISHED.json",dict(success=totals,source=sources,direct_comparison=by_pair[f"{direct_t} vs {direct_a}"]))


if __name__=="__main__":
    main()
