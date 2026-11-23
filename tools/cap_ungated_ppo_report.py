"""Learning curves and room-paired evaluation of the current A/B PPO."""
import csv
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
from tools import cap_first_choice_transfer as handoff
from tools import cap_ungated_ppo as t

r=t.r


def metrics():
    return [r.e.read(p) for p in sorted((t.OUT/"metrics").glob("*.json"))]


def plot(rows):
    if not rows:return
    t.DOC.mkdir(exist_ok=True,parents=True)
    fig,axes=plt.subplots(2,3,figsize=(13,6.8),constrained_layout=True)
    for ax,(field,title) in zip(axes.flat,(("success","Training success (50 episodes)"),("entropy","Action entropy (nats)"),
               ("critic_loss","Critic MSE"),("actor_loss","Clipped actor loss"),("kl","Approximate KL"),("clip_fraction","Clip fraction")),strict=True):
        for seed in r.SEEDS:
            records=sorted((q for q in rows if q["seed"]==seed),key=lambda q:q["stage"])
            ax.plot([q["stage"]*t.PER for q in records],
                    [q[field]/t.PER if field=="success" else q[field] for q in records],label=f"seed {seed}",lw=1.2)
        ax.axvline(200,ls=":",c="0.55",lw=1)
        ax.set(xlabel="Online episodes per seed",title=title,xlim=(0,t.ROUNDS*t.PER))
        ax.grid(alpha=.2)
    axes[0,0].set_ylim(0,1)
    axes[0,0].legend()
    path=t.DOC/"learning_curves.png"
    tmp=path.with_suffix(".tmp.png")
    fig.savefig(tmp,dpi=135)
    plt.close(fig)
    tmp.replace(path)


def watch():
    last=-1
    while True:
        data=metrics()
        if len(data)!=last:
            plot(data)
            last=len(data)
        if last==t.ROUNDS*len(r.SEEDS):return
        time.sleep(20)


def eval_handoff():
    plan=r.e.read(t.OUT/"protocol.json")
    assert (t.OUT/"locked.json").is_file() and (t.OUT/"timing"/"confirm.json").is_file()
    rooms=r.prior.fixtures(t.CONFIRM)
    assert sorted(rooms)==plan["confirm"] and r.prior.initialize()==plan["profile"]
    r.study.ROOMS.update(rooms)
    r.exp.OUT=t.OUT
    r.exp.WORKERS=r.prior.first.WORKERS
    jobs=[("confirm",k,"BA") for k in plan["confirm"] if not (t.OUT/"confirm"/"BA"/(k+".json")).exists()]
    r.exp.timed_pool("handoff",partial(handoff.episode,output=t.OUT),jobs)


def stats(a,b,mask,draws):
    diff=(b-a)*mask
    den=mask.sum(1)
    good=den[draws].sum(1)>0
    dst=100*diff.sum(1)[draws].sum(1)[good]/den[draws].sum(1)[good]
    return dict(n=int(den.sum()),reference=int((a*mask).sum()),ppo=int((b*mask).sum()),
                gained=int((diff==1).sum()),lost=int((diff==-1).sum()),
                delta_pp=100*diff.sum().item()/den.sum().item() if den.any() else None,
                ci=np.quantile(dst,[.025,.975]).tolist() if good.any() else None)


def panel_table(panel,keys):
    rows=[]
    for key in keys:
        root=t.OUT/panel if panel=="confirm" else r.OUT/"fresh"
        a,b=[r.e.read(root/name/(key+".json"))["success"] for name in ("A","B")]
        handoff_root=t.OUT/"confirm" if panel=="confirm" else handoff.OUT/"held"
        ba=r.e.read(handoff_root/"BA"/(key+".json"))["success"]
        for s in r.SEEDS:
            def get(root,name):
                return r.e.read(root/f"s{s}_{name}"/(key+".json"))["success"]
            results=dict(panel=panel,room=key,seed=s,A=a,B=b,BA=ba,
                         parent=get(root,"parent"),branch=(get(root,"branch") if panel=="confirm"
                         else get(t.baseline.OUT/"fresh","no_replay")),final=get(t.OUT/panel,"r40"))
            if panel=="diagnostic":results["early"]=get(t.OUT/panel,"r4")
            rows.append(results)
    def arr(name):
        return np.asarray([q[name] for q in rows],int).reshape(len(keys),len(r.SEEDS))
    draws=np.random.default_rng(71117).integers(len(keys),size=(10000,len(keys)))
    p=arr("parent")
    a,b=arr("A"),arr("B")
    masks=dict(all=np.ones_like(p,dtype=bool),parent_success=p==1,parent_fail=p==0,
               A_only=a>b,B_only=b>a,both=(a+b)==2,neither=(a+b)==0)
    totals={k:arr(k).sum(0).tolist() for k in ("parent","branch","final")}
    if panel=="diagnostic":totals["early"]=arr("early").sum(0).tolist()
    contrasts={}
    for endpoint in (("early","final") if panel=="diagnostic" else ("final",)):
        for ref in ("parent","branch","A","B","BA"):
            contrasts[f"{endpoint}_vs_{ref}"]={label:stats(arr(ref),arr(endpoint),m,draws) for label,m in masks.items()}
    return rows,dict(n=len(keys),totals=totals,A=int(a[:,0].sum()),B=int(b[:,0].sum()),
                     BA=int(arr("BA")[:,0].sum()),union=int((a|b)[:,0].sum()),contrasts=contrasts)


def main():
    protocol=r.e.read(t.OUT/"protocol.json")
    docs=t.DOC
    docs.mkdir(parents=True,exist_ok=True)
    all_rows=[]
    report={}
    for panel in ("diagnostic","confirm"):
        rows,report[panel]=panel_table(panel,protocol[panel])
        all_rows.extend(rows)
    with (docs/"rooms.csv").open("w") as f:
        wr=csv.DictWriter(f,fieldnames=list(all_rows[0]))
        wr.writeheader()
        wr.writerows(all_rows)
    data=metrics()
    assert len(data)==120
    plot(data)
    with (docs/"training.csv").open("w") as f:
        wr=csv.DictWriter(f,fieldnames=list(data[0]))
        wr.writeheader()
        wr.writerows(data)
    report["training"]=dict(episodes=sum(q["episodes"] for q in data),differing_boundaries=sum(q["n"] for q in data),
                            max_logp_discrepancy=max(q["old_logp_error"] for q in data),
                            ticks=sum(q["ticks"] for q in data),updates=sum(q["updates"] for q in data))
    schedule=protocol["schedule"]
    train_keys=sorted({key for group in schedule for key in group})
    train_types={}
    for key in train_keys:
        a,b=[r.e.read(handoff.OUT/"dev"/arm/(key+".json"))["success"] for arm in ("A","B")]
        train_types[key]=("neither","B_only","A_only","both")[2*a+b]
    def composition(keys):
        return {name:sum(train_types[k]==name for k in keys) for name in ("A_only","B_only","both","neither")}
    report["training"]["strata"]=dict(rooms=composition(train_keys),visits=composition([k for group in schedule for k in group]),
        cumulative_unique=[len({k for group in schedule[:stage] for k in group}) for stage in range(1,len(schedule)+1)])
    r.e.save(docs/"results.json",report)
    def ef(v):
        lo,hi=v["ci"]
        return f'{v["delta_pp"]:+.2f} [{lo:+.2f}, {hi:+.2f}]'
    cf=report["confirm"]
    primary=cf["contrasts"]["final_vs_parent"]["all"]
    branch=cf["contrasts"]["final_vs_branch"]["all"]
    result=f'新确认面板上，最终 PPO 三个种子分别成功 {cf["totals"]["final"]} /200。相对父模型的成功率变化为 {ef(primary)} 个百分点。'
    analysis=f'相对默认分叉更新 {ef(branch)} 个百分点，相对纯 B 为 {ef(cf["contrasts"]["final_vs_B"]["all"])} 个百分点。'
    lines=["# 当前无门控 A/B 的 PPO-Clip 对照","",result+analysis,"",
           "| 面板与模型 | seed17 | seed29 | seed43 | 平均成功率 |","|---|---:|---:|---:|---:|"]
    for panel,label in (("diagnostic","已查看"),("confirm","新确认")):
        st=report[panel]
        for arm,name in (("parent","父 actor"),("branch","默认分叉更新"),("early","PPO 200 局"),("final","PPO 2,000 局")):
            if arm in st["totals"]:
                vals=st["totals"][arm]
                lines.append(f'| {label} · {name} | {vals[0]} | {vals[1]} | {vals[2]} | {sum(vals)/600:.1%} |')
        lines.append(f'| {label} · 纯 A / 纯 B / 两者并集 | {st["A"]} / {st["B"]} / {st["union"]} | | | |')
    lines+=["","## 新确认面板的配对差异","","| 比较基线 | PPO 独赢 | 基线独赢 | PPO 减基线（百分点，95% 区间） |",
             "|---|---:|---:|---:|"]
    for ref,name in (("parent","父 actor"),("branch","默认分叉更新"),("A","纯 A"),("B","纯 B"),("BA","固定 B 五步后交给 A")):
        s=cf["contrasts"][f"final_vs_{ref}"]["all"]
        lines.append(f'| {name} | {s["gained"]} | {s["lost"]} | {ef(s)} |')
    lines+=["","| 房间条件 | 父 actor 成功 | 最终 PPO 成功 |","|---|---:|---:|"]
    for key,label in (("parent_success","父 actor 原成功"),("parent_fail","父 actor 原失败"),
                      ("A_only","仅 A 成功"),("B_only","仅 B 成功"),("both","两专家均成功"),("neither","两专家均失败")):
        s=cf["contrasts"]["final_vs_parent"][key]
        lines.append(f'| {label} | {s["reference"]}/{s["n"]} | {s["ppo"]}/{s["n"]} |')
    lines+=["","种子共享面板，计数为房间与种子组合。区间按房间配对重抽 10,000 次，每次保留三个种子，仅描述这三份模型的房间抽样差异。","",
            "## 训练与评估","","固定两位专家、72 束局部中心感知、公开历史和动作参数、64×64 actor，屏蔽九个教师语义量。"
            "每个种子从与默认无回放分叉方法相同的第四轮父 actor 开始，独立 critic 最后一层初始化为常数 0.5。"
            "从第一次不同提案起在 A/B 概率中采样，温度为 1；相同提案执行共同命令并使用规范编号 A。"
            "每个训练批次只用本轮新轨迹，终局成功奖励 1、其余为 0，耗尽任务预算按终止。"
            "GAE 只在可选的分歧边界推进，γ=1、λ=.95。PPO clip=.2，最多四个 epoch，平均近似 KL 超过 .015 提前结束本轮。"
            "minibatch 128、actor/critic 学习率 3e-5 / 3e-4，价值损失权重 .5，无熵奖励、分叉损失、全局软保持或成功回放。"
            "部署使用贪心选择。更新前逐边界复算行为策略对数概率。","",
            "每种子每轮 50 个房间。前四轮重复默认分叉研究中的同一批 50 间，检查点为 200 次采样；"
            "分叉对照每种子使用 50 个父轨迹加最多 150 次分叉重跑，模拟 ticks、优化器更新与数据结构各有差异。"
            "从第五轮起每轮在固定 200 间开发池中无放回抽 50 间，同房间每轮重新采样动作。"
            "训练抽样与两位专家的成败无关。200 个母房间中，A 独胜 10、B 独胜 25、双胜 77、双败 88 间；每种子 2,000 个预定训练 episode 对应 103、270、761、866 局。第 4、10、20、40 轮累计见过 50、167、199、200 个不同房间。较早的分层 rollout 刻意多抽独胜房间，后来的无教师分叉每轮引入 50 个新房间；二者与长程 PPO 的训练分布均有差异。","",
            "最终为预先指定的第 40 轮，无检查点或种子择优。另在训练中、查看 PPO 诊断或确认结果前登记上一轮的固定 B 五步转 A 作为次要参考。独立确认面板在训练前按场景内容哈希冻结，最终模型锁定后执行。"
            "已查看的诊断面板复用父模型、分叉模型、纯专家的既有同房间结果，两个 PPO 检查点均重新执行。","",
            "![PPO training curves](learning_curves.png)","","[逐房间数据](rooms.csv) · [配对结果](results.json) · [逐轮训练记录](training.csv) · [按四种房间类型分解梯度](strata.md)"]
    lines.extend(["","## 这些结果怎样约束能力上限","",
        "设 J 为同一目标房间分布下一次实际执行的成功率。相同动作和预算约束下，存在以下顺序。",
        "",r"\[J(\pi_{\mathrm{PPO,greedy}})\;\leq\;"
        r"\sup_{\theta} J(\pi_{\theta,\mathrm{greedy}})\;\leq\;"
        r"\sup_{\pi(\mathrm{public\ history})} J(\pi).\]","",
        "第一项能通过冻结面板估计。第一处差距涉及有限场景、轨迹覆盖和优化；当前策略很少访问的多步准备与接手序列很难给出学习信号。"
        "训练用随机动作而部署用贪心动作，训练目标和这个评测指标也存在差异。第二处差距涉及公开历史被压缩成固定输入以及网络表达能力。"
        "如果可部署策略在不可逆动作之前只能看到相同历史，即使有全历史也可能无法同时满足潜在状态之间的相反选择；特权状态规划的最优值还可更高。"
        "重复采样固定 200 间只能改善这些房间上的动作探索，不能消除对新房间的估计误差。"
        "房间分层是影响训练分布的一个选择，双败房间仍可能通过切换成功。","",
        "B 五步后交给 A 提供同一信息约束下可执行的固定参照。事后从多条完整路线挑选得到的覆盖只对这些候选路线的事后挑选有直接解释；"
        "它不能给任意公开历史路由提供已证上界。"
        " [策略梯度的最优性、近似误差与分布偏移](https://jmlr.org/papers/v22/19-736.html) · [PPO 原论文](https://arxiv.org/abs/1707.06347)"])
    (docs/"README.md").write_text("\n".join(lines)+"\n")
    registry=Path("docs/research/experiment_journal/entries.json")
    entries=json.loads(registry.read_text())
    next(e for e in entries if e["id"]=="ungated-ppo").update(result=result,analysis=analysis)
    registry.write_text(json.dumps(entries,ensure_ascii=False,indent=2)+"\n")
    r.e.save(t.OUT/"FINISHED.json",dict(primary=primary,report_sha=t.sha(docs/"results.json")))
    print(result,analysis)


if __name__=="__main__":
    if sys.argv[1:] == ["watch"]:watch()
    elif sys.argv[1:]==["handoff"]:eval_handoff()
    else:main()
