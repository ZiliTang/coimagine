"""Reproduce three PPO rounds and decompose actual actor minibatch gradients by pure-expert outcomes."""
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import numpy as np
import torch
from tools import cap_first_choice_transfer as pure
from tools import cap_ungated_ppo as t

r=t.r
NAMES=("neither","B_only","A_only","both")
ROUNDS=(1,20,40)


class Meter:
    def __init__(self,rows,labels):
        self.labels=torch.tensor(np.concatenate([np.full(len(v[4]),labels[v[1]],int) for v in rows]))
        self.success=torch.tensor(np.concatenate([np.full(len(v[4]),v[2],int) for v in rows]))
        self.rows=[]
        for code in range(4):
            group=[v for v in rows if labels[v[1]]==code]
            self.rows.append(dict(episodes=len(group),wins=sum(v[2] for v in group),
                boundaries=sum(len(v[4]) for v in group),grad_dot=0.,step_dot=0.))

    def record_advantages(self,adv):
        for code,row in enumerate(self.rows):
            fail=(self.labels==code)&(self.success==0)
            row.update(failed_boundaries=int(fail.sum()),failed_adv_sum=float(adv[fail].sum()),
                       failed_abs_adv_sum=float(adv[fail].abs().sum()))

    def before(self,ix,loss,model):
        params=tuple(model.actor.parameters())
        self.params=torch.cat([p.detach().flatten() for p in params])
        self.grad=[]
        for code in range(4):
            ds=torch.autograd.grad(loss[self.labels[ix]==code].sum(),params,retain_graph=True)
            self.grad.append(torch.cat([v.detach().flatten() for v in ds]).double())
        total=sum(self.grad)
        for g,row in zip(self.grad,self.rows,strict=True):row["grad_dot"]+=float(torch.dot(g,total))

    def after(self,model):
        step=(self.params-torch.cat([p.detach().flatten() for p in model.actor.parameters()])).double()
        for g,row in zip(self.grad,self.rows,strict=True):row["step_dot"]+=float(torch.dot(g,step))


def fraction(rows,i,k):
    return rows[i][k]/sum(v[k] for v in rows)


def main():
    plan=r.e.read(t.OUT/"protocol.json")
    assert r.prior.initialize()==plan["profile"]
    rooms=r.prior.fixtures(r.prior.TRAIN_SEED)
    r.study.ROOMS.update(rooms)
    labels={}
    for k in rooms:
        a,b=[r.e.read(pure.OUT/"dev"/name/(k+".json"))["success"] for name in ("A","B")]
        labels[k]=2*a+b
    records=[]
    with ProcessPoolExecutor(max_workers=r.prior.first.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage in ROUNDS:
            keys=plan["schedule"][stage-1]
            jobs=[(s,stage,k,idx) for idx,k in enumerate(keys) for s in r.SEEDS]
            groups={s:[] for s in r.SEEDS}
            for row in pool.map(t.collect,jobs):groups[row[0]].append(row)
            for s in r.SEEDS:
                torch.manual_seed(s)
                model=t.Policy(r.study.cold.weights(r.prior.checkpoint(s,4)))
                opt=torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),dict(params=model.critic.parameters(),lr=3e-4)])
                if stage>1:
                    ck=torch.load(t.OUT/"state"/f"s{s}_r{stage-1}.pt",map_location="cpu",weights_only=True)
                    model.load_state_dict(ck["model"])
                    opt.load_state_dict(ck["optimizer"])
                meter=Meter(groups[s],labels)
                tr=t.fit(model,opt,groups[s],s,stage,meter=meter)
                expected=r.e.read(t.OUT/"metrics"/f"r{stage:02d}_s{s}.json")
                assert tr==expected,(s,stage,tr,expected)
                ck=torch.load(t.OUT/"state"/f"s{s}_r{stage}.pt",map_location="cpu",weights_only=True)
                assert all(torch.equal(v,ck["model"][k]) for k,v in model.state_dict().items())
                records.append(dict(seed=s,stage=stage,groups=meter.rows))
            print("verified round",stage,flush=True)
    groups=[{k:sum(row["groups"][i][k] for row in records) for k in records[0]["groups"][i]} for i in range(4)]
    r.e.save(t.DOC/"strata.json",dict(order=NAMES,rounds=ROUNDS,total=groups,by_seed_round=records))
    line=["# 按纯 A/B 成败分解 PPO 的实际更新","",
          "重放第 1、20、40 轮的三个种子，共 450 局。每轮的随机轨迹、GAE 和全部优化步骤重算，"
          "九份更新后的 actor/critic 参数均与原检查点逐值相同。分层由同一训练房间的原生 A、B 完整结局定义。"
          "该诊断在训练期间指定，只反映抽查的三个轮次。","",
          "| 训练房间类型 | PPO 成功/局数 | 分歧边界占比 | actor 梯度投影份额 | 沿实际 Adam 步的投影份额 | 失败轨迹平均原始 GAE |",
          "|---|---:|---:|---:|---:|---:|"]
    for i,name in enumerate(("双败","仅 B 成功","仅 A 成功","双胜")):
        v=groups[i]
        line.append(f'| {name} | {v["wins"]}/{v["episodes"]} | {fraction(groups,i,"boundaries"):.1%} | {fraction(groups,i,"grad_dot"):.1%} | {fraction(groups,i,"step_dot"):.1%} | {v["failed_adv_sum"]/v["failed_boundaries"]:.3f} |')
    line+=["","每个 minibatch 在相同归一化优势、概率比与 clip 下分解 actor 损失。第一种投影为所有步骤累加的各类梯度与总梯度内积，除以总梯度平方和。第二种将总梯度换成实际负向参数步长，分母为总梯度在这些步长上的总投影。总和均为 100%，方向冲突可产生负值。Adam 含跨步动量，沿实际参数步的投影并不把每个参数变化因果分摊给房间。critic 使用独立网络。失败标签只取该次 PPO 实际轨迹；即使专家双败，PPO 仍可能成功。",
           "","[逐种子和逐轮明细](strata.json)"]
    (t.DOC/"strata.md").write_text("\n".join(line)+"\n")
    print("\n".join(line[4:10]))


if __name__=="__main__":main()
