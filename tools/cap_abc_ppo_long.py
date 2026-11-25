"""Continue the frozen basic ABC PPO run while tracking fixed-panel outcomes."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import shutil

import numpy as np
import torch

from tools import cap_abc_semantic_ppo as base

PARENT = base.OUT
OUT = PARENT.parent/"cap_abc_ppo_long_20261123_01"
DOC = base.DOC.parent/"2026-11-23_abc_ppo_long"
ARM = "basic"
SEED = 17
START = 4
END = 40
CHECKS = (0,4,8,12,16,20,24,28,32,36,40)
ROOM_SEEDS = tuple(range(662330501,662330510))
CONFIRM_SEED = 662330541
s, read, save, ppo = base.s, base.read, base.save, base.ppo


def state_path(folder, round_number):
    return folder/"state"/f"{ARM}_s{SEED}_r{round_number}.pt"


def model_path(folder, round_number):
    return folder/"models"/f"{ARM}_s{SEED}_r{round_number}.npz"


def setup():
    parent = base.setup()
    learned = read(PARENT/"locked.json")
    parent_state = state_path(PARENT, START)
    parent_models = {r: model_path(PARENT,r) for r in (0,START)}
    for path in parent_models.values():
        assert ppo.sha(path)==learned[path.name]
    known = s.collect_identities() | set(parent["evaluation"])
    known.update(k for k in parent["demonstration"])
    known.update(s.phase.input_hash(base.ROOMS[k]) for k in parent["demonstration"])
    known.update(k for group in parent["schedule"] for k in group)
    ab = read(ppo.OUT/"protocol.json")
    known.update(ab["diagnostic"]+ab["confirm"])
    third = read(base.third.OUT/"protocol.json")
    known.update(third["held_content"])
    schedule=[]
    for seed in ROOM_SEEDS:
        rooms = base.get_rooms(seed)
        assert len(rooms)==200 and not known.intersection(rooms)
        known.update(rooms)
        base.ROOMS.update(rooms)
        order = np.random.default_rng(seed).permutation(sorted(rooms)).tolist()
        schedule.extend(order[i:i+50] for i in range(0,200,50))
    fresh = base.get_rooms(CONFIRM_SEED)
    assert len(fresh)==200 and not known.intersection(fresh)
    base.ROOMS.update(fresh)
    diagnostic = sorted(parent["evaluation"], key=lambda k:base.third.pair.key("abc-ppo-long/"+k))[:64]
    plan = dict(parent=ppo.sha(PARENT/"protocol.json"), parent_state=ppo.sha(parent_state),
                checkpoint={str(r):learned[p.name] for r,p in parent_models.items()},
                driver=ppo.sha(Path(__file__)), profile=parent["profile"],
                expert_sha=parent["expert_sha"], executor=parent["executor"], seed=SEED,
                arm=ARM, temperature=parent["temperature"], start=START, end=END,
                room_seeds=list(ROOM_SEEDS), schedule=schedule, confirm_seed=CONFIRM_SEED,
                confirm=sorted(fresh), diagnostic=diagnostic, diagnostic_stages=list(CHECKS),
                primary="R40 vs R4 on untouched 200 rooms, R0 and pure ABC are supplemental. No checkpoint selection.")
    assert len(schedule)==END-START==36
    manifest = OUT/"protocol.json"
    if manifest.exists():
        assert read(manifest)==plan
    else:
        save(manifest,plan)
    (OUT/"models").mkdir(exist_ok=True, parents=True)
    for r,path in parent_models.items():
        dest=model_path(OUT,r)
        if not dest.exists():
            shutil.copyfile(path,dest)
        assert ppo.sha(dest)==plan["checkpoint"][str(r)]
    base.OUT=OUT
    ppo.r.exp.OUT=OUT
    return plan


def restore(path, model, optimizer):
    ck = torch.load(path,map_location="cpu",weights_only=True)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])


def monitor_result(round_number):
    return OUT/"diagnostic"/f"r{round_number:02d}.json"


def evaluate_one(job):
    key, tag, round_number = job
    dest = OUT/"evaluation"/tag/(key+".json")
    if dest.exists():
        return read(dest)
    if round_number is None:
        result = base.episode(key,teacher=tag)[0]
    else:
        result = base.episode(key,actor=base.load_actor(ARM,SEED,round_number),greedy=True)[0]
    row = {k:result[k] for k in ("success","history","ticks")}
    save(dest,row)
    return row


def diag(pool, plan, round_number):
    rows = []
    if round_number in (0,START):
        for key in plan["diagnostic"]:
            rows.append(read(PARENT/"evaluation"/f"{ARM}_s{SEED}_r{round_number}"/(key+".json")))
    else:
        jobs=[(key,f"diag_r{round_number:02d}",round_number) for key in plan["diagnostic"]]
        rows = list(pool.map(evaluate_one,jobs))
    a = np.asarray([r["success"] for r in rows],int)
    if round_number==0:
        ref=a
    else:
        ref=np.asarray(read(monitor_result(0))["outcome"],int)
    save(monitor_result(round_number),dict(round=round_number, success=int(a.sum()),n=len(a),
                                          gained=int(sum(a>ref)),lost=int(sum(a<ref)),
                                          outcome=a.tolist()))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows=[read(PARENT/"metrics"/f"{ARM}_s{SEED}_r{r}.json") for r in range(1,START+1)]
    rows.extend(read(p) for p in sorted((OUT/"metrics").glob("r*.json")))
    if not rows:
        return
    diag_rows=[read(p) for p in sorted((OUT/"diagnostic").glob("r*.json"))]
    ends=[50*r["stage"] for r in rows]
    fig, axes = plt.subplots(2,2,figsize=(10,6.2),layout="constrained")
    a, b, c, d = axes.ravel()
    p = [r["success"]/r["episodes"] for r in rows]
    a.plot(ends,p,"o-",ms=2.5,lw=1,color="#6599be",label="50-episode training batches")
    if len(p)>=5:
        a.plot(ends[4:],np.convolve(p,np.ones(5)/5,"valid"),lw=1.7,color="#1c608a",
               label="5-batch rolling mean")
    a.set(ylabel="Success fraction",title="Stochastic training success",ylim=(0,1))
    a.legend(loc="upper right",fontsize=8)
    if diag_rows:
        b.plot([50*r["round"] for r in diag_rows],[r["success"]/r["n"] for r in diag_rows],
               "o-",color="#087f71",lw=1.7,ms=4)
    b.set(ylabel="Success fraction",title="Greedy success, fixed 64 rooms",ylim=(0,1))
    c.plot(ends,[r["entropy"] for r in rows],color="#6e48b4",lw=1.7)
    c.set(ylabel="Mean entropy (nats)",title="Policy entropy on distinct proposals")
    d.plot(ends,[r["critic_loss"] for r in rows],color="#af6842",label="Critic MSE")
    d.plot(ends,[r["actor_loss"] for r in rows],color="#315cbb",label="Actor loss")
    d.set(ylabel="Loss",title="PPO losses")
    d.legend(loc="upper right",fontsize=8)
    for ax in axes.ravel():
        ax.axvline(START*50,color="#888888",ls=":",lw=1)
        ax.set_xlabel("Cumulative PPO episodes")
        ax.set_xlim(0,END*50)
        ax.grid(alpha=.22)
    fig.suptitle("ABC routing PPO  |  basic features  |  seed 17",fontsize=13)
    DOC.mkdir(exist_ok=True,parents=True)
    temp=DOC/"learning_curves.pending.png"
    fig.savefig(temp,dpi=155)
    temp.replace(DOC/"learning_curves.png")
    plt.close(fig)


def run(plan):
    torch.manual_seed(SEED)
    model = base.Policy()
    optimizer = torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                                   dict(params=model.critic.parameters(),lr=3e-4)])
    restore(state_path(PARENT,START),model,optimizer)
    actor = base.load_actor(ARM,SEED,START)[1]
    assert all(np.array_equal(v,w.detach().numpy().T) for v,w in zip(actor,model.actor.parameters(),strict=True))
    assert all(optimizer.state[p]["step"].item()>0 for p in model.parameters())
    state_path(OUT,START).parent.mkdir(exist_ok=True,parents=True)
    with ProcessPoolExecutor(base.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for r in (0, START):
            if not monitor_result(r).exists():
                diag(pool,plan,r)
        plot()
        for stage, keys in enumerate(plan["schedule"],START+1):
            state = state_path(OUT,stage)
            if state.is_file():
                meta = read(OUT/"metrics"/f"r{stage:02d}.json")
                assert ppo.sha(model_path(OUT,stage))==meta["actor_sha"]
                assert ppo.sha(state)==meta["state_sha"]
                restore(state,model,optimizer)
            else:
                jobs = [(ARM,SEED,stage,key,i) for i,key in enumerate(keys)]
                rows=list(pool.map(base.rollout,jobs))
                assert [r[1] for r in rows]==keys
                stats=ppo.fit(model,optimizer,rows,SEED,stage)
                base.export(ARM,SEED,stage,model)
                ppo.legacy.save_checkpoint(state,dict(model=model.state_dict(),optimizer=optimizer.state_dict()))
                save(OUT/"metrics"/f"r{stage:02d}.json",dict(stats,actor_sha=ppo.sha(model_path(OUT,stage)),
                                                            state_sha=ppo.sha(state)))
                print(f"{stage*50} episodes  train {stats['success']}/50 entropy {stats['entropy']:.3f}",flush=True)
            if stage in plan["diagnostic_stages"] and not monitor_result(stage).exists():
                diag(pool,plan,stage)
            plot()
    lock = {str(r):ppo.sha(model_path(OUT,r)) for r in plan["diagnostic_stages"]}
    save(OUT/"locked.json", lock)


def confirm(plan):
    for r,digest in read(OUT/"locked.json").items():
        assert ppo.sha(model_path(OUT,int(r)))==digest
    variants=[(a,None) for a in base.EXPERTS]+[(f"r{r}",r) for r in (0,START,END)]
    jobs = [(key,tag,r) for key in plan["confirm"] for tag,r in variants
            if not (OUT/"evaluation"/tag/(key+".json")).exists()]
    ppo.r.exp.timed_pool("confirm",evaluate_one,jobs)


if __name__=="__main__":
    torch.set_num_threads(1)
    plan=setup()
    if not (OUT/"locked.json").exists():
        run(plan)
    confirm(plan)
