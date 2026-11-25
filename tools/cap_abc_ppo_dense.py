"""Matched continuation with positive option-boundary progress rewards."""
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from tools import cap_abc_ppo_distance as terminal

binary, base, ppo = terminal.binary, terminal.base, terminal.ppo
CONTROL, PREVIOUS = terminal.CONTROL, terminal.OUT
OUT = CONTROL.parent/"cap_abc_ppo_dense_20261124_01"
DOC = terminal.DOC.parent/"2026-11-24_abc_ppo_dense"
ARM, SEED, START, END = terminal.ARM, terminal.SEED, terminal.START, terminal.END
PER_METRE = .10
read, save = terminal.read, terminal.save
original_tick = base.s.tick_action
original_inputs = base.candidate_inputs
trace = None


def capture_option(world, action, controller, update, counts):
    if trace is not None and not trace["distance"]:
        goal = base.s.observe(world)["goal"]
        trace["body"] = world.env.env.backend.bodies[goal["target_id"]]
        trace["target"] = goal["xy"]
        trace["distance"].append(math.dist(trace["body"].position, trace["target"]))
    original_tick(world, action, controller, update, counts)
    if trace is not None:
        trace["distance"].append(math.dist(trace["body"].position, trace["target"]))


def capture_choice(frame, history, obs, proposals):
    if trace is not None:
        trace["choices"].append(max(0,len(trace["distance"])-1))
    return original_inputs(frame, history, obs, proposals)


def rewards_for_choices(distance, choices, success, *, rate=PER_METRE):
    distance = np.asarray(distance,float)
    choices = np.asarray(choices,int)
    assert distance.ndim==1 and np.isfinite(distance).all()
    option_count = len(distance)-1
    assert all(0<=i<option_count for i in choices) and np.all(np.diff(choices)>0)
    approaching = np.maximum(0., -np.diff(distance))
    cumulative = np.concatenate(([0.],np.cumsum(approaching)*rate))
    reward = np.diff(cumulative[np.append(choices,option_count)]).astype(np.float32)
    if len(reward):
        reward[-1] += success
    return reward, float(cumulative[-1]), float(approaching.sum())


def setup():
    control = binary.setup()
    assert read(PREVIOUS/"progress/same_panel.json")["done"]==400
    plan = dict(control=ppo.sha(CONTROL/"protocol.json"), terminal=ppo.sha(PREVIOUS/"protocol.json"),
                initial_state=control["parent_state"], driver=ppo.sha(Path(__file__)),
                reward_per_metre=PER_METRE,
                reward="Positive net approach after each actual option; terminal success +1",
                treatment="dense reward from transition between option boundaries, success fitness held constant",
                checkpoint=END, seed=SEED, episode_count=1800, temperature=control["temperature"],
                primary="Dense minus binary on same viewed 200 rooms; terminal distance arm is secondary.")
    path = OUT/"protocol.json"
    if path.exists():
        assert read(path)==plan
    else:
        save(path,plan)
    base.load_actor(ARM,SEED,START)
    base.OUT=terminal.OUT=OUT
    ppo.r.exp.OUT=OUT
    base.s.tick_action=capture_option
    base.s.output=terminal.result_with_distance
    base.candidate_inputs=capture_choice
    return control


def rollout(job):
    global trace
    r, key, idx = job
    trace = dict(distance=[],choices=[])
    rng = np.random.default_rng([SEED,r,idx,1549])
    try:
        result,x,a,logp=base.episode(key,actor=base.load_actor(ARM,SEED,r-1),rng=rng)
        assert len(trace["choices"])==len(x)==len(a)
        assert math.isclose(trace["distance"][-1],result["distance_m"],abs_tol=1e-12)
        rewards,bonus,progress=rewards_for_choices(trace["distance"],trace["choices"],result["success"])
    finally:
        trace=None
    return (key,result["success"],result["ticks"],x,a,logp,result["distance_m"],
            rewards,bonus,progress)


def fit(model,opt,rows,stage):
    x,a,logp=(torch.from_numpy(np.concatenate([r[i] for r in rows])) for i in (3,4,5))
    with torch.no_grad():
        dist,values,_=model(x)
        entropy=dist.entropy().mean().item()
    offset=0
    advantages=[]
    for row in rows:
        reward=row[7]
        if len(reward):
            value=values[offset:offset+len(reward)].numpy()
            advantages.append(ppo.legacy.prior.advantages(reward,value))
            offset+=len(reward)
    assert offset==len(values)
    advantage=torch.from_numpy(np.concatenate(advantages))
    batch=dict(x=x,chosen=a,old_logp=logp,adv=advantage,returns=advantage+values)
    stats=ppo.legacy.update_batch(model,batch,opt,np.random.default_rng([SEED,stage,172]))
    fail=[r[6] for r in rows if not r[1]]
    return dict(stats,seed=SEED,stage=stage,episodes=len(rows),success=sum(r[1] for r in rows),
                ticks=sum(r[2] for r in rows),n=len(x),entropy=entropy,
                mean_dense_reward=float(np.mean([r[8] for r in rows])),
                mean_progress_m=float(np.mean([r[9] for r in rows])),
                bonus_at_least_one=sum(r[8]>=1 for r in rows),
                max_dense_reward=max(r[8] for r in rows),
                fail_distance_median=float(np.median(fail)) if fail else None)


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial=[read(binary.PARENT/"metrics"/f"basic_s17_r{r}.json") for r in range(1,START+1)]
    diag_initial=[read(CONTROL/"diagnostic"/f"r{r:02d}.json") for r in (0,START)]
    fig,axs=plt.subplots(2,2,figsize=(10,6.1),layout="constrained")
    train,diag,entropy,distance=axs.flat
    values=[]
    for folder,label,color in ((CONTROL,"Binary","#445c90"),(PREVIOUS,"Terminal distance","#8d5d9d"),
                               (OUT,"Dense progress","#087e69")):
        current=[read(f) for f in sorted((folder/"metrics").glob("r*.json"))]
        rows=initial+current
        if len(rows)>=5:
            span=np.convolve([q["success"]/q["episodes"] for q in rows],np.ones(5)/5,"valid")
            train.plot([50*q["stage"] for q in rows[4:]],span,color=color,label=label)
        entropy.plot([50*q["stage"] for q in rows],[q["entropy"] for q in rows],color=color)
        ds=diag_initial+[read(f) for f in sorted((folder/"diagnostic").glob("r*.json")) if int(f.stem[1:])>START]
        vals=[q["success"] for q in ds]
        values.extend(vals)
        diag.plot([50*q["round"] for q in ds],vals,color=color,marker=".",label=label)
        if folder!=CONTROL:
            distance.plot([50*q["stage"] for q in current],
                          [np.nan if q["fail_distance_median"] is None else q["fail_distance_median"] for q in current],
                          color=color,alpha=.8, label=label)
    train.set(title="Training success, rolling 250 episodes",ylabel="Binary success fraction")
    diag.set(title="Greedy on fixed 64 rooms",ylabel="Rooms solved",ylim=(min(values)-2,max(values)+2))
    entropy.set(title="Policy entropy",ylabel="Mean entropy (nats)")
    distance.set(title="Terminal distance among failed training runs",ylabel="Median distance (m)")
    for ax in axs.flat:
        ax.axvline(200,lw=1,color="#777777",ls=":")
        ax.set(xlabel="Cumulative PPO episodes",xlim=(0,2000))
        ax.grid(alpha=.2)
    for ax in (train,diag,distance):
        ax.legend(fontsize=8)
    temp=DOC/"learning_curves.pending.png"
    fig.savefig(temp,dpi=155)
    temp.replace(DOC/"learning_curves.png")
    plt.close(fig)


def verify_reward_path(rows,stage):
    terminal.verify_control_update([r[:7] for r in rows],stage)
    zeros = []
    for row in rows:
        y=np.zeros(len(row[7]),np.float32)
        if len(y):
            y[-1]=row[1]
        zeros.append((*row[:7],y,*row[8:]))
    model,opt=terminal.model_and_optimizer()
    fit(model,opt,zeros,stage)
    control=torch.load(binary.state_path(CONTROL,stage),weights_only=True,map_location="cpu")
    assert all(torch.equal(v,control["model"][k]) for k,v in model.state_dict().items())
    save(OUT/"control_replay.json",dict(model_parameters_identical_with_dense_disabled=True,
        stage=stage,rows=len(rows),control_state_sha=ppo.sha(binary.state_path(CONTROL,stage))))


def train(control):
    model,opt=terminal.model_and_optimizer()
    (OUT/"state").mkdir(parents=True,exist_ok=True)
    plot()
    with ProcessPoolExecutor(base.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage,keys in enumerate(control["schedule"],START+1):
            state=binary.state_path(OUT,stage)
            metrics=OUT/"metrics"/f"r{stage:02d}.json"
            if state.exists() and metrics.exists():
                m=read(metrics)
                assert ppo.sha(state)==m["state_sha"] and ppo.sha(binary.model_path(OUT,stage))==m["actor_sha"]
                binary.restore(state,model,opt)
            else:
                rows=list(pool.map(rollout,[(stage,key,i) for i,key in enumerate(keys)]))
                assert [r[0] for r in rows]==keys
                if stage==START+1:
                    verify_reward_path(rows,stage)
                stats=fit(model,opt,rows,stage)
                base.export(ARM,SEED,stage,model)
                ppo.legacy.save_checkpoint(state,dict(model=model.state_dict(),optimizer=opt.state_dict()))
                save(metrics,dict(stats,state_sha=ppo.sha(state),actor_sha=ppo.sha(binary.model_path(OUT,stage))))
                print(f"{stage*50} episodes  success {stats['success']}/50  dense bonus {stats['mean_dense_reward']:.3f}  bonus >=1 {stats['bonus_at_least_one']}/50",flush=True)
            if stage in binary.CHECKS and not (OUT/"diagnostic"/f"r{stage:02d}.json").exists():
                terminal.diagnostic(pool,control,stage)
            plot()
    save(OUT/"locked.json",{str(r):ppo.sha(binary.model_path(OUT,r)) for r in binary.CHECKS if r>START})


def assess(control):
    assert ppo.sha(binary.model_path(OUT,END))==read(OUT/"locked.json")[str(END)]
    jobs=[("shaped",END,key) for key in control["confirm"]
          if not (OUT/"evaluation"/f"shaped_r{END}"/(key+".json")).exists()]
    ppo.r.exp.timed_pool("same_panel",terminal.eval_one,jobs)


if __name__=="__main__":
    torch.set_num_threads(1)
    control=setup()
    if not (OUT/"locked.json").exists():
        train(control)
    assess(control)
