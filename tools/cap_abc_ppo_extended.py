"""Resume ABC PPO and generate a separate, unseen room batch for each update."""
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from tools import cap_abc_ppo_dense as dense

previous = dense.binary

base, ppo, read, save = previous.base, previous.ppo, previous.read, previous.save
BINARY = previous.OUT
PARENT = dense.OUT
OUT = PARENT.parent/"cap_abc_ppo_extended_20261124_01"
DOC = previous.DOC.parent/"2026-11-24_abc_ppo_extended"
ARM, SEED = previous.ARM, previous.SEED
START, END = 40, 400
BATCH = 50
CONFIRM_SEED = 662334521
FIRST_BATCH_SEED = 662334701
DIAG_CHECKS = tuple(range(60,END+1,20))
TEST_STAGES = (0,4,START,END)


def setup():
    parent = previous.setup()
    dense_protocol=read(PARENT/"protocol.json")
    assert dense_protocol["control"]==ppo.sha(BINARY/"protocol.json")
    assert dense_protocol["reward_per_metre"]==dense.PER_METRE
    assert dense_protocol["checkpoint"]==START and dense_protocol["seed"]==SEED
    assert read(PARENT/"progress/same_panel.json")["done"]==200
    assert parent["end"]==START and parent["start"]==4 and parent["arm"]==ARM and parent["seed"]==SEED
    state = previous.state_path(PARENT,START)
    locked = read(PARENT/"locked.json")
    record = read(PARENT/"metrics"/f"r{START:02d}.json")
    assert ppo.sha(state)==record["state_sha"]
    for r in TEST_STAGES[:-1]:
        base.OUT=PARENT if r==START else BINARY
        path=previous.model_path(base.OUT,r)
        assert ppo.sha(path)==read(base.OUT/"locked.json")[str(r)]
        base.load_actor(ARM,SEED,r)
    previous_keys=set(base.ROOMS)
    previous_keys.update(base.s.phase.input_hash(f) for f in base.ROOMS.values())
    previous_keys.update(base.s.collect_identities())
    new_test=base.get_rooms(CONFIRM_SEED)
    assert len(new_test)==200 and not previous_keys.intersection(new_test)
    base.ROOMS.update(new_test)
    plan=dict(parent_protocol=ppo.sha(PARENT/"protocol.json"),parent_state=record["state_sha"],
              parent_actor=locked[str(START)],driver=ppo.sha(Path(__file__)),seed=SEED,start=START,end=END,
              reward="0.10 times signed distance reduction in metres at executed option boundaries; success terminal +1",
              reward_per_metre=dense.PER_METRE,temperature=parent["temperature"],
              profile=parent["profile"],executors=parent["executor"],experts=parent["expert_sha"],
              batch=BATCH,room_seeds=list(range(FIRST_BATCH_SEED,FIRST_BATCH_SEED+END-START)),
              room_sampling="Each round uses a distinct generator seed, uniform shuffled unseen unique rooms, 50 per batch, no outcome screening",
              diagnostic=parent["diagnostic"],diagnostic_stages=list(DIAG_CHECKS),
              confirm=sorted(new_test),confirm_seed=CONFIRM_SEED,test_stages=list(TEST_STAGES),
              primary="R400 vs R40 on a new pre-frozen 200-room panel, R400 fixed regardless of curve")
    manifest=OUT/"protocol.json"
    if manifest.exists():
        assert plan==read(manifest)
    else:
        save(manifest,plan)
    base.OUT=OUT
    ppo.r.exp.OUT=OUT
    base.s.tick_action=dense.capture_option
    base.s.output=dense.terminal.result_with_distance
    base.candidate_inputs=dense.capture_choice
    dense.rewards_for_choices=signed_rewards
    return plan, previous_keys | new_test.keys()


def signed_rewards(distance, choices, success):
    distance=np.asarray(distance,float)
    assert distance.ndim==1 and len(distance)>=2 and np.isfinite(distance).all()
    choices=np.asarray(choices,int)
    final=len(distance)-1
    assert all(0<=i<final for i in choices) and np.all(np.diff(choices)>0)
    total=(distance[0]-distance)*dense.PER_METRE
    reward=np.diff(total[np.append(choices,final)]).astype(np.float32)
    if len(reward):
        reward[-1]+=success
    forward=float(np.maximum(0.,-np.diff(distance)).sum())
    return reward,float(total[-1]),forward


def sample(seed, used):
    candidates=base.get_rooms(seed)
    order=np.random.default_rng(seed).permutation(sorted(candidates)).tolist()
    keys=[key for key in order if key not in used][:BATCH]
    if len(keys)!=BATCH:
        raise ValueError(f"seed {seed} supplied only {len(keys)} distinct unused rooms")
    used.update(keys)
    return [(key,candidates[key]) for key in keys]


def rollout(job):
    stage,key,fixture,i=job
    base.ROOMS[key]=fixture
    try:
        return dense.rollout((stage,key,i))
    finally:
        del base.ROOMS[key]


def solve(job):
    key, tag, stage = job
    path=OUT/"evaluation"/tag/(key+".json")
    if path.is_file():
        return read(path)["success"]
    if stage is None:
        result=base.episode(key,teacher=tag)[0]
    else:
        result=base.episode(key,actor=base.load_actor(ARM,SEED,stage),greedy=True)[0]
    save(path,{k:result[k] for k in ("success","history","ticks")})
    return result["success"]


def run_diagnostic(pool,plan,stage):
    path=OUT/"diagnostic"/f"r{stage:03d}.json"
    if path.is_file():
        return
    outcomes=list(pool.map(solve,[(key,f"diag_r{stage:03d}",stage) for key in plan["diagnostic"]]))
    save(path,dict(round=stage,n=len(outcomes),success=sum(outcomes),outcome=outcomes))


def draw(parent_stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    extra=[read(f) for f in sorted((OUT/"metrics").glob("r*.json"))]
    metrics=parent_stats+extra
    diag=[read(BINARY/"diagnostic"/f"r{r:02d}.json") for r in (0,4)]
    diag.extend(read(f) for f in sorted((PARENT/"diagnostic").glob("r*.json")))
    diag.extend(read(f) for f in sorted((OUT/"diagnostic").glob("r*.json")))
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.3),layout="constrained")
    a,b,c,d=axes.ravel()
    x=np.array([m["stage"]*BATCH for m in metrics])
    rates=np.array([m["success"]/m["episodes"] for m in metrics])
    a.plot(x,rates,alpha=.34,lw=.65,color="#1d7198",label="50-episode batch")
    a.plot(x[19:],np.convolve(rates,np.ones(20)/20,"valid"),color="#165376",lw=2,label="1,000-episode mean")
    ys=[m["success"] for m in diag]
    b.plot([m["round"]*BATCH for m in diag],ys,"o-",color="#127c68",lw=1.6,ms=3.5)
    c.plot(x,[m["entropy"] for m in metrics],color="#6944a4",lw=1)
    d.plot(x,[m["critic_loss"] for m in metrics],color="#aa623f",lw=.85,label="Critic MSE")
    d.plot(x,[m["actor_loss"] for m in metrics],color="#3966a8",lw=.85,label="Actor")
    a.set(title="Binary success during training",ylabel="Success fraction",ylim=(0,1))
    b.set(title="Greedy success on fixed 64 rooms",ylabel="Rooms solved",ylim=(max(0,min(ys)-5),min(64,max(ys)+5)))
    c.set(title="Policy entropy on distinct proposals",ylabel="Mean entropy (nats)")
    d.set(title="PPO update losses",ylabel="Loss")
    for ax in (a,d):
        ax.legend(fontsize=8)
    for ax in axes.flat:
        ax.axvline(START*BATCH,color="#777777",ls=":",lw=1)
        ax.set(xlabel="Cumulative PPO episodes",xlim=(0,END*BATCH))
        ax.grid(alpha=.18)
    fig.suptitle("ABC PPO extension  |  signed progress + terminal success  |  seed 17",fontsize=13)
    temp=DOC/"learning_curves.pending.png"
    fig.savefig(temp,dpi=150)
    temp.replace(DOC/"learning_curves.png")
    plt.close(fig)


def training(plan,used):
    parent_stats=[read(previous.PARENT/"metrics"/f"{ARM}_s{SEED}_r{r}.json") for r in range(1,5)]
    parent_stats.extend(read(PARENT/"metrics"/f"r{r:02d}.json") for r in range(5,START+1))
    torch.manual_seed(SEED)
    model=base.Policy()
    opt=torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                          dict(params=model.critic.parameters(),lr=3e-4)])
    previous.restore(previous.state_path(PARENT,START),model,opt)
    assert all(np.array_equal(w,p.detach().numpy().T) for w,p in
               zip(base.load_actor(ARM,SEED,START)[1],model.actor.parameters(),strict=True))
    (OUT/"state").mkdir(parents=True,exist_ok=True)
    draw(parent_stats)
    with ProcessPoolExecutor(base.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage,seed in enumerate(plan["room_seeds"],START+1):
            batch=sample(seed,used)
            keys=[q[0] for q in batch]
            state=previous.state_path(OUT,stage)
            metrics=OUT/"metrics"/f"r{stage:03d}.json"
            if metrics.exists():
                m=read(metrics)
                assert m["rooms"]==keys and m["scene_seed"]==seed
                assert ppo.sha(state)==m["state_sha"]
                assert ppo.sha(previous.model_path(OUT,stage))==m["actor_sha"]
                previous.restore(state,model,opt)
            else:
                results=list(pool.map(rollout,[(stage,k,f,i) for i,(k,f) in enumerate(batch)]))
                assert [row[0] for row in results]==keys
                stats=dense.fit(model,opt,results,stage)
                base.export(ARM,SEED,stage,model)
                ppo.legacy.save_checkpoint(state,dict(model=model.state_dict(),optimizer=opt.state_dict()))
                save(metrics,dict(stats,scene_seed=seed,rooms=keys,state_sha=ppo.sha(state),
                                  actor_sha=ppo.sha(previous.model_path(OUT,stage))))
                print(f"{stage*BATCH} episodes  success {stats['success']}/{BATCH}  entropy {stats['entropy']:.3f}  signed progress {stats['mean_dense_reward']:+.3f}",flush=True)
            if stage in DIAG_CHECKS:
                run_diagnostic(pool,plan,stage)
            if stage%5==0:
                draw(parent_stats)
    save(OUT/"locked.json", {str(r):ppo.sha(previous.model_path(OUT,r)) for r in DIAG_CHECKS})


def final_eval(plan):
    final_path=previous.model_path(OUT,END)
    assert ppo.sha(final_path)==read(OUT/"locked.json")[str(END)]
    variants=[(a,None) for a in base.EXPERTS]+[(f"r{r}",r) for r in TEST_STAGES]
    jobs=[(key,a,r) for key in plan["confirm"] for a,r in variants
          if not (OUT/"evaluation"/a/(key+".json")).exists()]
    dest = OUT/"timing"/"fresh.json"
    if jobs or not dest.exists():
        ppo.r.exp.timed_pool("fresh",solve,jobs)


if __name__=="__main__":
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    DOC.mkdir(parents=True,exist_ok=True)
    with (OUT/".driver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan,used=setup()
        if not (OUT/"locked.json").exists():
            training(plan,used)
        final_eval(plan)
