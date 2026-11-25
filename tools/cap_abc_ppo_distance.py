"""Continue ABC PPO with a bounded terminal-distance bonus for failure."""
from concurrent.futures import ProcessPoolExecutor
import math
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from tools import cap_abc_ppo_long as binary

base, ppo, read, save = binary.base, binary.ppo, binary.read, binary.save
CONTROL = binary.OUT
OUT = CONTROL.parent/"cap_abc_ppo_distance_20261123_01"
DOC = binary.DOC.parent/"2026-11-23_abc_ppo_distance"
ARM, SEED, START, END = binary.ARM, binary.SEED, binary.START, binary.END
MAX_BONUS = .25
LENGTH_M = 1.
original_result = base.s.output


def shaped_reward(success, distance):
    return 1. if success else MAX_BONUS * math.exp(-distance / LENGTH_M)


def result_with_distance(world, counts):
    result = original_result(world, counts)
    goal = base.s.observe(world)["goal"]
    body = world.env.env.backend.bodies[goal["target_id"]]
    distance = math.dist(body.position, goal["xy"])
    if not math.isfinite(distance):
        raise ValueError("invalid terminal distance")
    result["distance_m"] = distance
    return result


def setup():
    control = binary.setup()
    assert read(CONTROL/"progress/confirm.json")["done"] == 1200
    final = binary.model_path(CONTROL,END)
    assert ppo.sha(final) == read(CONTROL/"locked.json")[str(END)]
    plan = dict(control_protocol=ppo.sha(CONTROL/"protocol.json"), control_final=ppo.sha(final),
                initial_state=control["parent_state"], driver_sha=ppo.sha(Path(__file__)),
                success=1., failure_scale_m=LENGTH_M, failure_max=MAX_BONUS,
                reward="zero before termination; 1 for success; 0.25*exp(-terminal_true_goal_distance_m) for failure",
                primary="R40 distance-trained minus R40 binary on same previously viewed confirmation panel",
                end=END, start=START, episode_count=1800,
                profile=control["profile"], experts=control["expert_sha"])
    manifest = OUT/"protocol.json"
    if manifest.exists():
        assert read(manifest)==plan
    else:
        save(manifest, plan)
    for r in (0,START):
        path = binary.model_path(CONTROL,r)
        base.WEIGHTS[ARM,SEED,r] = base.load_actor(ARM,SEED,r)[1]
        assert ppo.sha(path)==control["checkpoint"][str(r)]
    base.load_actor(ARM,SEED,END)
    base.WEIGHTS["binary",SEED,END] = base.WEIGHTS.pop((ARM,SEED,END))
    base.OUT=OUT
    ppo.r.exp.OUT=OUT
    base.s.output=result_with_distance
    return control


def rollout(job):
    stage, key, idx = job
    rng = np.random.default_rng([SEED,stage,idx,1549])
    result, x, actions, logps = base.episode(key,actor=base.load_actor(ARM,SEED,stage-1),rng=rng)
    return key, result["success"], result["ticks"], x, actions, logps, result["distance_m"]


def tensor_rows(rows, use_distance):
    return [(SEED,key,shaped_reward(won,d) if use_distance else won,ticks,x,a,p)
            for key,won,ticks,x,a,p,d in rows]


def eval_one(job):
    kind, r, key = job
    tag = f"{kind}_r{r}"
    path = OUT/"evaluation"/tag/(key+".json")
    if path.exists():
        return read(path)
    if kind == "binary":
        actor = ("basic",base.WEIGHTS["binary",SEED,r])
    else:
        actor = base.load_actor(ARM,SEED,r)
    result = base.episode(key,actor=actor,greedy=True)[0]
    row = {k:result[k] for k in ("success","history","ticks","distance_m")}
    if kind == "binary":
        stored = read(CONTROL/"evaluation"/"r40"/(key+".json"))
        assert all(row[k] == stored[k] for k in ("success","history","ticks"))
    save(path,row)
    return row


def diagnostic(pool, control, r):
    jobs = [("shaped",r,key) for key in control["diagnostic"]]
    rows = list(pool.map(eval_one,jobs))
    y = [x["success"] for x in rows]
    save(OUT/"diagnostic"/f"r{r:02d}.json",dict(round=r,success=sum(y),n=len(y),outcome=y,
        distance_m=[x["distance_m"] for x in rows]))


def plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    control_start = [read(binary.PARENT/"metrics"/f"basic_s17_r{r}.json") for r in range(1,START+1)]
    b = control_start + [read(f) for f in sorted((CONTROL/"metrics").glob("r*.json"))]
    d = [read(f) for f in sorted((OUT/"metrics").glob("r*.json"))]
    fig,axes=plt.subplots(2,2,figsize=(11,6.2),layout="constrained")
    a,g,e,m = axes.ravel()
    for rows,label,color in ((b,"Binary","#4463a2"),(control_start+d,"Distance bonus","#007d66")):
        reward = [row["success"]/row["episodes"] for row in rows]
        if len(rows)>=5:
            a.plot([50*x["stage"] for x in rows[4:]],np.convolve(reward,np.ones(5)/5,"valid"),
                   color=color,label=label)
        e.plot([50*x["stage"] for x in rows],[x["entropy"] for x in rows],color=color,label=label)
    base_diag=sorted((CONTROL/"diagnostic").glob("r*.json"))
    shap_diag=[f for f in base_diag if int(f.stem[1:])<=START]+sorted((OUT/"diagnostic").glob("r*.json"))
    for files,label,color in ((base_diag,"Binary","#4463a2"),(shap_diag,"Distance bonus","#007d66")):
        rows=[read(f) for f in files]
        g.plot([50*x["round"] for x in rows],[x["success"] for x in rows],"o-",ms=3, color=color,label=label)
    if d:
        m.plot([50*x["stage"] for x in d],[np.nan if x["fail_distance_median"] is None else x["fail_distance_median"] for x in d],
               "o-",color="#007d66",ms=2)
    a.set(title="Training success, rolling 250 episodes",ylabel="Binary success fraction",ylim=(.3,.8))
    g.set(title="Greedy on fixed 64 rooms",ylabel="Rooms solved",ylim=(29,47))
    e.set(title="Policy entropy",ylabel="Mean entropy (nats)")
    m.set(title="Terminal distance among failed training runs",ylabel="Median distance (m)")
    for ax in axes.ravel():
        ax.axvline(START*50,color="#777777",ls=":",lw=1)
        ax.set(xlim=(0,END*50),xlabel="Cumulative PPO episodes")
        ax.grid(alpha=.2)
    a.legend(loc="upper right",fontsize=8)
    g.legend(loc="upper right",fontsize=8)
    DOC.mkdir(parents=True,exist_ok=True)
    temp = DOC/"learning_curves.pending.png"
    fig.savefig(temp,dpi=155)
    temp.replace(DOC/"learning_curves.png")
    plt.close(fig)


def model_and_optimizer():
    torch.manual_seed(SEED)
    model=base.Policy()
    opt=torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                          dict(params=model.critic.parameters(),lr=3e-4)])
    binary.restore(binary.state_path(binary.PARENT,START),model,opt)
    return model,opt


def verify_control_update(rows, stage):
    model,opt = model_and_optimizer()
    ppo.fit(model,opt,tensor_rows(rows,use_distance=False),SEED,stage)
    control = torch.load(binary.state_path(CONTROL,stage),weights_only=True,map_location="cpu")
    assert all(torch.equal(v,control["model"][k]) for k,v in model.state_dict().items())
    save(OUT/"control_replay.json",dict(stage=stage,model_parameters_identical=True,
                                       rows=len(rows),control_state_sha=ppo.sha(binary.state_path(CONTROL,stage))))


def train(control):
    model, opt = model_and_optimizer()
    (OUT/"state").mkdir(parents=True,exist_ok=True)
    plot()
    with ProcessPoolExecutor(base.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage, keys in enumerate(control["schedule"],START+1):
            state=binary.state_path(OUT,stage)
            metrics=OUT/"metrics"/f"r{stage:02d}.json"
            if state.exists() and metrics.exists():
                record=read(metrics)
                assert record["state_sha"]==ppo.sha(state)
                assert record["actor_sha"]==ppo.sha(binary.model_path(OUT,stage))
                binary.restore(state,model,opt)
            else:
                rows = list(pool.map(rollout,[(stage,k,i) for i,k in enumerate(keys)]))
                assert [r[0] for r in rows] == keys
                if stage==START+1:
                    verify_control_update(rows,stage)
                stats=ppo.fit(model,opt,tensor_rows(rows,use_distance=True),SEED,stage)
                failure=[row[-1] for row in rows if not row[1]]
                terminal_return=stats.pop("success")/len(rows)
                stats.update(success=sum(row[1] for row in rows), terminal_return=terminal_return,
                             fail_distance_median=float(np.median(failure)) if failure else None)
                base.export(ARM,SEED,stage,model)
                ppo.legacy.save_checkpoint(state,dict(model=model.state_dict(),optimizer=opt.state_dict()))
                save(metrics,dict(stats,actor_sha=ppo.sha(binary.model_path(OUT,stage)),state_sha=ppo.sha(state)))
                print(f"{stage*50} episodes  binary success {stats['success']}/50  shaped return {terminal_return:.3f}",flush=True)
            if stage in binary.CHECKS and not (OUT/"diagnostic"/f"r{stage:02d}.json").exists():
                diagnostic(pool,control,stage)
            plot()
    save(OUT/"locked.json", {str(stage):ppo.sha(binary.model_path(OUT,stage)) for stage in binary.CHECKS if stage>START})


def assess(control):
    for r,h in read(OUT/"locked.json").items():
        assert ppo.sha(binary.model_path(OUT,int(r)))==h
    jobs = [(kind,END,k) for k in control["confirm"] for kind in ("binary","shaped")
            if not (OUT/"evaluation"/f"{kind}_r{END}"/(k+".json")).exists()]
    ppo.r.exp.timed_pool("same_panel",eval_one,jobs)


if __name__ == "__main__":
    torch.set_num_threads(1)
    control=setup()
    if not (OUT/"locked.json").exists():
        train(control)
    assess(control)
