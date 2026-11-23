"""Three-seed, on-policy PPO for the frozen A/B proposal library."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import multiprocessing as mp
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from tools import cap_regional_id_ppo as legacy
from tools import cap_ungated_no_replay as baseline

r=baseline.run
OUT=r.OUT.parent/"cap_ungated_ppo_20261123_01"
DOC=r.DOC.parent/"2026-11-23_ungated_ppo"
CONFIRM=662322761
ROUNDS=40
PER=50
CACHE={}
sha=r.d.source.graph.file_sha


class Policy(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.actor=nn.Sequential(nn.Linear(349,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.critic=nn.Sequential(nn.Linear(349,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.constant_(self.critic[-1].bias,.5)
        if parent is not None:
            with torch.no_grad():
                for p,w in zip(self.actor.parameters(),parent,strict=True):
                    p.copy_(torch.from_numpy(w.T.copy()))

    def forward(self,x):
        score=self.actor(x).flatten()
        dist=Categorical(logits=torch.stack([torch.zeros_like(score),score],1))
        return dist,self.critic(x).flatten(),torch.ones_like(score,dtype=torch.bool)[:,None]


def model_path(seed,stage):
    return OUT/"models"/f"s{seed}_r{stage}.npz"


def freeze():
    # Initialize through the original protocol so all public runtime/source settings match.
    historical_pool=r.exp.OUT
    ref=r.freeze()
    known=r.e.s.collect_identities() | set(ref["fresh"]) | set(ref["train"])
    for root,fields in ((r.prior.OUT,("train","trend","confirm")),(r.old.OUT,("train","fresh")),
                        (r.prior.first.OUT,("train","diagnostic","fresh")),(r.prior.first.PREVIOUS,("train","held"))):
        for field in fields:
            known.update(r.e.read(root/"protocol.json")[field])
    for group in r.e.read(historical_pool/"protocol.json")["ids"].values():
        known.update(group)
    for group in r.e.read(r.d.OUT/"protocol.json")["rooms"]:
        known.update(group.values())
    for path in (r.exp.trend.OUT/"confirmation_ids.json",r.exp.p.OUT/"fresh_ids.json",
                 r.d.source.OUT/"confirmation_ids.json",r.e.OUT/"confirmation_ids.json",r.e.old.OUT/"fresh_panel.json"):
        known.update(r.e.read(path).values())
    confirm=r.prior.fixtures(CONFIRM)
    assert not known.intersection(confirm)
    r.study.ROOMS.update(confirm)
    train=r.e.read(r.prior.OUT/"protocol.json")["train"]
    rng=np.random.default_rng(823116)
    schedule=[list(map(str,rng.choice(ref["train"] if stage<=4 else train,PER,replace=False)))
              for stage in range(1,ROUNDS+1)]
    protocol=dict(profile=ref["profile"],seeds=list(r.SEEDS),parents=ref["parents"],
        branch=r.e.read(baseline.OUT/"locked.json"),confirm_seed=CONFIRM,confirm=sorted(confirm),diagnostic=ref["fresh"],
        schedule=schedule,rounds=ROUNDS,episodes_per_round=PER,aggregate_episodes=ROUNDS*PER*len(r.SEEDS),
        actor="parent from round four; 349x64x64x1; masked nine semantics; no stall gate; softmax temperature one",
        critic="independent 349x64 tanh x1, output initialized to 0.5; disagreement boundaries",
        optimizer="legacy PPO-Clip; actor 3e-5, critic 3e-4; GAE .95, gamma 1, clip .2, epochs 4, KL stop .015, minibatch 128, value weight .5, norm clip .5; no entropy bonus",
        sample="sample A/B only at distinct proposals; equal proposals execute canonical A; GAE clock counts decision opportunities; terminal 1/0 incl budget",
        primary="prespecified final greedy PPO vs parent, default branch and pure A/B on new 200; paired room bootstrap; no selection",
        diagnostic_comparison="R4 (200 episodes on same 50 rooms) and R40 (2000) on viewed 200. Branch reference: 50 parent runs + at most 150 fork reruns per seed.")
    if (OUT/"protocol.json").exists():
        assert r.e.read(OUT/"protocol.json")==protocol
    else:r.e.save(OUT/"protocol.json",protocol)
    r.exp.OUT=OUT
    return protocol


def collect(job):
    seed,stage,key,idx=job
    token=(seed,stage-1)
    if token not in CACHE:
        CACHE[token]=r.study.cold.weights(model_path(*token))
    r.d.MODELS["history"]=CACHE[token]
    rng=np.random.default_rng(np.random.SeedSequence([seed,stage,idx,921]))
    hist=r.d.PublicHistory()
    xs,bs,logps=[],[],[]
    def choose(_,obs,a,b,c):
        x=hist.vector(obs,a,b,c)
        who="A"
        if a!=b:
            z=float(r.d.logits(x,"history"))
            p=np.exp(-np.logaddexp(0.,-z))
            chosen=int(rng.random()<p)
            xs.append(r.d.project(x,"history"))
            bs.append(chosen)
            logps.append(-np.logaddexp(0.,-z if chosen else z))
            who="B" if chosen else "A"
        hist.executed(who,a,b,c)
        return who
    r.d.source.ROUTE=choose
    row=r.d.source.play(r.study.ROOMS[key],stall_gate=False)["result"]
    return seed,key,row["success"],row["ticks"],np.asarray(xs,np.float32).reshape(-1,349),np.array(bs,int),np.asarray(logps,np.float32)


def fit(policy,opt,rows,seed,stage,*,meter=None):
    x=torch.from_numpy(np.concatenate([v[4] for v in rows]))
    bs=torch.from_numpy(np.concatenate([v[5] for v in rows]))
    logp=torch.from_numpy(np.concatenate([v[6] for v in rows]))
    with torch.no_grad():
        dist,values,_=policy(x)
        v=values.numpy()
        entropy=dist.entropy().mean().item()
    advs=[]
    offset=0
    for _,_,won,_,states,_,_ in rows:
        n=len(states)
        if n:
            rew=np.zeros(n,np.float32)
            rew[-1]=won
            advs.append(legacy.prior.advantages(rew,v[offset:offset+n]))
            offset+=n
    assert offset==len(v)
    adv=torch.from_numpy(np.concatenate(advs))
    batch=dict(x=x,chosen=bs,old_logp=logp,adv=adv,returns=adv+values)
    if meter is not None:
        meter.record_advantages(adv)
    stats=legacy.update_batch(policy,batch,opt,np.random.default_rng([seed,stage,172]),meter=meter)
    stats.update(entropy=entropy,n=len(v),success=sum(row[2] for row in rows),episodes=len(rows),
                 ticks=sum(row[3] for row in rows),seed=seed,stage=stage)
    return stats


def export(seed,stage,model):
    r.exp.save_npz(model_path(seed,stage),**{f"arr_{i}":p.detach().numpy().T.copy()
                                             for i,p in enumerate(model.actor.parameters())})


def train(plan):
    models,opts={},{}
    for seed in r.SEEDS:
        torch.manual_seed(seed)
        models[seed]=Policy(r.study.cold.weights(r.prior.checkpoint(seed,4)))
        m=models[seed]
        opts[seed]=torch.optim.Adam([dict(params=m.actor.parameters(),lr=3e-5),dict(params=m.critic.parameters(),lr=3e-4)])
        export(seed,0,m)
    with ProcessPoolExecutor(max_workers=r.exp.WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage,keys in enumerate(plan["schedule"],1):
            pending=[]
            for seed in r.SEEDS:
                statefile=OUT/"state"/f"s{seed}_r{stage}.pt"
                if statefile.is_file():
                    ck=torch.load(statefile,map_location="cpu",weights_only=True)
                    models[seed].load_state_dict(ck["model"])
                    opts[seed].load_state_dict(ck["optimizer"])
                else:pending.append(seed)
            jobs=[(s,stage,k,i) for i,k in enumerate(keys) for s in pending]
            groups={s:[] for s in pending}
            for row in pool.map(collect,jobs):
                groups[row[0]].append(row)
            for seed,rows in groups.items():
                assert [q[1] for q in rows]==keys
                stats=fit(models[seed],opts[seed],rows,seed,stage)
                export(seed,stage,models[seed])
                r.e.save(OUT/"metrics"/f"r{stage:02d}_s{seed}.json",stats)
                statefile=OUT/"state"/f"s{seed}_r{stage}.pt"
                statefile.parent.mkdir(exist_ok=True)
                legacy.save_checkpoint(statefile,dict(model=models[seed].state_dict(),optimizer=opts[seed].state_dict()))
            if pending:
                print("round",stage,{s:sum(v[2] for v in rows) for s,rows in groups.items()},flush=True)
    locks={}
    for s in r.SEEDS:
        for stage in (4,ROUNDS):
            locks[f"s{s}_r{stage}"]=sha(model_path(s,stage))
    r.e.save(OUT/"locked.json",locks)


def evaluate(plan):
    locks=r.e.read(OUT/"locked.json")
    for s in r.SEEDS:
        for k in (4,ROUNDS):
            name=f"s{s}_r{k}"
            assert sha(model_path(s,k))==locks[name]
            r.WEIGHTS[name]=r.study.cold.weights(model_path(s,k))
        path=baseline.OUT/"models"/f"s{s}_no_replay.npz"
        assert sha(path)==plan["branch"][f"s{s}_no_replay"]
        r.WEIGHTS[f"s{s}_branch"]=r.study.cold.weights(path)
    for panel,arms in (("diagnostic",("r4","r40")),("confirm",("parent","branch","r40"))):
        names=[f"s{s}_{arm}" for s in r.SEEDS for arm in arms]
        if panel=="confirm":names+=["A","B"]
        jobs=[(panel,k,name) for k in plan[panel] for name in names if not (OUT/panel/name/(k+".json")).exists()]
        r.exp.timed_pool(panel,partial(r.play,output=OUT),jobs)


if __name__=="__main__":
    protocol=freeze()
    train(protocol)
    evaluate(protocol)
