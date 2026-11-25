"""Paired ABC routing with public observations, optional semantics, and short PPO."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import runpy

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.planar_tasks.relational_search.motion import ACTIONS
from tools import cap_authored_third as third
from tools import cap_ungated_ppo as ppo

OUT = third.OUT.parent / "cap_abc_semantic_ppo_20261123_01"
DOC = third.DOC.parent / "2026-11-23_abc_semantic_ppo"
WORK = third.WORK.parent / "abc_semantics_1123"
ARMS = ("basic", "semantic")
EXPERTS = ("B", "A", "C")
SEEDS = (17, 29, 43)
EVAL_STAGES = (0, 4)
TRAIN_SEED = 662329137
TEST_SEED = 662329153
WORKERS = 24
WARM_UPDATES = 4000
STAGES = ("", "find", "approach", "final", "orient", "touch", "push", "wait_peer_contact",
          "standoff", "align", "ready", "view_insufficient", "wait", "peer_wait",
          "perception_wait", "blocked", "move_blocked", "move")
MAX_FEATURES = 5
EYE = np.eye(len(ACTIONS), dtype=np.float32)
STAGE_EYE = np.eye(len(STAGES)+1, dtype=np.float32)
ROOMS = {}
WEIGHTS = {}
s = third.s
read, save = third.read, third.save


def get_rooms(seed):
    return {s.phase.input_hash(row["fixture"]): row["fixture"]
            for row in s.phase.cap.rooms.generate(seed)["cases"]}


def setup():
    parent = read(third.OUT/"protocol.json")
    sources = third.load_pair()
    assert parent["parent_sha"] == {k: third.pair.base.digest(v) for k,v in sources.items()}
    member = read(third.OUT/"member.json")
    assert third.pair.base.host.RoomCampaign(third.CAM).verify_frozen() == member["seal"]
    path = third.OUT/member["path"]
    assert ppo.sha(path) == member["source_sha256"]
    sources["C"] = path.read_text()
    third.use(sources)
    dev = third.pair.dev()
    online = get_rooms(TRAIN_SEED)
    test = get_rooms(TEST_SEED)
    assert {k: s.phase.input_hash(f) for k,f in dev.items()} == parent["dev_content"]
    assert all(len(group)==200 for group in (dev, online, test))
    used = s.collect_identities() | set(parent["dev_content"].values()) | set(parent["held_content"])
    prior = read(ppo.OUT/"protocol.json")
    used.update(prior["confirm"] + prior["diagnostic"])
    assert not (set(online) & (used | set(test))) and not set(test) & used
    assert not (set(dev) & (set(online) | set(test)))
    ROOMS.update(dev | online | test)
    feature_source = DOC/"features.py"
    analysis = runpy.run_path(str(feature_source))
    names = analysis["feature_names"]
    assert len(names)==len(set(names)) and 3<=len(names)<=MAX_FEATURES
    global semantic
    semantic = analysis["features"]
    ids = np.random.default_rng(TRAIN_SEED).permutation(sorted(online)).tolist()
    plan = dict(profile=parent["profile"], executor=parent["executor"],
                expert_sha={k: third.pair.perception.fingerprint(src) for k,src in sources.items()},
                feature_sha=ppo.sha(feature_source), feature_names=list(names),
                driver=ppo.sha(Path(__file__)), ppo_source=ppo.sha(Path(ppo.__file__)),
                seed=list(SEEDS), demonstration=sorted(dev), evaluation=sorted(test),
                schedule=[ids[i:i+50] for i in range(0,200,50)], eval_stages=list(EVAL_STAGES),
                imitation_updates=WARM_UPDATES, temperature=1, critic_semantics=False,
                protocol="Uniform successful rooms, uniform successful expert within room, uniform disagreement within episode. Shared candidate scorer. First representative of each distinct complete command in B/A/C order. No stall gate.")
    manifest=OUT/"protocol.json"
    if manifest.exists():
        assert read(manifest)==plan
    else:
        save(manifest,plan)
    ppo.r.exp.OUT = OUT
    ppo.r.exp.WORKERS = WORKERS
    return plan


def action_vector(action, target):
    return [*EYE[action["a0"]], *EYE[action["a1"]], action["ticks"]/50,
            float(action["target_id"] == target), float(action["target_id"] is None)]


def public_frame(obs):
    row = third.pair.g.live_frame(obs)
    return dict(tick=int(row["tick"]), seen=row["seen"], gap=float(row["gap"]),
                xy=row["xy"].tolist(), rel=row["rel"].tolist(), touch=row["touch"],
                static=[int(r["contacting_static"]) for r in obs["robots"]],
                stages=[r["execution"].get("stage") or "" for r in obs["robots"]])


def frame_vector(frame):
    stages = [STAGE_EYE[STAGES.index(s) if s in STAGES else len(STAGES)] for s in frame["stages"]]
    return [frame["tick"]/1800, frame["seen"], frame["gap"]/3, *np.ravel(frame["xy"])/5,
            *np.asarray(frame["rel"])/3, *frame["touch"], *frame["static"], *stages[0], *stages[1]]


ACTION_SIZE = len(ACTIONS)*2 + 3
# The observation contains 15 scalar entries plus two one-hot execution stages.
FRAME_SIZE = 15 + 2*(len(STAGES)+1)
BASE_SIZE = FRAME_SIZE + 4*(1+FRAME_SIZE+ACTION_SIZE) + 2 + ACTION_SIZE + 3
SIZE = BASE_SIZE + MAX_FEATURES + 1


def candidate_inputs(frame, history, obs, proposals):
    target = obs["goal"]["target_id"]
    previous = list(history)
    common = frame_vector(frame)
    for _ in range(4-len(previous)):
        common.extend([0.]*(1+FRAME_SIZE+ACTION_SIZE))
    for past in previous:
        common.extend((1., *frame_vector(past), *action_vector(past["action"], target)))
    common.extend(np.asarray(obs["goal"]["xy"])/5)
    rows = []
    for i, action in enumerate(proposals):
        semantics = list(semantic(frame, previous, action, target))
        if not all(np.isfinite(v) and -1<=v<=1 for v in semantics):
            raise ValueError(f"invalid semantic values {semantics}")
        rows.append([*common, *action_vector(action, target),
                     *(float(item==action) for item in proposals),
                     *semantics, *([0.]*(MAX_FEATURES-len(semantics))),
                     float(action not in proposals[:i])])
    return np.asarray(rows, np.float32).reshape(3,SIZE)


def project(inputs, arm):
    if arm == "basic":
        inputs[...,BASE_SIZE:-1] = 0
    return inputs


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(SIZE-1,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
        self.critic = nn.Sequential(nn.Linear(BASE_SIZE,64),nn.Tanh(),nn.Linear(64,1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.constant_(self.critic[-1].bias,.5)

    def distribution(self, x):
        legal = x[..., -1].bool()
        logits = self.actor(x[..., :-1]).squeeze(-1).masked_fill(~legal, -torch.inf)
        return Categorical(logits=logits), legal

    def forward(self, x):
        dist, legal = self.distribution(x)
        v = self.critic(x[..., :BASE_SIZE].mean(1)).flatten()
        return dist, v, legal


def checkpoint(arm, seed, stage):
    return OUT/"models"/f"{arm}_s{seed}_r{stage}.npz"


def export(arm, seed, stage, model):
    fields = {f"a{i}":v.detach().numpy().T.copy() for i,v in enumerate(model.actor.parameters())}
    ppo.r.exp.save_npz(checkpoint(arm,seed,stage), **fields)


def scores(x, weights):
    w1,b1,w2,b2,w3,b3 = weights
    raw = (np.maximum(0,np.maximum(0,x[...,:-1]@w1+b1)@w2+b2)@w3+b3).ravel()
    return np.where(x[..., -1]>0, raw, -np.inf)


def episode(key, *, teacher=None, collect=False, actor=None, greedy=False, rng=None):
    names = (teacher,) if teacher and not collect else EXPERTS
    members = {a:s.phase.FrozenCAP(third.MEMBERS[a]) for a in names}
    counts, history, choices = Counter(), deque(maxlen=4), Counter()
    data, selected, logp = [], [], []
    with s.phase.world(ROOMS[key]) as (world,_):
        def update(events):
            for member in members.values():
                member.update(events)

        update(world.history)
        while not s.ended(world):
            obs = s.observe(world)
            if teacher and not collect:
                action = members[teacher].propose(obs)
            else:
                frame = public_frame(obs)
                proposals = [members[a].propose(obs) for a in names]
                distinct = any(action != proposals[0] for action in proposals[1:])
                if distinct:
                    x = candidate_inputs(frame, history, obs, proposals)
                index = 0
                if teacher:
                    index = proposals.index(proposals[names.index(teacher)])
                    if distinct:
                        data.append(x)
                        selected.append(index)
                elif distinct:
                    x = project(x,actor[0])
                    logits = scores(x,actor[1])
                    if greedy:
                        index = int(logits.argmax())
                    else:
                        z = np.exp((logits-logits.max()).astype(np.float64))
                        probabilities = z/z.sum()
                        index = int(rng.choice(3,p=probabilities))
                        data.append(x)
                        selected.append(index)
                        logp.append(np.log(probabilities[index]))
                choices[names[index]] += 1
                action = proposals[index]
                history.append(dict(frame, action=action))
            s.tick_action(world,action,third.EXECUTOR,update,counts)
        result = s.output(world,counts)
    result["choices"] = dict(choices)
    return result, np.asarray(data,np.float32).reshape(-1,3,SIZE), np.asarray(selected,int), np.asarray(logp,np.float32)


def demo_one(job):
    name, key = job
    dest = OUT/"demonstration"/name/(key+".json")
    if dest.exists():
        return
    result, x, labels, _ = episode(key,teacher=name,collect=True)
    if name in third.ANCHORS:
        old = read(third.OUT/"screen"/(key+".json"))[name]["result"]
        assert (result["success"],result["history"]) == (old["success"],old["history"])
    if result["success"] and len(labels):
        ppo.r.exp.save_npz(dest.with_suffix(".npz"),x=x,label=labels)
    save(dest,dict(success=result["success"],ticks=result["ticks"],boundaries=len(labels),history=result["history"]))


def demo(plan):
    ppo.r.exp.timed_pool("demonstration", demo_one, [(expert,key) for key in plan["demonstration"] for expert in EXPERTS])


def demonstration_arrays(plan):
    data, labels, weight = [], [], []
    by_expert, only_expert = Counter(), Counter()
    union = 0
    for k in plan["demonstration"]:
        found = []
        winners = []
        for a in EXPERTS:
            path = OUT/"demonstration"/a/(k+".json")
            if read(path)["success"]:
                winners.append(a)
                by_expert[a] += 1
            if path.with_suffix(".npz").exists():
                with np.load(path.with_suffix(".npz")) as d:
                    found.append((d["x"],d["label"]))
        if len(winners)==1:
            only_expert[winners[0]] += 1
        union += bool(winners)
        for x,y in found:
            data.append(x)
            labels.append(y)
            weight.extend([1/(len(found)*len(y))]*len(y))
    save(OUT/"demo_summary.json",dict(by_expert=by_expert,only=only_expert,union=union,boundaries=len(weight)))
    return np.concatenate(data), np.concatenate(labels), np.cumsum(weight)/sum(weight)


def warm_start(plan):
    data, labels, cdf = demonstration_arrays(plan)
    for seed in SEEDS:
        for arm in ARMS:
            if checkpoint(arm,seed,0).exists():
                continue
            x = torch.from_numpy(project(data.copy(),arm))
            y = torch.from_numpy(labels)
            torch.manual_seed(seed)
            model = Policy()
            opt = torch.optim.Adam(model.actor.parameters(),lr=.001)
            rng = np.random.default_rng(seed)
            for _ in range(WARM_UPDATES):
                indices = np.searchsorted(cdf,rng.random(128))
                dist,_ = model.distribution(x[indices])
                loss = -dist.log_prob(y[indices]).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            with torch.no_grad():
                predicted = torch.cat([model.distribution(b)[0].probs.argmax(1) for b in x.split(4096)])
            export(arm,seed,0,model)
            save(OUT/"supervision"/f"{arm}_s{seed}.json",dict(accuracy=(predicted==y).float().mean().item(),boundaries=len(y)))


def load_actor(arm, seed, stage):
    tag = (arm,seed,stage)
    if tag not in WEIGHTS:
        with np.load(checkpoint(*tag)) as d:
            WEIGHTS[tag] = [d[f"a{i}"] for i in range(6)]
    return arm, WEIGHTS[tag]


def rollout(job):
    arm, seed, stage, key, idx = job
    rng = np.random.default_rng([seed,stage,idx,1549])
    result, x, y, logp = episode(key,actor=load_actor(arm,seed,stage-1),rng=rng)
    return seed, key, result["success"], result["ticks"], x, y, logp


def train(plan):
    models, opts = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = Policy()
            with torch.no_grad():
                for p,w in zip(model.actor.parameters(),load_actor(arm,seed,0)[1],strict=True):
                    p.copy_(torch.from_numpy(w.T.copy()))
            tag = (arm,seed)
            models[tag] = model
            opts[tag] = torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                                          dict(params=model.critic.parameters(),lr=3e-4)])
    with ProcessPoolExecutor(WORKERS,mp_context=mp.get_context("fork")) as pool:
        for stage, keys in enumerate(plan["schedule"],1):
            pending = []
            for tag, model in models.items():
                arm,seed = tag
                state = OUT/"state"/f"{arm}_s{seed}_r{stage}.pt"
                if state.exists():
                    ck = torch.load(state,weights_only=True,map_location="cpu")
                    model.load_state_dict(ck["model"])
                    opts[tag].load_state_dict(ck["optimizer"])
                else:
                    pending.append(tag)
            groups = {tag:[] for tag in pending}
            jobs = [(arm,seed,stage,key,i) for i,key in enumerate(keys) for arm,seed in pending]
            for job,row in zip(jobs,pool.map(rollout,jobs),strict=True):
                groups[job[:2]].append(row)
            for tag, rows in groups.items():
                arm,seed = tag
                assert [r[1] for r in rows] == keys
                stats = ppo.fit(models[tag],opts[tag],rows,seed,stage)
                export(arm,seed,stage,models[tag])
                save(OUT/"metrics"/f"{arm}_s{seed}_r{stage}.json",stats)
                state = OUT/"state"/f"{arm}_s{seed}_r{stage}.pt"
                state.parent.mkdir(exist_ok=True)
                ppo.legacy.save_checkpoint(state,dict(model=models[tag].state_dict(),optimizer=opts[tag].state_dict()))
            print("PPO round",stage,"completed",flush=True)
    locking = {path.name: ppo.sha(path) for path in (OUT/"models").glob("*.npz")}
    save(OUT/"locked.json", locking)


def evaluate_one(job):
    key, tag, actor = job
    if actor is None:
        result = episode(key,teacher=tag)[0]
    else:
        result = episode(key,actor=load_actor(*actor),greedy=True)[0]
    save(OUT/"evaluation"/tag/(key+".json"),result)


def evaluate(plan):
    for name,digest in read(OUT/"locked.json").items():
        assert ppo.sha(OUT/"models"/name)==digest
    policies = [(name,None) for name in EXPERTS]
    policies += [(f"{a}_s{s}_r{r}", (a,s,r)) for a in ARMS for s in SEEDS for r in EVAL_STAGES]
    jobs = [(key,tag,actor) for tag,actor in policies for key in plan["evaluation"]
            if not (OUT/"evaluation"/tag/(key+".json")).exists()]
    ppo.r.exp.timed_pool("evaluation",evaluate_one,jobs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase",choices=("demo","run"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan = setup()
    demo(plan)
    if args.phase=="run":
        if not (OUT/"locked.json").exists():
            warm_start(plan)
            train(plan)
        evaluate(plan)
