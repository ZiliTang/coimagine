"""Candidate-level clipped PPO initialized from the frozen GLOBAL Q selector."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime, full_worker as worker
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import flatten, read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import candidate_arrays
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import load_checkpoint
from .cap_trace_value_long import sha, phase

TEMPERATURE = .03
EPOCHS = 4
TARGET_KL = .015
CACHE = {}


class Policy(nn.Module):
    def __init__(self, parent):
        super().__init__()
        self.actor, _ = load_checkpoint(parent)
        width = self.actor.layout['width']
        self.critic = nn.Sequential(nn.Linear(2*width, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.zeros_(self.critic[-1].weight)
        nn.init.constant_(self.critic[-1].bias, .5)

    def forward(self, x, candidates, sources, valid):
        q = self.actor.values(x, candidates, sources)
        logits = (q / TEMPERATURE).masked_fill(~valid, -torch.inf)
        value = self.critic(self.actor.encode(x)).squeeze(-1)
        return Categorical(logits=logits), value


def advantages(rewards, values):
    # Complete finite-horizon episodes, including time/decision limits. gamma=1.
    result = np.empty_like(values)
    following_value = residual = 0.
    for i in reversed(range(len(values))):
        residual = rewards[i] + following_value - values[i] + .95 * residual
        result[i] = residual
        following_value = values[i]
    return result


def rollout(job):
    case, directory, parent, checkpoint, action_seed, training = job
    directory.mkdir(parents=True)
    if checkpoint not in CACHE:
        CACHE.clear()
        model = Policy(parent)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        CACHE[checkpoint] = model.eval()
    model = CACHE[checkpoint]
    rng = np.random.default_rng(action_seed)
    env = fast_runtime.ScopedEnvironment(case['fixture'], worker.EventConfig(mode='duration'),
        fast_feedback=True, history_factory=fast_runtime.AuditedCachedHistory, record_details=True)
    store = {k: [] for k in ('x','candidates','sources','valid','chosen','old_logp','value','entropy')}
    decisions = []
    h = env.history
    public = [dict(worker.current_event(h), history_hash=h.sha256)]
    started = time.monotonic()
    try:
        worker.LIBRARY.reset()
        for i in range(300):
            if time.monotonic()-started > 1800:
                raise TimeoutError(case['case_id'])
            packet = worker.LIBRARY.advance(h)
            x = flatten(packet)
            candidates, sources, valid = candidate_arrays(packet, worker.LIBRARY)
            features = (x,candidates,sources,valid)
            with torch.inference_mode():
                dist, value = model(*(torch.from_numpy(a[None]) for a in features))
                probs = dist.probs[0].double().numpy()
                greedy = int(dist.logits[0].argmax())
                chosen = int(rng.choice(len(valid), p=probs/probs.sum())) if training else greedy
                logp = float(dist.logits[0,chosen])
                entropy = float(dist.entropy())
            event = env.step(packet['pool'][chosen], remaining_ticks=1800-env.env.total_periods)
            if training:
                for k,v in zip(store, (*features,chosen,logp,float(value),entropy), strict=True):
                    store[k].append(v)
            decisions.append(dict(chosen=chosen, greedy=greedy, candidates=packet['pool'], sources=packet['sources'],
                probabilities=probs[valid].tolist(), old_logp=logp, value=float(value), event=event))
            public.append(dict(worker.current_event(h), history_hash=h.sha256))
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        n = len(decisions)
        summary = dict(case_id=case['case_id'], success=bool(env.env.success), decisions=n, ticks=env.env.total_periods,
            complete=True, termination=env.env.termination, action_seed=action_seed, checkpoint=str(checkpoint),
            stop='terminal' if env.env.termination is not None else 'tick_budget' if env.env.total_periods>=1800 else 'decision_budget')
        if training:
            array = {k: np.asarray(v) for k,v in store.items()}
            rewards = np.zeros(n, dtype=np.float32)
            rewards[-1] = summary['success']
            adv = advantages(rewards, array['value'])
            array.update(adv=adv.astype('f'), returns=(adv+array['value']).astype('f'), reward=rewards)
            shard = directory/'transitions.npz'
            np.savez_compressed(shard, **array)
            summary.update(shard=str(shard), entropy=float(np.mean(array['entropy'])))
        with gzip.open(directory/'result.json.gz','wt',compresslevel=1) as f:
            f.write(json.dumps(dict(summary=summary,decisions=decisions,public_rows=public,replay=env.replay()),allow_nan=False))
        save(directory/'summary.json',summary)
        return summary
    finally:
        env.close_reads()
        fast_runtime.ACTIVE = None


def collect(pool, jobs, directory):
    futures = {pool.submit(rollout,j): i for i,j in enumerate(jobs)}
    rows = [None]*len(jobs)
    for n,f in enumerate(as_completed(futures),1):
        rows[futures[f]] = f.result()
        if n%10==0 or n==len(jobs):
            save(directory/'progress.json', dict(completed=n,total=len(jobs)))
    save(directory/'results.json',rows)
    return rows


def fit(policy, optimizer, rows, rng):
    episodes = []
    for row in rows:
        with np.load(row['shard']) as data:
            episodes.append({k:data[k] for k in data.files})
    batch = {k:torch.from_numpy(np.concatenate([e[k] for e in episodes])) for k in episodes[0]}
    del episodes
    adv = batch['adv']
    adv = (adv-adv.mean()) / adv.std(unbiased=False).clamp_min(1e-6)
    metrics = []
    def distribution(ids):
        return policy(*(batch[k][ids] for k in ('x','candidates','sources','valid')))
    with torch.no_grad():
        initial_error = max(float((distribution(ids)[0].log_prob(batch['chosen'][ids])-
                        batch['old_logp'][ids]).abs().max()) for ids in torch.arange(len(adv)).split(128))
    if initial_error > .001:
        raise RuntimeError(f'rollout log probability mismatch: {initial_error}')
    for epoch in range(EPOCHS):
        for ids in torch.from_numpy(rng.permutation(len(adv))).split(128):
            dist,value = distribution(ids)
            difference = dist.log_prob(batch['chosen'][ids])-batch['old_logp'][ids].float()
            ratio = difference.exp()
            actor = -torch.minimum(ratio*adv[ids], ratio.clamp(.8,1.2)*adv[ids]).mean()
            critic = (value-batch['returns'][ids]).square().mean()
            entropy = dist.entropy().mean()
            loss = actor+.5*critic
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
            metrics.append([float(z.detach()) for z in (actor,critic,entropy,((ratio-1).abs()>.2).float().mean())])
        with torch.no_grad():
            diffs = torch.cat([distribution(ids)[0].log_prob(batch['chosen'][ids])-batch['old_logp'][ids]
                               for ids in torch.arange(len(adv)).split(128)])
            kl = float((diffs.exp()-1-diffs).mean())
        if kl>TARGET_KL: break
    return dict(zip(('actor_loss','value_mse','entropy','clip_fraction'),np.mean(metrics,axis=0).tolist()),
        epochs=epoch+1, updates=len(metrics), kl=kl, old_logp_error=initial_error, boundaries=len(adv))


def run(parent, baseline, out):
    from tools.cap_trace_value_ppo_report import render
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    inputs = read(baseline/'inputs.json')
    source = parent/'models/GLOBAL_s17/online_r4.pt'
    assert sha(source)==inputs['checkpoint_sha256']
    assert sha(baseline/'schedule.json')==inputs['schedule_sha256']
    for filename in ('online_training','final_test'):
        panel = parent/'panels'/f'{filename}.json'
        assert sha(panel)==read(parent/'panels/manifest.json')['files'][panel.name]
    cases = {c['case_id']:c for c in read(parent/'panels/online_training.json')['cases']}
    development = {c['case_id']:c for c in read(Path(inputs['bundle'])/'scenes.json')}
    monitor = [development[c] for c in read(parent/'split.json')['monitor']]
    diagnostic = read(parent/'panels/final_test.json')['cases']
    for rel,digest in inputs['sources'].items():
        if sha(Path(rel))!=digest: raise ValueError(rel)
    save(out/'inputs.json',dict(parent=str(parent), baseline=str(baseline), checkpoint_sha256=sha(source),
        schedule_sha256=inputs['schedule_sha256'], seed=17, temperature=TEMPERATURE, gamma=1, gae_lambda=.95,
        clip=.2, epochs=EPOCHS, target_kl=TARGET_KL, minibatch=128, actor_lr=3e-5, critic_lr=3e-4,
        entropy_bonus=0, value_weight=.5, grad_clip=.5, episodes=1800, final_cumulative_episodes=2000,
        source_hashes={**inputs['sources'], **{str(p):sha(p) for p in (Path(__file__),Path('tools/cap_trace_value_ppo_report.py'))}}))
    model = Policy(source)
    optimizer = torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),dict(params=model.critic.parameters(),lr=3e-4)])
    checkpoint = out/'initial.pt'
    torch.save(model.state_dict(),checkpoint)
    original,_ = load_checkpoint(source)
    replay = parent/'replay/GLOBAL_s17/r4'
    sample = slice(None,None,11)
    tensors = [torch.from_numpy(np.load(replay/(k+'.npy'),mmap_mode='r')[sample].copy()) for k in ('x','candidates','sources','valid')]
    with torch.no_grad():
        matches = []
        for ids in torch.arange(len(tensors[0])).split(64):
            x,c,s,v=(a[ids] for a in tensors)
            matches.extend((model(x,c,s,v)[0].probs.argmax(-1)==original.values(x,c,s).masked_fill(~v,-torch.inf).argmax(-1)).tolist())
        assert all(matches)
    save(out/'initial_parity.json',dict(matched=sum(matches),boundaries=len(matches)))
    history = []
    with ProcessPoolExecutor(max_workers=12,mp_context=mp.get_context('spawn'),initializer=fast_runtime.initialize,
            initargs=(inputs['bundle'],inputs['priority_bundle'])) as pool:
        for block in read(baseline/'schedule.json'):
            r = block['round']
            directory = out/'training'/f'r{r}'
            phase(out,'collect',round=r)
            rows = collect(pool,[(cases[item['case_id']],directory/item['case_id'],source,checkpoint,item['action_seed'],True)
                                for item in block['cases']],directory)
            phase(out,'fit',round=r)
            metrics = fit(model,optimizer,rows,rng)
            checkpoint = out/f'actor_critic_r{r}.pt'
            torch.save(model.state_dict(),checkpoint)
            history.append(dict(round=r,episodes=r*50,success=sum(x['success'] for x in rows), count=len(rows),
                behavior_entropy=np.average([x['entropy'] for x in rows],weights=[x['decisions'] for x in rows]), **metrics))
            save(out/'history.json',history)
            for label,panel,enabled in [('monitor',monitor,r%4==0),('diagnostic',diagnostic,r in (8,16,24,40))]:
                if enabled:
                    phase(out,label,round=r)
                    dest = out/label/f'e{r*50}'
                    collect(pool,[(c,dest/c['case_id'],source,checkpoint,17,False) for c in panel],dest)
            render(out)
    phase(out,'report')
    render(out,final=True)
    phase(out,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('parent','baseline','out'): parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    run(args.parent.resolve(),args.baseline.resolve(),args.out.resolve())
