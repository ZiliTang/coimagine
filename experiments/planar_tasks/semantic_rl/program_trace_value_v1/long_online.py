"""Budget-only continuation, with observational metrics and partitioned replay.

The original experiment's learner and workers remain unchanged. Partitioning
changes storage only: sampling is uniform over the same ordered boundary table.
"""
import copy
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import read, save
from .full_data import Store
from .full_learner import checkpoint
from .q_model import load_checkpoint, tensors, double_dqn_target
from . import full_worker


class PartitionedStore:
    def __init__(self, paths):
        self.stores = [Store(p) for p in paths]
        self.ends = np.cumsum([len(s) for s in self.stores])
        self.starts = np.r_[0, self.ends[:-1]]
        self.arrays = {k: np.concatenate([s.arrays[k] for s in self.stores])
                       for k in ('reward', 'done', 'chosen')}

    def __len__(self):
        return int(self.ends[-1])

    def batch(self, ids, next_state=False):
        ids = np.asarray(ids)
        if np.any(ids < 0) or np.any(ids >= len(self)):
            raise IndexError('Replay index outside table')
        partitions = np.searchsorted(self.ends, ids, side='right')
        result = {}
        for i in np.unique(partitions):
            mask = partitions == i
            batch = self.stores[i].batch(ids[mask]-self.starts[i], next_state)
            for key, value in batch.items():
                if key not in result:
                    result[key] = np.empty((len(ids), *value.shape[1:]), dtype=value.dtype)
                result[key][mask] = value
        return result


def behavior_entropy(count, epsilon):
    """Exact entropy (nats) of the worker's deterministic-tie epsilon greedy."""
    if count < 1 or not 0 <= epsilon <= 1:
        raise ValueError((count, epsilon))
    probabilities = np.full(count, epsilon/count, dtype=np.float64)
    probabilities[0] += 1-epsilon
    positive = probabilities > 0
    return float(-np.sum(probabilities[positive]*np.log(probabilities[positive])))


def rollout_metrics(path):
    with gzip.open(path, 'rt') as f:
        data = json.load(f)
    rows = data['decisions']
    entropy, normalized, gaps, values, counts = [], [], [], [], []
    nongreedy = 0
    for row in rows:
        count = len(row['candidates'])
        h = behavior_entropy(count, row['epsilon'])
        entropy.append(h)
        normalized.append(h/np.log(count) if count > 1 else 0.)
        counts.append(count)
        score = sorted(row['scores'], reverse=True)
        if count > 1:
            gaps.append(score[0]-score[1])
        values.append(row['chosen_value'])
        nongreedy += row['chosen'] != row['greedy']
    return dict(decisions=len(rows), entropy_sum=sum(entropy),
                normalized_entropy_sum=sum(normalized), candidate_count_sum=sum(counts),
                nongreedy_decisions=int(nongreedy), q_gap_sum=sum(gaps), q_gap_count=len(gaps),
                chosen_q_sum=sum(values), chosen_q_saturated=sum(q < .01 or q > .99 for q in values))


def run_job(job):
    result = full_worker.run(job)
    if not job.get('cap') and result['complete']:
        path = Path(job['directory'])/'behavior_metrics.json'
        if not path.exists():
            save(path, rollout_metrics(Path(job['directory'])/'result.json.gz'))
        result = dict(result, behavior=read(path))
    return result


def train_round(parent, out, seed, round_number, replay_paths, updates=1500, log_every=100):
    parent, out = Path(parent), Path(out)
    directory = out/'models'/f'GLOBAL_s{seed}'
    previous = (parent/'models'/f'GLOBAL_s{seed}'/'online_r4.pt' if round_number == 5
                else directory/f'online_r{round_number-1}.pt')
    q, state = load_checkpoint(previous)
    q.train()
    target = copy.deepcopy(q).eval()
    target.load_state_dict(state['target_model'])
    optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)
    optimizer.load_state_dict(state['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    offline = Store(parent/'arrays/train')
    online = PartitionedStore(replay_paths)
    previous_updates = state['online_updates']
    directory.mkdir(parents=True, exist_ok=True)
    log = directory/f'online_r{round_number}_metrics.jsonl'
    if log.exists():
        raise FileExistsError('Refusing to overwrite an interrupted round: '+str(log))
    started = time.monotonic()
    window = []
    with log.open('x') as stream:
        for update in range(updates):
            if (out/'STOP').exists() or time.monotonic()-started > 1800:
                checkpoint(directory/f'online_r{round_number}_interrupted.pt', q, optimizer, seed,
                           update, 'Q', rng, target_model=target.state_dict(),
                           online_updates=previous_updates+update)
                raise RuntimeError('STOP or 1800-second update-round limit')
            selected = [(offline, rng.integers(len(offline), size=32)),
                        (online, rng.integers(len(online), size=32))]
            current_parts = [s.batch(ids) for s, ids in selected]
            next_parts = [s.batch(ids, next_state=True) for s, ids in selected]
            current = {k: np.concatenate([b[k] for b in current_parts])
                       for k in ('x', 'candidates', 'sources', 'valid')}
            following = {k: np.concatenate([b[k] for b in next_parts])
                         for k in ('x', 'candidates', 'sources', 'valid')}
            reward = torch.tensor(np.concatenate([s.arrays['reward'][ids] for s, ids in selected]))
            done = torch.tensor(np.concatenate([s.arrays['done'][ids] for s, ids in selected]))
            chosen = torch.tensor(np.concatenate([s.arrays['chosen'][ids] for s, ids in selected]), dtype=torch.long)
            current, following = tensors(current), tensors(following)
            label = double_dqn_target(q, target, following, reward, done)
            optimizer.zero_grad(set_to_none=True)
            prediction = q.values(current['x'], current['candidates'], current['sources']).gather(1, chosen[:, None]).squeeze(1)
            squared = (prediction-label)**2
            loss = squared.mean()
            if not torch.isfinite(loss):
                raise ValueError('nonfinite online loss')
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(q.parameters(), 1.)
            if not torch.isfinite(grad):
                raise ValueError('nonfinite gradient')
            optimizer.step()
            with torch.no_grad():
                window.append(dict(td_mse=float(loss), offline_td_mse=float(squared[:32].mean()),
                    online_td_mse=float(squared[32:].mean()), td_abs=float((prediction-label).abs().mean()),
                    mean_q=float(prediction.mean()), mean_target=float(label.mean()),
                    grad_norm_before_clip=float(grad), terminal_fraction=float(done.float().mean()),
                    reward_fraction=float(reward.mean()),
                    q_saturated_fraction=float(((prediction < .01) | (prediction > .99)).float().mean())))
            if (previous_updates+update+1) % 250 == 0:
                target.load_state_dict(q.state_dict())
            if (update+1) % log_every == 0 or update+1 == updates:
                row = {k:float(np.mean([r[k] for r in window])) for k in window[0]}
                row.update(seed=seed, round=round_number, updates=previous_updates+update+1,
                           episodes=round_number*50, online_boundaries=len(online),
                           window_updates=len(window), seconds=time.monotonic()-started)
                stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                save(directory/'progress.json', row)
                window.clear()
                if (update+1) % 500 == 0:
                    print('LONG_FIT', row, flush=True)
    destination = directory/f'online_r{round_number}.pt'
    checkpoint(destination, q, optimizer, seed, updates, 'Q', rng,
               online_updates=previous_updates+updates, collected_online_episodes=50*round_number,
               target_model=target.state_dict(), replay_partitions=[str(Path(p).resolve()) for p in replay_paths],
               offline_store=str((parent/'arrays/train').resolve()), parent_checkpoint=str(previous))
    result = dict(seed=seed, round=round_number, updates=previous_updates+updates,
                  episodes=50*round_number, online_boundaries=len(online),
                  seconds=time.monotonic()-started)
    save(directory/f'online_r{round_number}.json', result)
    return result
