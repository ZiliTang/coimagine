"""Same Double-DQN updates with externally provided lossless resident replay.

Continuation accepts the saved interrupted optimizer/RNG at any update boundary.
No minibatch order, model arithmetic, or reward change is introduced.
"""
import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from .dataset import save
from .full_learner import checkpoint
from .q_model import load_checkpoint, tensors, double_dqn_target


def train_round(previous, out, seed, round_number, offline, online, updates=1500, log_every=100):
    out = Path(out)
    directory = out/'models'/f'GLOBAL_s{seed}'
    q, state = load_checkpoint(previous)
    q.train()
    target = copy.deepcopy(q).eval()
    target.load_state_dict(state['target_model'])
    optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)
    optimizer.load_state_dict(state['optimizer'])
    rng = np.random.default_rng()
    rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
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
            sample_started = time.perf_counter()
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
            sample_seconds = time.perf_counter()-sample_started
            compute_started = time.perf_counter()
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
                window.append(dict(sample_seconds=sample_seconds, compute_seconds=time.perf_counter()-compute_started, td_mse=float(loss), offline_td_mse=float(squared[:32].mean()),
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
               target_model=target.state_dict(), replay_partitions=[str(s.path) for s in online.stores],
               offline_store=str(offline.path), parent_checkpoint=str(previous))
    result = dict(seed=seed, round=round_number, updates=previous_updates+updates,
                  episodes=50*round_number, online_boundaries=len(online),
                  seconds=time.monotonic()-started)
    save(directory/f'online_r{round_number}.json', result)
    return result
