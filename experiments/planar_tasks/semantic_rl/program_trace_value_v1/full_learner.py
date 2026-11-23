"""MC expert V -> one-step Q initialization -> real-transition Double DQN."""
import copy
import time
from pathlib import Path

import numpy as np
import torch

from .dataset import read, save
from .full_data import Store
from .model import ValueModel, statistics
from .q_model import QModel, all_v, tensors, double_dqn_target, load_checkpoint


def normalizers(store, layout, path):
    path = Path(path)
    if path.exists():
        return dict(np.load(path))
    mean, scale = statistics(store.arrays['x'])
    count = np.zeros(85, dtype=np.float64)
    total = count.copy()
    square = count.copy()
    g = layout['geometry_width']
    n_caps = layout.get('cap_count', 7)
    for start in range(0, len(store), 256):
        a = np.asarray(store.arrays['x'][start:start+256, g:g+n_caps*86]).reshape(-1, n_caps, 86)[:, :, :85].reshape(-1, 85).astype(np.float64)
        finite = np.isfinite(a)
        a = np.where(finite, a, 0)
        count += finite.sum(0)
        total += a.sum(0)
        square += (a*a).sum(0)
    cm = total/np.maximum(count, 1)
    cs = np.maximum(np.sqrt(np.maximum(square/np.maximum(count, 1)-cm*cm, 0)), .1)
    result = dict(mean=mean, scale=scale, candidate_mean=cm.astype(np.float32), candidate_scale=cs.astype(np.float32))
    np.savez(path, **result)
    return result


def checkpoint(path, model, optimizer, seed, updates, kind, rng, **extra):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = dict(kind=kind, arm=model.arm, layout=model.layout, seed=seed, updates=updates,
                 model=model.state_dict(), optimizer=optimizer.state_dict(), numpy_rng=rng.bit_generator.state,
                 torch_rng=torch.get_rng_state(), **extra)
    temporary = path.with_suffix('.pending.pt')
    torch.save(value, temporary)
    temporary.replace(path)


def value_metrics(model, store, regression_targets=None):
    predictions = np.empty(len(store), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(store), 256):
            end = min(start+256, len(store))
            x = torch.tensor(store.arrays['x'][start:end])
            cap = torch.tensor(store.arrays['cap'][start:end])
            predictions[start:end] = model(x, cap).numpy()
    if regression_targets is not None:
        squared = (predictions-regression_targets)**2
        initial = store.arrays['boundary'] == 0
        return dict(boundaries=len(store), target_mse=float(squared.mean()),
                    initial_target_mse=float(squared[initial].mean()))
    y = store.arrays['y']
    roots = store.arrays['root']
    squared = (predictions-y)**2
    correct = (predictions >= .5) == y
    initial = store.arrays['boundary'] == 0
    root_initial = []
    briers, accuracy = [], []
    for root in np.unique(roots):
        mask = roots == root
        briers.append(float(squared[mask].mean()))
        accuracy.append(float(correct[mask].mean()))
        ids = np.flatnonzero(mask & initial)
        ids = ids[np.argsort(store.arrays['cap'][ids])]
        assert len(ids) == model.cap_count
        chosen = int(np.argmax(predictions[ids]))
        root_initial.append(dict(root=int(root), scores=predictions[ids].tolist(),
                                 outcomes=y[ids].astype(int).tolist(), chosen=chosen,
                                 chosen_success=int(y[ids[chosen]])))
    return dict(boundaries=len(store), brier=float(squared.mean()), boundary_accuracy=float(correct.mean()),
                root_equal_brier=float(np.mean(briers)), root_equal_accuracy=float(np.mean(accuracy)),
                initial_brier=float(squared[initial].mean()), initial=root_initial,
                constant_train_rate=float(y.mean()))


def train_offline(out, arm, seed, v_updates=16000, q_updates=8000, *, train_store=None, v_sampler=None, evaluate_validation=True, v_targets=None):
    out = Path(out)
    directory = out/'models'/f'{arm}_s{seed}'
    if (directory/'offline_complete.json').exists():
        return read(directory/'offline_complete.json')
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    train = Store(out/'arrays/train') if train_store is None else train_store
    validation = Store(out/'arrays/validation') if evaluate_validation else None
    if v_targets is not None and (v_targets.shape != (len(train),) or
                                  not np.isfinite(v_targets).all() or
                                  np.any((v_targets < 0) | (v_targets > 1))):
        raise ValueError('one finite V target in [0, 1] required per training boundary')
    train_labels = train.arrays['y'] if v_targets is None else v_targets
    layout = read(out/'dataset.json')['layout']
    norms = normalizers(train, layout, out/'normalizers.npz')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    value = ValueModel(layout, norms['mean'], norms['scale'], arm)
    optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
    started = time.monotonic()
    losses = []
    for update in range(v_updates):
        if time.monotonic()-started > 3600 or (out/'STOP').exists():
            checkpoint(directory/'V_interrupted.pt', value, optimizer, seed, update, 'V', rng)
            raise RuntimeError('offline fit budget exhausted')
        ids = rng.integers(len(train), size=64)
        if v_sampler is not None:
            ids = v_sampler(ids)
        x = torch.tensor(train.arrays['x'][ids])
        cap = torch.tensor(train.arrays['cap'][ids])
        y = torch.tensor(train_labels[ids])
        optimizer.zero_grad(set_to_none=True)
        loss = ((value(x, cap)-y)**2).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite V loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(value.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (update+1) % 1000 == 0:
            progress = dict(arm=arm, seed=seed, stage='V', updates=update+1,
                            mse=float(np.mean(losses[-1000:])), seconds=time.monotonic()-started)
            save(directory/'progress.json', progress)
            print('FULL_FIT', progress, flush=True)
    checkpoint(directory/'V.pt', value, optimizer, seed, v_updates, 'V', rng)
    value.eval()
    fit_metrics = dict(train=value_metrics(value, train, v_targets), validation=value_metrics(value, validation) if validation is not None else None)
    save(directory/'V_metrics.json', fit_metrics)
    # The frozen V labeler sees only actual next public prefixes, never alternate futures.
    maximum = np.empty(len(train), dtype=np.float32)
    g = layout['geometry_width']
    n_caps = layout.get('cap_count', 7)
    with torch.no_grad():
        for start in range(0, len(train), 256):
            x = torch.tensor(train.arrays['x'][start:start+256])
            scores = all_v(value, x)
            legal = x[:, g:g+n_caps*86].reshape(-1, n_caps, 86)[:, :, -1] == 1
            scores = scores.masked_fill(~legal, -torch.inf)
            best = scores.max(1).values
            maximum[start:start+len(x)] = torch.where(legal.any(1), best, 0.).numpy()
    targets = np.asarray(train.arrays['reward']) + (~np.asarray(train.arrays['done'])).astype(np.float32)*maximum[train.arrays['next']]
    np.save(directory/'Q_init_targets.npy', targets)
    q = QModel(layout, norms['mean'], norms['scale'], arm, norms['candidate_mean'], norms['candidate_scale'])
    q.initialize_from_v(value)
    q_optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)
    # Reset sampling stream: paired arms see identical actual-transition indices.
    rng = np.random.default_rng(seed+100000)
    losses = []
    for update in range(q_updates):
        if time.monotonic()-started > 3600 or (out/'STOP').exists():
            checkpoint(directory/'Q_interrupted.pt', q, q_optimizer, seed, update, 'Q', rng)
            raise RuntimeError('offline Q fit budget exhausted')
        ids = rng.integers(len(train), size=64)
        batch = tensors(train.batch(ids))
        chosen = torch.as_tensor(np.array(train.arrays['chosen'][ids]), dtype=torch.long)
        q_optimizer.zero_grad(set_to_none=True)
        prediction = q.values(batch['x'], batch['candidates'], batch['sources']).gather(1, chosen[:, None]).squeeze(1)
        loss = ((prediction-torch.as_tensor(targets[ids]))**2).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite Q initialization loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.)
        q_optimizer.step()
        losses.append(float(loss.detach()))
        if (update+1) % 1000 == 0:
            progress = dict(arm=arm, seed=seed, stage='Q_init', updates=update+1,
                            mse=float(np.mean(losses[-1000:])), seconds=time.monotonic()-started)
            save(directory/'progress.json', progress)
            print('FULL_FIT', progress, flush=True)
    checkpoint(directory/'offline.pt', q, q_optimizer, seed, q_updates, 'Q', rng,
               v_updates=v_updates, online_updates=0, collected_online_episodes=0)
    result = dict(arm=arm, seed=seed, complete=True, v_updates=v_updates, q_updates=q_updates,
                  seconds=time.monotonic()-started, parameters=sum(p.numel() for p in q.parameters()),
                  q_init_recent_mse=float(np.mean(losses[-1000:])),
                  train_boundaries=len(train), validation_boundaries=len(validation) if validation is not None else 0)
    save(directory/'offline_complete.json', result)
    return result


def train_online_round(out, arm, seed, round_number, online_store_path, updates=1500):
    out = Path(out)
    directory = out/'models'/f'{arm}_s{seed}'
    previous = directory/('offline.pt' if round_number == 1 else f'online_r{round_number-1}.pt')
    q, state = load_checkpoint(previous)
    q.train()
    target = copy.deepcopy(q).eval()
    if 'target_model' in state:
        target.load_state_dict(state['target_model'])
    optimizer = torch.optim.Adam(q.parameters(), lr=3e-4)
    optimizer.load_state_dict(state['optimizer'])
    rng = np.random.default_rng(seed+200000) if round_number == 1 else np.random.default_rng()
    if round_number > 1:
        rng.bit_generator.state = state['numpy_rng']
    torch.set_rng_state(state['torch_rng'])
    offline = Store(out/'arrays/train')
    online = Store(online_store_path)
    previous_updates = state.get('online_updates', 0)
    losses = []
    started = time.monotonic()
    for update in range(updates):
        if (out/'STOP').exists():
            checkpoint(directory/f'online_r{round_number}_interrupted.pt', q, optimizer, seed, update, 'Q', rng,
                       target_model=target.state_dict(), online_updates=previous_updates+update)
            raise RuntimeError('STOP during online gradient updates')
        selected = [(offline, rng.integers(len(offline), size=32)), (online, rng.integers(len(online), size=32))]
        current_parts = [s.batch(ids) for s, ids in selected]
        next_parts = [s.batch(ids, next_state=True) for s, ids in selected]
        current = {k: np.concatenate([b[k] for b in current_parts]) for k in ('x', 'candidates', 'sources', 'valid')}
        following = {k: np.concatenate([b[k] for b in next_parts]) for k in ('x', 'candidates', 'sources', 'valid')}
        reward = torch.tensor(np.concatenate([s.arrays['reward'][ids] for s, ids in selected]))
        done = torch.tensor(np.concatenate([s.arrays['done'][ids] for s, ids in selected]))
        chosen = torch.tensor(np.concatenate([s.arrays['chosen'][ids] for s, ids in selected]), dtype=torch.long)
        current, following = tensors(current), tensors(following)
        label = double_dqn_target(q, target, following, reward, done)
        optimizer.zero_grad(set_to_none=True)
        prediction = q.values(current['x'], current['candidates'], current['sources']).gather(1, chosen[:, None]).squeeze(1)
        loss = ((prediction-label)**2).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite online loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
        if (previous_updates+update+1) % 250 == 0:
            target.load_state_dict(q.state_dict())
        if (update+1) % 500 == 0:
            progress = dict(arm=arm, seed=seed, stage='online', round=round_number,
                            updates=previous_updates+update+1, td_mse=float(np.mean(losses[-500:])))
            save(directory/'progress.json', progress)
            print('ONLINE_FIT', progress, flush=True)
    destination = directory/f'online_r{round_number}.pt'
    checkpoint(destination, q, optimizer, seed, updates, 'Q', rng, online_updates=previous_updates+updates,
               collected_online_episodes=50*round_number, target_model=target.state_dict(),
               replay_store=str(Path(online_store_path).resolve()), offline_store=str((out/'arrays/train').resolve()))
    result = dict(arm=arm, seed=seed, round=round_number, updates=previous_updates+updates,
                  online_boundaries=len(online), offline_boundaries=len(offline), seconds=time.monotonic()-started,
                  recent_td_mse=float(np.mean(losses[-500:])))
    save(directory/f'online_r{round_number}.json', result)
    return result
