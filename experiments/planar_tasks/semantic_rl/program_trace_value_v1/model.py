"""Matched architectures and strict trace masking for continuation-value arms."""
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .dataset import read, save

ARMS = ('GA', 'LOCAL', 'GLOBAL')


def statistics(x):
    count = np.zeros(x.shape[1], dtype=np.float64)
    total = count.copy()
    square = count.copy()
    for start in range(0, len(x), 256):
        chunk = np.asarray(x[start:start+256], dtype=np.float64)
        valid = np.isfinite(chunk)
        chunk = np.where(valid, chunk, 0.)
        count += valid.sum(0)
        total += chunk.sum(0)
        square += (chunk*chunk).sum(0)
    mean = total / np.maximum(count, 1)
    scale = np.sqrt(np.maximum(square/np.maximum(count, 1)-mean*mean, 0.))
    return mean.astype(np.float32), np.maximum(scale, .1).astype(np.float32)


class ValueModel(nn.Module):
    def __init__(self, layout, mean, scale, arm):
        super().__init__()
        self.layout = layout
        self.arm = arm
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32).clone())
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32).clone())
        width = layout['width']
        self.cap_count = layout.get('cap_count', 7)
        mask = torch.ones(self.cap_count, 2*width)
        offset = width-layout['trace_width']
        if arm != 'GLOBAL':
            mask[:, offset:width] = 0
            mask[:, width+offset:] = 0
        if arm == 'LOCAL':
            for i, size in enumerate(layout['trace_widths']):
                mask[i, offset:offset+size] = 1
                mask[i, width+offset:width+offset+size] = 1
                offset += size
        self.register_buffer('channel_mask', mask)
        self.net = nn.Sequential(nn.Linear(2*width+self.cap_count, 64), nn.ReLU(),
                                 nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def encode(self, x):
        finite = torch.isfinite(x)
        normalized = torch.where(finite, (x-self.mean)/self.scale, 0.).clamp(-10., 10.)
        return torch.cat([normalized, finite.float()], dim=-1)

    def encoded_forward(self, z, cap):
        z = z*self.channel_mask[cap]
        identity = torch.nn.functional.one_hot(cap, self.cap_count).float()
        return self.net(torch.cat([z, identity], dim=-1)).squeeze(-1)

    def forward(self, x, cap):
        return self.encoded_forward(self.encode(x), cap)

    def all_values(self, x):
        x = torch.as_tensor(x, dtype=torch.float32).reshape(1, -1)
        return self.encoded_forward(self.encode(x).expand(self.cap_count, -1), torch.arange(self.cap_count))


def metrics(model, z, labels):
    predictions = []
    with torch.no_grad():
        for start in range(0, len(z), 256):
            predictions.append(model.encoded_forward(z[start:start+256], torch.as_tensor(labels['cap'][start:start+256])).numpy())
    p = np.concatenate(predictions)
    y = labels['y']
    roots = labels['root']
    per_root = [(float(np.mean((p[roots == r]-y[roots == r])**2)),
                 float(np.mean((p[roots == r] >= .5) == y[roots == r]))) for r in np.unique(roots)]
    initial = labels['boundary'] == 0
    root_init = []
    for r in np.unique(roots):
        ids = np.flatnonzero(initial & (roots == r))
        order = np.argsort(labels['cap'][ids])
        ids = ids[order]
        assert len(ids) == 7
        chosen = int(np.argmax(p[ids]))
        root_init.append(dict(root=int(r), values=p[ids].tolist(), outcomes=y[ids].astype(int).tolist(),
                              chosen_cap=chosen, chosen_success=int(y[ids[chosen]])))
    return dict(boundaries=len(y), brier=float(np.mean((p-y)**2)),
                boundary_accuracy=float(np.mean((p >= .5) == y)),
                root_equal_brier=float(np.mean([r[0] for r in per_root])),
                root_equal_accuracy=float(np.mean([r[1] for r in per_root])),
                mean_prediction=float(p.mean()), success_fraction=float(y.mean()),
                initial_brier=float(np.mean((p[initial]-y[initial])**2)), initial=root_init)


def train_all(out, seed=17, updates=2000, wall_seconds=1200):
    out = Path(out)
    torch.set_num_threads(1)
    dataset = read(out / 'dataset.json')
    layout = dataset['layout']
    train_x = np.load(out / 'arrays/train_x.npy', mmap_mode='r')
    val_x = np.load(out / 'arrays/validation_x.npy', mmap_mode='r')
    train_labels = dict(np.load(out / 'arrays/train_labels.npz'))
    val_labels = dict(np.load(out / 'arrays/validation_labels.npz'))
    mean, scale = statistics(train_x)
    prototype = ValueModel(layout, mean, scale, 'GLOBAL')
    encoded = []
    for x in (train_x, val_x):
        z = torch.empty((len(x), layout['width']*2), dtype=torch.float32)
        for i in range(0, len(x), 256):
            z[i:i+256] = prototype.encode(torch.tensor(x[i:i+256]))
        encoded.append(z)
    train_z, val_z = encoded
    cap = torch.as_tensor(train_labels['cap'])
    y = torch.as_tensor(train_labels['y'])
    results = {}
    for arm in ARMS:
        directory = out / 'fit' / arm
        directory.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ValueModel(layout, mean, scale, arm)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        started = time.monotonic()
        losses = []
        completed = 0
        for update in range(updates):
            if time.monotonic()-started >= wall_seconds or (out/'STOP').exists():
                break
            ids = rng.integers(len(train_z), size=64)
            optimizer.zero_grad(set_to_none=True)
            prediction = model.encoded_forward(train_z[ids], cap[ids])
            loss = torch.mean((prediction-y[ids])**2)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            losses.append(float(loss.detach()))
            completed = update + 1
            if completed % 100 == 0:
                progress = dict(arm=arm, updates=completed, seconds=time.monotonic()-started,
                                recent_mse=float(np.mean(losses[-100:])))
                save(directory/'progress.json', progress)
                print('FIT', progress, flush=True)
        checkpoint = dict(arm=arm, seed=seed, updates=completed, layout=layout,
                          model=model.state_dict(), optimizer=optimizer.state_dict(),
                          numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state())
        torch.save(checkpoint, directory/'model.pt')
        result = dict(arm=arm, seed=seed, updates=completed, complete=completed == updates,
                      seconds=time.monotonic()-started, parameters=sum(p.numel() for p in model.parameters()),
                      train=metrics(model, train_z, train_labels), validation=metrics(model, val_z, val_labels))
        save(directory/'accuracy.json', result)
        results[arm] = result
        if completed != updates:
            raise RuntimeError('fit stopped before planned updates; checkpoint preserved')
    save(out/'fit_results.json', results)
    return results


def load_model(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    state = checkpoint['model']
    model = ValueModel(checkpoint['layout'], state['mean'], state['scale'], checkpoint['arm'])
    model.load_state_dict(state)
    model.eval()
    return model
