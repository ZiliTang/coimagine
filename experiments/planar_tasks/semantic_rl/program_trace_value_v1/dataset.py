"""Frozen root split and per-trajectory causal feature shards."""
import gzip
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ..memory_flow import MemoryHistory
from .runtime import CAPS, TraceLibrary, entries


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def split(bundle):
    scenes = read(Path(bundle) / 'scenes.json')
    groups = {}
    for row in scenes:
        f = row['fixture']
        content = dict(f)
        content.pop('scene_id', None)
        key = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
        n = len(f.get('movable_objects', [])) + 1
        groups.setdefault(n, []).append((hashlib.sha256(('2026111801:' + key).encode()).hexdigest(), row['case_id'], key))
    result = dict(seed=2026111801, train=[], validation=[], pilot_train=[], pilot_validation=[])
    contents = set()
    for n, group in sorted(groups.items()):
        group.sort()
        if len({r[2] for r in group}) != len(group):
            raise ValueError('duplicate content requires grouped split')
        assert not contents.intersection(r[2] for r in group)
        contents.update(r[2] for r in group)
        count = round(.2 * len(group))
        validation = [r[1] for r in group[:count]]
        train = [r[1] for r in group[count:]]
        result['validation'].extend(validation)
        result['train'].extend(train)
        result['pilot_validation'].extend(validation[:4])
        result['pilot_train'].extend(train[:8])
    assert len(result['pilot_train']) == 32 and len(result['pilot_validation']) == 16
    return result


def flatten(packet):
    return np.concatenate([packet['g'], packet['advice'].ravel(), packet['budget'], packet['trace']]).astype(np.float32)


_LIBRARY = None


def initialize(bundle):
    global _LIBRARY
    _LIBRARY = TraceLibrary(entries(bundle))


def extract(job):
    bundle, out, row, number = job
    out = Path(out)
    destination = out / 'shards' / (row['sha256'] + '.npz')
    metadata = destination.with_suffix('.json')
    if metadata.exists() and destination.exists():
        return read(metadata)
    started = time.monotonic()
    with gzip.open(Path(bundle) / row['path'], 'rt') as f:
        data = json.load(f)
    history = MemoryHistory()
    features = []
    pool_counts = []
    invalid = fallback = own_missing = 0
    times = []
    cap = CAPS.index(row['cap_id'])
    for i, event in enumerate(data['public_rows'][:-1]):
        if (out / 'STOP').exists():
            raise RuntimeError('STOP requested during extraction')
        # Only the current already-recorded public observation/action is exposed.
        history._append_projected(event['observation'], event['action'])
        packet = _LIBRARY.advance(history)
        expected = data['events'][i]['requested']
        if packet['proposals'][cap] != expected:
            own_missing += 1
        invalid += sum(not v for v in packet['legal'])
        fallback += int(packet['fallback'])
        pool_counts.append(len(packet['pool']))
        times.append(packet['seconds'])
        features.append(flatten(packet))
    if own_missing:
        raise ValueError(f"{row['case_id']}/{row['cap_id']}: {own_missing} behavior-CAP action mismatches")
    x = np.asarray(features, dtype=np.float32)
    n = len(x)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.pending.npz')
    np.savez_compressed(temporary, x=x, y=np.full(n, int(row['success']), dtype=np.float32),
                        cap=np.full(n, cap, dtype=np.int64), root=np.full(n, number, dtype=np.int64),
                        boundary=np.arange(n, dtype=np.int32))
    temporary.replace(destination)
    stats = dict(case_id=row['case_id'], cap_id=row['cap_id'], source_sha256=row['sha256'],
                 path=str(destination), boundaries=n, success=bool(row['success']),
                 candidate_counts=dict(zip(*[v.tolist() for v in np.unique(pool_counts, return_counts=True)])),
                 invalid_proposals=invalid, fallback_boundaries=fallback, behavior_mismatches=own_missing,
                 trace_seconds=sum(times), seconds=time.monotonic()-started, width=x.shape[1],
                 geometry_width=len(packet['g']), advice_width=packet['advice'].size,
                 trace_width=len(packet['trace']), trace_widths=[s['width'] for s in _LIBRARY.schemas])
    save(metadata, stats)
    return stats


def assemble(out, shards, roots):
    out = Path(out)
    arrays = out / 'arrays'
    arrays.mkdir(exist_ok=True)
    info = {}
    for name, selected in [('train', roots['pilot_train']), ('validation', roots['pilot_validation'])]:
        chosen = [r for r in shards if r['case_id'] in selected]
        count = sum(r['boundaries'] for r in chosen)
        width = chosen[0]['width']
        x = np.lib.format.open_memmap(arrays / (name + '_x.npy'), mode='w+', dtype=np.float32, shape=(count, width))
        labels = {key: [] for key in ('y', 'cap', 'root', 'boundary')}
        start = 0
        for row in chosen:
            with np.load(row['path']) as data:
                end = start + len(data['x'])
                x[start:end] = data['x']
                for key in labels:
                    labels[key].append(data[key])
                start = end
        x.flush()
        del x
        np.savez(arrays / (name + '_labels.npz'), **{k: np.concatenate(v) for k, v in labels.items()})
        info[name] = dict(boundaries=count, roots=len(selected), trajectories=len(chosen),
                          success_boundaries=int(sum(r['boundaries'] for r in chosen if r['success'])))
    first = shards[0]
    info['layout'] = {k: first[k] for k in ('width', 'geometry_width', 'advice_width', 'trace_width', 'trace_widths')}
    save(out / 'dataset.json', info)
    return info
