"""Disk-backed real-transition tables for full offline and online learning."""
import gzip
import time
from pathlib import Path

import numpy as np

from ..memory_flow import MemoryHistory
from ..relational_transfer_features import action_encoding
from .dataset import read, save, flatten
from .runtime import CAPS, TraceLibrary, entries

LIBRARY = None


def initialize(bundle):
    global LIBRARY
    LIBRARY = TraceLibrary(entries(bundle))


def candidate_arrays(packet, library):
    n = len(packet['advice'])
    candidates = np.zeros((n, 85), dtype=np.float32)
    sources = np.zeros((n, n), dtype=np.float32)
    valid = np.zeros(n, dtype=np.bool_)
    for j, (action, supporters) in enumerate(zip(packet['pool'], packet['sources'])):
        valid[j] = True
        if supporters:
            candidates[j] = packet['advice'][supporters[0], :85]
            sources[j, supporters] = 1
        else:
            geometry = library.geometry
            candidates[j] = np.concatenate([action_encoding(action, geometry.goal_id, geometry.theta),
                                             geometry.candidate_geometry(action)])
    return candidates, sources, valid


def extract(job):
    bundle, out, row, number = job
    out = Path(out)
    path = out/'transition_shards'/(row['sha256']+'.npz')
    metadata = path.with_suffix('.json')
    if path.exists() and metadata.exists():
        return read(metadata)
    start = time.monotonic()
    with gzip.open(Path(bundle)/row['path'], 'rt') as f:
        data = __import__('json').load(f)
    history = MemoryHistory()
    values = {k: [] for k in ('x', 'candidates', 'sources', 'valid', 'chosen', 'duration')}
    own = CAPS.index(row['cap_id'])
    invalid = fallback = 0
    counts = {}
    for i, event in enumerate(data['public_rows'][:-1]):
        if (out/'STOP').exists():
            raise RuntimeError('STOP during extraction')
        history._append_projected(event['observation'], event['action'])
        packet = LIBRARY.advance(history)
        action = data['events'][i]['requested']
        if packet['proposals'][own] != action or action not in packet['pool']:
            raise ValueError('recorded behavior action changed: '+row['case_id']+'/'+row['cap_id'])
        candidates, sources, valid = candidate_arrays(packet, LIBRARY)
        values['x'].append(flatten(packet))
        values['candidates'].append(candidates)
        values['sources'].append(sources)
        values['valid'].append(valid)
        values['chosen'].append(packet['pool'].index(action))
        values['duration'].append(data['events'][i]['periods'])
        invalid += sum(not v for v in packet['legal'])
        fallback += int(packet['fallback'])
        counts[len(packet['pool'])] = counts.get(len(packet['pool']), 0) + 1
    n = len(values['x'])
    arrays = {k: np.asarray(v) for k, v in values.items()}
    arrays.update(y=np.full(n, float(row['success']), dtype=np.float32), cap=np.full(n, own, dtype=np.int64),
                  root=np.full(n, number, dtype=np.int64), boundary=np.arange(n, dtype=np.int32),
                  reward=np.zeros(n, dtype=np.float32), done=np.zeros(n, dtype=np.bool_))
    arrays['reward'][-1] = float(row['success'])
    arrays['done'][-1] = True  # Fixed 1800-tick OR 300-decision task, including budget truncations.
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.pending.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)
    stats = dict(case_id=row['case_id'], cap_id=row['cap_id'], boundaries=n, success=row['success'],
                 path=str(path), seconds=time.monotonic()-start, invalid_proposals=invalid,
                 fallback_boundaries=fallback, candidate_counts=counts, behavior_mismatches=0,
                 layout=dict(width=arrays['x'].shape[1], geometry_width=len(packet['g']),
                             advice_width=packet['advice'].size, trace_width=len(packet['trace']),
                             trace_widths=[s['width'] for s in LIBRARY.schemas]))
    save(metadata, stats)
    return stats


KEYS = ('x', 'candidates', 'sources', 'valid', 'chosen', 'duration', 'y', 'cap', 'root', 'boundary', 'reward', 'done')


def assemble(path, shards):
    """Atomic metadata publication; one mmap per field, never concatenate all RAM."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    n = sum(r['boundaries'] for r in shards)
    with np.load(shards[0]['path']) as first:
        arrays = {key: np.lib.format.open_memmap(path/(key+'.npy'), mode='w+',
                    dtype=first[key].dtype, shape=(n, *first[key].shape[1:])) for key in KEYS}
    following = np.lib.format.open_memmap(path/'next.npy', mode='w+', dtype=np.int64, shape=(n,))
    offset = 0
    for row in shards:
        with np.load(row['path']) as data:
            count = len(data['x'])
            for key in KEYS:
                arrays[key][offset:offset+count] = data[key]
            following[offset:offset+count] = np.arange(offset+1, offset+count+1)
            following[offset+count-1] = offset+count-1
            offset += count
    for array in list(arrays.values())+[following]:
        array.flush()
    record = dict(boundaries=n, episodes=len(shards), width=arrays['x'].shape[1])
    save(path/'manifest.json', record)
    return record


class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.metadata = read(self.path/'manifest.json')
        self.arrays = {k: np.load(self.path/(k+'.npy'), mmap_mode='r') for k in (*KEYS, 'next')}

    def __len__(self):
        return self.metadata['boundaries']

    def batch(self, ids, next_state=False):
        if next_state:
            ids = self.arrays['next'][ids]
        return {k: np.array(self.arrays[k][ids], copy=True) for k in ('x', 'candidates', 'sources', 'valid')}
