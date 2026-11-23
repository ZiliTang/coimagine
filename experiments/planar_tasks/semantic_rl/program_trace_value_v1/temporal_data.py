"""Causal eight-boundary windows over complete expert episodes."""
from pathlib import Path

import numpy as np
import torch

from .dataset import read, save
from .full_data import Store
from .temporal_q import H

FIELDS = ('root', 'cap', 'boundary', 'y', 'reward', 'done', 'chosen', 'candidates', 'sources')


def prepare(path, locations, layout, root_limit=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    locations = [Path(p) for p in locations]
    stores = [Store(p) for p in locations]
    selected = []
    for store in stores:
        roots = np.unique(store.arrays['root'])
        if root_limit is not None:
            roots = np.random.default_rng(197).choice(roots, root_limit, replace=False)
        selected.append(np.flatnonzero(np.isin(store.arrays['root'], roots)))
    spec = dict(locations=[str(p.resolve()) for p in locations], root_limit=root_limit,
                source_manifests=[s.metadata for s in stores], rows=sum(map(len, selected)))
    if (path/'manifest.json').exists():
        assert read(path/'manifest.json') == spec
        return SequenceStore(path)
    size = spec['rows']
    template = stores[0].arrays
    keys = (*FIELDS, 'raw_id', 'feature')
    arrays = {}
    for k in keys:
        shape, dtype = (template[k].shape[1:], template[k].dtype) if k in FIELDS else (
            ((layout.row,), np.float32) if k == 'feature' else ((), np.int64))
        arrays[k] = np.lib.format.open_memmap(path/(k+'.npy'), mode='w+', dtype=dtype, shape=(size,*shape))
    pos = 0
    for ids,store in zip(selected, stores, strict=True):
        a = store.arrays
        for left in range(0, len(ids), 256):
            ix = ids[left:left+256]
            sl = slice(pos, pos+len(ix))
            for k in FIELDS:
                arrays[k][sl] = a[k][ix]
            arrays['raw_id'][sl] = ix
            middle = a['boundary'][ix] > 0
            prev = ix[middle]-1
            assert np.all(a['root'][prev]==a['root'][ix[middle]]) and np.all(a['cap'][prev]==a['cap'][ix[middle]])
            assert np.array_equal(a['boundary'][prev]+1, a['boundary'][ix[middle]])
            assert np.array_equal(a['next'][prev], ix[middle]) and not a['done'][prev].any()
            support = np.zeros((len(ix),7), np.float32)
            periods = np.zeros((len(ix),1), np.float32)
            support[middle] = a['sources'][prev,a['chosen'][prev]]
            periods[middle,0] = a['duration'][prev]/40.
            raw = a['x'][ix]
            arrays['feature'][sl] = layout.compact(raw, support, periods)
            pos += len(ix)
    assert pos == size
    assert np.array_equal(np.flatnonzero(arrays['boundary']==0)[1:],np.flatnonzero(arrays['done'])[:-1]+1)
    assert arrays['boundary'][0]==0 and arrays['done'][-1]
    for array in arrays.values():
        array.flush()
    save(path/'manifest.json', spec)
    return SequenceStore(path)


class SequenceStore:
    def __init__(self,path):
        self.a={k:np.load(path/(k+'.npy'),mmap_mode='r') for k in (*FIELDS,'raw_id','feature')}

    def __len__(self):
        return len(self.a['boundary'])

    def next(self,ids):
        return ids+(~self.a['done'][ids])

    def window(self,ids,frame):
        starts=ids-self.a['boundary'][ids]
        indices=ids[:,None]-np.arange(H-1,-1,-1)
        exists=indices>=starts[:,None]
        hist=self.a['feature'][np.maximum(indices,starts[:,None])]
        hist[~exists]=np.nan
        return torch.from_numpy(hist),torch.from_numpy(self.a['feature'][starts,:frame]),torch.from_numpy(exists)
