"""Train on nested, disk-backed partitions without re-copying all earlier doses."""
import argparse
import hashlib
import mmap
from pathlib import Path

import numpy as np

from . import cap_inverse_scale as study
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data, full_learner
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import LZ4, ResidentStore, pack

OUT, PARENT = study.OUT, study.PARENT


def rows_for(dose):
    prefix=read(OUT/'doses'/f'{dose}.json')
    rows=[]
    for b in range(1,prefix['batch']+1):
        rows.extend(read(OUT/'batches'/f'{b:03d}_results.json'))
    rows=rows[:prefix['runs']]
    assert study.stats(rows)=={k:prefix[k] for k in ('runs','success','distinct','families','boundaries')}
    return rows


def prepare(dose):
    part=OUT/'parts'/str(dose)
    dose_index=study.DOSES.index(dose)
    lower=read(OUT/'doses'/f'{study.DOSES[dose_index-1]}.json')['runs'] if dose_index else 0
    rows=rows_for(dose)[lower:]
    assert len(rows)%7==0 and all(r['cap']==study.CAPS[i%7] and r['shard']['behavior_mismatches']==0 for i,r in enumerate(rows))
    assert len({r['case_id'] for r in rows})*7==len(rows)
    if not (part/'manifest.json').exists():
        full_data.assemble(part,[r['shard'] for r in rows])
    size=read(part/'manifest.json')
    assert size['episodes']==len(rows) and size['boundaries']==sum(r['shard']['boundaries'] for r in rows)
    pack(part,part/'packed')


class PackedFeatures:
    """Row decoding only for random minibatches; sequential metrics read the raw contiguous mmap."""
    def __init__(self,store,packed):
        self.raw=store.arrays['x']
        meta=pack(store.path,packed)
        self.width=meta['shape'][1]
        self.codec=LZ4()
        self.blob=np.memmap(packed/'x.lz4rows',mode='r',dtype=np.uint8)
        self.blob._mmap.madvise(mmap.MADV_RANDOM)
        self.offsets=np.load(packed/'offsets.npy',mmap_mode='r')
        assert len(self.blob)==meta['packed_bytes'] and len(self.offsets)==len(store)+1

    def __len__(self):
        return len(self.raw)

    decode=ResidentStore.decode


class JoinedFeatures:
    def __init__(self,parts,ends):
        self.parts=parts
        self.ends=ends
        self.starts=np.r_[0,ends[:-1]]

    def __getitem__(self,ids):
        if isinstance(ids,slice):
            start,end,step=ids.indices(int(self.ends[-1]))
            assert step==1
            chunks=[p.raw[max(start-left,0):min(end-left,right-left)]
                    for p,left,right in zip(self.parts,self.starts,self.ends,strict=True) if start<right and end>left]
            return chunks[0] if len(chunks)==1 else np.concatenate(chunks)
        x=np.empty((len(ids),self.parts[0].width),dtype=np.float32)
        bins=np.searchsorted(self.ends,ids,side='right')
        for index in np.unique(bins):
            take=bins==index
            x[take]=self.parts[index].decode(ids[take]-self.starts[index])
        return x


class JoinedStore:
    def __init__(self,locations):
        self.stores=[full_data.Store(path) for path,_ in locations]
        self.ends=np.cumsum([len(p) for p in self.stores])
        self.starts=np.r_[0,self.ends[:-1]]
        pieces=[PackedFeatures(store,packed) for store,(_,packed) in zip(self.stores,locations,strict=True)]
        self.arrays={'x':JoinedFeatures(pieces,self.ends)}
        for key in ('cap','y','root','boundary','reward','done','chosen','next'):
            blocks=[p.arrays[key]+int(offset) if key=='next' else p.arrays[key]
                    for p,offset in zip(self.stores,self.starts,strict=True)]
            self.arrays[key]=np.concatenate(blocks)
        ids=np.unique(np.concatenate([self.starts,self.ends-1,np.random.default_rng(51).integers(len(self),size=64)]))
        packed=self.arrays['x'][ids]
        batch=self.batch(ids)
        for j,i in enumerate(ids):
            k=np.searchsorted(self.ends,i,side='right')
            local=i-self.starts[k]
            store=self.stores[k]
            assert packed[j].tobytes()==store.arrays['x'][local].tobytes()
            assert self.arrays['next'][i]==store.arrays['next'][local]+self.starts[k]
            for name in ('candidates','sources','valid'):
                assert batch[name][j].tobytes()==store.arrays[name][local].tobytes()

    def __len__(self):
        return int(self.ends[-1])

    def batch(self,ids,next_state=False):
        if next_state:ids=self.arrays['next'][ids]
        result={'x':self.arrays['x'][ids]}
        bins=np.searchsorted(self.ends,ids,side='right')
        for key in ('candidates','sources','valid'):
            a=self.stores[0].arrays[key]
            values=np.empty((len(ids),*a.shape[1:]),dtype=a.dtype)
            for k in np.unique(bins):
                take=bins==k
                values[take]=self.stores[k].arrays[key][ids[take]-self.starts[k]]
            result[key]=values
        return result


def fit(dose):
    folder=OUT/f'fit_{dose}'
    if (folder/'models/GA_s17/offline_complete.json').exists():return
    folder.mkdir(exist_ok=True,parents=True)
    original=PARENT/'arrays/train'
    cache_name=hashlib.sha256(str(original.resolve()).encode()).hexdigest()[:20]
    cached=study.family.BASE/'cap_trace_value_single_20261118_01/packed'/cache_name
    if not (cached/'manifest.json').exists():cached=OUT/'packed_original'
    locations=[(original,cached)]
    for d in study.DOSES:
        p=OUT/'parts'/str(d)
        if not (p/'packed/manifest.json').exists():prepare(d)
        locations.append((p,p/'packed'))
        if d==dose:break
    store=JoinedStore(locations)
    prefix=read(OUT/'doses'/f'{dose}.json')
    assert len(store)==len(store.stores[0])+prefix['boundaries']
    assert len(np.unique(store.arrays['root']))==608+prefix['neighbors']
    for file in ('dataset.json','normalizers.npz'):
        link=folder/file
        if not link.exists():link.symlink_to(PARENT/file)
        assert link.resolve()==(PARENT/file).resolve()
    source=full_learner.Store
    # Keep the production learner's numerical code, RNG stream and batch shapes.
    try:
        full_learner.Store=lambda path:store if path==folder/'arrays/train' else source(PARENT/'arrays/validation')
        full_learner.train_offline(folder,'GA',17)
    finally:
        full_learner.Store=source


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('dose',type=int,choices=study.DOSES)
    fit(parser.parse_args().dose)
