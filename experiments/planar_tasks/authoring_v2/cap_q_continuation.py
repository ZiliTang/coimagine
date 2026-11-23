"""Matched own-expert versus max-expert continuation on frozen GA-Q transitions."""
import argparse
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import cap_inverse_scale as scale
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_learner
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel, all_v, load_checkpoint, tensors
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ResidentStore, LZ4

OUT = scale.OUT.parent/'cap_q_continuation_20261119_01'
BASE = scale.OUT/'fit_1001/models/GA_s17'
Q = OUT/'own.pt'
SEED = 17


class Partition:
    """Read the existing losslessly packed rows, including relocated archives."""
    decode = ResidentStore.decode

    def __init__(self, arrays, packed):
        self.arrays = {k:np.load(arrays/(k+'.npy'), mmap_mode='r') for k in
                       ('candidates','sources','valid','chosen','cap','next','done','reward')}
        meta = read(packed/'manifest.json')
        self.width = meta['shape'][1]
        self.offsets = np.load(packed/'offsets.npy', mmap_mode='r')
        self.blob = np.memmap(packed/'x.lz4rows', mode='r', dtype=np.uint8)
        self.codec = LZ4()
        assert len(self)==meta['shape'][0] and self.offsets[-1]==len(self.blob)
        assert meta['every_row_bitwise_verified']

    def __len__(self):
        return len(self.arrays['done'])


class TrainingRows:
    def __init__(self, scratch):
        original = scale.PARENT/'arrays/train'
        cache_id = hashlib.sha256(str(original).encode()).hexdigest()[:20]
        packed = scale.OUT.parent/'cap_trace_value_single_20261118_01/packed'/cache_id
        self.parts = [Partition(scratch/'original', packed)]
        self.parts.extend(Partition(scratch/str(d),scratch/str(d)/'packed') for d in scale.DOSES)
        self.ends = np.cumsum([len(p) for p in self.parts])
        self.starts = np.r_[0,self.ends[:-1]]
        self.data = {k:np.concatenate([p.arrays[k]+start if k=='next' else p.arrays[k]
                                       for p,start in zip(self.parts,self.starts)])
                     for k in ('next','cap','done','reward','chosen')}
        assert self.ends[-1]==read(BASE/'offline_complete.json')['train_boundaries']

    def batch(self, ids, features_only=False):
        fields = ['x'] if features_only else ['x','candidates','sources','valid']
        result = {}
        bins = np.searchsorted(self.ends,ids,side='right')
        for k in np.unique(bins):
            part = self.parts[k]
            take = bins==k
            local = ids[take]-self.starts[k]
            for field in fields:
                values = part.decode(local) if field=='x' else part.arrays[field][local]
                if field not in result:
                    result[field] = np.empty((len(ids),*values.shape[1:]),dtype=values.dtype)
                result[field][take] = values
        return result


def protocol():
    reference = read(scale.OUT.parent/'cap_policy_switch_20261119_01/protocol.json')
    for path,key in ((BASE/'V.pt','v_sha256'),(BASE/'offline.pt','q_sha256'),
                     (scale.OUT/'evaluation_panel.json','panel_sha256')):
        assert sha(path)==reference[key]
    assert sha(scale.OUT/'per_scene.csv')==read(scale.OUT/'FINISHED.json')['files']['per_scene.csv']
    spec = dict(seed=SEED, dose=1001, v=reference['v_sha256'], baseline=reference['q_sha256'],
                panel=reference['panel_sha256'], baseline_rows=sha(scale.OUT/'per_scene.csv'),
                archived_targets=sha(BASE/'Q_init_targets.npy'), q_updates=8000, batch=64,
                objective='MSE on executed deduplicated action; Adam 0.0003; gamma=1; same saved V RNG state',
                treatment='only nonterminal legal continuation changes: max legal V versus behavior CAP V',
                primary='paired closed-loop success, max minus own, all 600; 10000 bootstrap draws stratified by movable count',
                secondary='unique expert, expert union, outside union; conditional on seed17; previously inspected panel')
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==spec
    else:
        save(OUT/'protocol.json',spec)


def targets(store):
    value,_ = load_checkpoint(BASE/'V.pt')
    n = store.ends[-1]
    cap = store.data['cap']
    current_max, current_own = np.empty(n,np.float32), np.empty(n,np.float32)
    own_legal = np.empty(n,bool)
    offset = value.layout['geometry_width']
    with torch.no_grad():
        for i in range(0,n,256):
            ids = np.arange(i,min(i+256,n))
            x = torch.from_numpy(store.batch(ids,True)['x'])
            v = all_v(value,x)
            legal = x[:,offset:offset+602].reshape(-1,7,86)[:,:,-1]==1
            best = v.masked_fill(~legal,-torch.inf).amax(1)
            current_max[ids] = torch.where(legal.any(1),best,0.).numpy()
            current_own[ids] = v[np.arange(len(ids)),cap[ids]].numpy()
            own_legal[ids] = legal[np.arange(len(ids)),cap[ids]].numpy()
    data = store.data
    live = ~data['done']
    nxt = data['next']
    assert np.all(cap[live]==cap[nxt[live]]) and own_legal[nxt[live]].all()
    y = data['reward'] + live*current_own[nxt]
    ref = np.load(BASE/'Q_init_targets.npy')
    assert np.array_equal(data['reward']+live*current_max[nxt], ref)
    assert np.array_equal(y[~live],ref[~live])
    diff = (ref-y)[live]
    assert diff.min()>=0
    np.save(OUT/'own_targets.npy',y)
    save(OUT/'targets.json',dict(boundaries=int(n), terminal=int((~live).sum()),
         nonterminal=int(live.sum()),max_targets_exact=True,positively_changed=int((diff>0).sum()),
         change_above_01=int((diff>.1).sum()), mean_gap=float(diff.mean()), median_gap=float(np.median(diff)),
         p90_gap=float(np.quantile(diff,.9)), max_gap=float(diff.max()), own_sha256=sha(OUT/'own_targets.npy')))


def fit(store, arm):
    v,state = load_checkpoint(BASE/'V.pt')
    norm = np.load(scale.PARENT/'normalizers.npz')
    torch.set_rng_state(state['torch_rng'])
    q = QModel(v.layout,norm['mean'],norm['scale'],'GA',norm['candidate_mean'],norm['candidate_scale'])
    q.initialize_from_v(v)
    opt = torch.optim.Adam(q.parameters(),lr=3e-4)
    rng = np.random.default_rng(SEED+100000)
    y = np.load(BASE/'Q_init_targets.npy' if arm=='max' else OUT/'own_targets.npy')
    for step in range(1,8001):
        ids = rng.integers(store.ends[-1],size=64)
        b = tensors(store.batch(ids))
        chosen = torch.from_numpy(store.data['chosen'][ids])
        opt.zero_grad(set_to_none=True)
        pred = q.values(b['x'],b['candidates'],b['sources']).gather(1,chosen[:,None]).squeeze(1)
        loss = ((pred-torch.as_tensor(y[ids]))**2).mean()
        if not torch.isfinite(loss):
            raise ValueError('nonfinite loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(q.parameters(),1.)
        opt.step()
        if step%500==0:
            save(OUT/(arm+'_progress.json'),dict(updates=step,last_batch_mse=loss.item()))
            print(arm,step,flush=True)
    if arm=='max':
        frozen,_ = load_checkpoint(BASE/'offline.pt')
        assert all(torch.equal(t,frozen.state_dict()[k]) for k,t in q.state_dict().items())
        save(OUT/'reproduction.json',dict(exact_state_dict=True,updates=8000))
    else:
        full_learner.checkpoint(Q,q,opt,SEED,8000,'Q',rng,v_updates=16000,online_updates=0,
                                continuation='own',collected_online_episodes=0)
        save(OUT/'own_frozen.json',dict(sha256=sha(Q),updates=8000))


def evaluate():
    assert sha(Q)==read(OUT/'own_frozen.json')['sha256']
    cases = read(scale.OUT/'evaluation_panel.json')
    assert len({c['case_id'] for c in cases})==len(cases)==600
    rows=[]
    with ProcessPoolExecutor(10,mp_context=mp.get_context('spawn'),initializer=scale.family.initialize) as pool:
        jobs=[pool.submit(scale.family.worker,dict(case=c,condition='own',checkpoint=str(Q),epsilon=0.,
              directory=str(OUT/'evaluation'/c['case_id']))) for c in cases]
        for f in as_completed(jobs):
            r = f.result()
            assert r['complete'] and r['error'] is None
            rows.append({k:r[k] for k in ('case_id','success','ticks','decisions','stop')})
            if len(rows)%10==0:
                save(OUT/'evaluation_progress.json',dict(complete=len(rows),total=len(cases)))
    save(OUT/'evaluation.json',sorted(rows,key=lambda r:r['case_id']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['targets','own','max','evaluate'])
    parser.add_argument('--scratch',type=Path)
    args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    protocol()
    torch.set_num_threads(1)
    if args.stage=='evaluate':
        evaluate()
    else:
        rows=TrainingRows(args.scratch)
        if args.stage=='targets':
            targets(rows)
        else:
            fit(rows,args.stage)
