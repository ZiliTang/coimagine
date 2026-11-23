"""Matched V reweighting on complete, singleton-success, seven-expert initial panels."""
import argparse
import csv
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import cap_q_continuation as prior
from .cap_ga_effect_data import W, environment, fast, step, tensors, terminal
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_learner
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v, load_checkpoint

scale = prior.scale
OUT = scale.OUT.parent / 'cap_unique_initial_weight_20261119_01'
MODELS = OUT / 'models/GA_s17'
PANEL = scale.OUT / 'evaluation_panel.json'
CAPS = scale.CAPS
NETWORKS = {}


class Features:
    def __init__(self, rows):
        self.rows = rows

    def __getitem__(self, ids):
        if isinstance(ids, slice):
            ids = np.arange(*ids.indices(len(self.rows)))
        return self.rows.batch(ids, features_only=True)['x']


class Rows(prior.TrainingRows):
    def __init__(self, staging):
        super().__init__(staging)
        self.arrays = dict(self.data, x=Features(self))
        for field in ('root', 'boundary', 'y'):
            self.arrays[field] = np.concatenate([
                np.load(staging / str(part) / (field + '.npy'))
                for part in ('original', *scale.DOSES)
            ])

    def __len__(self):
        return int(self.ends[-1])


def train(staging):
    spec = dict(seed=17, v_updates=16000, q_updates=8000, dose=1001,
                baseline_q=prior.sha(prior.BASE / 'offline.pt'), baseline_v=prior.sha(prior.BASE / 'V.pt'),
                panel=prior.sha(PANEL), source=prior.sha(__file__), learner=prior.sha(full_learner.__file__),
                sampling='50 ordinary boundary rows plus two independent uniform singleton training roots with all seven expert rows per update; original 64-index stream, replace final 14; separate root RNG seed 1707',
                targets='binary MC expert outcome; one-step max legal expert V at factual successor; gamma=1; unchanged Q initialization and sampling rule',
                primary='paired greedy Q successes on non-009 singleton roots on archived 600 development diagnostic rooms; report all and 009 singleton as well',
                ranking='initial top legal V vs seven archived outcomes; Brier averaged over legal experts; winner minus 009 V on non-009 singleton roots',
                evaluation='single training seed, inspected held-out families, no checkpoint or dose selection')
    if (OUT / 'protocol.json').exists():
        assert spec == read(OUT / 'protocol.json')
    else:
        save(OUT / 'protocol.json', spec)
    rows = Rows(staging)
    a = rows.arrays
    initial = np.flatnonzero(a['boundary'] == 0)
    initial = initial[np.lexsort((a['cap'][initial], a['root'][initial]))].reshape(-1, 7)
    assert len(initial) == 608 + 546
    assert (a['cap'][initial] == np.arange(7)).all() and (a['root'][initial] == a['root'][initial[:, :1]]).all()
    unique = initial[a['y'][initial].sum(1) == 1]
    rng = np.random.default_rng(1707)
    counts = np.bincount(a['y'][unique].argmax(1), minlength=7)
    save(OUT / 'sampling.json', dict(roots=len(initial), unique=len(unique), counts=counts.tolist(),
                                     boundaries=len(rows), root_group_draws=32000))
    def sample(ids):
        ids[-14:] = unique[rng.integers(len(unique), size=2)].reshape(-1)
        return ids

    (OUT / 'arrays').mkdir(exist_ok=True)
    for path in ('normalizers.npz', 'dataset.json', 'arrays/validation'):
        if not (OUT / path).exists():
            (OUT / path).symlink_to(scale.PARENT / path)
    full_learner.train_offline(OUT, 'GA', 17, train_store=rows, v_sampler=sample)
    save(OUT / 'frozen.json', {name:prior.sha(MODELS / name) for name in ('offline.pt', 'V.pt')})


def initialize():
    scale.family.initialize()
    torch.set_num_threads(1)
    for label, path in [('base_v', prior.BASE/'V.pt'), ('v', MODELS/'V.pt'), ('q', MODELS/'offline.pt')]:
        NETWORKS[label] = load_checkpoint(path)[0]


@torch.inference_mode()
def run(case):
    path = OUT/'cases'/(case['case_id']+'.json')
    if path.exists():
        return read(path)
    import time
    start = time.monotonic()
    env = environment(case['fixture'])
    W.LIBRARY.reset()
    result = dict(case_id=case['case_id'])
    try:
        while not terminal(env):
            if time.monotonic()-start > 1800:
                raise TimeoutError(case['case_id'])
            packet = W.LIBRARY.advance(env.history)
            x, actions, sources = tensors(packet)
            if env.decision_count == 0:
                result['legal'] = sources[0].any(0).tolist()
                for name in ('base_v','v'):
                    result[name] = all_v(NETWORKS[name],x)[0].tolist()
            scores = NETWORKS['q'].values(x, actions, sources)[0,:len(packet['pool'])]
            assert scores.isfinite().all()
            step(env, packet['pool'][int(scores.argmax())])
        result.update(success=int(env.env.success), ticks=env.env.total_periods, decisions=env.decision_count,
                      history=env.history.sha256, stop=env.env.termination or 'budget')
        save(path,result)
        return result
    finally:
        env.close_reads()
        fast.ACTIVE = None


def evaluate(workers):
    assert prior.sha(PANEL) == read(OUT/'protocol.json')['panel']
    for name, digest in read(OUT/'frozen.json').items():
        assert prior.sha(MODELS/name) == digest
    cases = read(PANEL)
    assert len({r['case_id'] for r in cases}) == len(cases) == 600
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        for n, job in enumerate(as_completed([pool.submit(run, c) for c in cases]), 1):
            job.result()
            if n % 10 == 0:
                save(OUT/'evaluation_progress.json', dict(completed=n,total=len(cases)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fit','evaluate'))
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--staging', type=Path, default=OUT/'staging')
    args = parser.parse_args()
    if args.stage == 'fit':
        train(args.staging)
    else:
        evaluate(args.workers)
