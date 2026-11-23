"""Replay only the six index samplers; check totals against the training log."""
import argparse
from pathlib import Path
import numpy as np
from experiments.planar_tasks.authoring_v2.cap_retention_supervised import ARMS,SEEDS,STEPS,TRAIN,Sampler
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save
from tools.cap_trace_value_long_report import write_csv


def main(out):
    store=Store(TRAIN);store.decode=lambda ids:np.asarray(store.arrays['x'][ids])
    sampler=Sampler(store)
    rows=[]
    for arm in ARMS:
        for seed in SEEDS:
            rng=np.random.default_rng(seed)
            ix=np.concatenate([sampler.batch(arm,rng) for _ in range(STEPS)])
            a=store.arrays;bd=a['boundary'][ix];y=a['y'][ix]
            baseline=read(out/f'{arm}_s{seed}_samples.json')
            assert np.bincount(a['root'][ix],minlength=760).tolist()==baseline['root_draws']
            assert int(y.sum())==baseline['positive']
            rows.append(dict(arm=arm,seed=seed,draws=len(ix),distinct_rows=len(np.unique(ix)),
                        initial_fraction=float((bd==0).mean()),first_five_fraction=float((bd<5).mean()),
                        median_boundary=float(np.median(bd)),late_20plus_fraction=float((bd>=20).mean()),
                        success_label_fraction=float(y.mean())))
    write_csv(out/'sample_exposure.csv',rows)
    print(rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
