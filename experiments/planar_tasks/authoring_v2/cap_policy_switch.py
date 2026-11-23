"""Evaluate greedy switching among seven frozen expert-continuation values."""
import multiprocessing as mp
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

from . import cap_inverse_scale as scale
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v, load_checkpoint

OUT = scale.OUT.parent/'cap_policy_switch_20261119_01'
MODEL = scale.OUT/'fit_1001/models/GA_s17'
V = str(MODEL/'V.pt')


class PolicySwitch:
    def __init__(self, network):
        self.network = network
        self.initial_file = None

    def values(self, x, candidates, sources):
        values = all_v(self.network, x)
        support = sources.bool()
        maximum = values[:, None].masked_fill(~support, -1.).amax(dim=-1)
        # For an empty proposal pool the default public fallback is the sole valid action.
        if self.initial_file is not None:
            pool = int(maximum[0].argmax())
            cap = int(values[0].masked_fill(~support[0, pool], -1.).argmax()) if support[0, pool].any() else None
            save(self.initial_file, dict(expert=cap, values=values[0].tolist(), candidate=pool))
            self.initial_file = None
        return maximum


def initialize():
    scale.family.initialize()
    network, state = load_checkpoint(V)
    assert state['kind']=='V' and state['arm']=='GA' and state['seed']==17 and state['updates']==16000
    fast_runtime.W.MODELS[V]=PolicySwitch(network)


def evaluate(case):
    directory=OUT/'evaluation'/case['case_id']
    fast_runtime.W.MODELS[V].initial_file=directory/'initial.json'
    row=scale.family.worker(dict(case=case, condition='ga_v_switch', checkpoint=V, epsilon=0., directory=str(directory)))
    assert row['complete'] and row['error'] is None
    return dict(case_id=row['case_id'], success=int(row['success']),decisions=row['decisions'],ticks=row['ticks'],
                initial_expert=read(directory/'initial.json')['expert'])


def main():
    protocol=read(scale.OUT/'protocol.json')
    panel=scale.OUT/'evaluation_panel.json'
    assert sha(panel)==protocol['files']['evaluation_panel.json']
    frozen=read(scale.OUT/'frozen.json')['1001']
    assert sha(MODEL/'offline.pt')==frozen
    spec=dict(seed=17, dose=1001, v_sha256=sha(V), q_sha256=frozen,panel_sha256=sha(panel),
              inputs='same GA geometry, proposals, candidate set and boundaries as GA-Q',
              v='recompute all seven expert V estimates at every boundary; each legal pool action takes maximum over its supporters; greedy first pool tie',
              primary='switch V minus offline Q on all 600 paired scenes',
              secondary='exactly one CAP, union, outside; reselecting vs committing to the first selected expert using recorded single CAP outcomes',
              uncertainty='10000 scene-paired bootstrap draws, stratified by movable count; conditional on seed17',
              optional_fallback='if no expert proposes a legal action at the opening, omit that scene only from commitment comparison')
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==spec
    else: save(OUT/'protocol.json',spec)
    cases=read(panel)
    assert len({r['case_id'] for r in cases})==len(cases)==600
    if not (OUT/'evaluation.json').exists():
        completed=[]
        save(OUT/'progress.json',dict(completed=0,total=len(cases)))
        with ProcessPoolExecutor(8,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
            for future in as_completed([pool.submit(evaluate,c) for c in cases]):
                completed.append(future.result())
                save(OUT/'progress.json',dict(completed=len(completed),total=len(cases)))
        save(OUT/'evaluation.json',sorted(completed,key=lambda r:r['case_id']))
    from tools.cap_policy_switch_report import report
    report()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        save(OUT/'error.json',dict(error=traceback.format_exc()))
        raise
