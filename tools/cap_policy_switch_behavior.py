"""Measure how often the switching rollout departs from the opening expert's advice."""
import gzip
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from experiments.planar_tasks.authoring_v2.cap_policy_switch import OUT,scale
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save


def measure(row):
    folder=OUT/'evaluation'/row['case_id']
    cap=row['initial_expert']
    with gzip.open(folder/'result.json.gz','rb') as f:
        rollout=json.load(f)
    steps=rollout['decisions']
    assert len(steps)==row['decisions']
    changed=sum(cap not in d['sources'][d['chosen']] for d in steps)
    values=read(folder/'initial.json')['values']
    eligible=[i for supporters in steps[0]['sources'] for i in supporters]
    margin=None
    if cap is not None:
        other=[values[j] for j in eligible if j!=cap]
        if other:margin=values[cap]-max(other)
    return cap,changed,len(steps),margin


def main():
    with ThreadPoolExecutor(3) as pool:
        data=list(pool.map(measure,read(OUT/'evaluation.json')))
    margins=[m for _,_,_,m in data if m is not None]
    result=dict(initial_choices=dict(Counter('fallback' if c is None else scale.CAPS[c] for c,_,_,_ in data)),
                differing_episodes=sum(d>0 for _,d,_,_ in data),differing_actions=sum(d for _,d,_,_ in data),
                decisions=sum(n for _,_,n,_ in data),opening_top_margin_median=float(np.median(margins)),
                opening_top_margin_zero=sum(m==0 for m in margins))
    save(OUT/'behavior.json',result)
    return result


if __name__=='__main__':print(main())
