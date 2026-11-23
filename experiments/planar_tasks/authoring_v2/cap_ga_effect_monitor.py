"""Publish counts from atomically completed collection shards."""
from pathlib import Path
import time
import numpy as np
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import save

OUT=Path('outputs/semantic_rl/cap_ga_effect_20261119_01')
if __name__=='__main__':
    seen=set()
    prefixes=transitions=mismatch=0
    while True:
        for p in (OUT/'samples').glob('*.npz'):
            if p.name.endswith('.tmp.npz') or p in seen:continue
            with np.load(p) as d:
                prefixes+=len(set(d['index']))
                transitions+=len(d['slot'])
                mismatch+=int(d['mismatch']>=0)
            seen.add(p)
        save(OUT/'live/counts.json',dict(rooms=len(seen),prefixes=prefixes,transitions=transitions,mismatch=mismatch))
        if len(seen)==608:break
        time.sleep(5)
