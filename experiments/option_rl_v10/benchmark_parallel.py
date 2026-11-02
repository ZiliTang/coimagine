"""Actual fixed-snapshot full-episode throughput; identical serial and worker jobs."""
from __future__ import annotations
import json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
import torch
from experiments.option_rl_v10 import parallel,scenes
from experiments.option_rl_v10.interface import Policy,make_env,observe
from experiments.option_rl_v45.train import RunningMeanStd


def main():
    torch.set_num_threads(1);torch.manual_seed(0)
    out=ROOT/'outputs/option_rl_v10/parallel_benchmark_001'
    out.mkdir(exist_ok=False);instances=scenes.panel(0,8,'train')
    dim=len(observe(make_env(instances[0]))[0]);policy=Policy(dim);stats=RunningMeanStd(dim)
    snap=parallel.snapshot(policy,stats)
    jobs=[dict(instance=i,snapshot=snap,seed=parallel.behaviour_seed(0,i.index),collect_raw=True) for i in instances]
    started=time.monotonic();serial=parallel.collect(None,jobs);serial_seconds=time.monotonic()-started
    started=time.monotonic()
    with parallel.pool(8) as executor:
        spawned=parallel.collect(executor,jobs);cold=time.monotonic()-started
        started=time.monotonic();warm=parallel.collect(executor,jobs);warm_seconds=time.monotonic()-started
    max_error=0.
    for a,b,c in zip(serial,spawned,warm):
        for key in ('return','success','progress','periods','actions'):
            assert a['summary'][key]==b['summary'][key]==c['summary'][key],(key,a['summary']['name'])
        for x,y,z in zip(a['steps'],b['steps'],c['steps']):
            max_error=max(max_error,abs(x.logp-y.logp),abs(x.value-y.value),abs(x.logp-z.logp),abs(x.value-z.value))
    result=dict(episodes=8,physical_horizon_seconds=90,workers=8,serial_seconds=serial_seconds,
        parallel_cold_seconds=cold,parallel_warm_seconds=warm_seconds,
        cold_speedup=serial_seconds/cold,warm_speedup=serial_seconds/warm_seconds,
        max_logp_value_error=max_error,worker_pids=sorted(set(r['summary']['worker_pid'] for r in spawned)),
        physics_returns_actions_equal=True,normalization='same frozen snapshot for serial and parallel',
        snapshot_hash=snap['hash'],results=[r['summary'] for r in spawned])
    (out/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='results'}),flush=True)
if __name__=='__main__':main()
