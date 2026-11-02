"""Exact pre/post fast-path comparison on equal physical work, without PPO."""
from __future__ import annotations
import argparse,gc,hashlib,importlib.util,json,statistics,sys,time,tracemalloc
from pathlib import Path
from coimagine.random_start import env as E,efficient_profile as EP
from experiments.option_testbench.option_efficiency_v2_eval import cases
from experiments.option_testbench.follow_peer_eval import instance_from_dict


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module


def run(World,inst,idle=False):
    start=time.perf_counter()
    w=World(inst,E.EnvConfig(lidar_backend='numpy_batch',peer_clearance_recovery='boundary_v1'),skill_factory=EP.skill_factory,action_table=EP.ACTIONS,action_version=EP.VERSION)
    actions=(60,60) if idle else (next(a.index for a in EP.ACTIONS if a.label=='PUSH_N[0.30m]'),61)
    boundaries=[]
    while True:
        boundaries.append(w.joint_step(*actions,record=False))
        if not idle or w.termination:break
    wall=time.perf_counter()-start
    return w,boundaries,wall


def main(out):
    backup=out.parent/'baseline_sources'
    before_native=load('coimagine.continuous2d._eff_before_environment',backup/'continuous2d_environment.py')
    before=load('coimagine.random_start._eff_before_environment',backup/'env.py')
    before.Continuous2DEnv=before_native.Continuous2DEnv
    inst=instance_from_dict(next(c for c in cases() if c['name']=='push_follow')['instance'])
    variants={'before':before.RandomStartEnv,'after':E.RandomStartEnv};records=[]
    for idle in (False,True):
        for cls in variants.values():run(cls,inst,idle)
        for k in range(3):
            order=list(variants.items())[::1 if k%2==0 else -1];outputs={}
            for label,cls in order:
                w,b,wall=run(cls,inst,idle);outputs[label]=(w,b)
                records.append(dict(work='fixed90s_wait' if idle else 'same_push_follow',repeat=k,variant=label,wall=wall,periods=w.total_periods,physical_seconds=w.total_periods*.05))
            a,ba=outputs['before'];b,bb=outputs['after']
            assert ba==bb and a.env.backend.prefix==b.env.backend.prefix
            assert a.object_state()==b.object_state() and a.robot_states()==b.robot_states()
            assert a.reward_total==b.reward_total
    memory={}
    for label,cls in variants.items():
        gc.collect();tracemalloc.start();w,b,wall=run(cls,inst,False);current,peak=tracemalloc.get_traced_memory();tracemalloc.stop();memory[label]=dict(python_peak_bytes=peak)
    summary={}
    for work in ('same_push_follow','fixed90s_wait'):
        med={v:statistics.median(r['wall'] for r in records if r['work']==work and r['variant']==v) for v in variants}
        summary[work]=dict(medians=med,speedup=med['before']/med['after'])
    result=dict(records=records,summary=summary,memory=memory,exact_boundary_and_prefix_equality=True,
        scope='one persistent process, three repeats, equal actions/controller/menu/scanner/physics; record=False; no PPO/network/IPC; memory is tracemalloc Python allocation peak, not total RSS',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'rollout_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'}))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
