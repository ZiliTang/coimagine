"""Matched moving-option check for combined scanner and rollout materialization speedup."""
import argparse,json,statistics,hashlib
from pathlib import Path
from experiments.option_testbench.rollout_efficiency_benchmark import load,run
from experiments.option_testbench.option_efficiency_v2_eval import cases
from experiments.option_testbench.follow_peer_eval import instance_from_dict
from coimagine.random_start import env as E


def main(out):
    root=out.parent/'baseline_sources';native=load('coimagine.continuous2d._moving_old_native',root/'continuous2d_environment.py')
    old=load('coimagine.random_start._moving_old_env',root/'env.py');old.Continuous2DEnv=native.Continuous2DEnv
    def before(instance,config,**kw):config.lidar_backend='scalar';return old.RandomStartEnv(instance,config,**kw)
    inst=instance_from_dict(next(c for c in cases() if c['name']=='push_follow')['instance']);rows=[]
    for cls in (before,E.RandomStartEnv):run(cls,inst,False)
    for k in range(3):
        values={}
        for name,cls in [('before_scalar',before),('after_batch',E.RandomStartEnv)][::1 if k%2==0 else -1]:
            w,b,wall=run(cls,inst,False);values[name]=(w,b);rows.append(dict(repeat=k,variant=name,wall=wall,physical_seconds=w.total_periods*.05))
        a,ba=values['before_scalar'];b,bb=values['after_batch'];assert ba==bb and a.env.backend.prefix==b.env.backend.prefix
    med={v:statistics.median(r['wall'] for r in rows if r['variant']==v) for v in ('before_scalar','after_batch')}
    result=dict(records=rows,medians=med,speedup=med['before_scalar']/med['after_batch'],exact_physics_and_boundaries=True,
        scope='same moving PUSH+FOLLOW, efficient62 controller/menu on both; only scanner and sampling data materialization differ; three warm-process repeats; no PPO/network/IPC',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'moving_stack_benchmark.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
