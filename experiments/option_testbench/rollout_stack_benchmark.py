"""Equal 90 simulated seconds: old scalar/full logging path vs efficient batch fast path."""
from pathlib import Path
import argparse,json,statistics,time,hashlib
from experiments.option_testbench.rollout_efficiency_benchmark import load,run
from experiments.option_testbench.option_efficiency_v2_eval import cases
from experiments.option_testbench.follow_peer_eval import instance_from_dict
from coimagine.random_start import env as E


def main(out):
    root=out.parent/'baseline_sources'
    native=load('coimagine.continuous2d._stack_old_native',root/'continuous2d_environment.py')
    old=load('coimagine.random_start._stack_old_env',root/'env.py');old.Continuous2DEnv=native.Continuous2DEnv
    def before(instance,config,**kw):
        config.lidar_backend='scalar'
        return old.RandomStartEnv(instance,config,**kw)
    inst=instance_from_dict(next(c for c in cases() if c['name']=='push_follow')['instance']);rows=[]
    for cls in (before,E.RandomStartEnv):run(cls,inst,True)
    for repeat in range(3):
        result={}
        for label,cls in [('before_scalar',before),('after_batch',E.RandomStartEnv)][::1 if repeat%2==0 else -1]:
            w,b,wall=run(cls,inst,True);result[label]=(w,b)
            rows.append(dict(repeat=repeat,variant=label,wall=wall,physical_seconds=w.total_periods*.05,periods=w.total_periods))
        a,ba=result['before_scalar'];b,bb=result['after_batch']
        assert ba==bb and a.env.backend.prefix==b.env.backend.prefix
        assert a.object_state()==b.object_state() and a.robot_states()==b.robot_states()
    medians={k:statistics.median(r['wall'] for r in rows if r['variant']==k) for k in ('before_scalar','after_batch')}
    data=dict(records=rows,medians=medians,speedup=medians['before_scalar']/medians['after_batch'],identical_physical_work=True,
        scope='same 90s/1800 periods and efficient62 controller/menu on both; only sensor computation and diagnostic/record materialization differ; no PPO/network/IPC; three warm-process repeats',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/'rollout_stack_benchmark.json').write_text(json.dumps(data,indent=2)+'\n');print(json.dumps(data))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
