"""Bounded training readiness audit: historical invalid roots, new legality and first ticks."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import layout,spec
from experiments.option_rl_v45.scenes import sample_scenes
from experiments.option_rl_v11 import scenes
from experiments.option_rl_v11.interface import ACTIONS,make_env,observe,metadata


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--count',type=int,default=10000);p.add_argument('--physical-roots',type=int,default=64)
    p.add_argument('--seconds',type=float,default=90.);args=p.parse_args()
    if args.out.exists():raise ValueError('preserve existing preflight output')
    args.out.mkdir(parents=True);started=time.monotonic();old={};rejections=Counter();new=[]
    for split,seed,count in [('train',1,64),('test',2,16)]:
        roots=sample_scenes(seed,count,split);results=[dict(name=i.name,**scenes.geometry_audit(i)) for i in roots]
        old[split]=dict(count=count,invalid=sum(not x['legal'] for x in results),roots=results)
    minima=dict(robot_object=float('inf'),robot_wall=float('inf'),robot_pair=float('inf'),object_wall=float('inf'))
    complete=True
    for index in range(args.count):
        if time.monotonic()-started>args.seconds:complete=False;break
        i=scenes.scene(0,index,'train',rejection_counts=rejections);a=scenes.geometry_audit(i)
        assert a['legal'];new.append(i)
        minima['robot_object']=min(minima['robot_object'],*a['robot_object_gaps'])
        minima['robot_wall']=min(minima['robot_wall'],*a['robot_wall_gaps'])
        minima['robot_pair']=min(minima['robot_pair'],a['robot_robot_gap'])
        minima['object_wall']=min(minima['object_wall'],a['object_wall_gap'])
    validations=scenes.panel(0,64,'validation');tests=scenes.panel(0,64,'test')
    splits=layout.audit_split_content({'train':new,'validation':validations,'test':tests});assert splits['ok']
    physical=[];wait=next(a.index for a in ACTIONS if a.kind=='wait' and a.duration==2.)
    roots=(new[:args.physical_roots]+validations[:args.physical_roots]+tests[:args.physical_roots])
    for instance in roots:
        if time.monotonic()-started>args.seconds:complete=False;break
        env=make_env(instance);vector,m0,m1=observe(env)
        # Restrict this diagnostic window to ONE genuine physics tick; rewards/task
        # conditions for the actual training environment are not changed.
        from dataclasses import replace
        env.config=replace(env.config,joint_max_seconds=spec.CONTROL_DT)
        b=env.joint_step(wait,wait,record=True)
        physical.append(dict(name=instance.name,termination=b['termination'],periods=b['periods'],
            penetration=env.rows[0]['penetration'],finite=bool(np.isfinite(vector).all()),
            mask_sizes=[int(m0.sum()),int(m1.sum())],scene_hash=layout.content_hash(instance)))
    result=dict(metadata(),scene_version=scenes.VERSION,status='complete' if complete else 'budget_incomplete',
        legacy=old,requested_new_roots=args.count,audited_new_roots=len(new),new_invalid=0,min_clearances=minima,
        proposal_rejections=dict(rejections),split_audit=splits,physical_first_tick=physical,
        physical_anomalies=sum(r['termination'] in ('penetration_limit','numeric_failure','out_of_bounds') for r in physical),
        seconds=time.monotonic()-started)
    (args.out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.out/'validation_instances.json').write_text(json.dumps([i.as_dict() for i in validations],indent=1)+'\n')
    (args.out/'test_instances.json').write_text(json.dumps([i.as_dict() for i in tests],indent=1)+'\n')
    print(json.dumps({k:result[k] for k in ('status','audited_new_roots','new_invalid','min_clearances','physical_anomalies','seconds')}))
if __name__=='__main__':main()
