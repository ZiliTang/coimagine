"""Bounded run12 failure replays and independent lifecycle comparisons, no training."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from experiments.option_rl_v12.mode_env import ContinuousModeEnv as Baseline
from experiments.option_rl_v13.mode_env import ContinuousModeEnv as Fixed
from experiments.option_rl_v13 import controller as C
from experiments.option_testbench.effect_push_eval import instance_from_dict, make_case, reposition_peer_cases
from experiments.option_testbench.peer_clearance_eval import frame

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'outputs/option_rl_v12/continuous_5hz_pilot_20261029/run12_001'

def read(p):return json.loads(p.read_text())
def rp(direction):return next(a.index for a in C.ACTIONS if a.kind=='reposition' and a.direction==direction and a.speed==.12)

def cases():
    manifest=read(SOURCE/'manifest.json');ev=read(SOURCE/'evaluations/iteration_000890.json');out=[]
    for idx in [1,8,18,19]:
        root=ev['roots'][idx]
        first=next(j for j,a in enumerate(root['actions']) if any(e!='continuing' for e in a['exits']))
        primary=next(i for i in (0,1) if root['actions'][first]['exits'][i] not in ('continuing','peer_mode_failed'))
        commands=[[a['a0'],a['a1']] for a in root['actions'][:first+1]]
        same=commands[-1]
        for suffix in ['repeat','escape']:
            cmd=same.copy()
            if suffix=='escape':
                cmd[primary]=rp('CW' if idx==8 else 'CCW')
                if idx==8:cmd[1-primary]=C.STOP
            # Let the peer-failure propagation manifest before requesting escape.
            seq=commands+[same]*2+[cmd]*40
            out.append(dict(name=f'val_{idx:02d}_{suffix}',instance=manifest['validation_instances'][idx],
                commands=seq,source_root=idx,source_prefix_count=len(commands),primary_robot=primary,
                expected_prefix=root['actions'][:first+1]))
    blocked=next(c for c in reposition_peer_cases() if c['name']=='detour_rp_CCW_static_gap_0.26')
    out.append(dict(name='static_corridor_negative',instance=blocked['instance'],commands=[[rp('CCW'),C.STOP]]*80))
    single=make_case('healthy_single',offset=.18)
    push=next(a.index for a in C.ACTIONS if a.kind=='push' and a.type_index==2 and a.speed==.16)
    for name,base,cmd in [('healthy_single',single,[push,C.STOP]),('healthy_dual',make_case('healthy_dual',cooperative=True),[push,push])]:
        out.append(dict(name=name,instance=base['instance'],commands=[cmd]*25+[[C.STOP,C.STOP]]*10))
    return out

def run(case,variant):
    w=(Baseline if variant=='baseline' else Fixed)(instance_from_dict(case['instance']))
    initial=frame(w);boundaries=[];prefix_exact=True;start=time.monotonic();branch_start=None
    for j,cmd in enumerate(case['commands']):
        b=w.joint_step(*cmd,record=True);boundaries.append(b)
        if variant=='baseline' and j<case.get('source_prefix_count',0):
            old=case['expected_prefix'][j]
            prefix_exact &= abs(b['end_time']-old['time'])<1e-8 and [b['skill_ends'][i]['exit_reason'] for i in (0,1)]==old['exits']
        if j+1==case.get('source_prefix_count',-1):
            branch_start=[list(s.as_tuple()) for s in w.robot_states()]
        if w.termination:break
    rows=w.rows;first=initial['true']['object'];last=rows[-1]['true']['object'];phases=Counter();reasons=Counter()
    for b in boundaries:
        for end in b['skill_ends'].values():
            phases.update([end['stage']]);reasons.update([end['exit_reason']])
    refs=[[r[f'r{i}']['reference_velocity'] for r in rows] for i in (0,1)]
    gaps=[math.dist(r['true']['robots'][0][:2],r['true']['robots'][1][:2])-.36 for r in rows]
    metric=dict(case=case['name'],variant=variant,push_outcomes=[],seconds=w.total_periods*.05,
        decisions=len(boundaries),wall_seconds=time.monotonic()-start,
        exits=[s.skill.exit_reason for s in w.slots],controller_failed=[s.skill.done for s in w.slots],
        max_penetration=max(r['penetration'] for r in rows),min_peer_gap=min(gaps),
        max_force=max(math.hypot(*f) for r in rows for f in r['forces']),
        max_reference_step=max(math.dist(a,b) for ref in refs for a,b in zip([(0,0)]+ref,ref)),
        object_delta=[last[k]-first[k] for k in (0,1)],object_rotation_rad=last[2]-first[2],
        phase_counts=phases,exit_state_counts=reasons,termination=w.termination,
        original_prefix_exit_time_exact=prefix_exact if variant=='baseline' and 'source_root' in case else None,
        branch_robot_displacement=[math.dist(branch_start[i][:2],rows[-1]['true']['robots'][i][:2]) for i in (0,1)] if branch_start else None)
    assert prefix_exact
    return dict(case={k:v for k,v in case.items() if k!='expected_prefix'},variant=variant,instance=case['instance'],initial=initial,rows=rows,metric=metric,boundaries=boundaries)

def render(out,records):
    data=[]
    for r in records:
        frames=[]
        for row in [r['initial']]+r['rows']:
            ss=[]
            for i in (0,1):
                s=row.get(f'r{i}',{});e=s.get('effect') or {}
                ss.append(dict(stage=s.get('stage','initial'),skill=s.get('skill',''),exit=s.get('exit_reason'),
                    gap=s.get('gap'),travel=e.get('normal_travel'),request=None,
                    normal=e.get('face_normal'),contact_target=e.get('contact_target'),preferred=None,
                    pause_reason=s.get('pause_reason')))
            frames.append(dict(t=row['time'],o=row['true']['object'][:3],r=[x[:3] for x in row['true']['robots']],force=row['forces'],s=ss))
        data.append(dict(case=r['case']['name'],variant=r['variant'],metric=r['metric'],instance=r['instance'],frames=frames))
    template=Path(__file__).with_name('recovery_template.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')).replace('__WARNING__','Development diagnostics; no RL success claim.'))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=60.)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False);(a.out/'logs').mkdir();panel=cases();results=[];start=time.monotonic()
    sources=[Path(__file__),Path(__file__).with_name('mode_env.py'),Path(__file__).with_name('controller.py'),Path(__file__).with_name('push_controller.py'),Path(__file__).with_name('interface.py')]
    (a.out/'manifest.json').write_text(json.dumps(dict(source=str(SOURCE),cases=panel,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2))
    for c in panel:
        for v in ['baseline','fixed']:
            if time.monotonic()-start>a.seconds:raise TimeoutError('diagnostic batch exceeded declared wall budget')
            result=run(c,v);results.append(result);(a.out/'logs'/f"{c['name']}__{v}.json").write_text(json.dumps(result))
    render(a.out,results)
    metrics=[r['metric'] for r in results];(a.out/'metrics.json').write_text(json.dumps(metrics,indent=2))
    healthy=[]
    for name in ['healthy_single','healthy_dual']:
        x,y=[r for r in results if r['case']['name']==name]
        healthy.append(dict(case=name,same_physical_states_and_forces=len(x['rows'])==len(y['rows']) and all(a['true']==b['true'] and a['forces']==b['forces'] for a,b in zip(x['rows'],y['rows']))))
    (a.out/'healthy_equivalence.json').write_text(json.dumps(healthy,indent=2))
    print(json.dumps(dict(seconds=time.monotonic()-start,branches=len(results),healthy=healthy,metrics=metrics),indent=2))

if __name__=='__main__':main()
