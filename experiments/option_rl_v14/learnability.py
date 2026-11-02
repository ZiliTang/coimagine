"""Explicit developer tasks, witnessed controls, and small position perturbations.

Witnesses are offline diagnostics using simulator distance to choose braking.
They are neither policy inputs nor PPO demonstrations. Every attempted case is
retained, including failures; the training panel is declared before the audit.
"""
from dataclasses import replace
from pathlib import Path
import argparse,json,math,time
import numpy as np
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import layout,spec
from experiments.option_testbench.effect_push_eval import make_case
from experiments.option_testbench.peer_clearance_eval import frame
from experiments.option_rl_v13.diagnose_recovery import render as render_witnesses
from experiments.option_rl_v14.controller import ACTIONS,make_env,mask
from experiments.option_rl_v14.render_episodes import instance_from_dict


def action(kind,*,direction=None,speed=.16):
    return next(a.index for a in ACTIONS if a.kind==kind and
        (kind=='wait' or (a.direction==direction if kind=='reposition' else a.type_index==direction) and abs(a.speed-speed)<1e-9))


def proposals():
    cases=[]
    for category in ('contact','approach','dual','reposition'):
        for j,direction in enumerate((2,0)):
            c=make_case(f'{category}_{direction}',direction=direction,cooperative=category=='dual')
            inst=instance_from_dict(c['instance']);n=(math.cos(direction*math.pi/4),math.sin(direction*math.pi/4))
            gap=.008 if category=='contact' else .10 if category=='approach' else .10
            shift=.405-(spec.OBJECT_HEIGHT/2+spec.ROBOT_RADIUS+gap)
            robots=list(inst.robot_poses)
            robots[0]=replace(robots[0],x=robots[0].x+shift*n[0],y=robots[0].y+shift*n[1])
            if category=='dual':robots[1]=replace(robots[1],x=robots[1].x+shift*n[0],y=robots[1].y+shift*n[1])
            target=(direction+4)%8 if category=='reposition' else direction
            goal=(inst.object_pose.x+.45*math.cos(target*math.pi/4),inst.object_pose.y+.45*math.sin(target*math.pi/4))
            inst=replace(inst,split='train',index=len(cases),seed=12000+len(cases),robot_poses=tuple(robots),goal_centre=goal)
            cases.append(dict(category=category,direction=target,orbit='CW' if j==0 else 'CCW',instance=inst.as_dict()))
    return cases


def witness(case,perturbation=0.):
    instance=instance_from_dict(case['instance'])
    if perturbation:
        r=instance.robot_poses[0]
        tangent=(-math.sin(r.angle),math.cos(r.angle))
        instance=replace(instance,name=instance.name+f'_tangent{perturbation:+.3f}',split='dev-perturb',index=100+instance.index,
            robot_poses=(replace(r,x=r.x+perturbation*tangent[0],y=r.y+perturbation*tangent[1]),instance.robot_poses[1]))
    env=make_env(instance);initial=frame(env);braking=False;started=False;decisions=[]
    for k in range(450):
        if env.object_distance()<.13:braking=True
        obs=env.boundary_observations()
        available=mask(obs[0])
        push=action('push',direction=case['direction'])
        if not started and case['category']=='reposition':
            # Diagnostic reference only: wait until a scan-available desired push
            # also lies on the goal-opposite side of the object.
            n=(math.cos(case['direction']*math.pi/4),math.sin(case['direction']*math.pi/4))
            p=env.robot_states()[0];obj=env.object_state()
            behind=(p.x-obj.x)*n[0]+(p.y-obj.y)*n[1]<-.30
            started=bool(available[push] and behind)
            a0=push if started else action('reposition',direction=case['orbit'],speed=.12)
        else:a0=push
        if braking:a0=action('wait')
        a1=push if case['category']=='dual' and not braking else action('wait')
        b=env.joint_step(a0,a1,record=True);decisions.append([a0,a1])
        if b['termination']:break
    rows=env.rows
    metric=dict(case=instance.name,variant='witness',category=case['category'],success=env.success,
        seconds=env.total_periods*.05,termination=env.termination,final_distance=env.object_distance(),
        max_penetration=max(r['penetration'] for r in rows),
        exits=[s.skill.exit_reason for s in env.slots],witness_privileged_braking=True,
        initial_distance=math.dist((instance.object_pose.x,instance.object_pose.y),instance.goal_centre))
    return dict(case=dict(name=instance.name),variant='witness',metric=metric,instance=instance.as_dict(),
        initial=initial,rows=rows,commands=decisions)


def load_panel(path):
    if path is None:raise ValueError('micro panel requires --panel-path with an audited manifest')
    data=json.loads(Path(path).read_text())
    train=[instance_from_dict(d['instance']) for d in data['cases']]
    validation=[instance_from_dict(d) for d in data['perturbation_instances']]
    return train,validation


def audit(out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic()
    cases=proposals();records=[];perturbed=[]
    for c in cases:
        for dx in (0.,.01):
            result=witness(c,dx);records.append(result)
            (out/'logs'/f"{result['metric']['case']}.json").write_text(json.dumps(result))
            if dx:perturbed.append(result['instance'])
    render_witnesses(out,records)
    result=dict(version='declared-developer-panel-v1',cases=cases,perturbation_instances=perturbed,
        metrics=[r['metric'] for r in records],train_count=len(cases),perturbation_count=len(perturbed),
        all_witnessed=all(r['metric']['success'] for r in records),seconds=time.monotonic()-start,
        usage='fixed training is deliberate overfitting diagnostic; perturbations are development data, not final test; failed witnesses retained')
    (out/'panel.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','perturbation_instances')},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    audit(p.parse_args().out)
