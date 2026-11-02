"""Recompute retention and replay dual-task selection/recovery counterfactuals.

Original evaluation commands are replayed from reset, not approximated using a
controller snapshot. Diagnostic controls use truth for braking only, never as
learned-policy inputs. Every declared branch is kept, including failed controls.
"""
import argparse,json,math,time
from pathlib import Path
from collections import Counter
from experiments.option_rl_v15.interface import ACTIONS,make_env
from experiments.option_rl_v15.learnability import load_panel
from experiments.option_rl_v15.render_episodes import instance_from_dict
from experiments.option_rl_v14.learnability import action
from experiments.option_rl_v15.live import atomic_json


def frame(env):
    obj=env.object_state();robots=env.robot_states();states=[]
    for slot in env.slots:
        info=slot.last_info;effect=info.get('effect') or {}
        states.append(dict(stage=info.get('stage',slot.skill.stage),skill=slot.action.label,
            exit=slot.skill.exit_reason,pause_reason=slot.pause_reason,gap=info.get('gap'),
            normal=effect.get('face_normal'),contact_target=effect.get('contact_target'),
            request=slot.pending.label if slot.pending else None))
    return dict(t=env.total_periods*.05,o=list(obj.as_tuple())[:3],r=[list(r.as_tuple())[:3] for r in robots],
        force=[slot.last_info.get('force') or (0.,0.) for slot in env.slots],s=states,
        distance=env.object_distance(),linear_speed=math.hypot(obj.vx,obj.vy),angular_speed=abs(obj.omega))


def branch(instance,commands,variant,direction,cut=None):
    env=make_env(instance);frames=[frame(env)];phases=Counter();exits=Counter();prefix_match=True;first_fail=None
    def advance(cmd):
        nonlocal first_fail
        b=env.joint_step(*cmd)
        f=frame(env);frames.append(f)
        for i in (0,1):
            phases[b['skill_ends'][i]['stage']]+=b['periods']*.05
            exits[b['skill_ends'][i]['exit_reason']]+=b['periods']*.05
        if first_fail is None and any(s.skill.done for s in env.slots):first_fail=env.total_periods*.05
        return b
    replay=commands if variant in ('greedy30','greedy40') else commands[:cut]
    for a in replay:
        start=env.total_periods
        while env.total_periods-start<round(a['dt']/.05):
            b=advance((a['a0'],a['a1']))
            if env.termination:break
        prefix_match &= abs(env.total_periods*.05-a['time'])<1e-8
        if env.termination:break
    prefix_distance=env.object_distance();prefix_time=env.total_periods*.05
    if variant=='reset_after_failure':
        for _ in range(4):
            if not env.termination:advance((44,44))
    push=action('push',direction=direction,speed=.16);braking=False
    while not env.termination:
        if env.object_distance()<.13:braking=True
        cmd=(44,44) if braking else (push,push)
        # Original greedy commands should cover their entire episode.
        if variant in ('greedy30','greedy40'):raise RuntimeError('incomplete saved greedy episode')
        advance(cmd)
    metric=dict(variant=variant,name=instance.name,success=env.success,seconds=env.total_periods*.05,
        min_distance=min(f['distance'] for f in frames),final_distance=env.object_distance(),return_=env.reward_total,
        prefix_time=prefix_time,prefix_distance=prefix_distance,prefix_clock_match=prefix_match,first_failure_time=first_fail,
        exits=[s.skill.exit_reason for s in env.slots],phase_robot_seconds=dict(phases),exit_robot_seconds=dict(exits),
        offline_truth_braking=variant not in ('greedy30','greedy40'))
    if not prefix_match:raise RuntimeError('replay clock mismatch')
    return dict(case=instance.name,variant=variant,instance=instance.as_dict(),metric=metric,frames=frames)


def run(source,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic()
    source=Path(source);manifest=json.loads((source/'manifest.json').read_text());evaluations={}
    for p in sorted((source/'evaluations').glob('*.json')):
        evaluations[int(p.stem.split('_')[-1])]=json.loads(p.read_text())
    retention={}
    for it,e in evaluations.items():
        for r in e['development_training']['roots']:
            retention.setdefault(r['name'],[]).append(dict(iteration=it,success=r['success'],progress=r['progress']))
    records=[]
    for name,direction in [('dual_2',2),('dual_0',0)]:
        instance=next(instance_from_dict(d) for d in manifest['development_train_instances'] if d['name']==name)
        roots={it:next(r for r in evaluations[it]['development_training']['roots'] if r['name']==name) for it in (30,40)}
        commands=roots[40]['actions'];failure=next(i for i,a in enumerate(commands) if any(e!='continuing' for e in a['exits']))+1
        variants=[('greedy30',roots[30]['actions'],None),('greedy40',commands,None),
            ('aligned_reset',commands,0),('aligned_after_2s',commands,3),
            ('same_intent_after_failure',commands,failure),('reset_after_failure',commands,failure)]
        for variant,cmd,cut in variants:
            r=branch(instance,cmd,variant,direction,cut);records.append(r)
            atomic_json(out/'logs'/f'{name}__{variant}.json',r)
    template=Path('experiments/option_rl_v13/recovery_template.html').read_text()
    template=template.replace('持续模式 · 失败恢复对照','双机任务 · 成功保留与执行分支')
    template=template.replace('原 run12 与生命周期修复对照。同一初态与指令序列完整重放；保留真实失败与静态通路阻塞，不无条件重启失败动作。右侧可切换两个机器人。',
        '第30/40轮实际贪心命令，以及相同前缀后的诊断控制。诊断分支使用真值决定停车，仅作离线分析；右侧切换机器人。')
    template=template.replace("$('variant').value='baseline'","$('variant').value='greedy30'")
    template=template.replace('__WARNING__','开发场景，不是泛化评测。所有失败分支保留；same_intent故意保留失败锁存，reset先显式BRAKE再建立新控制器。')
    template=template.replace('__DATA__',json.dumps(records,ensure_ascii=False).replace('<','\\u003c'))
    (out/'index.html').write_text(template)
    result=dict(retention=retention,branches=[r['metric'] for r in records],seconds=time.monotonic()-start,
        scope='old policy retention and controller counterfactuals only; no training')
    atomic_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    run(**vars(p.parse_args()))
