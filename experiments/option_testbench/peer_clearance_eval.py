"""Bounded frozen-boundary comparison for the blocked run10 training examples."""
from __future__ import annotations
import argparse,hashlib,json,math,pickle,subprocess,sys,time,html
from dataclasses import asdict,replace
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import env as E,layout,spec
from coimagine.continuous2d.scenes import Pose,Box
from coimagine.random_start.effect_push import ACTIONS_REPOSITION_V2 as ACTIONS,REPOSITION_V2_VERSION,make_effect_skill
from coimagine.random_start.peer_clearance import public_gap,VERSION
from experiments.option_rl_v10.interface import CONFIG
from experiments.option_testbench.effect_push_eval import instance_from_dict
SOURCE=ROOT/'outputs/option_rl_v10/overnight_parallel_20261028/run10_001'


def make_world(instance,enabled):
    return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1' if enabled else 'off'),
        skill_factory=partial(make_effect_skill,config=CONFIG),action_table=ACTIONS,action_version=REPOSITION_V2_VERSION)


def capture(out):
    parents=[]
    for index in (26735,26673,26602):
        raw=json.loads((SOURCE/'rollouts'/f'{index:07d}.json').read_text());world=make_world(instance_from_dict(raw['instance']),False)
        exact=True
        for k,d in enumerate(raw['summary']['actions']):
            b=world.joint_step(d['a0'],d['a1'],record=False)
            exact&=([b['skill_ends'][i]['exit_reason'] for i in (0,1)]==d['exits'] and abs(world.total_periods*.05-d['time'])<1e-8)
            if public_gap(world.boundary_observations()[0])[0]<=.03 and 'safety_abort' in d['exits']:
                assert exact,'original prefix not reproduced'
                snapshot=world.export_branch_state();path=out/f'parent_{index}.pkl'
                path.write_bytes(pickle.dumps(dict(state=snapshot,instance=raw['instance'],remaining=raw['summary']['actions'][k+1:]),protocol=5))
                parents.append(dict(index=index,after_decision=k+1,time=world.total_periods*.05,public_gap=public_gap(world.boundary_observations()[0])[0],
                    snapshot=str(path),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),original_prefix_exact=exact))
                break
        else:raise RuntimeError('no close-peer boundary found')
    return parents


def cases(parents):
    def rp(direction):return next(a.index for a in ACTIONS if a.kind=='reposition' and a.direction==direction and a.distance==.4)
    wait=next(a.index for a in ACTIONS if a.kind=='wait' and a.duration==2.)
    result=[]
    for parent in parents:
        data=pickle.loads(Path(parent['snapshot']).read_bytes());next_action=data['remaining'][0]
        for mode,actions in [('original_next',[next_action['a0'],next_action['a1']]),('RP_CW_WAIT',[rp('CW'),wait]),
                             ('RP_CCW_WAIT',[rp('CCW'),wait]),('recorded_suffix',None)]:
            result.append(dict(name=f"train_{parent['index']}_{mode}",source=parent,mode=mode,actions=actions))
    for mode in ('caged','one_free','clear_peer'):
        poses=[Pose(.85,1.5,0.),Pose(1.22 if mode!='clear_peer' else 1.55,1.5,math.pi)]
        boxes=[]
        if mode=='caged':
            lo,hi=poses[0].x-.184,poses[1].x+.184
            boxes=[Box(lo-.02,1.5,.02,.225),Box(hi+.02,1.5,.02,.225),
                   Box((lo+hi)/2,1.5-.204,(hi-lo)/2+.04,.02),
                   Box((lo+hi)/2,1.5+.204,(hi-lo)/2+.04,.02)]
        for i in ([1] if mode=='one_free' else []):
            x,y=poses[i].x,poses[i].y
            # 4 mm body clearance: there is no observed retreat with the fixed 5 mm buffer.
            boxes += [Box(x+.204,y,.02,.225),Box(x,y-.204,.225,.02),Box(x,y+.204,.225,.02)]
        inst=layout.InstanceSpec(name=mode,map='dev',split='dev',index=0,seed=0,arena=(4.,3.),obstacles=tuple(boxes),
            object_pose=Pose(2.8,2.3,0.),robot_poses=tuple(poses),goal_centre=(3.,1.5),goal_radius=.2)
        result.append(dict(name=mode,mode=mode,instance=inst.as_dict(),actions=[rp('CCW'),wait]))
    from experiments.option_testbench.effect_push_eval import reposition_scale_cases
    healthy=next(c for c in reposition_scale_cases() if c['name']=='scale2_rp_CCW_solo_1.80')
    result.append(dict(name='healthy_rp_solo',mode='healthy',instance=healthy['instance'],actions=healthy['actions']))
    return result


def frame(world):
    return dict(time=world.total_periods*.05,true=dict(object=list(world.object_state().as_tuple()),robots=[list(s.as_tuple()) for s in world.robot_states()]),
                forces=[[0.,0.],[0.,0.]],r0={},r1={})


def run(case,enabled):
    initial={};data=None
    if 'source' in case:
        data=pickle.loads(Path(case['source']['snapshot']).read_bytes());world=make_world(instance_from_dict(data['instance']),enabled)
        world.restore_branch_state(data['state'],allow_execution_change=True)
    else:world=make_world(instance_from_dict(case['instance']),enabled)
    initial=frame(world);before=(world.object_state().x,world.object_state().y);start=world.total_periods
    boundaries=[];exact=[];requested=[];fallback=0
    sequence=([(d['a0'],d['a1']) for d in data['remaining']] if case['mode']=='recorded_suffix' else [case['actions']])
    for k,actions in enumerate(sequence):
        b=world.joint_step(*actions,record=True);boundaries.append(b);requested.append([ACTIONS[a].label for a in actions])
        if not enabled and data and case['mode']=='recorded_suffix':
            d=data['remaining'][k];exact.append(abs(b['end_time']-d['time'])<1e-8 and [b['skill_ends'][i]['exit_reason'] for i in (0,1)]==d['exits'])
        if world.termination:break
    if case['mode']=='recorded_suffix':
        wait=next(a.index for a in ACTIONS if a.kind=='wait' and a.duration==2.)
        while not world.termination:
            boundaries.append(world.joint_step(wait,wait,record=True));requested.append([ACTIONS[wait].label]*2);fallback+=1
    rows=world.rows
    prefixes=[r for r in rows if r['r0'].get('peer_clearance')]
    released=[r for r in prefixes if r['r0']['peer_clearance']['phase']=='released']
    failures=[r for r in prefixes if r['r0']['peer_clearance']['phase']=='failed']
    min_gap=min(math.dist(r['true']['robots'][0][:2],r['true']['robots'][1][:2])-2*spec.ROBOT_RADIUS for r in rows)
    metric=dict(case=case['name'],enabled=enabled,seconds=(world.total_periods-start)*.05,source_time=initial['time'],
        decisions=len(boundaries),very_short=sum(b['periods']<=3 for b in boundaries),
        both_safety_abort=sum(all(b['skill_ends'][i]['exit_reason']=='safety_abort' for i in (0,1)) for b in boundaries),
        exits=[boundaries[-1]['skill_ends'][i]['exit_reason'] for i in (0,1)],
        prefix_entered=sum(bool(b['peer_clearance']['entered']) for b in boundaries),prefix_releases=len(released),prefix_failures=len(failures),
        first_release_seconds=released[0]['time']-initial['time'] if released else None,
        first_release_gap=released[0]['r0']['peer_clearance']['public_gap'] if released else None,
        min_peer_gap=min_gap,final_peer_gap=public_gap(world.boundary_observations()[0])[0],max_penetration=max(r['penetration'] for r in rows),
        nominal_credit_during_prefix=max((abs((r[f'r{i}'].get('completion_progress') or r[f'r{i}'].get('signed_progress') or 0.)) for r in prefixes for i in (0,1)),default=0.),
        requested_time_budgets=boundaries[0]['execution']['requested_time_budgets'],
        object_displacement=math.dist(before,(world.object_state().x,world.object_state().y)),termination=world.termination,
        original_suffix_exact=all(exact) if exact else None,wait_fallbacks=fallback,
        reward=sum(b['reward']['total'] for b in boundaries))
    audit=E.recompute_episode_reward(rows,boundaries,before,world.config)
    metric['reward_reconstruction_error']=abs(audit['total']-metric['reward'])
    assert metric['reward_reconstruction_error']<1e-7
    return dict(case=case,metric=metric,instance=world.instance.as_dict(),initial=initial,requested=requested,boundaries=boundaries,rows=rows)


def render(out):
    all_records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))]
    records=[r for r in all_records if r.get('rows')]
    incomplete=[dict(case=r['case']['name'],enabled=r['enabled'],status='incomplete',error=r.get('error')) for r in all_records if not r.get('rows')]
    (out/'incomplete.json').write_text(json.dumps(incomplete,indent=2)+'\n')
    data=[]
    for r in records:
        frames=[r['initial']]+r['rows']
        data.append(dict(case=r['case']['name'],enabled=r['metric']['enabled'],metric=r['metric'],instance=r['instance'],
            frames=[dict(t=x['time'],o=x['true']['object'][:3],r=[p[:3] for p in x['true']['robots']],
                force=x['forces'],s=[dict(stage=x.get(f'r{i}',{}).get('stage','boundary'),skill=x.get(f'r{i}',{}).get('skill',''),
                prefix=x.get(f'r{i}',{}).get('peer_clearance'),progress=x.get(f'r{i}',{}).get('signed_progress'),
                gap=x.get(f'r{i}',{}).get('gap'),exit=x.get(f'r{i}',{}).get('exit_reason')) for i in (0,1)]) for x in frames]))
    template=Path(__file__).with_name('peer_clearance_template.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')).replace('__WARNING__',html.escape('未完成分支：'+str(len(incomplete))+('；详见 incomplete.json' if incomplete else ''))))
    metrics=[r['metric'] for r in records]+incomplete;(out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    lines=['# 近距离受阻：同一物理边界的原控制/前置分离对照','','前置分离计入原时钟和成本，不计入主动行程。固定后续指令不是新 RL 成功率。',
           '', '| 场景 | 分离 | 决策数 | 极短数 | 双 safety 数 | 释放 / 失败 | 最后退出 |', '|---|---|---:|---:|---:|---|---|']
    for m in [r['metric'] for r in records]:lines.append(f"| {m['case']} | {m['enabled']} | {m['decisions']} | {m['very_short']} | {m['both_safety_abort']} | {m['prefix_releases']} / {m['prefix_failures']} | {m['exits']} |")
    lines+=['',f'未完成分支：{len(incomplete)}。详见 incomplete.json。']
    (out/'render_manifest.json').write_text(json.dumps({'renderer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'template_sha256':hashlib.sha256(Path(__file__).with_name('peer_clearance_template.html').read_bytes()).hexdigest(),'records':len(records),'incomplete':len(incomplete)},indent=2)+'\n')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--cases',nargs='*')
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8)
    p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true');args=p.parse_args()
    if args.worker:
        task=json.load(sys.stdin);Path(task['path']).write_text(json.dumps(run(task['case'],task['enabled']))+'\n');return
    if args.render_only:render(args.out);return
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();started=time.monotonic()
    parents=capture(out);panel=cases(parents)
    if args.cases:panel=[c for c in panel if c['name'] in args.cases]
    sources=[ROOT/'src/coimagine/random_start'/n for n in ('env.py','skills.py','effect_push.py','peer_clearance.py')]+[Path(__file__)]
    manifest=dict(version=VERSION,cases=panel,parents=parents,seconds=args.seconds,branch_seconds=args.branch_seconds,
        variables='peer_clearance_recovery off / boundary_v1; same boundary, requests, physics, task reward and windows',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for c in panel:
        for enabled in (False,True):
            path=out/'logs'/f"{c['name']}__{int(enabled)}.json";remaining=args.seconds-(time.monotonic()-started)
            try:
                if remaining<=0:raise TimeoutError('batch budget exhausted')
                r=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--out',str(out)],input=json.dumps(dict(case=c,enabled=enabled,path=str(path))),
                    text=True,capture_output=True,timeout=min(args.branch_seconds,remaining),cwd=ROOT)
                if r.returncode:raise RuntimeError(r.stderr[-2500:])
            except Exception as error:path.write_text(json.dumps(dict(case=c,enabled=enabled,status='incomplete',error=repr(error)))+'\n')
    render(out);print(json.dumps(dict(out=str(out),branches=len(panel)*2,seconds=time.monotonic()-started)))
if __name__=='__main__':main()
