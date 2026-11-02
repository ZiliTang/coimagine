"""Bounded standalone FOLLOW guard/gap ablation on frozen run10 boundaries."""
from __future__ import annotations
import argparse, hashlib, json, math, pickle, subprocess, sys, time
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path: sys.path.insert(0,str(p))
from coimagine.random_start import env as E, geometry as G, layout, spec
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start.effect_push import ACTIONS_REPOSITION_V2 as ACTIONS, REPOSITION_V2_VERSION, make_effect_skill
from experiments.option_rl_v10.interface import CONFIG
from experiments.option_testbench.peer_clearance_eval import frame, instance_from_dict, cases as clearance_cases
PARENTS=ROOT/'outputs/option_testbench_v1/peer_clearance_recovery_v1/round_003'
VERSION='standalone-follow-peer-v2'
VARIANTS={
    'legacy_5cm': CONFIG,
    'guard_5cm': replace(CONFIG,standalone_follow_peer_guard='scan_projection_v1'),
    'guard_2cm': replace(CONFIG,standalone_follow_peer_guard='scan_projection_v1',standalone_follow_gap=.02),
}

def request(kind,*,duration=None,distance=None,direction=None):
    return next(a.index for a in ACTIONS if a.kind==kind and (duration is None or a.duration==duration)
        and (distance is None or a.distance==distance) and (direction is None or
            (a.label.startswith('PUSH_'+direction+'[') if kind=='push' else a.direction==direction)))

def cases():
    result=[]
    for index in (26735,26673,26602):
        path=PARENTS/f'parent_{index}.pkl';d=pickle.loads(path.read_bytes());a=d['remaining'][0]
        result.append(dict(name=f'train_{index}_original_next',snapshot=str(path),snapshot_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            instance=d['instance'],actions=[a['a0'],a['a1']]))
    for name,poses,actions in [
        ('healthy_static',[(2.2,1.5,math.pi/2),(3.3,1.5,math.pi/2)],[request('follow',duration=6),request('wait',duration=6)]),
        ('same_face_pair',[(2.285,1.5,math.pi/2),(2.715,1.5,math.pi/2)],[request('follow',duration=6)]*2),
        ('push_follow',[(2.2,1.5,math.pi/2),(2.8,1.5,math.pi/2)],[request('push',distance=.3,direction='N'),request('follow',duration=6)]),
    ]:
        inst=layout.InstanceSpec(name=name,map='dev',split='dev',index=0,seed=0,arena=(5.,4.),obstacles=(),
            object_pose=Pose(2.5,2.,0.),robot_poses=tuple(Pose(*p) for p in poses),goal_centre=(2.5,3.3),goal_radius=.2)
        result.append(dict(name=name,instance=inst.as_dict(),actions=actions))
    for c in clearance_cases([]):
        if c['name'] in ('caged','one_free'):
            c=dict(c,name='follow_'+c['name'],actions=[request('follow',duration=2),request('wait',duration=2)])
            result.append(c)
    return result

def make_world(instance,variant):
    return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1'),
        skill_factory=partial(make_effect_skill,config=VARIANTS[variant]),action_table=ACTIONS,action_version=REPOSITION_V2_VERSION)

def run(case,variant,*,world_factory=None):
    world=(world_factory or make_world)(instance_from_dict(case['instance']),variant)
    if case.get('snapshot'):world.restore_branch_state(pickle.loads(Path(case['snapshot']).read_bytes())['state'],allow_execution_change=True)
    initial=frame(world);obj=(world.object_state().x,world.object_state().y)
    boundary=world.joint_step(*case['actions'],record=True);rows=world.rows
    gaps=[];objects=[[],[]];guard=[0,0];follow=[0,0];contact=[0,0]
    for r in rows:
        robots=r['true']['robots'];gaps.append(math.dist(robots[0][:2],robots[1][:2])-2*spec.ROBOT_RADIUS)
        o=r['true']['object'];poly=G.box_polygon(*o[:3],spec.OBJECT_WIDTH/2,spec.OBJECT_HEIGHT/2)
        for i in (0,1):
            info=r[f'r{i}'];objects[i].append(G.point_polygon_distance(robots[i][:2],poly)-spec.ROBOT_RADIUS)
            guard[i]+=bool((info.get('follow_peer_guard') or {}).get('active'))
            follow[i]=max(follow[i],info.get('follow_time') or 0.)
            contact[i]+=bool(info.get('contacting_object'))
    audit=E.recompute_episode_reward(rows,[boundary],obj,world.config)
    metric=dict(case=case['name'],variant=variant,seconds=boundary['periods']*.05,
        exits=[boundary['skill_ends'][i]['exit_reason'] for i in (0,1)],
        safety_triggers=[r[f'r{i}']['safety_trigger'] for r in rows for i in (0,1) if r[f'r{i}'].get('safety_trigger')],
        min_peer_gap=min(gaps),min_nominal_peer_gap=min((g for g,r in zip(gaps,rows) if not r['r0'].get('peer_clearance')),default=None),
        final_peer_gap=gaps[-1],max_penetration=max(r['penetration'] for r in rows),
        guard_seconds=[n*.05 for n in guard],follow_seconds=follow,contact_seconds=[n*.05 for n in contact],
        final_true_object_gaps=[v[-1] for v in objects],
        final_second_true_object_gaps=[sum(v[-20:])/len(v[-20:]) for v in objects],
        time_budgets=boundary['execution']['requested_time_budgets'],reward_error=abs(audit['total']-boundary['reward']['total']),
        prefix=boundary['peer_clearance'],termination=world.termination)
    assert metric['reward_error']<1e-7
    return dict(case=case,variant=variant,config=asdict(VARIANTS[variant]),metric=metric,initial=initial,instance=world.instance.as_dict(),rows=rows,boundary=boundary)

def render(out):
    records=[];incomplete=[]
    for p in sorted((out/'logs').glob('*.json')):
        r=json.loads(p.read_text())
        if not r.get('rows'):incomplete.append(r);continue
        frames=[r['initial']]+r['rows'];data=[]
        for x in frames:
            data.append(dict(t=x['time'],o=x['true']['object'][:3],r=[p[:3] for p in x['true']['robots']],force=x['forces'],
                s=[dict(stage=x.get(f'r{i}',{}).get('stage','boundary'),skill=x.get(f'r{i}',{}).get('skill',''),
                    prefix=x.get(f'r{i}',{}).get('peer_clearance'),guard=x.get(f'r{i}',{}).get('follow_peer_guard'),
                    gap=x.get(f'r{i}',{}).get('gap'),target=x.get(f'r{i}',{}).get('follow_target_gap'),
                    progress=x.get(f'r{i}',{}).get('follow_time'),exit=x.get(f'r{i}',{}).get('exit_reason')) for i in (0,1)]))
        records.append(dict(case=r['case']['name'],variant=r['variant'],metric=r['metric'],instance=r['instance'],frames=data))
    template=Path(__file__).with_name('follow_peer_template.html').read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(records,ensure_ascii=False).replace('<','\\u003c')).replace('__WARNING__',f'未完成分支：{len(incomplete)}'))
    (out/'metrics.json').write_text(json.dumps([r['metric'] for r in records],indent=2)+'\n')
    (out/'incomplete.json').write_text(json.dumps(incomplete,indent=2)+'\n')
    lines=['# Standalone FOLLOW guard and 2 cm gap','','All variants use the same boundary separation, requests, timeouts, costs and physics.','','| Case | Variant | Exits | Min peer gap m | Guard s | Follow s | Max penetration m |','|---|---|---|---:|---|---|---:|']
    for r in records:
        m=r['metric'];lines.append(f"| {m['case']} | {m['variant']} | {m['exits']} | {m['min_peer_gap']:.4f} | {m['guard_seconds']} | {m['follow_seconds']} | {m['max_penetration']:.6f} |")
    lines.append(f'\nIncomplete: {len(incomplete)}. These are local option comparisons, not RL success rates.')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true')
    p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8);p.add_argument('--cases',nargs='*');args=p.parse_args()
    if args.worker:
        task=json.load(sys.stdin);Path(task['path']).write_text(json.dumps(run(task['case'],task['variant']))+'\n');return
    if args.render_only:render(args.out);return
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic();panel=cases()
    if args.cases:panel=[c for c in panel if c['name'] in args.cases]
    sources=[ROOT/'src/coimagine/random_start'/n for n in ('env.py','skills.py','effect_push.py','peer_clearance.py')]+[Path(__file__),Path(__file__).with_name('follow_peer_template.html')]
    manifest=dict(version=VERSION,cases=panel,configs={k:asdict(v) for k,v in VARIANTS.items()},seconds=args.seconds,branch_seconds=args.branch_seconds,
        variables='legacy 5cm / peer guard 5cm / peer guard 2cm; all with boundary prefix',source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for c in panel:
        for variant in VARIANTS:
            path=out/'logs'/f"{c['name']}__{variant}.json";remaining=args.seconds-(time.monotonic()-start)
            try:
                if remaining<=0:raise TimeoutError('batch budget exhausted')
                r=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--out',str(out)],input=json.dumps(dict(case=c,variant=variant,path=str(path))),text=True,capture_output=True,timeout=min(args.branch_seconds,remaining),cwd=ROOT)
                if r.returncode:raise RuntimeError(r.stderr[-2000:])
            except Exception as error:path.write_text(json.dumps(dict(case=c['name'],variant=variant,status='incomplete',error=repr(error)))+'\n')
    render(out);print(json.dumps(dict(out=str(out),branches=len(panel)*len(VARIANTS),seconds=time.monotonic()-start)))
if __name__=='__main__':main()
