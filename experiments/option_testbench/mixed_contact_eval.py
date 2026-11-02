"""Bounded PUSH target projection around an independently executing peer."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,subprocess,sys,time
from dataclasses import asdict,replace
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import spec
from experiments.option_testbench import follow_peer_eval as F
from experiments.option_testbench.effect_push_eval import reposition_scale_cases
VERSION='mixed-peer-occupied-v1'
VARIANTS={'baseline':F.VARIANTS['guard_2cm'],'projected':replace(F.VARIANTS['guard_2cm'],mixed_contact_allocation=True)}
F.VARIANTS.update(VARIANTS)

def cases():
    base=next(c for c in F.cases() if c['name']=='push_follow');result=[base]
    for dx in (-.01,.01):
        c=copy.deepcopy(base);c['name']=f'push_follow_dx_{dx:+.2f}';c['instance']['robots'][0]['x']+=dx;result.append(c)
    c=copy.deepcopy(base);c['name']='push_follow_swapped';c['instance']['robots'].reverse();c['actions'].reverse();result.append(c)
    c=copy.deepcopy(base);c['name']='push_wait';c['actions'][1]=F.request('wait',duration=6);result.append(c)
    # Free single-pusher reference: no outward same-face peer occupancy.
    c=copy.deepcopy(base);c['name']='push_free_peer';c['instance']['robots'][1].update(x=3.5,y=2.6);c['actions'][1]=F.request('wait',duration=6);result.append(c)
    c=copy.deepcopy(base);c['name']='short_face_occupied';c['instance']['object']['angle']=math.pi/2
    for i,x in enumerate((2.3,2.7)):c['instance']['robots'][i].update(x=x,y=1.25)
    c['actions'][1]=F.request('wait',duration=6);result.append(c)
    result.append(next(c for c in F.cases() if c['name']=='train_26602_original_next'))
    return result

def run(case,variant,*,world_factory=None,actions=F.ACTIONS):
    r=F.run(case,variant,world_factory=world_factory);m=r['metric'];initial=r['initial']['true']['object'];last=r['rows'][-1]['true']['object']
    outcomes=[]
    for i,action in enumerate(case['actions']):
        a=actions[action]
        if a.kind!='push':continue
        push=[row[f'r{i}'] for row in r['rows'] if row[f'r{i}']['kind']=='push']
        effects=[x['effect'] for x in push if x.get('effect')]
        reservations=[e['same_face'] for e in effects if e.get('same_face')]
        targets=[e['contact_target'] for e in effects if e.get('contact_target')]
        outside=[]
        for e in effects:
            q=e.get('same_face')
            if not q or q['status']!='allocated':continue
            n=e['face_normal'];t=(-n[1],n[0]);target=q['target'];peer=q['peer']
            separation=sum((target[j]-peer[j])*t[j] for j in (0,1))*q['side']
            outside.append(separation-q['target_separation'])
        first_contact=next((row['time']-r['initial']['time'] for row in r['rows'] if row[f'r{i}']['contacting_object']),None)
        outcomes.append(dict(robot=i,action=a.label,exit=m['exits'][i],first_contact_seconds=first_contact,
            contact_seconds=m['contact_seconds'][i],normal_travel=max((e.get('normal_travel',0) for e in effects),default=0),
            requested_distance=a.distance,initial_target=targets[0] if targets else None,
            allocated_samples=sum(q['status']=='allocated' for q in reservations),allocation_failures=sorted({q['status'] for q in reservations if q['status'] not in ('allocated','not_same_face')}),
            min_target_reserve_slack=min(outside) if outside else None,
            approach_seconds=sum(x['stage'] in ('approach','ready') for x in push)*spec.CONTROL_DT))
    m.update(push_outcomes=outcomes,object_displacement=math.dist(initial[:2],last[:2]),object_delta=[last[j]-initial[j] for j in range(2)],object_rotation_rad=last[2]-initial[2])
    return r

def render(out,*,template_path=None):
    records=[];incomplete=[]
    for path in sorted((out/'logs').glob('*.json')):
        r=json.loads(path.read_text())
        if not r.get('rows'):incomplete.append(r);continue
        frames=[]
        for x in [r['initial']]+r['rows']:
            sides=[]
            for i in (0,1):
                s=x.get(f'r{i}',{});e=s.get('effect') or {};reservation=e.get('same_face')
                sides.append(dict(stage=s.get('stage','boundary'),skill=s.get('skill',''),exit=s.get('exit_reason'),
                    prefix=s.get('peer_clearance'),guard=s.get('follow_peer_guard'),gap=s.get('gap'),target=s.get('follow_target_gap'),
                    progress=s.get('follow_time'),contact_target=e.get('contact_target'),preferred=reservation.get('preferred') if reservation else None,
                    binding=s.get('follow_binding'),
                    allocation=reservation,normal=e.get('face_normal'),travel=e.get('normal_travel'),request=e.get('normal_travel_target')))
            frames.append(dict(t=x['time'],o=x['true']['object'][:3],r=[p[:3] for p in x['true']['robots']],force=x['forces'],s=sides))
        records.append(dict(case=r['case']['name'],variant=r['variant'],metric=r['metric'],instance=r['instance'],frames=frames))
    template=(template_path or Path(__file__).with_name('mixed_contact_template.html')).read_text()
    (out/'index.html').write_text(template.replace('__DATA__',json.dumps(records,ensure_ascii=False).replace('<','\\u003c')).replace('__WARNING__',f'未完成分支：{len(incomplete)}'))
    (out/'metrics.json').write_text(json.dumps([r['metric'] for r in records],indent=2)+'\n');(out/'incomplete.json').write_text(json.dumps(incomplete,indent=2)+'\n')
    lines=['# Mixed PUSH: nearest reachable unoccupied contact','','Same FOLLOW guard and 2 cm target; only mixed contact projection differs. Original requests, timeouts, physics, costs and observations retained.','','| Case | Variant | Exits | PUSH first contact s | PUSH travel m | Object delta m | Rotation rad | Penetration m |','|---|---|---|---|---|---|---:|---:|']
    for r in records:
        m=r['metric'];o=m['push_outcomes'];lines.append(f"| {m['case']} | {m['variant']} | {m['exits']} | {[a['first_contact_seconds'] for a in o]} | {[a['normal_travel'] for a in o]} | {m['object_delta']} | {m['object_rotation_rad']:.3f} | {m['max_penetration']:.6f} |")
    lines.append(f'\nIncomplete: {len(incomplete)}. Local controller comparisons, not RL success rates.')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8);p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true');a=p.parse_args()
    if a.worker:
        t=json.load(sys.stdin);Path(t['path']).write_text(json.dumps(run(t['case'],t['variant']))+'\n');return
    if a.render_only:render(a.out);return
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic();panel=cases()
    sources=[ROOT/'src/coimagine/random_start'/n for n in ('same_face.py','effect_push.py','skills.py','env.py','peer_clearance.py')]+[ROOT/'experiments/option_rl_v10/interface.py',Path(__file__),Path(__file__).with_name('mixed_contact_template.html')]
    (out/'manifest.json').write_text(json.dumps(dict(version=VERSION,cases=panel,configs={k:asdict(v) for k,v in VARIANTS.items()},
        seconds=a.seconds,branch_seconds=a.branch_seconds,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    for c in panel:
        for variant in VARIANTS:
            path=out/'logs'/f"{c['name']}__{variant}.json";remaining=a.seconds-(time.monotonic()-start)
            try:
                if remaining<=0:raise TimeoutError('batch budget exhausted')
                result=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--out',str(out)],input=json.dumps(dict(case=c,variant=variant,path=str(path))),text=True,capture_output=True,timeout=min(a.branch_seconds,remaining),cwd=ROOT)
                if result.returncode:raise RuntimeError(result.stderr[-1800:])
            except Exception as error:path.write_text(json.dumps(dict(case=c['name'],variant=variant,status='incomplete',error=repr(error)))+'\n')
    render(out);print(json.dumps(dict(branches=len(panel)*2,seconds=time.monotonic()-start,out=str(out))))
if __name__=='__main__':main()
