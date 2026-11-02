"""Frozen 20 Hz sensing/control: batch rays, 60 Hz physics, faster translation."""
from __future__ import annotations
import argparse,copy,hashlib,json,statistics,subprocess,sys,time
from dataclasses import asdict,replace
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import env as E,spec
from coimagine.random_start.effect_push import make_effect_skill
from coimagine.continuous2d.backend import PymunkBackend
from experiments.option_testbench import push_bound_follow_eval as P
from experiments.option_testbench import mixed_contact_eval as M
from experiments.option_testbench import follow_peer_eval as F
VERSION='execution-efficiency-v1'
BASE=P.CONFIGS['bound']
FAST=replace(BASE,speed=.18,translation_slack=6.,approach_normal_speed=.15,
    approach_tangential_speed=.30,approach_tangential_gain=3.3,point_rate=.09,joint_engage_speed=.06,standalone_follow_speed=.24)
PREPARE=replace(BASE,approach_normal_speed=.15,approach_tangential_speed=.30,approach_tangential_gain=3.3)
SETTINGS={
 'eff_scalar':dict(scanner='scalar',physics_hz=120,fast=False),
 'eff_batch':dict(scanner='numpy_batch',physics_hz=120,fast=False),
 'eff_physics60':dict(scanner='numpy_batch',physics_hz=60,fast=False),
 'eff_fast':dict(scanner='numpy_batch',physics_hz=120,fast=True),
 'eff_fast60':dict(scanner='numpy_batch',physics_hz=60,fast=True),
 'eff_prepare':dict(scanner='numpy_batch',physics_hz=120,fast=False,preparation=True),
}
F.VARIANTS.update({k:PREPARE if v.get('preparation') else FAST if v['fast'] else BASE for k,v in SETTINGS.items()})
A=P.TABLES['bound']


def cases():
    source={c['name']:c for c in P.cases()}
    result=[P.translated(source[n],'bound') for n in ('push_follow','push_0.1_follow','push_0.5_follow')]
    c=copy.deepcopy(result[0]);c['name']='dual_push_0.5_0.1';c['actions']=[F.request('push',direction='N',distance=.5),F.request('push',direction='N',distance=.1)];result.append(c)
    c=copy.deepcopy(result[0]);c['name']='reposition_0.8_wait';c['actions']=[F.request('reposition',direction='CCW',distance=.8),F.request('wait',duration=6.)];result.append(c)
    result.append(next(c for c in M.cases() if c['name']=='short_face_occupied'))
    return result


def make_world(instance,variant):
    setting=SETTINGS[variant];cfg=F.VARIANTS[variant]
    config=E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend=setting['scanner'])
    config.base=replace(config.base,physics=replace(config.base.physics,physics_dt=1./setting['physics_hz']))
    def factory(action,limit):
        # This speed candidate is deliberately scoped to translation and its follower.
        chosen=BASE if getattr(action,'effect_kind',None)=='rotate' else cfg
        return make_effect_skill(action,limit,config=chosen)
    return E.RandomStartEnv(instance,config,skill_factory=factory,action_table=A,action_version=P.VERSIONS['bound'])


def run(case,variant):
    # Non-cProfile wall timers; nested scanner counters are inclusive subsets.
    totals=dict(scan_seconds=0.,physics_seconds=0.,scan_calls=0,physics_calls=0)
    scalar,batch,step=E.scan_lidar,E.scan_lidar_batch,PymunkBackend.apply_and_step
    def wrap(fn,key):
        def call(*a,**kw):
            start=time.perf_counter()
            try:return fn(*a,**kw)
            finally:totals[key+'_seconds']+=time.perf_counter()-start;totals[key+'_calls']+=1
        return call
    E.scan_lidar=wrap(scalar,'scan');E.scan_lidar_batch=wrap(batch,'scan');PymunkBackend.apply_and_step=wrap(step,'physics')
    start=time.perf_counter()
    try:r=M.run(case,variant,world_factory=make_world,actions=A)
    finally:
        wall=time.perf_counter()-start
        E.scan_lidar,E.scan_lidar_batch,PymunkBackend.apply_and_step=scalar,batch,step
    r['metric'].update(wall_seconds=wall,real_time_factor=r['metric']['seconds']/wall,**totals)
    r['metric']['settings']=SETTINGS[variant]
    r['manifest_actions']=[asdict(a) for a in A]
    # PREDECLARED quality indicators; a completion label alone is insufficient.
    push=r['metric']['push_outcomes'];m=r['metric']
    forward=sum(m['object_delta'][j]*(0.,1.)[j] for j in (0,1))
    distance=max((p['requested_distance'] for p in push),default=0.)
    m['translation_quality']=None if not distance or case['name']=='short_face_occupied' else dict(
        contracts_reached=all(p['normal_travel']>=p['requested_distance']-1e-8 for p in push),
        forward_displacement=forward,forward_ratio=forward/distance,
        side_displacement=abs(m['object_delta'][0]),rotation_deg=abs(m['object_rotation_rad'])*180./3.141592653589793,
        basic_effect_retained=(all(p['normal_travel']>=p['requested_distance']-1e-8 for p in push)
          and .75<=forward/distance<=1.30 and abs(m['object_delta'][0])<=.05
          and abs(m['object_rotation_rad'])<=.08726646259971647 and m['max_penetration']<=.002
          and m['min_nominal_peer_gap']>.02))
    return r


def render(out):
    M.render(out,template_path=Path(__file__).with_name('execution_efficiency_template.html'))
    records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))];metrics=[r['metric'] for r in records if r.get('rows')]
    lines=['# Execution efficiency: same task, cost and 20 Hz control','','Translation quality gate: reached nominal body travel; forward 0.75–1.30 x request; lateral <= 5 cm; turn <= 5 deg; penetration <= 2 mm; peer gap > 2 cm. This is a small local gate, not a task success claim.','','| Case | Variant | Sim s | Wall s | Exits | Delta m | Turn deg | Quality |','|---|---|---:|---:|---|---|---:|---|']
    for m in metrics:
        q=m['translation_quality'];lines.append(f"| {m['case']} | {m['variant']} | {m['seconds']:.2f} | {m['wall_seconds']:.4f} | {m['exits']} | {m['object_delta']} | {m['object_rotation_rad']*180/3.141592653589793:.2f} | {q['basic_effect_retained'] if q else 'n/a'} |")
    lines.append(f'\nIncomplete: {len(records)-len(metrics)}. Wall timing excludes subprocess startup and JSON output, includes simulation, metrics and small timer overhead. Physics changes are separate ablations, not an assumed equivalent model.')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true');p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8);p.add_argument('--variants',nargs='+',choices=list(SETTINGS));a=p.parse_args()
    if a.worker:
        t=json.load(sys.stdin);Path(t['path']).write_text(json.dumps(run(t['case'],t['variant']))+'\n');return
    if a.render_only:render(a.out);return
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic();panel=cases()
    source=[ROOT/'src/coimagine/random_start'/n for n in ('lidar.py','env.py','skills.py','effect_push.py')]+[ROOT/'experiments/option_testbench'/n for n in ('execution_efficiency_eval.py','execution_efficiency_template.html','mixed_contact_eval.py','follow_peer_eval.py')]
    selected=a.variants or list(SETTINGS)
    manifest=dict(version=VERSION,cases=panel,settings={k:SETTINGS[k] for k in selected},baseline_config=asdict(BASE),fast_config=asdict(FAST),prepare_config=asdict(PREPARE),actions=[asdict(a) for a in A],control_hz=20,scan_hz=20,real_scan_rays=72,
        quality_gate=dict(forward_ratio=[.75,1.30],max_side=.05,max_rotation_deg=5,max_penetration=.002,min_peer_gap=.02),seconds=a.seconds,branch_seconds=a.branch_seconds,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for c in panel:
        for variant in selected:
            path=out/'logs'/f"{c['name']}__{variant}.json";remaining=a.seconds-(time.monotonic()-start)
            try:
                if remaining<=0:raise TimeoutError('batch budget exhausted')
                done=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--out',str(out)],input=json.dumps(dict(case=c,variant=variant,path=str(path))),text=True,capture_output=True,timeout=min(a.branch_seconds,remaining),cwd=ROOT)
                if done.returncode:raise RuntimeError(done.stderr[-1800:])
            except Exception as error:path.write_text(json.dumps(dict(case=c['name'],variant=variant,status='incomplete',error=repr(error)))+'\n')
    render(out);print(json.dumps(dict(branches=len(panel)*len(selected),seconds=time.monotonic()-start,out=str(out))))
if __name__=='__main__':main()
