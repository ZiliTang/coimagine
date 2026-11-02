"""Quality-constrained option efficiency: smooth preparation, stall guard, short WAIT menu."""
from __future__ import annotations
import argparse,copy,hashlib,json,math,subprocess,sys,time
from dataclasses import asdict,replace
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import env as E,efficient_profile as EP
from coimagine.random_start.effect_push import ACTIONS_PUSH_BOUND_FOLLOW,make_effect_skill,PUSH_BOUND_FOLLOW_VERSION
from experiments.option_testbench import execution_efficiency_eval as X,mixed_contact_eval as M,follow_peer_eval as F
VERSION='option-efficiency-v2'
CONFIGS={'baseline':X.PREPARE,'smooth':replace(EP.CONFIG,progress_guard=False),'efficient':EP.CONFIG}
F.VARIANTS.update({'v2_'+k:v for k,v in CONFIGS.items()})
TABLES={'baseline':ACTIONS_PUSH_BOUND_FOLLOW,'smooth':ACTIONS_PUSH_BOUND_FOLLOW,'efficient':EP.ACTIONS}
VERSIONS={'baseline':PUSH_BOUND_FOLLOW_VERSION,'smooth':PUSH_BOUND_FOLLOW_VERSION,'efficient':EP.VERSION}
GATE=dict(max_forward_change=.03,max_added_side=.015,max_added_rotation_deg=2.,absolute_side_allowance=.03,
          absolute_rotation_allowance_deg=5.,max_penetration=.002,min_peer_gap=.02)


def cases():
    raw=X.cases();base=copy.deepcopy(next(c for c in raw if c['name']=='push_follow'))
    result=[copy.deepcopy(c) for c in raw[:3]]
    for dx in (-.01,.01):
        c=copy.deepcopy(base);c['name']=f'push_dx_{dx:+.2f}';c['instance']['robots'][0]['x']+=dx;result.append(c)
    for tilt in (-5.,5.):
        c=copy.deepcopy(base);c['name']=f'push_tilt_{tilt:+.0f}';c['instance']['object']['angle']=math.radians(tilt);result.append(c)
    c=copy.deepcopy(base);c['name']='push_swapped';c['actions'].reverse();c['instance']['robots'].reverse();result.append(c)
    for duration in (6.,24.):
        c=copy.deepcopy(base);c['name']=f'push_wait{duration:.0f}';c['actions'][1]=next(a.index for a in TABLES['baseline'] if a.kind=='wait' and a.duration==duration);result.append(c)
    result += [copy.deepcopy(c) for c in raw[3:]]
    c=copy.deepcopy(base);c['name']='rotation_CCW_0.1_wait2';c['actions']=[next(a.index for a in TABLES['baseline'] if a.label=='ROTATE_CCW[0.10rad]'),next(a.index for a in TABLES['baseline'] if a.kind=='wait' and a.duration==2.)];result.append(c)
    # The static obstacle is genuinely visible and leaves an unfavorable approach lane.
    c=copy.deepcopy(base);c['name']='static_approach';c['instance']['obstacles']=[dict(cx=2.5,cy=1.52,half_w=.06,half_h=.15,angle=0.)];result.append(c)
    from experiments.option_testbench.effect_push_eval import make_case,ACTIONS as ORIGINAL,reposition_scale_cases
    from coimagine.random_start.effect_push import ACTIONS_REPOSITION_V2
    for sign in (1,-1):
        c=make_case(f'rotation_solo_{sign}',rotation=sign,angle=.1)
        c['actions']=[next(a.index for a in TABLES['baseline'] if a.label==ORIGINAL[i].label) for i in c['actions']]
        result.append(c)
    for direction in ('CW','CCW'):
        c=copy.deepcopy(next(c for c in reposition_scale_cases() if c['name']==f'scale2_rp_{direction}_solo_1.80'))
        c['actions']=[next(a.index for a in TABLES['baseline'] if a.label==ACTIONS_REPOSITION_V2[i].label) for i in c['actions']]
        result.append(c)
    assert len(result)==19 and len({c['name'] for c in result})==19
    return result


def translate(case,variant):
    c=copy.deepcopy(case)
    c['actions']=[next(b.index for b in TABLES[variant] if b.kind==a.kind and
                  (a.kind=='follow' or a.kind=='wait' and b.duration==(2. if variant=='efficient' else a.duration)
                   or a.kind not in ('follow','wait') and b.label==a.label))
                  for a in (TABLES['baseline'][i] for i in case['actions'])]
    return c


def make_world(instance,variant):
    name=variant.removeprefix('v2_');cfg=CONFIGS[name]
    def factory(action,limit):
        # Rotation motion is deliberately kept at the previously tested baseline;
        # only its target-direction plateau detector changes in the efficient profile.
        rcfg=replace(X.BASE,progress_guard=cfg.progress_guard)
        return make_effect_skill(action,limit,config=rcfg if getattr(action,'effect_kind',None)=='rotate' else cfg)
    return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1',lidar_backend='numpy_batch'),
        skill_factory=factory,action_table=TABLES[name],action_version=VERSIONS[name])


def run(case,variant):
    c=translate(case,variant);key='v2_'+variant;start=time.perf_counter()
    r=M.run(c,key,world_factory=make_world,actions=TABLES[variant]);m=r['metric']
    m.update(variant=variant,wall_seconds=time.perf_counter()-start,action_version=VERSIONS[variant],
             actual_actions=[TABLES[variant][i].label for i in c['actions']],source_actions=[TABLES['baseline'][i].label for i in case['actions']])
    r['variant']=variant;r['manifest_actions']=[asdict(a) for a in TABLES[variant]]
    m['preparation_seconds']=[sum(row[f'r{i}']['stage'] in ('approach','ready','align') and row[f'r{i}']['kind'] in ('push','reposition') for row in r['rows'])*.05 for i in (0,1)]
    return r


def compare(records):
    by={(r['case']['name'],r['variant']):r for r in records};results=[]
    for name in sorted({r['case']['name'] for r in records}):
        base=by.get((name,'baseline'))
        if base is None:continue
        bm=base['metric'];active=[i for i,a in enumerate(base['case']['actions']) if TABLES['baseline'][a].kind in ('push','reposition')]
        successful=all(bm['exits'][i] in ('normal_travel_complete','reposition_complete','effect_complete','local_effect_reached') for i in active)
        for variant in ('smooth','efficient'):
            if (name,variant) not in by:continue
            m=by[name,variant]['metric'];retained=all(m['exits'][i] in ('normal_travel_complete','reposition_complete','effect_complete','local_effect_reached') for i in active)
            side=abs(m['object_delta'][0]);bs=abs(bm['object_delta'][0]);angle=abs(m['object_rotation_rad']);ba=abs(bm['object_rotation_rad'])
            gate=(retained and abs(m['object_delta'][1]-bm['object_delta'][1])<=GATE['max_forward_change']
                and side<=max(bs+GATE['max_added_side'],GATE['absolute_side_allowance'])
                and angle<=max(ba+math.radians(GATE['max_added_rotation_deg']),math.radians(GATE['absolute_rotation_allowance_deg']))
                and m['max_penetration']<=GATE['max_penetration'] and m['min_peer_gap']>GATE['min_peer_gap'])
            results.append(dict(case=name,variant=variant,baseline_active_complete=successful,quality_retained=gate if successful else None,
                active_complete=retained,sim_ratio=m['seconds']/bm['seconds'],baseline_seconds=bm['seconds'],seconds=m['seconds']))
    return results


def render(out):
    M.render(out,template_path=Path(__file__).with_name('option_efficiency_v2_template.html'))
    records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))];valid=[r for r in records if r.get('rows')]
    comparisons=compare(valid);(out/'quality_comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    lines=['# Option efficiency v2','','Quality gate preregistered in manifest. WAIT duration removal is an explicit new codebook, not the same requested timed action.','','| Case | Variant | Sim s | Exits | Delta m | Rotation deg |','|---|---|---:|---|---|---:|']
    for r in valid:
        m=r['metric'];lines.append(f"| {m['case']} | {m['variant']} | {m['seconds']:.2f} | {m['exits']} | {m['object_delta']} | {math.degrees(m['object_rotation_rad']):.2f} |")
    lines.append(f'\nIncomplete: {len(records)-len(valid)}. See quality_comparison.json; failures remain in denominators.')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true');p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8);p.add_argument('--variants',nargs='+',choices=list(CONFIGS));a=p.parse_args()
    if a.worker:
        t=json.load(sys.stdin);Path(t['path']).write_text(json.dumps(run(t['case'],t['variant']))+'\n');return
    if a.render_only:render(a.out);return
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic();panel=cases()
    source=[ROOT/'src/coimagine/random_start'/n for n in ('effect_push.py','efficient_profile.py','progress_guard.py','env.py','skills.py','lidar.py')]+[ROOT/'src/coimagine/continuous2d/environment.py',Path(__file__),Path(__file__).with_name('option_efficiency_v2_template.html')]
    (out/'manifest.json').write_text(json.dumps(dict(version=VERSION,cases=panel,configs={k:asdict(v) for k,v in CONFIGS.items()},menus={k:[asdict(a) for a in v] for k,v in TABLES.items()},gate=GATE,
        seconds=a.seconds,branch_seconds=a.branch_seconds,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}),indent=2)+'\n')
    selected=a.variants or list(CONFIGS)
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
