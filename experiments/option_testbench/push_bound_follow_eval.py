"""Compare timed FOLLOW with a single PUSH-bound FOLLOW and bounded handoff."""
from __future__ import annotations
import argparse,copy,hashlib,json,subprocess,sys,time
from dataclasses import asdict,replace
from functools import partial
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import env as E,spec
from coimagine.random_start.effect_push import (ACTIONS_REPOSITION_V2,ACTIONS_PUSH_BOUND_FOLLOW,PUSH_BOUND_FOLLOW_VERSION,
    REPOSITION_V2_VERSION,make_effect_skill)
from experiments.option_testbench import mixed_contact_eval as M
from experiments.option_testbench import follow_peer_eval as F
VERSION='push-bound-follow-v1'
CONFIGS={'timed':M.VARIANTS['projected'],'bound':replace(M.VARIANTS['projected'],follow_completion='peer_push')}
F.VARIANTS.update(CONFIGS)
TABLES={'timed':ACTIONS_REPOSITION_V2,'bound':ACTIONS_PUSH_BOUND_FOLLOW}
VERSIONS={'timed':REPOSITION_V2_VERSION,'bound':PUSH_BOUND_FOLLOW_VERSION}


def cases():
    main=next(c for c in M.cases() if c['name']=='push_follow');result=[main]
    result.append(next(c for c in M.cases() if c['name']=='push_follow_swapped'))
    for distance in (.1,.5):
        c=copy.deepcopy(main);c['name']=f'push_{distance:.1f}_follow';c['actions'][0]=F.request('push',direction='N',distance=distance);result.append(c)
    for duration in (2.,24.):
        c=copy.deepcopy(main);c['name']=f'former_follow_{duration:.0f}s';c['actions'][1]=F.request('follow',duration=duration);result.append(c)
    c=copy.deepcopy(main);c['name']='push_unavailable';c['actions'][0]=F.request('push',direction='S',distance=.3);result.append(c)
    c=copy.deepcopy(main);c['name']='follower_unobserved';c['instance']['robots'][1].update(x=4.5,y=3.5);result.append(c)
    return result


def translated(case,variant):
    c=copy.deepcopy(case)
    if variant=='bound':c['actions']=[next(a.index for a in TABLES['bound'] if a.kind=='follow') if F.ACTIONS[i].kind=='follow' else i for i in c['actions']]
    return c


def make_world(instance,variant):
    return E.RandomStartEnv(instance,E.EnvConfig(peer_clearance_recovery='boundary_v1'),
        skill_factory=partial(make_effect_skill,config=CONFIGS[variant]),action_table=TABLES[variant],action_version=VERSIONS[variant])


def run(case,variant):
    c=translated(case,variant);r=M.run(c,variant,world_factory=make_world,actions=TABLES[variant]);m=r['metric']
    kinds=[TABLES[variant][i].kind for i in c['actions']];p=kinds.index('push');f=kinds.index('follow');ends=r['boundary']['skill_ends']
    push_end=ends[p]['time'];follow_end=ends[f]['time']
    m.update(push_end_time=push_end,follow_end_time=follow_end,follow_minus_push_seconds=follow_end-push_end,
        follower_held_before_push_end=any(row['time']<=push_end+1e-9 and row[f'r{f}']['kind']=='hold' for row in r['rows']),
        bound_tracking_seconds=max((row[f'r{f}'].get('follow_binding') or {}).get('during_push_seconds',0.) for row in r['rows']),
        action_version=VERSIONS[variant],actual_actions=[TABLES[variant][i].label for i in c['actions']],
        original_actions=[F.ACTIONS[i].label for i in case['actions']])
    r['manifest_actions']=[asdict(a) for a in TABLES[variant]]
    return r


def render(out):
    M.render(out,template_path=Path(__file__).with_name('push_bound_follow_template.html'))
    records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))];metrics=[r['metric'] for r in records if r.get('rows')]
    lines=['# PUSH-bound FOLLOW lifecycle','','Only FOLLOW lifetime/codebook changes. Gap, peer avoidance, contact allocation, PUSH request/timeout, physics and cost are the same. Bound handoff has <=0.85 s additional charged window, within joint/task caps.','','| Case | Mode | Exits | PUSH end s | FOLLOW end s | Early HOLD | Final true FOLLOW gap m |','|---|---|---|---:|---:|---|---:|']
    for m in metrics:
        f=next(i for i,label in enumerate(m['actual_actions']) if 'FOLLOW' in label)
        lines.append(f"| {m['case']} | {m['variant']} | {m['exits']} | {m['push_end_time']:.2f} | {m['follow_end_time']:.2f} | {m['follower_held_before_push_end']} | {m['final_true_object_gaps'][f]:.4f} |")
    lines.append(f'\nIncomplete: {len(records)-len(metrics)}. Inapplicable PUSH and lost-observation cases are retained failures.')
    (out/'SUMMARY.md').write_text('\n'.join(lines)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--worker',action='store_true');p.add_argument('--render-only',action='store_true');p.add_argument('--seconds',type=float,default=60);p.add_argument('--branch-seconds',type=float,default=8);a=p.parse_args()
    if a.worker:
        t=json.load(sys.stdin);Path(t['path']).write_text(json.dumps(run(t['case'],t['variant']))+'\n');return
    if a.render_only:render(a.out);return
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();start=time.monotonic();panel=cases()
    sources=[ROOT/'src/coimagine/random_start'/n for n in ('env.py','skills.py','effect_push.py','peer_clearance.py','same_face.py')]+[
        ROOT/'experiments/option_testbench'/n for n in ('push_bound_follow_eval.py','push_bound_follow_template.html','mixed_contact_eval.py','follow_peer_eval.py')]
    (out/'manifest.json').write_text(json.dumps(dict(version=VERSION,cases=panel,configs={k:asdict(v) for k,v in CONFIGS.items()},actions={k:[asdict(a) for a in v] for k,v in TABLES.items()},versions=VERSIONS,
        seconds=a.seconds,branch_seconds=a.branch_seconds,source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}),indent=2)+'\n')
    for c in panel:
        for variant in CONFIGS:
            path=out/'logs'/f"{c['name']}__{variant}.json";remaining=a.seconds-(time.monotonic()-start)
            try:
                if remaining<=0:raise TimeoutError('batch budget exhausted')
                completed=subprocess.run([sys.executable,str(Path(__file__)),'--worker','--out',str(out)],input=json.dumps(dict(case=c,variant=variant,path=str(path))),text=True,capture_output=True,timeout=min(a.branch_seconds,remaining),cwd=ROOT)
                if completed.returncode:raise RuntimeError(completed.stderr[-2000:])
            except Exception as error:path.write_text(json.dumps(dict(case=c['name'],variant=variant,status='incomplete',error=repr(error)))+'\n')
    render(out);print(json.dumps(dict(branches=len(panel)*2,seconds=time.monotonic()-start,out=str(out))))
if __name__=='__main__':main()
