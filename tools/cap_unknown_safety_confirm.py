"""Freeze the unknown-space testbed before paired, unseen-layout acceptance."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import argparse
import random
from tools.cap_symbolic_general import ROOT,read,save,sha
from tools.cap_unknown_safety_audit import OUT
from tools.cap_unknown_safety import run,fixture
from experiments.planar_tasks.authoring_v2.symbolic_unknown.policy import program,schema


def freeze():
    from tools.cap_symbolic_general_evaluate import verify_freeze
    old=verify_freeze()
    path=OUT/'acceptance_frozen.json'
    if path.exists():raise FileExistsError(path)
    rng=random.Random(2026112507)
    panels={f'pair{i:02d}':dict(x=rng.uniform(2.35,2.65),y=rng.uniform(2.2,2.6),
        robot_gap=rng.uniform(.65,.70),obstacle_gap=rng.uniform(.11,.14))for i in range(12)}
    files=[*sorted((ROOT/'experiments/planar_tasks/authoring_v2/symbolic_unknown').glob('*.py')),
           ROOT/'tools/cap_unknown_safety.py',Path(__file__),ROOT/'tests/test_symbolic_unknown.py']
    sources={str(p.relative_to(ROOT)):sha(p)for p in files}
    for p in files:
        dst=OUT/'acceptance_sources'/p.relative_to(ROOT);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
    save(OUT/'literal_program.json',program());save(OUT/'semantic_schema.json',schema())
    save(path,dict(sources=sources,panels=panels,seed=2026112507,
        arms=['empty_observe','blocked_observe','empty_no_observe'],
        old_policy_sha256=old['source_sha256'],
        endpoint='at least .02m x progress and complete 40 tick brake; all physical contacts and envelope violations count',
        scope='local conditional-safety acceptance; no whole-task or online-cost improvement claim'))
    return read(path)


def verify():
    frozen=read(OUT/'acceptance_frozen.json')
    for file,value in frozen['sources'].items():
        if sha(ROOT/file)!=value:raise ValueError('source drift: '+file)
    return frozen


def worker(pair,arm,parameters):
    verify();p=OUT/'acceptance'/arm/(pair+'.json.gz')
    if p.exists():raise FileExistsError(p)
    result=run(fixture(arm=='blocked_observe',**parameters),observe=arm!='empty_no_observe')
    result['case']=dict(pair=pair,arm=arm,parameters=parameters);save(p,result)
    return dict(pair=pair,arm=arm,**result['summary'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    f=freeze()if a.freeze else verify();rows=[]
    with ProcessPoolExecutor(max_workers=a.workers)as pool:
        jobs=[pool.submit(worker,pair,arm,parameters)for pair,parameters in f['panels'].items()for arm in f['arms']]
        for future in as_completed(jobs):
            row=future.result();rows.append(row);print(row,flush=True)
    rows.sort(key=lambda r:(r['pair'],r['arm']))
    grouped={arm:[r for r in rows if r['arm']==arm]for arm in f['arms']}
    metrics={arm:dict(episodes=len(rs),safe_progress=sum(r['termination']=='safe_progress_and_stop'for r in rs),
        obstacle_rejected=sum(r['termination']=='obstacle_rejected'for r in rs),
        contact_episodes=sum(r['contact_commands']>0 for r in rs),
        envelope_violation_episodes=sum(r['envelope_violations']>0 for r in rs),
        stopping_passed=sum(r['stopping_passed']for r in rs),
        uncertified_emergency_stops=sum(r['uncertified_emergency_stops']for r in rs),
        views=sum(r['views']for r in rs),ticks=sum(r['ticks']for r in rs))for arm,rs in grouped.items()}
    identical=sum(len({r['initial_public_sha256']for r in rows if r['pair']==pair})==1 for pair in f['panels'])
    save(OUT/'acceptance_metrics.json',dict(arms=metrics,initial_public_identical_pairs=identical,pairs=12,rows=rows))
    print(metrics,flush=True);verify()


if __name__=='__main__':main()
