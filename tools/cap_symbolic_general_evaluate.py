"""Frozen paired candidate and whole-task evaluation; truth is scoring-only."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from tools.cap_symbolic_general import ROOT, OUT, AUTHOR, read, save, immutable, sha, worker
from tools.cap_symbolic_general_study import candidate_actions, fork_worker, state
from tools.cap_symbolic_general_compile import compile_policy
from experiments.planar_tasks.authoring_v2.symbolic_general import frontend, scaffold, simulation
from experiments.planar_tasks.authoring_v2.symbolic_general.online import OnlineRollout
from tools.cap_symbolic_general_runtime import Controller, assess, VERSION as RUNTIME_VERSION
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel, outcome_valid


def freeze_evaluation(source_path):
    frozen=read(OUT/'policy_frozen.json')
    from experiments.planar_tasks.authoring_v2.random_room_author import RoomCampaign
    from experiments.planar_tasks.authoring_v2.checked_submission import verify_checked
    campaign=RoomCampaign(AUTHOR/'campaign').verify_frozen()
    checked=verify_checked(Path(source_path).parent)
    if sha(source_path)!=frozen['source_sha256']:raise ValueError('wrong submitted policy')
    models=[OUT/'models'/f'v{k}.json' for k in range(3)]
    for p in models:EffectModel(read(p))
    host_sources=dict(read(AUTHOR/'campaign/sources.json'))
    source_amendments={}
    for rel,old_hash in list(host_sources.items()):
        current=sha(ROOT/rel)
        if current!=old_hash:
            source_amendments[rel]=dict(author_snapshot=old_hash,evaluation_snapshot=current)
            host_sources[rel]=current
            target=OUT/'host_sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists()and sha(target)!=current:raise ValueError('host snapshot differs')
            target.write_bytes((ROOT/rel).read_bytes())
    for name in ('cap_symbolic_general','cap_symbolic_general_study','cap_symbolic_general_evaluate',
                 'cap_symbolic_general_compile','cap_symbolic_general_audit','cap_symbolic_general_runtime','cap_symbolic_general_finish'):
        path=ROOT/'tools'/(name+'.py');rel=str(path.relative_to(ROOT));host_sources[rel]=sha(path)
        target=OUT/'host_sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists()and sha(target)!=sha(path):raise ValueError('host source snapshot differs')
        target.write_bytes(path.read_bytes())
    packet=dict(source=str(Path(source_path).resolve()),source_sha256=sha(source_path),host_sources=host_sources,
        source_amendments=source_amendments,runtime_version=RUNTIME_VERSION,
        campaign_verification=campaign,checked_submission_source=checked['source_sha256'],
        model_files=[str(p) for p in models],model_sha256=[sha(p) for p in models],
        protocol_sha256=sha(OUT/'protocol.json'),runtime_protocol_sha256=sha(OUT/'runtime_protocol.json'),panels_sha256=sha(OUT/'panels.json'),
        roots_at_ticks=[0,120,300,600],candidates_per_root=4,
        pairing_numeric_tolerance=1e-9,
        root_generator='frozen pre-author starter; first nonbrake at fixed clocks',
        candidate_order='actual unguarded selected, alternate duration, orbit alternatives, other ranked',
        candidate_queries_per_boundary=2,online_reset='per room per version',
        candidate_denominator='all fixed legal candidates, including rejected and evaluation errors',
        primary_safety='target contact with static geometry at any root or successor physics tick',
        other_safety=['robot_static','robot_peer'],
        next_option_measure='version-specific runtime gate then actual policy selects after root; repeat that selected command at most 120 ticks to measure completion',
        paired_online_metric='all root-screening and actual successor-decision rollouts / (fixed root boundaries plus actual successor decision boundaries) * 100',
        whole_task='fresh start, runtime controller each boundary, actual policy reselects after evidence',
        bootstrap_seed=851125113,bootstrap_replicates=2000,
        scope='same diagonal start and layout family; geometry held compact20',
        confidence='room-paired bootstrap percentile95%; empirical, no universal safety claim')
    immutable(OUT/'evaluation_frozen.json',packet)
    return packet


def verify_freeze():
    packet=read(OUT/'evaluation_frozen.json')
    checks=[(packet['source'],packet['source_sha256']),
            (OUT/'protocol.json',packet['protocol_sha256']),(OUT/'panels.json',packet['panels_sha256'])]
    checks.append((OUT/'runtime_protocol.json',packet['runtime_protocol_sha256']))
    checks+=list(zip(packet['model_files'],packet['model_sha256']))
    checks+=[(ROOT/rel,digest)for rel,digest in packet['host_sources'].items()]
    for path,digest in checks:
        if sha(path)!=digest:raise ValueError('freeze changed: '+str(path))
    return packet


def prepare(workers=4):
    packet=verify_freeze();source=Path(packet['source']).read_text()
    policy=compile_policy(scaffold.verify_source(source))
    fixtures=read(OUT/'panels.json')['groups']['confirmation']
    starter=(AUTHOR/'starter.py').read_text()
    if hashlib.sha256(starter.encode()).hexdigest()!=read(OUT/'bootstrap_source.json')['sha256']:
        raise ValueError('prefix starter changed')
    jobs=[]
    for room,fixture in fixtures.items():
        path=OUT/'confirmation/prefixes'/(room+'.json.gz')
        if not path.exists():jobs.append((room,fixture,starter,path,dict(keep_observations=True,replay=False)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,result in enumerate(pool.map(worker,jobs)):
            state('confirmation_prefixes',completed=i+1,total=len(jobs),last=result)
    manifest=[]
    for room in fixtures:
        path=OUT/'confirmation/prefixes'/(room+'.json.gz');trace=read(path)
        rows=trace['records'];indices=[]
        for threshold in packet['roots_at_ticks']:
            index=next((i for i,r in enumerate(rows) if r['tick']>=threshold and r['choice']['option']!='brake'),None)
            if index is not None and index not in indices:indices.append(index)
        for i in indices:
            r=rows[i];actual=policy.select(r['semantic'])['option']
            actions=candidate_actions(policy,r['semantic'],actual or 'brake')
            manifest.append(dict(room=room,boundary=i,tick=r['tick'],actions=actions,trace=str(path),trace_sha256=sha(path)))
    immutable(OUT/'confirmation/roots.json',manifest)
    print(json.dumps(dict(rooms=len(fixtures),roots=len(manifest),candidates=sum(len(r['actions'])for r in manifest))))


def inside(next_values, outcomes, path_safe=None):
    """Actual joint set coverage, with epistemic permit reset excluded."""
    def contains(row):
        if path_safe is not None and row.get('path_safe')is not None and row['path_safe']!=path_safe:return False
        for name,bounds in row['updates'].items():
            if name.startswith('permit:'):continue
            value=next_values[name]
            if type(bounds)is bool:
                if value!=bounds:return False
            elif bounds is not None and not bounds[0]-1e-10<=value<=bounds[1]+1e-10:return False
        return True
    return any(contains(row)for row in outcomes)


def same_public(first,second,tolerance=1e-9):
    if type(first)is bool or type(second)is bool:return type(first)is type(second)and first==second
    if isinstance(first,(int,float))and isinstance(second,(int,float)):return abs(first-second)<=tolerance
    if isinstance(first,dict)and isinstance(second,dict):
        return first.keys()==second.keys()and all(same_public(first[k],second[k],tolerance)for k in first)
    if isinstance(first,(list,tuple))and isinstance(second,(list,tuple)):
        return len(first)==len(second)and all(same_public(a,b,tolerance)for a,b in zip(first,second))
    return type(first)is type(second)and first==second


def gate_action(policy,model,rollout,memory,observation,z,commands,action,queries_left):
    """Same admission predicate as Controller.decide, for a fixed candidate.

    Root forcing defines the diagnostic candidate experiment only. The next
    selector is always the unmodified submitted policy; full-task routing below
    uses Controller directly. All inputs here are public.
    """
    return assess(policy,model,rollout,memory,observation,z,commands,action,queries_left)



def paired_room(job):
    room,roots,fixture,source,model_files=job
    policy=compile_policy(scaffold.verify_source(source))
    models=[EffectModel(read(p))for p in model_files];rollouts=[OnlineRollout()for _ in models]
    controllers=[Controller(model,rollout=rollout)for model,rollout in zip(models,rollouts)]
    traces={};results=[]
    for root in roots:
        path=Path(root['trace'])
        if str(path)not in traces:
            if sha(path)!=root['trace_sha256']:raise ValueError('prefix trace changed')
            traces[str(path)]=read(path)
        trace=traces[str(path)];index=root['boundary']
        public=trace['public'][index];memory=deepcopy(public['memory']);o=public['observation']
        z,commands=frontend.semantic_packet(memory,o)
        prefix=[r['command']for r in trace['records'][:index]]
        # Save all predictions before any real branch labels exist.
        evaluations=[]
        for version,model in enumerate(models):
            remaining=2;decisions=[]
            for action in root['actions']:
                d=gate_action(policy,model,rollouts[version],deepcopy(memory),deepcopy(o),deepcopy(z),commands,action,remaining)
                remaining-=d['queries'];decisions.append(d)
            evaluations.append(decisions)
        prediction_path=OUT/'confirmation/predictions'/(room+f'_{index:04d}.json.gz')
        save(prediction_path,dict(root=root,versions=evaluations))
        root_signatures={}
        for version,decisions in enumerate(evaluations):
            physical=[]
            for d in decisions:
                before=deepcopy(rollouts[version].ledger);old_boundaries=controllers[version].ledger['boundaries']
                try:
                    actual=simulation.branch(fixture,prefix,d['action'],source,controller=controllers[version],retain_root_public=True)
                except Exception as exc:
                    actual=dict(action=d['action'],error=type(exc).__name__+': '+str(exc),
                                reservation_ticks=sum(a['ticks']for a in prefix)+160)
                physical.append(actual)
                follow_delta={k:rollouts[version].ledger[k]-before[k]for k in before}
                follow_boundaries=controllers[version].ledger['boundaries']-old_boundaries
                error=actual.get('error');valid=not error
                signature=actual.get('root_public_sha256')
                if valid and d['action']not in root_signatures:root_signatures[d['action']]=signature
                paired_root_bitwise_matches=bool(valid and signature==root_signatures.get(d['action']))
                paired_root_matches=bool(valid and same_public(o,actual['root_public']))
                complete=bool(actual.get('successor_completed',False));safe=bool(actual.get('safe_effective',False))
                outcomes=d.pop('interval_outcomes');pred=d.pop('prediction')
                row=dict(room=room,boundary=index,tick=root['tick'],version=version,
                    action=d['action'],admitted=d['allowed'],source=d['source'],predicted_risk=d['predicted_risk'],
                    authored_action=any(r['option']==d['action']for r in policy.rules),
                    nominal_first_choice=policy.select(z)['option']==d['action'],
                    supported=d['supported'],calibrated=d['calibrated'],local_rooms=d['local_rooms'],
                    actual_completed=complete,safe_effective=safe,error=error,
                    root_static=actual.get('audit',{}).get('object_static'),
                    full_static=actual.get('full_object_static'),
                    root_progress=actual.get('audit',{}).get('goal_gain'),
                    successor=actual.get('successor'),
                    paired_root_matches=paired_root_matches,paired_root_bitwise_matches=paired_root_bitwise_matches,root_public_sha256=signature,
                    joint_set_covered=inside(actual['next_semantic'],outcomes,
                        path_safe=not actual['audit']['object_static'])if valid and outcomes else None,
                    actual_next_executable=actual.get('successor',{}).get('executable',False),
                    actual_contract_valid=(outcome_valid(policy,d['action'],actual.get('successor_semantic',actual['next_semantic']),
                        path_safe=not actual['audit']['object_static'])[0]if valid else False),
                    successor_boundaries=follow_boundaries,
                    online_root_branches=d['rollout_delta']['branches'],online_successor_branches=follow_delta['branches'],
                    online_branches=d['rollout_delta']['branches']+follow_delta['branches'],
                    online_ticks=d['rollout_delta']['ticks']+follow_delta['ticks'],
                    online_models=d['rollout_delta']['models']+follow_delta['models'],
                    online_incompatible=d['rollout_delta']['incompatible']+follow_delta['incompatible'],
                    online_queries=d['queries'],symbolic_status=(d['symbolic']or {}).get('status'))
                if valid and pred['samples']:
                    row['distance_mae']=abs(np.mean([s['values']['distance']for s in pred['samples']])-actual['next_semantic']['distance'])
                results.append(row)
            save(OUT/'confirmation/branches'/f'v{version}'/(room+f'_{index:04d}.json.gz'),
                 dict(prefix=prefix,actions=root['actions'],rows=physical,version=version,
                      source_sha256=hashlib.sha256(source.encode()).hexdigest(),model_sha256=models[version].sha256))
    packet=dict(room=room,roots=len(roots),rows=results,online=[r.ledger for r in rollouts])
    for version,rollout in enumerate(rollouts):
        measured=[r for r in results if r['version']==version]
        if sum(r['online_branches']for r in measured)!=rollout.ledger['branches']:
            raise ValueError('unaccounted online branches')
        if sum(r['online_ticks']for r in measured)!=rollout.ledger['ticks']:
            raise ValueError('unaccounted online physics ticks')
    save(OUT/'confirmation/paired'/(room+'.json.gz'),packet)
    return dict(room=room,roots=len(roots),rows=len(results))


def paired(workers=4):
    freeze=verify_freeze();source=Path(freeze['source']).read_text()
    fixtures=read(OUT/'panels.json')['groups']['confirmation'];roots=read(OUT/'confirmation/roots.json')
    jobs=[]
    for room,fixture in fixtures.items():
        if (OUT/'confirmation/paired'/(room+'.json.gz')).exists():continue
        jobs.append((room,[r for r in roots if r['room']==room],fixture,source,freeze['model_files']))
    with ProcessPoolExecutor(max_workers=workers)as pool:
        futures=[pool.submit(paired_room,j)for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            result=f.result();state('paired_confirmation',completed=i+1,total=len(jobs),last=result)


def task_worker(job):
    room,fixture,source,version,model_path=job
    compile_policy(scaffold.verify_source(source))
    controller=Controller(EffectModel(read(model_path)))if model_path else None
    start=time.monotonic()
    result=simulation.episode(fixture,source,controller=controller,keep_observations=False,replay=True,
                              wall_seconds=240)
    result['provenance']=dict(room=room,version=version,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                              model_sha256=controller.model.sha256 if controller else None)
    save(OUT/'confirmation/tasks'/str(version)/(room+'.json.gz'),result)
    return dict(room=room,version=version,**result['summary'])


def tasks(workers=4,split='confirmation',limit=None):
    freeze=verify_freeze();source=Path(freeze['source']).read_text()
    fixtures=read(OUT/'panels.json')['groups'][split]
    if split!='confirmation':raise ValueError('use separate development run output, never confirmation paths')
    jobs=[]
    for version,model in list(enumerate(freeze['model_files']))+[('ungated',None)]:
        for room,fixture in list(fixtures.items())[:limit]:
            if not(OUT/'confirmation/tasks'/str(version)/(room+'.json.gz')).exists():
                jobs.append((room,fixture,source,version,model))
    with ProcessPoolExecutor(max_workers=workers)as pool:
        futures=[pool.submit(task_worker,j)for j in jobs]
        for i,f in enumerate(as_completed(futures)):
            result=f.result();state('whole_task_confirmation',completed=i+1,total=len(jobs),last=result)


def metric(rows,roots):
    admitted=[r for r in rows if r['admitted']]
    covered=[r for r in admitted if r['joint_set_covered']is not None]
    model_covered=[r for r in covered if r['source']=='model']
    authored=[r for r in rows if r.get('authored_action',True)]
    authored_admitted=[r for r in authored if r['admitted']]
    first=[r for r in rows if r.get('nominal_first_choice',False)]
    boundaries=roots+sum(r.get('successor_boundaries',0)for r in rows)
    return dict(candidates=len(rows),root_boundaries=roots,boundaries=boundaries,admitted=len(admitted),
        authored_candidates=len(authored),authored_admitted=len(authored_admitted),
        authored_C=sum(r['safe_effective']for r in authored_admitted)/len(authored)if authored else None,
        authored_F=sum(not r['actual_completed']for r in authored_admitted)/len(authored_admitted)if authored_admitted else None,
        nominal_first_candidates=len(first),
        nominal_first_C=sum(r['admitted']and r['safe_effective']for r in first)/len(first)if first else None,
        effective=sum(r['safe_effective']for r in admitted),
        successor_failed=sum(not r['actual_completed']for r in admitted),
        admitted_static=sum(r['full_static']is True for r in admitted),
        admitted_root_static=sum(r['root_static']is True for r in admitted),
        admitted_root_progress_2cm=sum((r.get('root_progress')or 0)>=.02 for r in admitted),
        model_admissions=sum(r['source']=='model'for r in admitted),
        online_admissions=sum(r['source']=='online_empirical'for r in admitted),
        admitted_contract_violations=sum(not r.get('actual_contract_valid',False)for r in admitted),
        rejected_effective=sum(r['safe_effective']for r in rows if not r['admitted']),
        errors=sum(bool(r['error'])for r in rows),
        pairing_mismatches=sum(not r.get('paired_root_matches',True)for r in rows if not r['error']),
        pairing_bitwise_mismatches=sum(not r.get('paired_root_bitwise_matches',True)for r in rows if not r['error']),
        mean_distance_prediction_error=float(np.mean([r['distance_mae']for r in rows if 'distance_mae'in r]))if any('distance_mae'in r for r in rows)else None,
        C=sum(r['safe_effective']for r in admitted)/len(rows)if rows else None,
        F=sum(not r['actual_completed']for r in admitted)/len(admitted)if admitted else None,
        K=100*sum(r['online_branches']for r in rows)/boundaries if boundaries else None,
        online_branches=sum(r['online_branches']for r in rows),online_ticks=sum(r['online_ticks']for r in rows),
        joint_coverage=sum(r['joint_set_covered']for r in covered)/len(covered)if covered else None,
        coverage_denominator=len(covered),
        model_admitted_joint_coverage=sum(r['joint_set_covered']for r in model_covered)/len(model_covered)if model_covered else None,
        model_covered_contract_violations=sum(r['joint_set_covered']and not r['actual_contract_valid']for r in model_covered),
        model_admitted_coverage_denominator=len(model_covered))


def summarize():
    freeze=verify_freeze();rooms=[];all_rows=[]
    for path in sorted((OUT/'confirmation/paired').glob('*.json.gz')):
        p=read(path);rooms.append(p);all_rows+=p['rows']
    metrics=[metric([r for r in all_rows if r['version']==k],sum(r['roots']for r in rooms))for k in range(3)]
    rng=np.random.default_rng(freeze['bootstrap_seed']);differences=[]
    for before,after in ((0,1),(1,2),(0,2)):
        boot={k:[]for k in ('C','F','K')}
        for _ in range(freeze['bootstrap_replicates']):
            sample=[rooms[i]for i in rng.integers(0,len(rooms),size=len(rooms))]
            selected=[r for p in sample for r in p['rows']];n=sum(p['roots']for p in sample)
            a=metric([r for r in selected if r['version']==before],n);b=metric([r for r in selected if r['version']==after],n)
            for k in boot:
                if a[k]is not None and b[k]is not None:boot[k].append(b[k]-a[k])
        differences.append(dict(before=before,after=after,metrics={k:dict(
            difference=metrics[after][k]-metrics[before][k]if metrics[after][k]is not None and metrics[before][k]is not None else None,
            ci95=np.quantile(v,[.025,.975]).tolist()if v else None,valid_replicates=len(v))for k,v in boot.items()}))
    task_summaries={}
    from scipy.stats import beta
    for version in ('0','1','2','ungated'):
        rs=[dict(room=p.name[:-8],**read(p)['summary'])for p in sorted((OUT/'confirmation/tasks'/version).glob('*.json.gz'))]
        task_summaries[version]=dict(rooms=len(rs),success=sum(r['success']for r in rs),
            object_static_rooms=sum(r['object_static_commands']>0 for r in rs),
            robot_static_rooms=sum(r['robot_static_commands']>0 for r in rs),
            peer_contact_rooms=sum(r['peer_contact_commands']>0 for r in rs),
            mean_cost=float(np.mean([r['task_cost_seconds']for r in rs]))if rs else None,
            errors=sum(bool(r['error'])for r in rs),
            online_branches=sum(r.get('online',{}).get('rollout',{}).get('branches',0)for r in rs),
            boundaries=sum(r['decisions']for r in rs),rows=rs)
        incidents=task_summaries[version]['object_static_rooms'];n=len(rs)
        task_summaries[version]['static_room_risk_upper95']=(float(beta.ppf(.95,incidents+1,n-incidents))
            if n>incidents else 1. if n else None)
        task_summaries[version]['risk_interval_assumption']='one-sided binomial bound over independently generated rooms, not per-action or universal physical guarantee'
    final=differences[-1]['metrics'];c,f,k=(final[x]['difference']for x in ('C','F','K'))
    point=bool(c is not None and f is not None and k is not None and c>=-.05 and f<=.05 and k<0)
    supported=bool(all(final[x]['ci95']is not None for x in ('C','F','K'))and
                   final['C']['ci95'][0]>=-.05 and final['F']['ci95'][1]<=.05 and final['K']['ci95'][1]<0)
    out=dict(paired_rooms=len(rooms),versions=metrics,differences=differences,tasks=task_summaries,
             main_hypothesis=dict(point_criteria_met=point,paired_intervals_support_all=supported,delta_C=.05,delta_F=.05))
    save(OUT/'metrics.json',out)
    import csv
    if all_rows:
        fields=list(dict.fromkeys(k for r in all_rows for k in r))
        with (OUT/'paired.csv').open('w')as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(all_rows)
    print(json.dumps(dict(versions=metrics,differences=differences,tasks={k:{f:v[f]for f in ('rooms','success','object_static_rooms','errors')}for k,v in task_summaries.items()})))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    f=sub.add_parser('freeze');f.add_argument('--policy',type=Path,required=True)
    for op in ('prepare','paired','tasks'):
        s=sub.add_parser(op);s.add_argument('--workers',type=int,default=4)
    sub.add_parser('summarize');a=p.parse_args()
    if a.op=='freeze':print(json.dumps(freeze_evaluation(a.policy)))
    elif a.op=='prepare':prepare(a.workers)
    elif a.op=='paired':paired(a.workers)
    elif a.op=='tasks':tasks(a.workers)
    elif a.op=='summarize':summarize()


if __name__=='__main__':main()
