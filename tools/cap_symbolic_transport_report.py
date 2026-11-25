"""Paired validation of the live symbolic policy, its decisions and the frozen voting comparator."""
from collections import Counter, defaultdict
import csv
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from tools.cap_symbolic_transport import ROOT, base, previous, transport, loop, prepared, measure_ticks, physical, execute_face_change_benchmark

DEST=Path('docs/research/key_findings/2026-11-25_symbolic_transport')


def evaluate():
    plan=loop.read(ROOT/'plan.json')
    cases=[loop.read(ROOT/'fresh'/(s['id']+'.json')) for s in plan['fresh']]
    dev=[loop.read(ROOT/'development'/(s['id']+'.json')) for s in plan['development']]
    votes=loop.read(previous.ROOT/'models.json')['envelope']
    oldcmd=base.setup()[1]['original_action']
    stats=defaultdict(Counter)
    tight=Counter()
    witnesses=Counter()
    actions=Counter()
    audit=[]
    pairs=[]
    paired_changes=Counter()
    for p in cases:
        if p['status']!='ready':
            continue
        branches=p['branches']
        checks=p['live']['checks']
        assert checks.keys()==branches.keys()
        live=p['live']
        keys=[transport.option_key(oldcmd,d,'ccw') for d in transport.DOSES]
        matched=all(k in branches for k in keys)
        fixed=branches[keys[-1]]['actual'] if matched else None
        def outcome(actual):
            return 'wait' if actual is None else ('valid' if actual['valid'] else 'invalid')
        raw=branches[live['selected']]['actual'] if live['selected'] else None
        used=live.get('measured')
        if used and live.get('followup_reason')!='entry_held' and used!=raw:
            raise ValueError('live branch differs from paired counterfactual')
        stats['symbolic_live'][outcome(used)]+=1
        stats['symbolic_before_recheck'][outcome(raw)]+=1
        if live.get('followup_reason')=='entry_held':
            stats['endpoint_hold'][outcome(raw)]+=1
        actions[str(len(p['commands']))]+=1
        if live['selected']:
            br=branches[live['selected']]
            stats['selected_action'][f"{br['cmd']['a0']}/{br['cmd']['a1']} {br['dose']} {br['route']}"]+=1
        stats['union']['any_valid']+=int(any(b['actual']['valid'] for b in branches.values()))
        for key,br in branches.items():
            c=checks[key];o=br['actual']
            tight['branches']+=1
            if c['admitted']:
                tight['admitted']+=1
                tight['invalid_admitted']+=int(not o['valid'])
                tight['admitted_push_contact']+=int(o['push_contact'])
                tight['admitted_follow_contact']+=int(o['follow_contact'])
                tight['admitted_incomplete_face']+=int(not o['completed'])
            else:
                tight['withheld_valid']+=int(o['valid'])
            for metric,interval in c['effects'].items():
                lo,hi=interval
                if lo is None and hi is None:
                    continue
                tight[metric+'_tested']+=1
                tight[metric+'_outside']+=int(lo is not None and 1000*o[metric]<lo or hi is not None and 1000*o[metric]>hi)
            for reason in {i.get('metric',i['property']) for i in c['issues'] if i['property']!='model_quality'}:
                witnesses[reason]+=1
            audit.append(dict(id=p['id'],candidate=key,admitted=c['admitted'],status=c['status'],
                support=c['support_rooms'],E_lower_mm=c['cell']['clearance'][0],
                required_entry_mm=c['route_width_mm'],predicted_entry_lower_mm=c['post']['route_entry_gap'][0],
                predicted_push_min_mm=c['post']['push_min_gap'][0],predicted_full_min_mm=c['post']['full_min_gap'][0],
                actual_push_contact=o['push_contact'],actual_follow_contact=o['follow_contact'],
                actual_face=o['completed'],valid=o['valid'],selected=(key==live['selected'])))
        rec=dict(id=p['id'],symbolic_key=live['selected'],symbolic_live=outcome(used),before_recheck=outcome(raw),
                 followup_reason=live.get('followup_reason'))
        if matched:
            oldrow=p['prefix_baseline'] if 'prefix_baseline' in p else None
            if oldrow is None:
                spec=next(s for s in plan['fresh'] if s['id']==p['id'])
                feat=previous.prefix(base.room(spec)[0])
                if feat['public_history']!=p['public_history']:
                    raise ValueError('baseline cannot see a different history')
                wall=feat['walls']['envelope']; feats=feat['features']
            else:
                wall,feats=oldrow['wall'],oldrow['features']
            v=next((k for k,d in zip(keys,transport.DOSES) if base.risk_decision(votes,feats,wall,d,'wall_scan')=='admit'),None)
            k=next((k for k in keys if checks[k]['admitted']),None)
            vo=outcome(branches[v]['actual'] if v else None)
            so=outcome(branches[k]['actual'] if k else None)
            stats['matched_symbolic'][so]+=1
            stats['matched_vote'][vo]+=1
            stats['matched_short'][outcome(fixed)]+=1
            paired_changes[f'{vo}->{so}']+=1
            short_ok=fixed['valid']; long_ok=branches[keys[0]]['actual']['valid']
            group=('neither','long_only','short_only','both')[2*int(short_ok)+int(long_ok)]
            stats['matched_'+group+':symbolic'][so]+=1
            stats['matched_'+group+':vote'][vo]+=1
            rec.update(group=group,vote=vo,symbolic_ccw=so)
        pairs.append(rec)
    return (dict(development=Counter(p['status'] for p in dev),fresh=Counter(p['status'] for p in cases),
                dev_forced=sum(len(p['branches']) for p in dev),
                dev_twins=sum(b['free'] is not None for p in dev for b in p['branches'].values()),
                ready_headings=actions,stats=stats,candidates=tight,obligations=witnesses,
                paired_changes=paired_changes),audit,pairs)


class OriginalCandidateSet(transport.TransportPlanner):
    """Use the same supported doses and counterclockwise continuation as the voting gate."""
    def choose(self,perception,obs,commands):
        _,checks=super().choose(perception,obs,commands)
        restricted={k:c for k,c in checks.items() if c['continuation']=='ccw'}
        return transport.contract_planner.first_admitted(restricted),restricted


def restricted_live(job):
    spec,model=job
    p=loop.read(ROOT/'fresh'/(spec['id']+'.json'))
    original=base.setup()[1]['original_action']
    keys=[transport.option_key(original,d,'ccw') for d in transport.DOSES]
    if not all(k in p['branches'] for k in keys):
        return None
    expected=next((k for k in keys if p['live']['checks'][k]['admitted']),None)
    if expected is None:
        return dict(id=p['id'],choice='wait')
    fixture,plane,angle=base.room(spec)
    with prepared(fixture) as (world,memory,obs,commands):
        assert world.history.sha256==p['public_history'] and original in commands
        with measure_ticks(world,plane,angle) as samples:
            live=execute_face_change_benchmark(world,memory,OriginalCandidateSet(model),obs,[original])
        assert live['selected']==expected
        _,axis=transport.push_phases.pusher(original)
        actual=physical(samples,live['push_end_tick'],axis,live['followup_completed'])
        frozen=p['branches'][expected]['actual']
        if live.get('followup_reason')!='entry_held' and actual!=frozen:
            raise ValueError('matched live execution diverged from physical fork')
        return dict(id=p['id'],choice='valid' if actual['valid'] else 'invalid',reason=live.get('followup_reason'),
                    contact=actual['push_contact'] or actual['follow_contact'])


def restricted_audit():
    plan=loop.read(ROOT/'plan.json')
    model=loop.read(ROOT/'physical_model.json')
    with ProcessPoolExecutor(8) as pool:
        rows=[r for r in pool.map(restricted_live,[(s,model) for s in plan['fresh']]) if r]
    return dict(counts=Counter(r['choice'] for r in rows),rows=rows)


def obligation_audit():
    """Delete only named geometric obligations and reuse the same forced physical branches."""
    cases=[loop.read(f) for f in sorted((ROOT/'fresh').glob('*.json'))]
    variants={'forward_and_support':{'push_min_gap','route_entry_gap','full_min_gap'},
              'push_path_only':{'route_entry_gap','full_min_gap'},
              'both_sweeps':{'route_entry_gap'},'full_contract':set()}
    result={}
    for name,ignored in variants.items():
        counts=Counter()
        for p in cases:
            if p['status']!='ready':
                continue
            admitted=[]
            for k,c in p['live']['checks'].items():
                if any(i['property']!='model_quality' and i.get('metric') not in ignored for i in c['issues']):
                    continue
                a=p['branches'][k]['actual']
                admitted.append(a)
                counts['admitted']+=1
                counts['valid']+=int(a['valid'])
                counts['static_contact']+=int(a['push_contact'] or a['follow_contact'])
            if admitted:
                counts['selected']+=1
                counts['selected_valid']+=int(admitted[0]['valid'])
                counts['selected_static_contact']+=int(admitted[0]['push_contact'] or admitted[0]['follow_contact'])
        result[name]=counts
    return result


if __name__=='__main__':
    DEST.mkdir(parents=True,exist_ok=True)
    stats,audit,pairs=evaluate()
    loop.seal(DEST/'metrics.json',stats)
    for name,rows in [('contracts',audit),('paired',pairs)]:
        with (DEST/(name+'.csv')).open('w') as f:
            fields=list(dict.fromkeys(k for r in rows for k in r))
            w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
    loop.seal(DEST/'obligations.json',obligation_audit())
    loop.seal(DEST/'restricted_live.json',restricted_audit())
    print(stats)
