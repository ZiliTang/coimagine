"""Independent study receipts, selector obligations and model-change dependency audit."""
from copy import deepcopy
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import math

import z3

from tools.cap_symbolic_general import OUT, AUTHOR, read, save, immutable, sha
from experiments.planar_tasks.authoring_v2.symbolic_general import scaffold, frontend
from experiments.planar_tasks.authoring_v2.symbolic_general.core import evaluate
from experiments.planar_tasks.authoring_v2.checked_submission import verify_checked
from experiments.planar_tasks.authoring_v2.random_room_author import RoomCampaign


def freeze_author():
    final=read(AUTHOR/'FINAL.json')
    submission=Path(final['checked_submission']);verified=verify_checked(submission)
    source_path=submission/'policy.py';source=source_path.read_text()
    if sha(final['policy'])!=sha(source_path):raise ValueError('final and checked policy differ')
    policy=scaffold.verify_source(source)
    clock=read(AUTHOR/'clock.json')
    if any(c['status']!='complete'for c in clock['calls']):raise ValueError('unsettled author work')
    campaign=RoomCampaign(AUTHOR/'campaign')
    if read(campaign.path/'campaign.json')['status']!='frozen':
        campaign.verify_sources()
        (campaign.path/'composed.py').write_text(source)
        save(campaign.path/'symbolic_author_receipt.json',dict(final=final,checked=verified,clock=clock,
            source_sha256=sha(source_path),implementation='metered symbolic author tools on authoring_v2 RoomCampaign',
            note='native subagent token usage unavailable; clock and physical simulation measured independently'))
        s=read(campaign.path/'campaign.json')
        s['used'].update(execution_ticks=clock['ticks_reserved'],resets=clock['resets'],attempts=len(clock['calls']))
        s['stage1']['max_decisions']=450
        s['symbolic_submission']=dict(source_sha256=sha(source_path),policy_file='composed.py',checked_receipt='symbolic_author_receipt.json')
        save(campaign.path/'campaign.json',s)
        campaign.freeze()
    checked_campaign=campaign.verify_frozen()
    frozen=dict(source=str(source_path.resolve()),source_sha256=sha(source_path),policy_sha256=policy.sha256,
                submission=verified,campaign=checked_campaign,author_final=final,
                author_seconds=time.time()-clock['start'],author_budget_seconds=clock['wall_seconds'],
                execution_ticks=clock['ticks_reserved'],resets=clock['resets'])
    # Re-running audit does not replace the original completion timestamp.
    if not(OUT/'policy_frozen.json').exists():immutable(OUT/'policy_frozen.json',frozen)
    elif read(OUT/'policy_frozen.json')['source_sha256']!=frozen['source_sha256']:raise ValueError('policy freeze differs')
    return read(OUT/'policy_frozen.json')


def selection_proof(source_path):
    policy=scaffold.verify_source(Path(source_path).read_text());v=policy._symbolic('runtime:')
    # Every non-stop permit may change after runtime evidence acquisition.
    # The brake candidate is never vetoed and must remain available if the menu
    # offers it. No relation between safety evidence and true geometry is assumed.
    constraints=policy._box(v,{'legal:brake':True,'permit:brake':True})
    guards=[evaluate(r['when'],v,True)for r in policy.rules]
    availability=policy._query(constraints,z3.Not(z3.Or(*guards)),v,10000)
    implications=[]
    for guard,rule in zip(guards,policy.rules):
        bad=z3.And(guard,z3.Not(evaluate(policy.host['options'][rule['option']]['requires'],v,True)))
        implications.append(dict(rule=rule['name'],**policy._query(constraints,bad,v,1000)))
    passed=availability['status']=='proved' and all(r['status']=='proved'for r in implications)
    report=dict(status='proved'if passed else 'unknown',availability=availability,rules=implications,
        argument='Finite typed real scores select an enabled maximum with fixed ties. Every enabled rule implies its own executable condition; an enabled rule always exists under the stated brake assumption.')
    packet=dict(policy_sha256=policy.sha256,query='actual selected action executable under arbitrary non-brake evidence permissions',
        assumptions=['brake publicly legal','brake permission true','primitive values within declared domains'],
        result=report,scope='startability only; neither completion nor physical safety follows')
    save(OUT/'runtime_selection_proof.json',packet)
    return packet


def rechecks():
    policy=scaffold.verify_source(Path(read(OUT/'policy_frozen.json')['source']).read_text())
    rows=[]
    for path in sorted((OUT/'confirmation/predictions').glob('*.json.gz')):
        data=read(path);versions=data['versions']
        for a,b in ((0,1),(1,2),(0,2)):
            for old,new in zip(versions[a],versions[b]):
                if old['action']!=new['action']:raise ValueError('candidate mismatch')
                changed=[]
                for name in policy.base:
                    first=[x['updates'].get(name)for x in old['interval_outcomes']]
                    second=[x['updates'].get(name)for x in new['interval_outcomes']]
                    if first!=second:changed.append(name)
                affected=policy.dependencies(changed)
                category=('maintained'if old['allowed']and new['allowed']else
                          'withdrawn'if old['allowed']else 'relaxed'if new['allowed']else 'still_withheld')
                rows.append(dict(room=data['root']['room'],boundary=data['root']['boundary'],action=old['action'],
                    before=a,after=b,changed_metrics=sorted(changed),affected=affected,
                    old_symbolic=(old['symbolic']or {}).get('status'),new_symbolic=(new['symbolic']or {}).get('status'),
                    old_basis=old['source'],new_basis=new['source'],admission_change=category,
                    path_prediction_changed=[x['path_safe']for x in old['interval_outcomes']]!=[x['path_safe']for x in new['interval_outcomes']]))
    save(OUT/'model_rechecks.json.gz',rows)
    summary={}
    for a,b in ((0,1),(1,2),(0,2)):
        selected=[r for r in rows if r['before']==a and r['after']==b]
        summary[f'{a}->{b}']={k:sum(r['admission_change']==k for r in selected)for k in ('maintained','withdrawn','relaxed','still_withheld')}
    save(OUT/'model_recheck_summary.json',summary)
    return summary


def program_rechecks():
    final=read(AUTHOR/'FINAL.json')
    search=next(e for e in final['development_evidence']if e['kind']=='development_search')
    trials=search['trials'];rows=[]
    for first,second in zip(trials,trials[1:]):
        paths=[AUTHOR/'trials'/f"{r['trial']:04d}"/'policy.py'for r in (first,second)]
        before,after=[scaffold.verify_source(p.read_text())for p in paths]
        names=set(before.derived)|set(after.derived)
        changed=[n for n in names if before.derived.get(n)!=after.derived.get(n)]
        direct=[];old_rules={r['name']:r for r in before.rules};new_rules={r['name']:r for r in after.rules}
        for name in set(old_rules)|set(new_rules):
            if old_rules.get(name)!=new_rules.get(name):direct.append(name)
        effects=[n for n in set(before.program.get('expect',{}))|set(after.program.get('expect',{}))
                 if before.program.get('expect',{}).get(n)!=after.program.get('expect',{}).get(n)]
        impacts=[p.dependencies(changed)for p in (before,after)]
        affected_rules=set(direct)|{name for impact in impacts for name in impact['rules']}
        competitors=set(old_rules)|set(new_rules)if affected_rules else set()
        rows.append(dict(before_trial=first['trial'],after_trial=second['trial'],before_sha256=sha(paths[0]),after_sha256=sha(paths[1]),
            changed_derived=sorted(changed),direct_rule_edits=sorted(direct),changed_effect_promises=sorted(effects),
            transitive_impacts=impacts,affected_rules=sorted(affected_rules),
            selection_competitors=sorted(competitors),
            affected_selected_options=sorted({r['option']for p in (before,after)for r in p.rules if r['name']in competitors}),
            development_success_before=first['successes'],development_success_after=second['successes'],
            scope='syntactic dependency and recorded development outcome; not a causal attribution of the edit'))
    save(OUT/'program_rechecks.json',rows)
    return dict(revisions=len(rows),path=str(OUT/'program_rechecks.json'))


def semantic_registry():
    host=scaffold.host_schema();descriptions={
        'seen':'Current target has at least three public returns sufficient for the local fit; not full-shape visibility.',
        'distance':'Distance from public fitted/last-estimated target centre to public goal; not true body distance.',
        'gx':'Public goal x minus fitted/last-estimated centre x; meaningful target geometry requires seen evidence.',
        'gy':'Public goal y minus fitted/last-estimated centre y; meaningful target geometry requires seen evidence.',
        'uncertainty':'Declared heuristic fit scale (.08/.15/1 m), not a calibrated confidence interval.',
        'stagnation':'Public elapsed time without at least .002 m decrease in estimated goal distance; can reflect fit noise.',
        'speed':'Maximum public robot translational speed, not target body velocity.',
        'contact0':'Robot0 public contacting_object flag for the active target.',
        'contact1':'Robot1 public contacting_object flag for the active target.',
        'static_contact':'Any public robot contacting_static flag; the host may classify contact with a non-target entity as an obstacle.',
        'peer_contact':'Any public robot contacting_teammate flag.'}
    rows={}
    for name,spec in host['metrics'].items():
        if name.startswith('legal:'):description='Current public target/global menu membership, plus target fit availability for target-bound commands; startability only.'
        elif name.startswith('permit:'):description='Runtime evidence gate result; reset at each real boundary before inquiry, then recomputed from frozen model and public-compatible rollout.'
        elif name.startswith('gap_'):description='Nearest retained static hit in a directional corridor relative to visible fitted extent. 5 m means no retained hit in that corridor, not certified free space or a clearance lower bound.'
        elif name.startswith('wall_seen_'):description='At least one retained observed static hit in the directional corridor; not complete corridor coverage.'
        elif name.startswith('align_'):description='Cosine between public estimated target-to-goal vector and the declared world push/move direction.'
        else:description=descriptions[name]
        rows[name]=dict(spec,description=description,source='public observation/history'if not name.startswith('permit:')else 'public-input evidence computation',
            refresh='recompute after each actual boundary; same-tick memory update is idempotent',
            invalidated_by_actions=[action for action,option in host['options'].items()if name in option['writes']])
    packet=dict(frontend_sha256=sha(Path(frontend.__file__)),host_schema_sha256=hashlib.sha256(json.dumps(host,sort_keys=True).encode()).hexdigest(),
        metrics=rows,options=host['options'],information_boundary='No real environment/body/fixture object or hidden truth is passed to the frontend or runtime gate.',
        temporal_state='Observed static hit memory is retained but bounded to96 spatial cells; stale target centre is explicitly accompanied by seen=False.')
    save(OUT/'semantic_registry.json',packet)
    return dict(metrics=len(rows),options=len(host['options']),path=str(OUT/'semantic_registry.json'))


def public_boundaries(source_path,count=8):
    """Host-chosen development mid-trajectory checks bind real CAP carrier."""
    from tools.cap_policy_check import run_boundary
    source=Path(source_path).read_text();digest=sha(source_path)
    reports=[]
    paths=sorted((OUT/'bootstrap/calibration').glob('*.json.gz'))[:4]
    for path in paths:
        trace=read(path)
        for threshold in (120,600):
            index=next((i for i,r in enumerate(trace['records'])if r['tick']>=threshold),None)
            if index is None:continue
            public=trace['public'][index]
            z,commands=frontend.semantic_packet(public['memory'],public['observation'])
            choice=scaffold.verify_source(source).select(z)
            if choice['option']is None:raise ValueError('missing actual action')
            boundary=dict(source_sha256=digest,memory=public['memory'],observation=public['observation'],action=commands[choice['option']])
            report=run_boundary(source,boundary)
            if not report['selection_checked']or report.get('findings'):raise ValueError('actual CAP carrier binding failed')
            reports.append(dict(room=path.name[:-8],boundary=index,report=report))
    save(OUT/'host_checked_boundaries.json',reports)
    return dict(checked=len(reports),source_sha256=digest)


def costs():
    groups={}
    clock=read(AUTHOR/'clock.json')
    groups['author']=dict(ticks=clock['ticks_reserved'],resets=clock['resets'],calls=len(clock['calls']))
    for name,folder in [('bootstrap_development',OUT/'bootstrap'),('on_policy_development',OUT/'on_policy'),
                        ('confirmation_prefixes',OUT/'confirmation/prefixes')]:
        paths=list(folder.rglob('*.json.gz'));rs=[read(p)['summary']for p in paths]
        groups[name]=dict(ticks=sum(r['ticks']for r in rs),episodes=len(rs),
                         reservation_errors=sum(bool(r.get('ticks_are_reservation'))for r in rs))
    for name,folder in [('development_forks',OUT/'forks'),('confirmation_forks',OUT/'confirmation/branches')]:
        rs=[r for p in folder.rglob('*.json.gz')for r in read(p)['rows']]
        ok=[r for r in rs if not r.get('error')]
        groups[name]=dict(branches=len(rs),prefix_ticks=sum(r['prefix_ticks']for r in ok),
            action_ticks=sum(r['root_ticks']for r in ok),successor_ticks=sum(r['follow_ticks']for r in ok),
            errors=len(rs)-len(ok),unsettled_reserved_ticks=sum(r.get('reservation_ticks',0)for r in rs if r.get('error')))
    for version in (1,2):
        folder=OUT/'active'/f'v{version}'
        episodes=[read(p)['summary']for p in folder.glob('*/episodes/*.json.gz')]
        branches=[r for p in folder.glob('*/forks/*.json.gz')for r in read(p)['rows']]
        ok=[r for r in branches if not r.get('error')]
        online=[r['online']['rollout']for r in episodes if 'online'in r]
        online += [r['development_online']['rollout']for r in branches if 'development_online'in r]
        groups[f'active_model_round_{version}']=dict(episodes=len(episodes),episode_ticks=sum(r['ticks']for r in episodes),
            branches=len(branches),prefix_ticks=sum(r['prefix_ticks']for r in ok),root_ticks=sum(r['root_ticks']for r in ok),
            successor_ticks=sum(r['follow_ticks']for r in ok),errors=len(branches)-len(ok),
            online_branches=sum(r['branches']for r in online),online_ticks=sum(r['ticks']for r in online))
    for kind in ('paired','tasks'):
        rs=[]
        for p in (OUT/'confirmation'/kind).rglob('*.json.gz'):
            record=read(p)
            if kind=='paired':rs+=record['online']
            elif 'online'in record['summary']:rs.append(record['summary']['online']['rollout'])
        groups['online_'+kind]={k:sum(r[k]for r in rs)for k in ('branches','ticks','models','incompatible','errors','seconds','budget_exhausted')}
    paths=[p for p in (OUT/'engineering').rglob('*.json.gz')if p.name!='model_rechecks.json.gz']
    rs=[]
    for p in paths:
        r=read(p)
        if 'summary'in r:rs.append(r['summary'])
    groups['recorded_engineering_episodes']=dict(episodes=len(rs),ticks=sum(r['ticks']for r in rs),
        online_branches=sum(r.get('online',{}).get('rollout',{}).get('branches',0)for r in rs))
    packet=dict(scope='saved experiment artifacts; includes repeated physical prefixes and conservative unresolved reservations; model fitting and SMT CPU cost are not simulation ticks',groups=groups)
    save(OUT/'cost_accounting.json',packet);return packet


def raw_replay_audit():
    """Recompute endpoint progress and polygon overlap from stored every-tick poses."""
    from experiments.planar_tasks.authoring_v2.random_room_tasks import polygon,separated
    rows=[]
    for path in sorted((OUT/'confirmation/tasks').rglob('*.json.gz')):
        data=read(path);replay=data['replay'];scene=replay['scene'];frames=replay['frames']
        goal=[scene['goal']['x'],scene['goal']['y']];target=scene['primary_id'];outline=scene['object_polygon']
        obstacles=[polygon(o['cx'],o['cy'],2*o['half_w'],2*o['half_h'],o['angle'])for o in scene['obstacles']]
        overlaps=0;out_of_arena=0
        for frame in frames[1:]:
            x,y,a=frame['entities'][target][:3];c,s=math.cos(a),math.sin(a)
            points=[(x+c*u-s*v,y+s*u+c*v)for u,v in outline]
            overlaps+=any(not separated(points,wall,gap=0.)for wall in obstacles)
            out_of_arena+=any(px<0 or px>scene['arena'][0]or py<0 or py>scene['arena'][1]for px,py in points)
        first=frames[0]['entities'][target][:2];last=frames[-1]['entities'][target][:2]
        gain=math.dist(first,goal)-math.dist(last,goal);summary=data['summary']
        row=dict(room=path.name[:-8],version=path.parent.name,frames=len(frames),
            ticks=summary['ticks'],frame_tick_matches=frames[-1]['tick']==summary['ticks'],
            goal_gain_recomputed=gain,goal_gain_reported=summary['goal_gain'],
            goal_gain_error=abs(gain-summary['goal_gain']),final_centre_distance=math.dist(last,goal),
            success_reported=summary['success'],success_within_goal=not summary['success']or math.dist(last,goal)<=scene['goal']['radius']+1e-9,
            polygon_overlap_ticks=overlaps,out_of_arena_ticks=out_of_arena,
            backend_static_contact_ticks=summary['object_static_ticks'],
            overlap_without_backend_contact=bool(overlaps and not summary['object_static_ticks']))
        rows.append(row)
    packet=dict(scope='independent pose-based SAT audit; exact polygon overlap differs from physics contact tolerance, so both are retained',
        rooms=len(rows),goal_gain_mismatches=sum(r['goal_gain_error']>1e-8 for r in rows),
        frame_tick_mismatches=sum(not r['frame_tick_matches']for r in rows),
        success_outside_goal=sum(not r['success_within_goal']for r in rows),
        overlap_without_backend_contact=sum(r['overlap_without_backend_contact']for r in rows),rows=rows)
    save(OUT/'raw_replay_audit.json',packet);return {k:v for k,v in packet.items()if k!='rows'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('op',choices=['freeze-author','selection','boundaries','rechecks','program-rechecks','semantic-registry','costs','raw-replay']);p.add_argument('--policy',type=Path)
    a=p.parse_args()
    if a.op=='freeze-author':r=freeze_author()
    elif a.op=='selection':r=selection_proof(a.policy)
    elif a.op=='boundaries':r=public_boundaries(a.policy)
    elif a.op=='rechecks':r=rechecks()
    elif a.op=='program-rechecks':r=program_rechecks()
    elif a.op=='semantic-registry':r=semantic_registry()
    elif a.op=='costs':r=costs()
    else:r=raw_replay_audit()
    print(json.dumps(r))


if __name__=='__main__':main()
