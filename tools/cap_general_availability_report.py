"""Summarize the fixed-command and actor-resumption controls by paired outcome."""
import csv
from collections import Counter
from pathlib import Path
from statistics import median
from math import dist

from tools import cap_general_actor_availability as audit

OUT=audit.OUT
DOCS=Path(__file__).resolve().parents[1]/'docs/research/key_findings/2026-11-20_hard_actor_fit'


def mask_loss(s):
    if s['allowed']:
        return 'allowed'
    if not s['rays']:
        return 'no_target_rays'
    if not s['faces']:
        return 'no_usable_face'
    if s['matching']==0:
        return 'no_matching_normal'
    return 'other_mask'


def paired(rows,key):
    eligible=[r for r in rows if key in r]
    changes=Counter((r['perturbed'],r[key]) for r in eligible)
    return dict(tested=len(eligible),zero_to_one=changes[0,1],one_to_zero=changes[1,0],
                initial_success=sum(r['perturbed'] for r in eligible),success=sum(r[key] for r in eligible))


def geometry(rows,folder):
    stages={}
    for stage in ('first','illegal'):
        robot_cm,object_cm,tick_gap=[],[],[]
        for row in rows:
            data=audit.ref.read(folder/(row['room']+'_'+str(row['actor'])+'.json'))
            a,b=data['original'][stage],data['perturbed'][stage]
            robot_cm.append(100*max(dist(x,y) for x,y in zip(a['robot_xy'],b['robot_xy'])))
            object_cm.append(100*dist(a['object_xy'],b['object_xy']))
            tick_gap.append(abs(a['tick']-b['tick']))
        stages[stage]=dict(median_max_robot_cm=median(robot_cm),median_object_cm=median(object_cm),
                           median_abs_tick_gap=median(tick_gap))
    degree=[]
    alternatives=Counter()
    rank_pool=Counter()
    for r in rows:
        data=audit.ref.read(folder/(r['room']+'_'+str(r['actor'])+'.json'))
        for j,robot in enumerate(r['rejection']['now']['robot_support'] or []):
            if robot['allowed'] or robot['matching'] is None:
                continue
            before,after=[data[m]['illegal']['angles'][j] for m in ('original','perturbed')]
            assert before is not None and before<=35.000001
            if after is not None:
                assert after>34.999999
                degree.append(after)
        if r['first']['stage']=='rank':
            old,new=[data[m]['first']['pool'] for m in ('original','perturbed')]
            rank_pool['identical_candidates' if len(old)==len(new) and all(p in new for p in old) else 'changed_candidates']+=1
        if r['first']['stage']!='proposal':
            continue
        original=r['first']['action']
        opts=data['perturbed']['first']['pool']
        same_target=[v for v in opts if v['target_id']==original['target_id']]
        same_type=[v for v in same_target if all(audit.ref.ACTIONS[v[k]].type_index==audit.ref.ACTIONS[original[k]].type_index for k in ('a0','a1'))]
        if any(all(v[k]==original[k] for k in ('a0','a1')) for v in same_target):
            key='duration_only'
        elif same_type:
            key='same_types_different_scale'
        elif same_target:
            key='same_target_different_types'
        else:
            key='target_absent_from_pool'
        alternatives[key]+=1
    return dict(stages=stages,alternatives=dict(alternatives),rank_pool=dict(rank_pool),
                visible_incompatible_faces=dict(n=len(degree),within_five_degrees=sum(a<=40 for a in degree),
                                                median_angle=median(degree) if degree else None))


def summarize(rows):
    blocked=[r for r in rows if r['rejection']]
    first_stage=lambda r:r['first']['stage'] if r['first'] else 'no_difference'
    channels=Counter()
    causes=Counter()
    macros=Counter()
    for r in blocked:
        n=r['rejection']['now']
        cause='target_not_visible' if n['target_points']==0 and r['rejection']['action']['target_id'] is not None else 'visible_target_pair_mask' if n['target_points'] else 'global_menu'
        causes[cause]+=1
        if cause=='visible_target_pair_mask':
            assert not n['target_pair']
            sides=[s for s in n['robot_support'] if not s['allowed']]
            assert sides
            for s in sides:
                channels[mask_loss(s)]+=1
                macros[s['label'].split('[')[0]]+=1
    earlier=[r for r in blocked if r['first']['step']<r['rejection']['step']]
    for r in rows:
        if 'resume_at_rejection' in r and first_stage(r)=='illegal':
            assert r['resume_at_rejection']==r['perturbed']
    return dict(n=len(rows),rooms=len({r['room'] for r in rows}),ref_success=sum(r['original'] for r in rows),
                actual_success=sum(r['perturbed'] for r in rows),fixed_status=dict(Counter(r['fixed_status'] for r in rows)),
                first=dict(Counter(first_stage(r) for r in rows)),causes=dict(causes),mask_by_robot=dict(channels),
                mask_actions=dict(macros),earlier=len(earlier),
                rank_old_winner_was_available=sum(r['first']['winner_also_in_old'] for r in rows if first_stage(r)=='rank'),
                duration_only_available=sum(r['first']['now']['alt_same_parameters'] for r in rows if first_stage(r)=='proposal'),
                errors=dict(Counter(r['rejection']['now']['reason'] for r in blocked)),
                median_steps_to_rejection=median([r['rejection']['offset'] for r in blocked]) if blocked else None,
                median_first_to_rejection=median(r['rejection']['step']-r['first']['step'] for r in earlier) if earlier else None,
                force_one=paired(rows,'one_forced_original'),until_block=paired(rows,'resume_at_rejection'))


def report():
    rows=[]
    for path in sorted((OUT/'rooms').glob('*.json')):
        case=audit.ref.read(path)
        for r in case['branches']:
            rows.append(dict(room=case['room'],**r))
    primary=[r for r in rows if r['primary']]
    assert len(rows)==378 and len(primary)==71 and sum(r['flip'] for r in rows)==92
    stats={}
    groups=dict(primary=primary,flipped=[r for r in rows if r['flip']],unchanged=[r for r in rows if not r['flip']],whole=rows)
    for name in ('flipped','unchanged','primary'):
        for z in (0,1):
            groups[name+('_old_success' if z else '_old_failure')]=[r for r in groups[name] if r['original']==z]
    for stage in ('proposal','rank','illegal'):
        groups['primary_'+stage]=[r for r in primary if r['first']['stage']==stage]
    for name,subset in groups.items():
        stats[name]=summarize(subset)
    if len(list((OUT/'geometry').glob('*.json')))==len(primary):
        stats['geometry']=geometry(primary,OUT/'geometry')
    audit.ref.save(OUT/'FINISHED.json',stats)
    with (DOCS/'availability_per_branch.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['room','actor','factual_success','perturbed_success','selected_71','fixed_status',
                    'first_change','first_boundary','first_illegal_boundary','legality_failure','target_public_points',
                    'per_robot_local_mask','after_one_forced_command','after_forcing_until_rejection'])
        for r in rows:
            reject=r['rejection']
            now=reject['now'] if reject else {}
            w.writerow([r['room'],['old','layernorm'][r['actor']],r['original'],r['perturbed'],int(r['primary']),r['fixed_status'],
                        r['first']['stage'] if r['first'] else '',r['first']['step'] if r['first'] else '',
                        reject['step'] if reject else '',now.get('reason'),now.get('target_points'),
                        '|'.join(mask_loss(s) for s in now.get('robot_support') or []),
                        r.get('one_forced_original'),r.get('resume_at_rejection')])
    return stats


if __name__=='__main__':
    report()
