"""Summarize same-prefix causal interventions without counting censored scripts as failures."""
from collections import Counter
from statistics import median
import csv
from pathlib import Path

from tools import cap_general_actor_forks as study

OUT=study.OUT
read,save=study.read,study.save


def cell(row):
    return ('both_fail','rescued','lost','both_succeed')[2*row['actual'][0]+row['actual'][1]]


def attribution(row):
    winner=row['actual'].index(1)
    rescue=bool(row['cross'][1-winner])
    break_win=not row['cross'][winner]
    return ('suffix','break_only','rescue_only','first')[2*rescue+break_win]


def teacher_stats(rows):
    eligible=[r for r in rows if 'teacher' in r]
    results=[]
    for actor in range(2):
        counts=Counter()
        for r in eligible:
            item=r['teacher']['actors'][actor]
            counts['hits']+=item['hits']
            counts['multi']+=r['teacher']['multi']
            counts['failed' if not r['actual'][actor] else 'won']+=1
            if item['first_error'] is not None:
                counts['tested']+=1
                counts['rescued' if item['repair'] and not r['actual'][actor] else 'lost' if r['actual'][actor] and not item['repair'] else 'unchanged']+=1
        results.append(dict(counts))
    return dict(eligible=len(eligible),actors=results)


def micro_stats(rows):
    pairs=[(r,i,m) for r in rows if r['cut'] is not None for i,m in enumerate(r['micro'])]
    flips=[(r,i,m) for r,i,m in pairs if r['actual'][i]!=m['success']]
    changed=[(r,i,m) for r,i,m in pairs if m['dt_delta']]
    def summary(v):
        return dict(n=len(v),flips=sum(r['actual'][i]!=m['success'] for r,i,m in v),
                    rooms=len({r['room'] for r,i,m in v}),
                    robot_median_cm=median(m['robot_delta'] for _,_,m in v)*100 if v else None,
                    object_median_cm=median(m['xy_delta'] for _,_,m in v)*100 if v else None)
    cmds=Counter()
    for r,i,m in flips:
        if m['fixed_status']=='complete':
            assert m['fixed_success'] in (0,1)
            cmds['flipped' if m['fixed_success']!=r['actual'][i] else 'same_as_original']+=1
        else:
            cmds[m['fixed_status']]+=1
    return dict(all=summary(pairs),executed_change=summary(changed),flips=summary(flips),
                flip_durations=dict(Counter(m['dt_delta'] for _,_,m in flips)),
                rescued=sum(not r['actual'][i] for r,i,m in flips),
                lost=sum(r['actual'][i] for r,i,m in flips),fixed=dict(cmds),
                same_next_cmd=sum(m['next_diff'] is None or m['next_diff']>0 for r,i,m in flips
                                  if m['next_diff'] is not None),by_actor=[summary([p for p in pairs if p[1]==i]) for i in (0,1)],
                by_kind={k:summary([p for p in pairs if k in p[0]['kinds'][p[1]]]) for k in sorted({k for r,i,_ in pairs for k in r['kinds'][i]})})


def report():
    rooms=[read(p) for p in sorted((OUT/'rooms').glob('*.json'))]
    late={r['room']:r for r in (read(p) for p in (OUT/'later').glob('*.json'))}
    assert len(rooms)==200 and len(late)==61
    result={}
    for name in ('all','both_fail','rescued','lost','both_succeed','discordant'):
        rows=[r for r in rooms if name=='all' or name==cell(r) or name=='discordant' and r['actual'][0]!=r['actual'][1]]
        split=[r for r in rows if r['cut'] is not None]
        discord=[r for r in split if r['actual'][0]!=r['actual'][1]]
        x=dict(rooms=len(rows),forks=len(split),teacher=teacher_stats(rows),micro=micro_stats(rows))
        if split:
            x.update(initial=sum(r['cut']==0 for r in split),median_step=median(r['cut'] for r in split),
                     cap_only_duration=sum(all(r['actions'][0][k]==r['actions'][1][k] for k in ('a0','a1','target_id')) for r in split),
                     same_kinds=sum(r['kinds'][0]==r['kinds'][1] for r in split),
                     teacher_at=dict(Counter(str(r['teacher_at_cut']) for r in split)),
                     supported=dict(Counter(str([int(bool(s)) for s in r['supported']]) for r in split)))
        if discord:
            assert all(late[r['room']]['first']==r['cut'] for r in discord)
            ls=[late[r['room']] for r in discord]
            later=lambda r:any(t['success'] for t in late[r['room']]['tested'])
            early=lambda r:r['cross'][r['actual'].index(0)]
            x.update(attribution=dict(Counter(attribution(r) for r in discord)),later=dict(
                rooms_tested=sum(bool(l['tested']) for l in ls),
                trials=sum(len(l['tested']) for l in ls),hits=sum(t['success'] for l in ls for t in l['tested']),
                rescued_rooms=sum(later(r) for r in discord),first_rescued_rooms=sum(early(r) for r in discord),
                extra_rooms=sum(later(r) and not early(r) for r in discord),
                any_room=sum(bool(early(r) or later(r)) for r in discord),
                rescued_teacher=dict(Counter(t['teacher'] for l in ls for t in l['tested'] if t['success']))))
        result[name]=x
    a=result['all']['teacher']['actors']
    assert [(v['hits'],v['multi']) for v in a]==[(1866,3162),(2302,3162)]
    matched={r['room']:r for r in rooms}
    labels=[read(p) for p in (OUT/'teacher_disagreements').glob('*.json')]
    assert {r['room'] for r in labels}=={r['room'] for r in rooms if 'teacher' in r}
    audit={}
    for name in ('old_only','ln_only'):
        cases=[(matched[r['room']],r[name]) for r in labels if r[name]['count']]
        outcome=lambda subset:dict(Counter(''.join(map(str,branch['success'])) for _,branch in subset))
        natural=[(room,branch) for room,branch in cases if room['teacher']['actors'][1]['first_error'] is None
                 or room['teacher']['actors'][1]['first_error']>=branch['step']]
        audit[name]=dict(boundaries=sum(v['count'] for _,v in cases),tested_rooms=len(cases),matrix=outcome(cases),
                         reachable_by_ln=len(natural),reachable_matrix=outcome(natural))
    assert audit['ln_only']['boundaries']-audit['old_only']['boundaries']==436
    result['teacher_disagreements']=audit
    result['probe_sources']={name:study.campaign.sha(Path(__file__).with_name(name)) for name in (
        'cap_general_actor_forks.py','cap_general_actor_later_forks.py','cap_general_actor_teacher_forks.py','cap_general_actor_fork_report.py')}
    save(OUT/'FINISHED.json',result)
    docs=Path(__file__).resolve().parents[1]/'docs/research/key_findings/2026-11-20_hard_actor_fit'
    with (docs/'forks_per_room.csv').open('w',newline='') as f:
        out=csv.writer(f)
        out.writerow(['room','old','layernorm','first_step_zero_based','first_tick','old_suffix_with_ln_action',
                      'ln_suffix_with_old_action','fastest_at_fork','old_first_teacher_error','ln_first_teacher_error',
                      'ln_repair_outcome','old_one_tick','ln_one_tick','later_repaired_steps'])
        for row in rooms:
            t=row.get('teacher',{}).get('actors',[{},{}])
            micro=row.get('micro',[{},{}])
            out.writerow([row['room'],*row['actual'],row['cut'],row.get('tick'),*row.get('cross',[None,None]),
                          row.get('teacher_at_cut'),t[0].get('first_error'),t[1].get('first_error'),t[1].get('repair'),
                          micro[0].get('success'),micro[1].get('success'),
                          ','.join(str(v['step']) for v in late.get(row['room'],{}).get('tested',[]) if v['success'])])


if __name__=='__main__':
    report()
