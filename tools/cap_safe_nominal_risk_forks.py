"""Frozen-source shadow attribution and late intervention after an executed old-P prefix."""
import argparse
from collections import Counter
from functools import partial
from statistics import median
from unittest.mock import patch

from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from tools import cap_safe_nominal_forks as first

study = first.study
OUT = first.OUT / 'risk'
REASONS = ('observed_sweep_too_short','no_shared_continuation_in_sampled_envelope')
FINAL = 'adf79ad691026c2eefb3a516fd717ba129f9b782cdf905f7e343d8569c8cb432'


def audit_source(source):
    """Copy the frozen loop verbatim and annotate candidates at its two relevant rejection sites."""
    before,rest=source.split('def candidate_packet(memory,o):\n')
    loop=rest.split('def update_run(')[0]
    for line,mark in [('    if reason:\n      continue','      memory["check"].append([act,reason])\n'),
                      ("      reason = 'no_shared_continuation_in_sampled_envelope'\n      continue",'      memory["check"].append([act,reason])\n')]:
        if loop.count(line)!=1:
            raise ValueError('unexpected candidate loop')
        loop=loop.replace(line,line.replace('      continue',mark+'      continue'))
    return before+'def candidate_packet(memory,o):\n'+loop+'''
def update_memory(memory,event):
  memory['check']=[]
  p=candidate_packet(memory,event['observation'])
  return {'packet':p,'checks':memory['check']}
def prepare_context(m,o):
  return dict(values=[0],state=[None]*5,proposals=[])
'''


def directed_action(action):
    return tuple((i,action[f'a{i}']//4) for i in first.translating(action))


def identify(job):
    room,fixture,base,safe,audit=job
    safe_worker=StreamProgram(safe)
    audit_worker=StreamProgram(audit)
    shadow=study.ISO.IsolatedCAP(safe_worker)
    original=study.host.old.prefix.run_member
    counts=Counter()
    anchor=None
    def with_shadow(job,member,preview,boundary,**kw):
        class Both:
            def update(self,history):
                member.update(history)
                shadow.update(history)
            def propose(self,obs):
                return member.propose(obs)
        def choose(w,obs,action,pending):
            nonlocal anchor
            issued=boundary(w,obs,action,pending)
            if pending or obs['tick']==0 or anchor:
                return issued
            if issued['target_id']!=obs['goal']['target_id'] or not first.translating(issued):
                return issued
            counts['old_target_push']+=1
            candidate=shadow.propose(obs)
            permit=shadow.memory['permit']
            if permit['command']!=candidate or permit['tick']!=obs['tick']:
                raise ValueError(f'{room}: shadow decision and committed permit differ')
            new_push=directed_action(candidate)
            changed=(new_push and new_push!=directed_action(issued)) or (
                permit['decision']=='reposition' and max(candidate['a0'],candidate['a1'])>=47)
            if not changed:
                return issued
            counts['alternative_direction_or_positioning']+=1
            diagnostic=audit_worker.request(dict(op='advance',memory=shadow.memory,event=dict(observation=obs)))['memory']
            if diagnostic['packet']!=permit:
                raise ValueError(f'{room}: attribution changed candidate packet')
            counts['verified_packet']+=1
            exact=[dict(command=act,reason=reason) for act,reason in diagnostic['checks']
                   if all(act[v]==issued[v] for v in ('target_id','a0','a1')) and reason in REASONS]
            if exact:
                anchor=dict(tick=obs['tick'],history=w.history.sha256,observation=first.digest(obs),old_action=issued,
                            new_action=candidate,permit=permit,rejection=exact)
            return issued
        return original(job,Both(),preview,choose,**kw)
    try:
        with patch.object(study.host.old.prefix,'run_member',with_shadow):
            _,row=study.host.episode((room,fixture,True,None,'boundary',False,study.host.enabled(base),None))
    finally:
        safe_worker.close()
        audit_worker.close()
    reference=study.read(first.OUT/'traces/base'/(room+'.json'))
    if row['result']!=reference['result']:
        raise ValueError(f'{room}: shadow changed the old trajectory')
    return room,dict(first=anchor,counts_before_first=dict(counts),result=row['result'])


def suffix(job):
    room,fixture,base,safe,anchor=job
    trace=first.Recorder(None)
    with patch.object(study.S,'tick_action',partial(trace.tick,study.S.tick_action)):
        _,row=study.host.episode((room,fixture,True,anchor['tick'],'boundary',False,study.host.enabled(safe),base))
    d=row['reached']
    if d['history']!=anchor['history'] or d['observation_sha256']!=anchor['observation']:
        raise ValueError(f'{room}: failed attribution prefix replay')
    return room,dict(result=row['result'],frames=trace.frames,decisions=trace.decisions,target=trace.target,
                     goal=[fixture['goal'][v] for v in ('x','y')])


def run(workers):
    fixtures=study.dev_rooms()
    locked=study.read(study.OUT/'locked.json')
    if locked['sha256']!=FINAL or study.sha(study.PUBLIC/'FINAL.json')!=locked['final_sha256']:
        raise ValueError('late attribution is specific to this frozen loop')
    programs=[]
    for name,sha in [('P',study.BASE_SHA),(locked['name'],FINAL)]:
        directory=study.OUT/'sealed'/name
        study.Campaign(directory).verify_frozen()
        src=(directory/'composed.py').read_text()
        if study.host.digest(src)!=sha:
            raise ValueError('changed frozen source')
        programs.append(src)
    base,safe=programs
    audit=audit_source(safe)
    immutable=first.immutable
    immutable(OUT/'protocol.json',dict(rooms=list(fixtures),input_sha256=[study.host.digest(s) for s in programs],
        eligibility="ordinary non-pending boundary on old P, tick>0, issued target translation PUSH; the frozen candidate loop on the same public committed history actually visits the matching target/a0/a1 and rejects on exactly one of the listed reasons, and the final candidate command chooses a different pusher/direction or repositions.",
        rejection_reasons=REASONS, max_branches=6,ranking='first eligible boundary in each room, then room IDs ascending, outcome-blind',
        common_prefix='old P supplies every prior action; candidate memory maintained on the same committed public events. Natural candidate reachability not asserted.',
        scope='Late branch is ongoing old P versus ongoing new P. Identical-history natural-fork outcomes are reported separately.',
        attribution_sha256=study.host.digest(audit),runner_sha256=study.sha(study.ROOT/'tools/cap_safe_nominal_risk_forks.py'),
        measured_physics=study.sha(study.ROOT/'tools/cap_safe_nominal_forks.py'),workers=workers))
    path=OUT/'shadow_attribution_source.py'
    if path.exists() and path.read_text()!=audit:
        raise ValueError('shadow source changed')
    path.write_text(audit)
    study.host.old.collect(base,[(k,f,base,safe,audit) for k,f in fixtures.items()],identify,workers,OUT/'shadow')
    rows={k:study.read(OUT/'shadow'/(k+'.json')) for k in sorted(fixtures)}
    hits=[k for k,v in rows.items() if v['first']]
    chosen=hits[:6]
    if chosen:
        study.host.old.collect(safe,[(k,fixtures[k],base,safe,rows[k]['first']) for k in chosen],
                               suffix,min(workers,len(chosen)),OUT/'suffix')
    outcomes=[]
    for k in chosen:
        a=study.read(first.OUT/'traces/base'/(k+'.json'))
        b=study.read(OUT/'suffix'/(k+'.json'))
        tick=rows[k]['first']['tick']
        status,pair=first.find_fork(a,b)
        if status!='matched_fork' or pair[0]['tick']!=tick or pair[1]['choice']!=rows[k]['first']['new_action']:
            raise ValueError(f'{k}: fork is not the registered action on the old physical history')
        outcomes.append(dict(room=k,prefix=rows[k]['first'],anchor=pair[0],base=first.effects(a,tick),safe=first.effects(b,tick)))
    summary=dict(n=len(rows),matched_rooms=hits,selected_rooms=chosen,no_match=[k for k,v in rows.items() if not v['first']],
                 beyond_cap=hits[6:],compared=len(chosen),old_queries_at_eligible_boundary=sum(v['counts_before_first'].get('old_target_push',0) for v in rows.values()),
                 alternative_queries_checked=sum(v['counts_before_first'].get('verified_packet',0) for v in rows.values()),
                 rescues=sum(v['safe']['terminal_success']>v['base']['terminal_success'] for v in outcomes),
                 losses=sum(v['safe']['terminal_success']<v['base']['terminal_success'] for v in outcomes))
    for arm in ('base','safe'):
        vals=[v[arm] for v in outcomes]
        summary[arm]=dict(terminal_success=sum(v['terminal_success'] for v in vals),
            window_advance_ge_2cm=sum(v['goal_advance_m']>=.02 for v in vals),
            median_advance_m=median(v['goal_advance_m'] for v in vals) if vals else None,
            next_effective=sum(v['next_effective_push'] is not None for v in vals),
            after_window=dict(Counter(v['followup_status'] for v in vals)),
            with_static_contact=sum(v['object_static_ticks']>0 for v in vals))
    study.save(OUT/'pairs.json',outcomes)
    study.save(OUT/'summary.json',summary)
    print(summary,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=4)
    run(parser.parse_args().workers)
