"""Fit fixed one-boundary physical effect probes, then test their relation to terminal success."""
import csv
import gzip
import itertools
import json
from collections import defaultdict

import numpy as np
from coimagine.random_start.spec import CONTROL_DT
from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl.candidate_semantics import direction
from experiments.planar_tasks.semantic_rl.corridor_semantics import corridor_semantics
from tools import cap_semantic_fork_ranking as rank
from tools.cap_option_effect_replay import DOC, OUT, PREFIX, ref, run

TARGETS = ('push','move','goal')
ARMS = ('nominal','actions','semantics')
MARGIN = .01


def effects(cmd, goal_id, goal, step, semantic):
    a = [ACTIONS[cmd[f'a{i}']] for i in range(2)]
    seconds = cmd['ticks']*CONTROL_DT
    truth = np.subtract(step['target1'],step['target0'])
    target_goal = np.subtract(goal,step['target0'])
    obs0,obs1 = step['observed0'],step['observed1']
    values = {'goal':(np.linalg.norm(target_goal)-np.linalg.norm(target_goal-truth),
                      semantic['push_goal_progress'],None)}
    moves = [(i,direction(v),v.speed) for i,v in enumerate(a) if v.kind=='move']
    if moves:
        actual = np.mean([u@step['robots_delta'][i] for i,u,_ in moves])
        nominal = seconds*np.mean([speed for _,_,speed in moves])
        values['move'] = (actual,nominal,None)
    pushes = [direction(v)*v.speed for v in a if getattr(v,'effect_kind',None)=='translate']
    if pushes:
        assert cmd['target_id']==goal_id
        net = np.mean(pushes,axis=0)
        speed = np.linalg.norm(net)
        if speed>1e-8:
            axis = net/speed
            observed = None if obs0 is None or obs1 is None else np.subtract(obs1,obs0)@axis
            values['push'] = (truth@axis,seconds*speed,observed)
    return values


def predict(samples, arm, target):
    train=[x for x in samples if x['split']=='development']
    def array(x):
        return x[arm+'_x'] if target=='goal' else np.r_[x[arm+'_x'],x['nominal']]
    x=np.stack([array(s) for s in train])
    count=defaultdict(int)
    for s in train:
        count[s['room']]+=1
    weights=np.array([1/count[s['room']] for s in train])
    weights/=weights.sum()
    mean=weights@x
    scale=np.maximum(np.sqrt(weights@((x-mean)**2)),.05)
    z=(x-mean)/scale
    y=np.array([s['actual'] for s in train])
    intercept=weights@y
    beta=np.linalg.solve((z.T*weights)@z+.1*np.eye(z.shape[1]),z.T@(weights*(y-intercept)))
    for s in samples:
        s[arm]=float(intercept+((array(s)-mean)/scale)@beta)


def read_gzip(path):
    with gzip.open(path,'rt') as f:
        return [json.loads(line) for line in f]


def room_ci(rows,fn):
    groups=defaultdict(list)
    for r in rows:
        groups[r['room']].append(fn(r))
    if not groups:
        return None
    means=np.array([np.mean(v) for v in groups.values()])
    rng=np.random.default_rng(17)
    bs=rng.choice(means,(5000,len(means))).mean(1)
    return dict(rooms=len(means),mean=float(means.mean()),ci=np.quantile(bs,[.025,.975]).tolist())


def order(x,y):
    return .5 if abs(x)<=1e-9 or abs(y)<=1e-9 else float(x*y>0)


def describe(pairs):
    informative=[p for p in pairs if abs(p['actual'])>=MARGIN]
    opposite=[p for p in informative if p['terminal_diff']]
    out=dict(all_pairs=len(pairs),measurable=len(informative),opposite=len(opposite),
             opposite_before_margin=sum(p['terminal_diff']!=0 for p in pairs))
    for a in ARMS:
        fn=lambda p:order(p[a],p['actual'])
        out[a]=dict(correct=sum(map(fn,informative)),balanced=room_ci(informative,fn))
    out['semantic_increment']=room_ci(informative,lambda p:order(p['semantics'],p['actual'])-order(p['actions'],p['actual']))
    for a in ('actual','semantics'):
        f=lambda p:order(p[a],p['terminal_diff'])
        out[a+'_terminal']=dict(correct=sum(map(f,opposite)),balanced=room_ci(opposite,f))
    matched=[p for p in informative if abs(p['nominal'])<=1e-8]
    out['same_nominal']=dict(n=len(matched),rooms=len({p['room'] for p in matched}),
                             semantics_correct=sum(order(p['semantics'],p['actual']) for p in matched))
    obs=[p for p in informative if p['observed'] is not None]
    out['public_post_vs_true']=dict(n=len(obs),correct=sum(order(p['observed'],p['actual']) for p in obs))
    return out


def main():
    data=read_gzip(PREFIX/'prefixes.jsonl.gz')
    replay={(s['room'],s['i']):s for s in read_gzip(OUT/'steps.jsonl.gz')}
    assert len(replay)==878
    samples={t:[] for t in TARGETS}
    for raw in data:
        if 'actions' not in raw:
            continue
        row=rank.prepare(raw)
        obs=row['current']
        subtype=corridor_semantics(obs)['kind']
        leaf=ref.read(run.OUT/'evaluation'/(row['room']+'.json'))['branches'] if row['split']=='held' else None
        for i,cmd in enumerate(row['actions']):
            step=replay[row['room'],i]
            assert (cmd,row['prefix'])==(step['action'],step['prefix'])
            values=effects(cmd,obs['goal']['target_id'],obs['goal']['xy'],step,row['extra'][i])
            for t,(actual,nominal,observed) in values.items():
                samples[t].append(dict(room=row['room'],split=row['split'],i=i,subtype=subtype,
                                       actual=actual,nominal=nominal,observed=observed,success=row['outcomes'][i],
                                       pref=i==row['pref'],dt=step['dt'],ticks=cmd['ticks'],
                                       immediate_terminal=None if leaf is None else leaf[i]['leaf']['1']['terminal'],
                                       actions_x=row['actions_x'][i],semantics_x=row['semantics_x'][i]))
    results={}; allpairs=[]; candidates=[]
    for t,ss in samples.items():
        for arm in ARMS[1:]:
            predict(ss,arm,t)
        pairs=[]
        for room,group in itertools.groupby(ss,key=lambda s:s['room']):
            for a,b in itertools.combinations(group,2):
                pairs.append(dict(target=t,room=room,split=a['split'],subtype=a['subtype'],i=a['i'],j=b['i'],
                                  terminal_diff=a['success']-b['success'],end_first=a['immediate_terminal'] or b['immediate_terminal'],
                                  observed=None if a['observed'] is None or b['observed'] is None else a['observed']-b['observed'],
                                  **{k:a[k]-b[k] for k in ('actual',*ARMS)}))
        held=[p for p in pairs if p['split']=='held']
        held_s=[s for s in ss if s['split']=='held']
        res=describe(held)
        res['budget']=dict(dev_candidates=len(ss)-len(held_s),dev_rooms=len(set(s['room'] for s in ss if s['split']=='development')),
                           dev_pairs=sum(p['split']=='development' for p in pairs),
                           held_candidates=len(held_s),held_rooms=len(set(s['room'] for s in held_s)))
        res['mae_m']={a:float(np.mean([abs(s[a]-s['actual']) for s in held_s])) for a in ARMS}
        res['delta_mae_m']={a:float(np.mean([abs(p[a]-p['actual']) for p in held])) for a in ARMS}
        res['measured_terminal_excluding_immediate']=describe([p for p in held if not p['end_first']])['actual_terminal']
        res['terminal_nonimmediate_n']=sum(bool(not p["end_first"] and abs(p["actual"])>=MARGIN and p["terminal_diff"]) for p in held)
        if t=='goal':
            res['by_subtype']={k:describe([p for p in held if p['subtype']==k]) for k in sorted({p['subtype'] for p in held})}
            base=sum(r.get('unchanged',0) for r in data if r['split']=='held')
            selection={k:dict(wins=base,rescued=0,lost=0) for k in ('actual',*ARMS)}
            for room,group in itertools.groupby(held_s,key=lambda s:s['room']):
                group=list(group)
                actor=next(s for s in group if s['pref'])
                for arm,stats in selection.items():
                    highest=max(s[arm] for s in group)
                    chosen=max((s for s in group if highest-s[arm]<=1e-9),key=lambda s:(s['pref'],-s['i']))
                    y=chosen['success']
                    stats['wins']+=y
                    stats['rescued']+=int(y>actor['success'])
                    stats['lost']+=int(y<actor['success'])
            res['once_over_200']=selection
        results[t]=res
        allpairs.extend(pairs)
        candidates.extend(dict(target=t,**{k:v for k,v in s.items() if not k.endswith('_x')}) for s in ss)
    DOC.mkdir(exist_ok=True,parents=True)
    rank.write_table(DOC/'pairs.csv',allpairs)
    rank.write_table(DOC/'candidates.csv',candidates)
    (DOC/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
