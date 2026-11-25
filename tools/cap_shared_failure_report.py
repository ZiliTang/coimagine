"""Summarise observed mechanics; room classes describe the final 20 seconds."""
import csv
from collections import Counter
from math import cos, sin, pi, dist, hypot
from statistics import median
import numpy as np
from tools.cap_shared_failure_audit import DEST, r

DOC=r.DOC/'shared_failures'
GOAL=(4.25,3.90)
def intervals(room):
    replay=r.read(DEST/'replays'/(room+'.json'))
    old=r.read(r.OUT/'evaluation/parent'/(room+'.json'))
    data=[]
    for z,b,e in zip(replay['rows'],old['boundaries'],old['effects'],strict=True):
        dt=z['dt']; contacts=z['contacts']
        assert contacts['samples']==dt
        a=b['action'];push=a['target_id']=='entity-a' and min(a['a0'],a['a1'])<32
        xy=z['object'][:2];end=z['end'][:2]
        displacement=dist(xy,end)
        goal=dist(xy,GOAL)-dist(end,GOAL)
        dst=z['route'] if z['route'] and dist(xy,GOAL)>.63 else GOAL
        progress=dist(xy,dst)-dist(end,dst)
        pair=[a['a0'],a['a1']]
        def kind(i):
            return i//4 if i<32 else (i-47)//4+100 if i>=47 else i
        direction=[sum((cos if j==0 else sin)(pi*(i//4)/4)*(1+i%4) for i in pair if i<32) for j in range(2)]
        length=hypot(*direction)
        along=sum(v*(q-p) for v,p,q in zip(direction,xy,end))/length if length else 0
        data.append(dict(t=z['t'],dt=dt,xy=xy,end=end,pose=z['object'],robot=z['robot'],goal=goal,move=displacement,
            along=along,progress=progress,route=z['route'],err=dist(xy,z['compact'][:2]) if z['compact'] else None,
            sig=tuple(map(kind,pair)),push=push,node=e['node'],touch=contacts.get('robot_object',0),wall=contacts.get('object_wall',0),
            movable=contacts.get('object_object',0),peer=contacts.get('robot_robot',0),seen=z['seen'],
            role=z['slot_plan'] or z['outer_plan'],stage=b['execution'],walk=sum(dist(p,q) for p,q in zip(z['robot'],z['robot_end'])),
            null=a['target_id'] is None))
    return data

def summarize(room,d,window=400):
    T=d[-1]['t']+d[-1]['dt']
    tail=[v for v in d if v['t']>=T-window] or d[-1:]
    ticks=sum(v['dt'] for v in tail)
    def part(predicate):return sum(v['dt'] for v in tail if predicate(v))/ticks
    def ratio(key):return sum(v[key] for v in tail)/ticks
    start=tail[0]
    push=part(lambda z:z['push']); touch=ratio('touch')
    close=part(lambda z:z['node'].split('/')[1]!='open')
    station=part(lambda z:z['node'].startswith('unavailable/'))
    if push>=.5 and touch>=.5:cat='contact_push'
    elif push>=.5:cat='seeking_contact'
    else:cat='mostly_nonpush'
    flips=sum(v['sig']!=w['sig'] for v,w in zip(tail,tail[1:]))
    cycle=sum(v['sig']==x['sig']!=w['sig'] for v,w,x in zip(tail,tail[1:],tail[2:]))
    gaps=[dist(z['xy'],GOAL) for z in d]+[dist(d[-1]['end'],GOAL)]
    # Last boundary crossing the "more than 0.15 m from final pose" threshold.
    far=[z for z in d if dist(z['xy'],d[-1]['end'])>.15]
    dwell=T-(far[-1]['t']+far[-1]['dt']) if far else T
    pos_errors=[z['err'] for z in tail if z['err'] is not None]
    return dict(room=room[-3:],signature=cat,termination_tick=T,tail_start=start['t'],best_gap=round(min(gaps),3),final_gap=round(gaps[-1],3),
        final_neighborhood_ticks=dwell,tail_object_path=round(sum(z['move'] for z in tail),3),tail_goal_gain=round(sum(z['goal'] for z in tail),3),
        tail_robot_paths=round(sum(z['walk'] for z in tail),3),push=round(push,4),target_touch=round(touch,4),object_wall=round(ratio('wall'),4),
        object_movable=round(ratio('movable'),4),robot_robot=round(ratio('peer'),4),goal_sweep_blocked=round(close,4),rear_stations_unavailable=round(station,4),
        waypoint=round(part(lambda z:z['route'] is not None),4),visible=round(part(lambda z:z['seen']),4),center_error_m=round(median(pos_errors),3) if pos_errors else None,
        null_target=round(part(lambda z:z['null']),4),changes=flips,ABA=cycle,boundaries=len(tail))
def export(name,rows):
    with (DOC/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def main():
    DOC.mkdir(exist_ok=True)
    g=r.read(DEST/'groups.json'); pairs=g['pairs']
    cases=[];controls=[]
    for k,c in pairs.items():
        a=summarize(k,intervals(k));cases.append(dict(**a,stratum=g['strata'][k],control=c[-3:] if c else ''))
        if c:controls.append(dict(**summarize(c,intervals(c)),stratum=g['strata'][c],case=k[-3:]))
    export('rooms.csv',cases);export('controls.csv',controls)
    for name,rows in [('failure',cases),('success',controls)]:
        print(name,'n',len(rows), Counter(x['signature'] for x in rows),Counter(x['stratum'] for x in rows))
        print('medians', {key:median([x[key] for x in rows if x[key] is not None]) for key in ('tail_object_path','tail_goal_gain','push','target_touch','object_wall','object_movable','robot_robot','changes','ABA','tail_robot_paths','waypoint','visible','center_error_m','final_neighborhood_ticks')})
        for cat in ['contact_push','seeking_contact','mostly_nonpush']:
            group=[x for x in rows if x['signature']==cat]
            if group:print(cat,','.join(x['room'] for x in group),{key:round(median(x[key] for x in group),3) for key in ('push','target_touch','tail_object_path','tail_goal_gain','goal_sweep_blocked','rear_stations_unavailable','tail_robot_paths','changes','ABA')})
if __name__=='__main__':main()
