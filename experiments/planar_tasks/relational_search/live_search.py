"""Receding-horizon profile: fitted expansion ranks, comparable frontier costs.

All legal commands remain in the lazy stream. Early coverage is stratified by
entity, robot role, direction and duration rather than adjacent numeric IDs.
No authored sequence or privileged execution world is accepted here.
"""
from dataclasses import dataclass
from copy import deepcopy
from itertools import zip_longest, islice
import math
import time
from .contract import action_space, validate_action, finite
from .motion import move_id


@dataclass(frozen=True)
class Budget:
    calls: int = 12
    prediction_ticks: int = 600
    horizon_ticks: int = 120
    batch: int = 72
    fair_every: int = 4
    seconds: float = 15.

    def __post_init__(self):
        for k in ('calls','prediction_ticks','horizon_ticks','batch','fair_every'):
            if type(getattr(self,k)) is not int or getattr(self,k)<1:raise ValueError(k)
        if self.batch>256 or not finite(self.seconds) or self.seconds<=0:raise ValueError('search budget')


def candidates(observation):
    preferred=[2,10,18,26,6,14,22,30,33,36,40,43,44,46]
    groups=[]
    for entity in observation['entities']:
        target=entity['id'];menu=observation.get('entity_legal_pairs',{}).get(target,observation['legal_pairs'])
        allowed={tuple(p) for p in menu}
        pairs=[pair for a in preferred for pair in ((a,a),(44,a),(a,44))]
        pairs+=sorted(allowed)
        pairs=list(dict.fromkeys(p for p in pairs if p in allowed))
        groups.append([(target,*p) for p in pairs])
    global_menu={tuple(p) for p in observation.get('global_legal_pairs',[[44,44]])}
    moves=[move_id(d) for d in ('E','N','W','S','NE','NW','SW','SE')]
    pairs=[p for a in moves for p in ((44,a),(a,44),(a,a))]+[(44,44)]
    groups.append([(None,*p) for p in pairs if p in global_menu])
    seen=set()
    for row in zip_longest(*groups):
        for value in row:
            if value is None:continue
            target,a0,a1=value
            for ticks in (40,20,4):
                key=(target,a0,a1,ticks)
                action=dict(target_id=target,a0=a0,a1=a1,ticks=ticks)
                try:validate_action(action,observation)
                except ValueError:continue
                if key not in seen:seen.add(key);yield action
    for action in action_space(observation):
        key=tuple(action[k] for k in ('target_id','a0','a1','ticks'))
        if key not in seen:seen.add(key);yield action


def leaf_seconds(history):
    """Shared visible-support distance/nominal speed heuristic; no clearing bonus."""
    o=history.events[-1]['observation'];goal=o['goal'];points=None
    for event in reversed(history.events):
        entity=next((e for e in event['observation']['entities'] if e['id']==goal['target_id'] and e['points']),None)
        if entity:points=entity['points'];break
    if not points:return 90.
    centre=[sum(p[k] for p in points)/len(points) for k in (0,1)]
    return max(0.,math.dist(centre,goal['xy'])-goal['radius'])/.3


def choose(history,model,score_batch,budget=Budget(),*,experience=None,reserved_root_calls=0):
    if type(reserved_root_calls) is not int or not 0<=reserved_root_calls<=budget.calls:raise ValueError('reserved root calls')
    started=time.monotonic();nodes=[];pending=[];trace=[];scoring=[]
    ticks=calls=0
    def add(h,handle,path,elapsed,cost,root_score=0.,terminal=False,success=False):
        retrieved=[] if experience is None else experience.retrieve(h)
        for item in retrieved:validate_action(item['action'],h.events[-1]['observation'])
        def stream():
            seen=set()
            for action in [item['action'] for item in retrieved]:
                key=tuple(action[k] for k in ('target_id','a0','a1','ticks'))
                if key not in seen:seen.add(key);yield action
            for action in candidates(h.events[-1]['observation']):
                key=tuple(action[k] for k in ('target_id','a0','a1','ticks'))
                if key not in seen:seen.add(key);yield action
        node=dict(id=len(nodes),history=h,handle=handle,path=path,elapsed=elapsed,cost=cost,
            root_score=root_score,terminal=terminal,success=success,children=0,pending=0,
            retrieved=retrieved,actions=iter(stream()),exhausted=terminal or elapsed>=budget.horizon_ticks)
        nodes.append(node);return node
    add(history.fork(),model.root,[],0,0.)
    while calls<budget.calls and time.monotonic()-started<budget.seconds:
        for n in list(nodes):
            if n['exhausted'] or n['pending']:continue
            batch=[]
            while not batch and time.monotonic()-started<budget.seconds:
                raw=list(islice(n['actions'],budget.batch))
                if not raw:n['exhausted']=True;break
                batch=[a for a in raw if n['elapsed']+a['ticks']<=budget.horizon_ticks]
            if not batch:continue
            scores=score_batch(n['history'].fork(),deepcopy(batch))
            if len(scores)!=len(batch) or any(not finite(v) for v in scores):raise ValueError('candidate score schema')
            scoring.append(dict(node=n['id'],history_tick=n['history'].events[-1]['observation']['tick'],
                                actions=batch,scores=scores,retrieved=n['retrieved']))
            for a,q in zip(batch,scores):pending.append((q,n,a));n['pending']+=1
        affordable=[]
        for i,(_,n,a) in enumerate(pending):
            quote=model.cost_ticks(n['handle'],a)
            if type(quote) is not int or quote<a['ticks']:raise ValueError('prediction quote')
            if ticks+quote<=budget.prediction_ticks:affordable.append(i)
        if not affordable:break
        fair=(calls+1)%budget.fair_every==0
        protected=[i for i in affordable if pending[i][1]['id']==0 and any(pending[i][2]==r['action'] for r in nodes[0]['retrieved'])]
        reserve=calls<reserved_root_calls and bool(protected)
        i=protected[0] if reserve else affordable[0] if fair else max(affordable,key=lambda i:(pending[i][0],pending[i][1]['elapsed']))
        q,n,a=pending.pop(i);n['pending']-=1
        quote=model.cost_ticks(n['handle'],a);calls+=1;charged=quote
        event=dict(node=n['id'],action=a,score=q,fair=fair and not reserve,reserved_experience=reserve,depth=len(n['path']),requested_ticks=a['ticks'])
        try:
            response=model.predict(n['handle'],a)
            if response['success'] and not response['terminal']:raise ValueError('success must terminate')
            actual=response['ticks']
            if type(actual) is not int or not 1<=actual<=a['ticks'] or (actual<a['ticks'] and not response['terminal']):
                raise ValueError('prediction silently shortened action')
            if response['observation']['tick']!=n['history'].events[-1]['observation']['tick']+actual:
                raise ValueError('prediction time mismatch')
            charged=response.get('charged_ticks',actual)
            if type(charged) is not int or not actual<=charged<=quote:raise ValueError('prediction charge')
            if not finite(response['cost_seconds']) or response['cost_seconds']<0:raise ValueError('task cost')
            h=n['history'].fork();h.append(response['observation'],a);n['children']+=1
            child=add(h,response['handle'],n['path']+[a],n['elapsed']+actual,n['cost']+response['cost_seconds'],
                n['root_score'] if n['path'] else q,response['terminal'],response['success'])
            event.update(status='predicted',child=child['id'],actual_ticks=actual,
                         elapsed_ticks=child['elapsed'],leaf_seconds=leaf_seconds(h),success=response['success'])
        except Exception as exc:event.update(status='error',error=str(exc))
        ticks+=charged;event['charged_ticks']=charged;trace.append(event)
    success=[n for n in nodes if n['success']]
    frontier=[n for n in nodes if n['path'] and not n['children'] and not n['terminal']]
    # A partially expanded short prefix is not a competing completed plan.
    max_elapsed=max((n['elapsed'] for n in frontier),default=0)
    pool=success or [n for n in frontier if n['elapsed']==max_elapsed]
    winner=min(pool,key=lambda n:(n['cost']+(0 if n['success'] else leaf_seconds(n['history'])),-n['root_score'],n['id'])) if pool else None
    return dict(action=winner['path'][0] if winner else None,
        status='predicted' if winner else 'needs_common_fallback',calls=calls,prediction_ticks=ticks,
        selection_horizon_ticks=winner['elapsed'] if winner else 0,
        selected_path=winner['path'] if winner else [],selected_node=winner['id'] if winner else None,
        trace=trace,scoring=scoring,wall_seconds=time.monotonic()-started)
