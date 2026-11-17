"""Standard-library option search. Real legality alone governs executable roots.

Approximate prediction failures are evidence, never execution prohibitions.
No pose merging, beam pruning, progress requirement or duration quantization.
"""
from dataclasses import dataclass, asdict
import heapq
import math
import time


@dataclass(frozen=True, slots=True)
class Action:
    a0: int
    a1: int
    ticks: int

    def packet(self): return asdict(self)


def action_space(observation):
    """Exactly the existing protocol's legal_pairs x integer [1,40]."""
    seen=set()
    for pair in observation['legal_pairs']:
        if len(pair)!=2 or any(type(i) is not int for i in pair):
            raise ValueError('malformed legal pair')
        key=tuple(pair)
        if key in seen:continue
        seen.add(key)
        for ticks in range(1,41):yield Action(*pair,ticks)


def validate(action, observation):
    if not isinstance(action,dict) or set(action)!={'a0','a1','ticks'}:
        raise ValueError('action fields must be a0/a1/ticks')
    if any(type(v) is not int for v in action.values()):raise ValueError('integer actions required')
    if not 1<=action['ticks']<=40 or [action['a0'],action['a1']] not in observation['legal_pairs']:
        raise ValueError('action outside original legal menu')
    return action


@dataclass(frozen=True)
class Budget:
    expansions: int=6
    prediction_ticks: int=480
    prediction_calls: int=40
    search_seconds: float=2.5
    max_depth: object=None

    def __post_init__(self):
        for name in ('expansions','prediction_ticks','prediction_calls'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError(name)
        if type(self.search_seconds) not in (int,float) or not math.isfinite(self.search_seconds) or self.search_seconds<=0:
            raise ValueError('positive finite search_seconds required')
        if self.max_depth is not None and (type(self.max_depth) is not int or self.max_depth<1):raise ValueError('max_depth')


class Search:
    def __init__(self, scorer, budget=None):
        self.scorer=scorer;self.budget=budget or Budget()

    def choose(self, observation, context, model):
        started=time.monotonic();deadline=started+self.budget.search_seconds
        nodes=[];heap=[];fifo=[];events=[];scoring_errors=[];expanded=set();blocked=set()
        reserved_ticks=reserved_calls=attempts=0
        # Hooks see the same legal history in the worker. They cannot mutate this
        # process's observations, search queues or action validity checks.
        try:self.scorer.begin(context,deadline)
        except Exception as exc:scoring_errors.append(dict(type=type(exc).__name__,message=str(exc)))

        def score(ids):
            for offset in range(0,len(ids),128):
                batch=ids[offset:offset+128]
                records=[dict(id=i,parent_id=nodes[i]['parent'],plan=[a.packet() for a in nodes[i]['plan']],
                              cost_seconds=nodes[i]['g'],status=nodes[i]['status']) for i in batch]
                try:
                    values=self.scorer.score(records,deadline)
                    if not isinstance(values,list) or len(values)!=len(batch):raise ValueError('one numeric heuristic per node required')
                    if any(type(v) not in (int,float) or not math.isfinite(v) for v in values):raise ValueError('finite numeric heuristics required')
                except Exception as exc:
                    values=[nodes[i].get('h',0.) for i in batch]
                    if len(scoring_errors)<8:scoring_errors.append(dict(type=type(exc).__name__,message=str(exc)))
                for i,h in zip(batch,values):
                    nodes[i]['h']=h
                    value=nodes[i]['g']+h
                    if not math.isfinite(value):value=nodes[i]['g']
                    nodes[i]['score']=value
                    heapq.heappush(heap,(value,i))

        def add_children(parent,menu):
            parent_plan=nodes[parent]['plan'] if parent is not None else ()
            ids=[]
            for action in action_space(menu):
                i=len(nodes);plan=parent_plan+(action,)
                nodes.append(dict(parent=parent,plan=plan,g=sum(a.ticks for a in plan)*.05,
                    status='unexpanded',score=float('inf')))
                fifo.append(i);ids.append(i)
            score(ids)
            return ids

        roots=add_children(None,observation)
        if not roots:raise ValueError('empty original legal menu')
        focus=[];fifo_cursor=0;stop='expansion_budget_exhausted'
        while attempts<self.budget.expansions:
            if time.monotonic()>=deadline:stop='search_deadline';break
            eligible=lambda i:i not in expanded and i not in blocked and nodes[i]['status']=='unexpanded'
            # Every fourth selection follows insertion order; two selections per
            # cycle inspect a child of the last expanded node. Others are best-first.
            # This changes allocation only. It never deletes an unvisited branch.
            chosen=None;schedule='best_first'
            if attempts%4==3:
                schedule='fifo'
                while fifo_cursor<len(fifo):
                    i=fifo[fifo_cursor];fifo_cursor+=1
                    if eligible(i):chosen=i;break
            elif attempts%4 in (1,2):
                options=[i for i in focus if eligible(i)]
                if options:chosen=min(options,key=lambda i:(nodes[i]['score'],i));schedule='lookahead'
            if chosen is None:
                while heap:
                    _,i=heapq.heappop(heap)
                    if eligible(i):chosen=i;break
            if chosen is None:stop='frontier_exhausted';break
            node=nodes[chosen];plan=node['plan'];cost=sum(a.ticks for a in plan);calls=len(plan)+2
            if self.budget.max_depth is not None and len(plan)>self.budget.max_depth:
                blocked.add(chosen);continue
            if reserved_ticks+cost>self.budget.prediction_ticks or reserved_calls+calls>self.budget.prediction_calls:
                blocked.add(chosen);continue
            expanded.add(chosen);attempts+=1;reserved_ticks+=cost;reserved_calls+=calls
            try:
                result=model.evaluate([a.packet() for a in plan],deadline)
                node['status']='predicted'
                node['g']=result.get('ticks',sum(a.ticks for a in plan))*.05
                self.scorer.register(chosen,result,deadline)
                score([chosen])
                events.append(dict(node=chosen,schedule=schedule,plan=[a.packet() for a in plan],status='predicted',score=node['score']))
                if result.get('termination') is None and (self.budget.max_depth is None or len(plan)<self.budget.max_depth):
                    focus=add_children(chosen,result['observation'])
                else:focus=[]
            except Exception as exc:
                node['status']='model_error';score([chosen]);focus=[]
                events.append(dict(node=chosen,schedule=schedule,plan=[a.packet() for a in plan],status='model_error',error=dict(type=type(exc).__name__,message=str(exc))))
        # All real-legal roots remain executable, even when unexpanded or rejected
        # by the model. Predicted descendants are additional evidence for a root.
        eligible_outputs=roots+[i for i in expanded if nodes[i]['status']=='predicted' and i not in roots]
        winner=min(eligible_outputs,key=lambda i:(nodes[i]['score'],-len(nodes[i]['plan']),i))
        action=nodes[winner]['plan'][0].packet();validate(action,observation)
        trace=dict(kind='option_search',root_action_count=len(roots),nodes_materialized=len(nodes),
            expansions=attempts,prediction_ticks_reserved=reserved_ticks,prediction_calls_reserved=reserved_calls,
            stop=stop,wall_seconds=time.monotonic()-started,selected_node=winner,
            selected_status=nodes[winner]['status'],selected_plan=[a.packet() for a in nodes[winner]['plan']],
            selected_score=nodes[winner]['score'],scoring_errors=scoring_errors,events=events,
            root_coverage='all original legal pairs and all 40 durations; no simulation-required execution gate')
        return action,trace
