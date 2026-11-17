"""Budgeted lazy Option search with local action costs and explicit evidence levels."""
from dataclasses import dataclass, asdict
import math
import time


@dataclass(frozen=True,slots=True)
class Action:
    a0:int
    a1:int
    ticks:int
    def packet(self):return asdict(self)


def validate(action,observation):
    if not isinstance(action,dict) or set(action)!={'a0','a1','ticks'}:raise ValueError('exact a0/a1/ticks command required')
    if any(type(v)is not int for v in action.values()):raise ValueError('integer command required')
    if not 1<=action['ticks']<=40 or [action['a0'],action['a1']] not in observation['legal_pairs']:raise ValueError('outside observable legal menu')
    return action


def action_space(observation,preferred=None):
    """Lazy exhaustive support, useful windows first; no duration/pair removed."""
    pairs=list(dict.fromkeys(tuple(p) for p in observation['legal_pairs']))
    if any(len(p)!=2 or any(type(x)is not int for x in p) for p in pairs):raise ValueError('malformed legal menu')
    seen=set()
    if preferred is not None:
        try:validate(preferred,observation)
        except ValueError:pass
        else:
            a=Action(**preferred);seen.add(a);yield a
    durations=(40,20,10,1)+tuple(t for t in range(39,0,-1) if t not in (40,20,10,1))
    for ticks in durations:
        for p in pairs:
            a=Action(*p,ticks)
            if a not in seen:yield a


@dataclass(frozen=True)
class Budget:
    expansions:int=6
    prediction_ticks:int=480
    prediction_calls:int=40
    search_seconds:float=2.5
    candidate_batch:int=24
    widen_batch:int=8
    proposal_limit:int=8
    lookahead_ticks:int=120
    max_depth:object=None
    def __post_init__(self):
        for name in ('expansions','prediction_ticks','prediction_calls'):
            if type(getattr(self,name))is not int or getattr(self,name)<0:raise ValueError(name)
        for name in ('candidate_batch','widen_batch','lookahead_ticks','proposal_limit'):
            if type(getattr(self,name))is not int or getattr(self,name)<1:raise ValueError(name)
        if type(self.search_seconds)not in (int,float) or not math.isfinite(self.search_seconds) or self.search_seconds<=0:raise ValueError('search_seconds')
        if self.max_depth is not None and (type(self.max_depth)is not int or self.max_depth<1):raise ValueError('max_depth')


def finite(value):return type(value) in (int,float) and math.isfinite(value)


def root_state(hypothesis):
    if hypothesis is None:return None
    keys=('x','y','angle','vx','vy','omega')
    return dict(object=dict(zip(keys,hypothesis['object'])),robots=[dict(zip(keys,r)) for r in hypothesis['robots']])


class Search:
    def __init__(self,scorer,budget=None):self.scorer=scorer;self.budget=budget or Budget()

    def choose(self,observation,context,model):
        started=time.monotonic();deadline=started+self.budget.search_seconds;b=self.budget
        errors=[];events=[];nodes={};next_id=0;candidate_count=0;rank_calls=0;flat_batches=0;reserved_ticks=reserved_calls=0
        def error(stage,exc):
            if len(errors)<12:errors.append(dict(stage=stage,type=type(exc).__name__,message=str(exc)))
        try:self.scorer.begin(context,deadline)
        except Exception as exc:error('begin',exc)

        def node(parent,action,result):
            nonlocal next_id
            ident=next_id;next_id+=1
            path=parent['packet']['path_context']+[action] if parent else []
            packet=dict(id=ident,source='predicted' if parent else 'real_observation_estimate',
                state=result['state'] if parent else root_state(context.get('hypothesis')),
                observation=result['observation'] if parent else observation,
                elapsed_ticks=result['ticks'] if parent else 0,
                depth=len(path),incoming_action=action,path_context=path,
                goal=context['observation']['robots'][0]['goal_xy'],
                terminal=result.get('termination') if parent else None,
                success=bool(result.get('success')) if parent else False)
            try:
                value=self.scorer.register(ident,packet,deadline)
                if not finite(value):raise ValueError('evaluate_state must return one finite remaining-cost estimate')
            except Exception as exc:error('evaluate_state',exc);value=0.
            if packet['terminal']:value=0. if packet['success'] else 180.
            n=dict(packet=packet,value=value,edges={},stream=action_space(packet['observation'],action if parent else context.get('previous_action')),batches=0,exhausted=False)
            nodes[ident]=n;return n

        root=node(None,None,{})
        def candidates(n):
            nonlocal candidate_count,rank_calls,flat_batches
            if n['packet']['terminal']:return
            batch=[];limit=b.candidate_batch if not n['batches'] else b.widen_batch
            while len(batch)<limit and not n['exhausted']:
                try:a=next(n['stream']).packet()
                except StopIteration:n['exhausted']=True;break
                key=(a['a0'],a['a1'],a['ticks'])
                if key not in n['edges']:batch.append(a)
            if not batch:return
            rank_calls+=1;n['batches']+=1
            try:
                reply=self.scorer.rank(n['packet']['id'],batch,deadline)
                if not isinstance(reply,dict) or set(reply)-{'costs','proposals'} or 'costs' not in reply:raise ValueError('rank_actions requires costs and optional proposals')
                costs=reply['costs'];proposals=reply.get('proposals',[])
                if not isinstance(costs,list) or len(costs)!=len(batch) or not all(finite(x) for x in costs):raise ValueError('one finite local total-cost estimate per candidate')
                if not isinstance(proposals,list) or len(proposals)>b.proposal_limit:raise ValueError('proposal count limit')
                for p in proposals:
                    if not isinstance(p,dict) or set(p)!={'action','cost'} or not finite(p['cost']):raise ValueError('proposal requires action and finite cost')
                    validate(p['action'],n['packet']['observation'])
            except Exception as exc:error('rank_actions',exc);costs=[n['value']]*len(batch);proposals=[]
            if len(set(costs))==1 and len(costs)>1:flat_batches+=1
            for a,q in list(zip(batch,costs))+[(p['action'],p['cost']) for p in proposals]:
                key=(a['a0'],a['a1'],a['ticks'])
                if key not in n['edges']:
                    n['edges'][key]=dict(action=dict(a),prior=q,status='unexpanded',child=None,blocked=False);candidate_count+=1
                elif n['edges'][key]['status']=='unexpanded':n['edges'][key]['prior']=min(n['edges'][key]['prior'],q)

        candidates(root)
        if not root['edges']:raise ValueError('empty actual legal menu')
        focus=root;stop='expansion_budget_exhausted'
        for attempt in range(b.expansions):
            if time.monotonic()>=deadline:stop='search_deadline';break
            # Explicit finite allocation: one root then up to two local extensions.
            # Duration costs are proposed by the hook, never inferred as equal gain.
            n=root if attempt%3==0 or focus is None else focus
            if n['packet']['terminal'] or n['packet']['elapsed_ticks']>=b.lookahead_ticks or (b.max_depth is not None and n['packet']['depth']>=b.max_depth):n=root
            if not n['batches'] or any(e['status']!='unexpanded' for e in n['edges'].values()):candidates(n)
            choices=[e for e in n['edges'].values() if e['status']=='unexpanded' and not e['blocked']]
            chosen=None
            for e in sorted(choices,key=lambda e:e['prior']):
                ticks,calls=model.cost(n['packet'],e['action'])
                if reserved_ticks+ticks<=b.prediction_ticks and reserved_calls+calls<=b.prediction_calls:
                    chosen=e;break
                e['blocked']=True
            if chosen is None:
                focus=None
                if n is root:stop='no_affordable_generated_candidate';break
                continue
            reserved_ticks+=ticks;reserved_calls+=calls
            action=chosen['action'];plan=n['packet']['path_context']+[action]
            event=dict(parent_id=n['packet']['id'],local_action=action,plan=plan,
                prior_total_seconds=chosen['prior'],requested_horizon_ticks=sum(a['ticks'] for a in plan))
            try:
                result=model.expand(n['packet'],next_id,action,deadline)
                child=node(n,action,result);chosen.update(status='predicted',child=child['packet']['id'],edge_seconds=(child['packet']['elapsed_ticks']-n['packet']['elapsed_ticks'])*.05)
                focus=child;event.update(status='predicted',child_id=child['packet']['id'],state_value=child['value'],prefix_reused=model.last_reused)
            except Exception as exc:
                chosen['status']='model_error';focus=None;event.update(status='model_error',error=dict(type=type(exc).__name__,message=str(exc)))
            events.append(event)

        def backed(n):
            successful=[e for e in n['edges'].values() if e['status']=='predicted']
            if not successful:return n['value'],[]
            values=[]
            for e in successful:
                value,path=backed(nodes[e['child']]);values.append((e['edge_seconds']+value,[e['action']]+path))
            return min(values,key=lambda pair:pair[0])
        output=[]
        for e in root['edges'].values():
            if e['status']=='predicted':
                value,path=backed(nodes[e['child']]);value+=e['edge_seconds'];path=[e['action']]+path
            else:value=e['prior'];path=[e['action']]
            output.append((value,e,path))
        value,winner,path=min(output,key=lambda x:x[0]);action=validate(dict(winner['action']),observation)
        suffix=[a['ticks'] for e in events for a in e['plan'][1:]]
        predicted=[e for e in events if e['status']=='predicted']
        max_horizon=max([e['requested_horizon_ticks'] for e in predicted] or [0])
        warnings=[]
        if suffix and all(t==1 for t in suffix):warnings.append('all_forecast_suffixes_one_tick')
        if max_horizon<b.lookahead_ticks:warnings.append('lookahead_time_target_not_reached')
        if flat_batches:warnings.append('flat_local_candidate_costs')
        if winner['status']!='predicted':warnings.append('selected_root_uses_unverified_prior')
        trace=dict(kind='option_search_v3',root_action_support=len(set(tuple(p) for p in observation['legal_pairs']))*40,
            generated_candidates=candidate_count,root_candidates_generated=len(root['edges']),state_nodes=len(nodes),rank_calls=rank_calls,flat_rank_batches=flat_batches,
            expansions=len(events),prediction_ticks_reserved=reserved_ticks,prediction_calls_reserved=reserved_calls,
            selected_status=winner['status'],selected_plan=path,selected_cost_seconds=value,
            selected_horizon_ticks=sum(a['ticks'] for a in path),maximum_forecast_ticks=max_horizon,
            suffix_ticks=suffix,distinct_forecast_root_commands=len(set(tuple(e['plan'][0][k] for k in ('a0','a1','ticks')) for e in events)),
            diagnostics=warnings,hook_errors=errors,events=events,stop=stop,wall_seconds=time.monotonic()-started,
            selected_prediction_state=(nodes[winner['child']]['packet']['state'] if winner['status']=='predicted' else None),
            selection_semantics='predicted root uses backed-up predicted frontier; unexpanded/model-error root retains its explicit local total-cost prior')
        return action,trace
