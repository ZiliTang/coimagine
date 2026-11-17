"""Actual Option-boundary replanning with a frozen fitted scorer."""
from copy import deepcopy
from dataclasses import asdict
import time
from .contract import History,validate_action
from .pipeline_execution import fixture,RecordedEnv,scene_metadata,validate_commands
from .live_search import Budget,choose
from .live_scorer import FittedScorer
from .predictor import NominalPredictor


def run(value,source,fitted,*,condition='M',prefix=(),max_ticks=400,max_decisions=30,
        budget=Budget(),wall_seconds=300,on_prediction=None,on_progress=None,
        model_factory=NominalPredictor,score_factory=FittedScorer,experience=None,reserved_root_calls=0):
    prefix=validate_commands(list(prefix),True)
    s,boxes=fixture(value);env=RecordedEnv(s,boxes,overrides=value['overrides'])
    h=History();h.append(env.public());scorer=score_factory(source,fitted,condition)
    rows=[dict(command_index=0,observation=h.events[-1]['observation'],action=None,history_hash=h.sha256)]
    events=[];decisions=[];commands=[];error=None;stop='execution_budget';prediction_ticks=prediction_calls=0
    started=time.monotonic()
    def step(action):
        env.command_index=len(commands)
        event=env.step_command(action);events.append(event);commands.append(deepcopy(action))
        h.append(env.public(),action)
        rows.append(dict(command_index=len(commands),observation=h.events[-1]['observation'],action=deepcopy(action),history_hash=h.sha256))
    for a in prefix:
        if env.termination is not None:break
        try:step(a)
        except ValueError as exc:error=dict(kind='invalid_prefix',message=str(exc));break
    origin=env.total_periods;failures=0
    for index in range(max_decisions):
        if error or env.termination is not None:break
        if env.total_periods-origin>=max_ticks:break
        if time.monotonic()-started>wall_seconds:
            error=dict(kind='wall_timeout',message='closed-loop episode wall limit');break
        decision=dict(index=index,real_tick=env.total_periods,real_history_hash=h.sha256,condition=condition)
        model=None;reservation=False;charged=calls=0
        try:
            if on_prediction:on_prediction('reserve',budget.prediction_ticks,budget.calls)
            reservation=True;charged=budget.prediction_ticks;calls=budget.calls
            model=model_factory(h.fork(),ticks=budget.prediction_ticks,calls=budget.calls,seconds=budget.seconds)
            result=choose(h.fork(),model,scorer,budget,experience=experience,reserved_root_calls=reserved_root_calls)
            charged=result['prediction_ticks'];calls=result['calls']
            decision.update(result);decision['prediction']=model.record()
        except Exception as exc:
            decision.update(status='planning_error',error=str(exc),action=None)
        finally:
            if model is not None:model.close()
            if reservation and on_prediction:on_prediction('settle',charged,calls)
            prediction_ticks+=charged;prediction_calls+=calls
        action=decision.get('action')
        if action is None:
            failures+=1;action=dict(target_id=None,a0=44,a1=44,ticks=4)
            decision['fallback']='common BRAKE, no privileged recovery'
        else:failures=0
        decision['executed_action']=action
        if env.total_periods-origin+action['ticks']>max_ticks:
            decision['stop']='remaining execution budget smaller than full action';decisions.append(decision);break
        try:validate_action(action,h.events[-1]['observation']);step(action)
        except ValueError as exc:error=dict(kind='invalid_policy_action',message=str(exc))
        decision['real_end_tick']=env.total_periods;decisions.append(decision)
        if on_progress:on_progress(len(decisions),env.total_periods,decision)
        if failures>=3:
            error=dict(kind='planner_unavailable',message='three consecutive planning failures');break
    else:stop='decision_budget'
    status='tool_error' if error else 'success' if env.success else 'task_failure'
    return dict(summary=dict(status=status,success=bool(env.success),ticks=env.total_periods,
        task_cost_seconds=None if error else env.total_periods*.05 if env.success else 2*env.config.time_budget,
        termination=env.termination or stop,error=error,boundaries=len(rows),decisions=len(decisions),
        prediction_ticks=prediction_ticks,prediction_calls=prediction_calls,prefix_ticks=origin,
        wall_seconds=time.monotonic()-started),public_rows=rows,events=events,decisions=decisions,
        commands=commands,replay=dict(scope='author/human privileged replay; not deployed inputs',
        scene=scene_metadata(s,boxes),frames=env.frames),search_profile=asdict(budget))
