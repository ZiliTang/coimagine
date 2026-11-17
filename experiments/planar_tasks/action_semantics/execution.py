"""Proposal identity / nonlinear choice / separately labelled inherited search."""
import time
from copy import deepcopy
import numpy as np
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import History,validate_action,digest
from experiments.planar_tasks.relational_search.pipeline_semantics import embedding
from experiments.planar_tasks.relational_search.live_search import choose,Budget
from experiments.planar_tasks.relational_search.predictor import NominalPredictor
from .program import Program,action_encoding
from .network import MLP


class Scorer:
    def __init__(self,program,fitted,condition='S'):
        if fitted['semantic_sha256']!=program.sha256:raise ValueError('source/fit mismatch')
        if condition not in ('S','E','M'):raise ValueError('condition')
        self.program=program;self.condition=condition
        self.model=MLP.restore(fitted['models'][condition]['checkpoint'])
        self.base=MLP.restore(fitted['models'][condition]['semantic_checkpoint']) if condition=='M' else None

    def packet_scores(self,h,packet):
        raw=[embedding(h,a)+action_encoding(h.events[-1]['observation'],a) for a in packet['actions']]
        scores=self.model.score(packet['semantic'] if self.condition=='S' else raw)
        if self.base is not None:scores+=self.base.score(packet['semantic'])
        return scores.tolist()

    def __call__(self,h,actions):
        packet=self.program.evaluate(h,actions,generic=False);scores=self.packet_scores(h,packet)
        return [scores[packet['actions'].index(a)] for a in actions]


class SemanticProposals:
    def __init__(self,program):self.program=program
    def retrieve(self,h):return [dict(action=a,provenance='action_semantics') for a in self.program.evaluate(h,generic=False)['proposals']]


def run(value,source,*,mode='proposal',fitted=None,condition='S',max_ticks=900,max_decisions=225,wall_seconds=300,on_progress=None,on_prediction=None,on_feature=None):
    if mode not in ('proposal','learned','search'):raise ValueError('mode')
    program=Program(source,on_feature);scorer=None if mode=='proposal' else Scorer(program,fitted,condition)
    s,b=X.fixture(value);env=X.RecordedEnv(s,b,overrides=value['overrides']);h=History();h.append(env.public())
    rows=[dict(command_index=0,action=None,observation=h.events[-1]['observation'],history_hash=h.sha256)]
    decisions=[];events=[];commands=[];error=None;prediction_ticks=prediction_calls=0;started=time.monotonic();stop='decision_budget'
    for index in range(max_decisions):
        if env.termination is not None:break
        if time.monotonic()-started>wall_seconds:error=dict(kind='wall_timeout',message='episode wall limit');break
        if env.total_periods>=max_ticks:stop='execution_budget';break
        d=dict(index=index,real_tick=env.total_periods,real_history_hash=h.sha256,mode=mode,condition=condition)
        try:
            packet=program.evaluate(h,generic=mode!='proposal');d.update(proposals=packet['proposals'],invalid_proposals=packet['invalid_proposals'])
            if mode=='proposal':
                if not packet['first_legal']:raise ValueError('CAP-equivalence first proposal missing/illegal')
                action=packet['raw_first']
            elif mode=='learned':
                scores=scorer.packet_scores(h,packet);selected=int(np.argmax(scores));action=packet['actions'][selected]
                d.update(candidates=packet['actions'],scores=scores,selected=selected)
            else:
                # Explicit diagnostic arm: unchanged nominal cost/leaf selector.
                budget=Budget();model=None;charged=budget.prediction_ticks;calls=budget.calls
                if on_prediction:on_prediction('reserve',charged,calls)
                try:
                    model=NominalPredictor(h,ticks=budget.prediction_ticks,calls=budget.calls,seconds=budget.seconds)
                    result=choose(h,model,scorer,budget,experience=SemanticProposals(program),reserved_root_calls=2)
                    charged=result['prediction_ticks'];calls=result['calls'];d.update(result,prediction=model.record());action=result['action']
                    if action is None:raise ValueError('no nominal executable frontier')
                finally:
                    if model is not None:model.close()
                    prediction_ticks+=charged;prediction_calls+=calls
                    if on_prediction:on_prediction('settle',charged,calls)
            validate_action(action,h.events[-1]['observation'])
            if env.total_periods+action['ticks']>max_ticks:stop='execution_budget';break
            env.command_index=len(commands);events.append(env.step_command(action));commands.append(deepcopy(action))
            h.append(env.public(),action);rows.append(dict(command_index=len(commands),action=deepcopy(action),observation=h.events[-1]['observation'],history_hash=h.sha256))
            d.update(executed_action=action,real_end_tick=env.total_periods);decisions.append(d)
            if on_progress:on_progress(len(decisions),env.total_periods,d)
        except Exception as exc:
            error=dict(kind='policy_or_planning_error',message=str(exc));d['error']=error;decisions.append(d);break
    return dict(summary=dict(status='tool_error' if error else 'success' if env.success else 'task_failure',success=bool(env.success),ticks=env.total_periods,
        task_cost_seconds=None if error else env.total_periods*.05 if env.success else 180.,termination=env.termination or stop,error=error,decisions=len(decisions),feature_batches=program.calls,prediction_ticks=prediction_ticks,prediction_calls=prediction_calls,wall_seconds=time.monotonic()-started),
        commands=commands,decisions=decisions,public_rows=rows,events=events,replay=dict(scene=X.scene_metadata(s,b),frames=env.frames))
