"""Versioned public-only runtime: rollouts cannot erase plausible unsafe modes.

Frozen before v0 and confirmation. The original engineering Controller remains
in the author campaign source snapshot so its optimistic override is auditable.
"""
from copy import deepcopy

from experiments.planar_tasks.authoring_v2.symbolic_general.online import OnlineRollout
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import outcome_valid
from tools.cap_symbolic_general_compile import compile_policy

VERSION='preserve-predicted-modes-v2'


def assess(policy,model,rollout,memory,observation,z,commands,action,queries_left,risk_threshold=.10):
    compile_policy(policy)
    before=deepcopy(rollout.ledger)
    pred=model.predict(action,z,policy.host)
    tests=[outcome_valid(policy,action,s['values'],path_safe=s['path_safe'])[0]for s in pred['samples']]
    risk=1-sum(tests)/len(tests)if tests else None
    outcomes=model.interval_outcomes(pred,policy.host)
    report=None;allowed=False;source='insufficient';online=None;used=0
    box={k:v if type(v)is bool else [v,v]for k,v in z.items()}
    if not z['legal:'+action]:source='current_precondition'
    else:
        # Even an uncalibrated or sparsely supported forecast can reveal a
        # counterexample. New nominal successes do not eliminate that outcome.
        if outcomes:
            report=policy.check(box,action,outcomes,quality='empirical',enumerate_rules=False)
        prior_safe=report is None or report['status']=='conditional'
        if pred['supported']and pred['calibrated']and prior_safe and risk is not None and risk<=risk_threshold:
            allowed=True;source='model'
        elif not prior_safe or risk is not None and risk>risk_threshold:
            source='unresolved_predicted_mode'
        elif queries_left:
            online=rollout.run(memory,observation,action,commands[action]);used=1
            samples=online['samples']
            tests=[outcome_valid(policy,action,s['values'],path_safe=s['path_safe'])[0]for s in samples]
            if len(tests)>=2 and all(tests):
                additional=[dict(updates={k:v if type(v)is bool else [v,v]for k,v in s['values'].items()},
                                 path_safe=s['path_safe'],label='observed-compatible online hypothesis')for s in samples]
                online['joint_symbolic']=policy.check(box,action,outcomes+additional,
                                                      quality='empirical',enumerate_rules=False)
                allowed=online['joint_symbolic']['status']=='conditional'
            source='online_empirical'if allowed else 'online_insufficient'
    return dict(action=action,allowed=allowed,source=source,predicted_risk=risk,
        supported=pred['supported'],calibrated=pred['calibrated'],local_rooms=pred['local_rooms'],
        symbolic=report,online=online,queries=used,
        rollout_delta={k:rollout.ledger[k]-before[k]for k in before},
        prediction=pred,interval_outcomes=outcomes)


class Controller:
    def __init__(self,model,*,risk=.10,max_queries=2,rollout=None):
        self.model=model;self.risk=risk;self.max_queries=max_queries
        self.rollout=rollout or OnlineRollout()
        self.ledger=dict(version=VERSION,boundaries=0,model_admissions=0,model_rejections=0,
                         online_queries=0,online_admissions=0,withheld=0,rollout=self.rollout.ledger)

    def decide(self,memory,observation,z,commands,policy):
        self.ledger['boundaries']+=1
        permissions={};checks=[];local=dict(z);remaining=self.max_queries
        for _ in range(len(commands)):
            selected=policy.select(local);action=selected['option']
            if action is None:break
            if action=='brake':
                permissions[action]=dict(permit=True,source='fallback_stop');break
            decision=assess(policy,self.model,self.rollout,memory,observation,local,commands,action,remaining,self.risk)
            remaining-=decision['queries'];self.ledger['online_queries']+=decision['queries']
            if decision['source']=='model':self.ledger['model_admissions']+=1
            if decision['source']=='unresolved_predicted_mode':self.ledger['model_rejections']+=1
            if decision['source']=='online_empirical':self.ledger['online_admissions']+=1
            if not decision['allowed']:self.ledger['withheld']+=1
            permissions[action]=dict(permit=decision['allowed'],source=decision['source'])
            # Keep the evidence report, not a copy of every training neighbor.
            checks.append({k:v for k,v in decision.items()if k not in ('prediction','interval_outcomes')})
            if decision['allowed']:break
            local['permit:'+action]=False
        return dict(permissions=permissions,checks=checks,model_sha256=self.model.sha256,runtime_version=VERSION)
