"""Fixed deployment adapter: one original-format action per real observation."""
import json
import sys
from search import Search, Budget
from model import ReplayModel
from estimator import Estimator
from hook_client import HookClient
from predictor_client import Predictor


class Policy:
    def __init__(self,contract):
        self.contract=contract;self.estimator=Estimator();self.scorer=HookClient();self.previous=None
        with open('planner_config.json') as stream:self.config=json.load(stream)
        self.search=Search(self.scorer,Budget(**self.config))

    def act(self,observation):
        error=None
        try:hypothesis=self.estimator.update(observation)
        except Exception as exc:hypothesis=None;error=str(exc)
        context=dict(observation=observation,hypothesis=hypothesis,previous_action=self.previous,
            initialization='cold legal hypothesis; synthetic paths preserve their own option history',estimation_error=error)
        model=ReplayModel(Predictor(),hypothesis)
        action,trace=self.search.choose(observation,context,model)
        trace.update(decision=observation['decision'],prediction_rpc_calls=model.calls,
            prediction_ticks_requested=model.requested_ticks,cleanup_errors=model.cleanup_errors,
            initialization=context['initialization'])
        print(json.dumps(trace,allow_nan=False),file=sys.stderr,flush=True)
        self.previous=action
        return action
