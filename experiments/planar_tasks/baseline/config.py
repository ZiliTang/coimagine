"""Explicit finite campaign budgets; no paid or physics work on import/init."""
from dataclasses import asdict, dataclass, fields
import math


@dataclass(frozen=True)
class Budget:
    wall_seconds: float
    tool_calls: int
    model_calls: int
    model_input_bytes: int
    model_output_tokens: int  # conservative reservation, including failed calls
    model_call_output_tokens: int
    model_call_seconds: float
    python_call_seconds: float
    python_total_seconds: float
    resets: int
    control_ticks: int
    policy_tests: int
    workspace_bytes: int
    output_bytes: int
    episode_wall_seconds: float
    response_seconds: float

    def __post_init__(self):
        for f in fields(self):
            value=getattr(self,f.name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'{f.name} must be positive and finite')
            if f.type is int and type(value) is not int:
                raise ValueError(f'{f.name} must be an integer')
        if self.model_call_output_tokens>self.model_output_tokens:
            raise ValueError('single model output reservation exceeds total')

    def as_dict(self):
        return asdict(self)


def validate_config(config):
    required={'model','budget','development_tasks','purpose'}
    if not required<=set(config) or set(config)-required-{'finalization','submission_settings','prediction'}:
        raise ValueError('config requires model, budget, development_tasks, purpose; optional finalization/submission_settings')
    required={'provider','name','settings','source_access','accounting'}
    if not isinstance(config['model'],dict) or set(config['model'])!=required:
        raise ValueError('model requires provider/name/settings/source_access/accounting')
    model=config['model']
    if model['source_access']!='hidden':
        raise ValueError('v1 only implements source-hidden tools; source-visible requires another profile')
    if model['accounting']!='reserved_output_tokens_and_input_bytes':
        raise ValueError('unsupported model accounting; no implicit monetary estimate')
    if not isinstance(model['settings'],dict) or not all(isinstance(model[k],str) and model[k] for k in ('provider','name')):
        raise ValueError('explicit model name/provider/settings required')
    if config['purpose'] not in ('development','infrastructure_smoke'):
        raise ValueError('v1 has no independent research-test panel')
    names=config['development_tasks']
    if not isinstance(names,list) or not names or len(set(names))!=len(names) or not all(isinstance(n,str) for n in names):
        raise ValueError('explicit distinct development tasks required')
    budget=Budget(**config['budget'])
    if 'prediction' in config:
        from .prediction import validate_profile
        validate_profile(config['prediction'])
        if config['purpose'] != 'development':
            raise ValueError('prediction requires development purpose')
    reserve=config.get('finalization',{})
    keys={'model_calls','model_input_bytes','model_output_tokens','tool_calls','wall_seconds'}
    if reserve and set(reserve)!=keys:
        raise ValueError('finalization requires model_calls/input_bytes/output_tokens, tool_calls, wall_seconds')
    for key,value in reserve.items():
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not 0<value<getattr(budget,key):
            raise ValueError('finalization reserve must be positive and below total: '+key)
        if key!='wall_seconds' and type(value)is not int:raise ValueError('integer finalization reserve required')
    if reserve and reserve['model_output_tokens']<reserve['model_calls']*budget.model_call_output_tokens:
        raise ValueError('reserve must cover finalization model output caps')
    if reserve and (reserve['model_calls']<2 or reserve['tool_calls']<reserve['model_calls']):
        raise ValueError('submission stage requires at least two calls and one tool allowance per call')
    submission=config.get('submission_settings',{})
    if not isinstance(submission,dict) or set(submission)-{'thinking','reasoning_effort'}:
        raise ValueError('submission_settings may only override thinking and reasoning_effort')
    if submission and not reserve:
        raise ValueError('submission_settings requires finalization')
    if 'thinking' in submission and submission['thinking'] not in ({'type':'enabled'},{'type':'disabled'}):
        raise ValueError('invalid submission thinking setting')
    if 'reasoning_effort' in submission and submission['reasoning_effort'] not in ('none','low','high','max'):
        raise ValueError('invalid submission reasoning_effort')
    effective=request_settings(config,finalizing=True)
    if reserve and model['provider']=='deepseek' and effective.get('thinking',{}).get('type')!='disabled':
        raise ValueError('DeepSeek forced submit requires explicitly disabled thinking; no silent mode switch')
    return budget


def request_settings(config, *, finalizing=False):
    """Apply only a predeclared phase override; never silently disable thinking."""
    settings=dict(config['model']['settings'])
    if finalizing:
        settings.update(config.get('submission_settings',{}))
    return settings
