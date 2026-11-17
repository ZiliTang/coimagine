"""Explicit larger development budgets; deployment prediction/search limits unchanged."""
from copy import deepcopy
import json
from pathlib import Path
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.baseline.config import validate_config
from experiments.planar_tasks.baseline.prediction import DEVELOPMENT_LIMITS

DEFAULT_PROFILE = dict(
    version='experiment-authoring-v2',
    prediction_limits={**DEVELOPMENT_LIMITS, 'calls':100000, 'ticks':1000000,
                       'models':30000, 'cpu_seconds':2400.},
    tool_reply_bytes=12000, detail_bytes=8000, context_trigger_bytes=64000,
    context_max_bytes=131072, recent_turns=2, notebook_bytes=8000,
    max_branches=128, max_prefix_commands=200, max_branch_commands=12,
    batch_wall_seconds=120.)


def validate_profile(profile):
    if set(profile) != set(DEFAULT_PROFILE) or profile['version'] != DEFAULT_PROFILE['version']:
        raise ValueError('unsupported experiment profile')
    # This release fixes summary/context semantics and development prediction accounting.
    # New profile values need a new release rather than an author-side override.
    if profile != DEFAULT_PROFILE:
        raise ValueError('experiment profile differs from frozen defaults')
    return deepcopy(profile)


def campaign_config(model, development_tasks=None):
    """Prepare, never execute; explicit provider/name/settings supplied by caller."""
    c=json.loads((seal.BASE/'development_config.json').read_text())
    c['model']=deepcopy(model)
    if development_tasks is not None:c['development_tasks']=list(development_tasks)
    c['prediction']=dict(version='nominal-v1',runtime=True)
    c['budget'].update(wall_seconds=5400,tool_calls=100000,model_calls=80,
        model_input_bytes=33554432,model_output_tokens=1310720,
        model_call_output_tokens=16384,model_call_seconds=600,
        python_call_seconds=120,python_total_seconds=4200,
        resets=2000,control_ticks=250000,policy_tests=100,
        episode_wall_seconds=120,response_seconds=5)
    c['finalization']=dict(model_calls=2,model_input_bytes=1048576,
        model_output_tokens=32768,tool_calls=12,wall_seconds=300)
    # DeepSeek forced-submit transport requires this existing explicit declaration.
    if model['provider']=='deepseek':c['submission_settings']={'thinking':{'type':'disabled'}}
    validate_config(c)
    return c
