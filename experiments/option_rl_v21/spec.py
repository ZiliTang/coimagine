"""Versioned V2 menu; append YIELD, preserving all historical numeric IDs."""
from dataclasses import asdict, replace
import hashlib
import json
import numpy as np

from coimagine.random_start.spec import RobotAction
from experiments.option_rl_v19 import spec as V1
from experiments.option_rl_v21.yield_skill import YieldConfig, YieldSkill

VERSION = 'option-v2-yield-signed-push'
PUSH_SPEEDS, REPOSITION_SPEED = V1.PUSH_SPEEDS, V1.REPOSITION_SPEED
ACTION_ALIASES = dict(V1.ACTION_ALIASES)
STOP, FOLLOW, YIELD = V1.STOP, V1.FOLLOW, 46
ACTIONS = V1.ACTIONS + (RobotAction(kind='yield', speed=.40, scale=.40, scale_kind='persistent',
                                   type_index=14, scale_index=0, index=YIELD, label='YIELD_CLEARANCE'),)
ACTIVE_ACTIONS = tuple(a for a in ACTIONS if a.index not in ACTION_ALIASES)
CODEBOOK_HASH = hashlib.sha256(json.dumps([asdict(a) for a in ACTIONS], sort_keys=True).encode()).hexdigest()


def canonical_index(index):
    index = int(index)
    if not 0 <= index < len(ACTIONS):
        raise ValueError('option index out of range')
    return ACTION_ALIASES.get(index, index)


def canonical_sequence(sequence):
    return [dict(s, a0=canonical_index(s['a0']), a1=canonical_index(s['a1'])) for s in sequence]


def mask(obs):
    from experiments.option_rl_v19.unknown_shape import visible_faces, direction_matches
    result = np.zeros(len(ACTIONS), bool)
    faces = visible_faces(obs)
    valid_types = {}
    for action in ACTIVE_ACTIONS:
        if action.kind == 'push' and action.effect_kind == 'translate':
            if action.type_index not in valid_types:
                valid_types[action.type_index] = any(direction_matches(f[1], action.effect_direction, obs) for f in faces)
            result[action.index] = valid_types[action.type_index]
        elif action.kind in ('wait', 'yield', 'follow'):
            result[action.index] = True
        elif action.kind in ('reposition', 'push'):
            result[action.index] = any(p.valid for p in obs.patches)
    return result


YIELD_PARAMS = tuple(V1.ParamSpec(k, v, .001, max(1., 3*v), '', 'local YIELD '+k)
                     for k, v in asdict(YieldConfig()).items())
YIELD_SPEC = V1.OptionSpec(key='YIELD:46:YIELD_CLEARANCE', family='YIELD', action_index=YIELD,
    label=ACTIONS[YIELD].label, action=ACTIONS[YIELD], params=YIELD_PARAMS,
    observations=('own', 'scan', 'peer', 'public_arena', 'estimated_surface_velocity'),
    states=('active', 'side', 'min_gap', 'predicted_min_gap', 'blocked', 'candidates'),
    stages=('yield_idle', 'yield_escape', 'yield_blocked'), joint_requirements=(), source=__file__)
SPECS = V1.SPECS + (YIELD_SPEC,)
ACTIVE_SPECS = tuple(s for s in SPECS if s.action_index not in ACTION_ALIASES)
SPEC_BY_INDEX = {s.action_index: s for s in SPECS}


def make_skill(action, overrides=None, *, estimator=None):
    if action.kind == 'yield':
        values = (overrides or {}).get('YIELD', {})
        allowed = set(asdict(YieldConfig()))
        if set(values)-allowed:
            raise ValueError('unknown YIELD parameter')
        return YieldSkill(action, estimator=estimator, config=replace(YieldConfig(), **values))
    return V1.make_skill(action, overrides)
