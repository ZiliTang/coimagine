"""Current testbed actions and contracts; frozen v13 IDs remain readable.

Only canonical actions enter the menu/mask. Old reposition speed IDs are read
aliases for the single .30 m/s action in each direction, never extra choices.
"""
from __future__ import annotations
import csv
import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v13.push_controller import ContinuousPush
from coimagine.random_start.skills import RepositionSkill, SupportFollowSkill

VERSION = 'testbed-speeds-v1'
PUSH_SPEEDS = (.10, .20, .30, .40)
REPOSITION_SPEED = .30
# Keep IDs 0..45 to avoid silently reinterpreting saved commands (44 = STOP).
ACTION_ALIASES = {38: 40, 39: 40, 41: 43, 42: 43}


def _current_action(action):
    if action.kind == 'reposition':
        return replace(action, speed=REPOSITION_SPEED, scale=REPOSITION_SPEED,
                       scale_index=0,
                       label=f'REPOSITION_{action.direction}[v={REPOSITION_SPEED:.2f}m/s]')
    if action.kind == 'push' and action.effect_kind == 'translate':
        speed = PUSH_SPEEDS[action.scale_index]
        direction = C.DIRECTION_NAMES[action.type_index]
        return replace(action, speed=speed, scale=speed,
                       label=f'PUSH_{direction}[v={speed:.2f}m/s]')
    return action


ACTIONS = tuple(_current_action(a) for a in C.ACTIONS)
ACTIVE_ACTIONS = tuple(a for a in ACTIONS if a.index not in ACTION_ALIASES)
STOP, FOLLOW = C.STOP, C.FOLLOW
CODEBOOK_HASH = hashlib.sha256(json.dumps(
    [asdict(a) for a in ACTIONS], sort_keys=True).encode()).hexdigest()


def canonical_index(index):
    index = int(index)
    if not 0 <= index < len(ACTIONS):
        raise ValueError('option index out of range')
    return ACTION_ALIASES.get(index, index)


def canonical_sequence(sequence):
    return [dict(segment, a0=canonical_index(segment['a0']),
                 a1=canonical_index(segment['a1'])) for segment in sequence]


def mask(obs):
    # The geometry mask depends on directions; use the current speed config
    # and hide all legacy aliases from policy selection.
    import numpy as np
    push = tuple(a for a in ACTIVE_ACTIONS if a.kind == 'push')
    local = C.action_mask(obs, C.CONFIG, actions=push)
    result = np.zeros(len(ACTIONS), dtype=bool)
    for action, valid in zip(push, local):
        result[action.index] = valid
    for action in ACTIVE_ACTIONS:
        if action.kind == 'reposition':
            result[action.index] = obs.patch is not None
        elif action.kind in ('wait', 'follow'):
            result[action.index] = True
    return result

STATE_DIR = Path(__file__).resolve().parents[2] / 'docs/agent_exchange/state'
OPTION_TABLE = STATE_DIR / 'option_table.csv'
STATE_TABLE = STATE_DIR / 'state_table.csv'
PARAM_TABLE = STATE_DIR / 'param_table.csv'

PUSH_STAGES = ('approach', 'ready', 'push', 'recover')
ROTATE_STAGES = ('approach', 'ready', 'push', 'recover')
MOVE_STAGES = ('move', 'align', 'blocked')
FOLLOW_STAGES = ('follow', 'blocked')
WAIT_STAGES = ('brake',)

_COMMON_STATES = ('action_label', 'stage', 'ready', 'contact_ready', 'release', 'joint_active',
                  'joint_pair', 'cooperative', 'done', 'exit_reason', 'pause_reason', 'pending',
                  'recovery_requested', 'recovery_ready', 'skill_time', 'reference', 'force',
                  'skill_uid', 'stage_fallback', 'force_gated')
_PUSH_STATES = _COMMON_STATES + ('face', 'initial_face', 'contact_target', 'contact_seen',
                                 'contact_loss_seconds', 'effective_contact', 'normal_travel',
                                 'effect_progress', 'gap', 'tangent_error', 'preparation_gap_target',
                                 'best_preparation_error', 'progress_guard', 'recovery_count')
_MOVE_STATES = _COMMON_STATES + ('tangent', 'cw', 'tangent_direction', 'boundary_mode', 'peer_detour')
_FOLLOW_STATES = _COMMON_STATES + ('bound_push', 'bound_push_phase', 'follow_seconds', 'follow_max_seconds')
_WAIT_STATES = ('action_label', 'stage', 'done', 'exit_reason', 'skill_time', 'reference', 'force',
                'skill_uid', 'stage_fallback', 'force_gated')

PUSH_JOINT = ('same_family', 'both_ready', 'both_contact')
ROTATE_JOINT = ('same_family',)
FOLLOW_JOINT = ('peer_push',)


@dataclass(frozen=True)
class ParamSpec:
    name: str
    default: object
    low: object
    high: object
    unit: str
    description: str


@dataclass(frozen=True)
class OptionSpec:
    key: str
    family: str
    action_index: int
    label: str
    action: object
    params: tuple[ParamSpec, ...]
    observations: tuple[str, ...]
    states: tuple[str, ...]
    stages: tuple[str, ...]
    joint_requirements: tuple[str, ...]
    source: str

    @property
    def kind(self):
        return self.action.kind

    @property
    def speed(self):
        return float(getattr(self.action, 'speed', 0.))


def _params_from_config(config, names, family):
    out = []
    for name in names:
        if not hasattr(config, name):
            continue
        value = getattr(config, name)
        if isinstance(value, bool):
            low, high = False, True
        elif isinstance(value, int):
            low, high = 0, max(1, value * 4)
        elif isinstance(value, float):
            low, high = 0., max(0.001, value * 4.)
        else:
            continue
        out.append(ParamSpec(name, value, low, high, '', f'{family} controller parameter'))
    return tuple(out)


_PUSH_PARAM_NAMES = ('approach_normal_speed', 'approach_tangential_speed', 'approach_tangential_gain',
                     'approach_settle_speed', 'joint_engage_speed', 'gap_gain', 'tangential_gain',
                     'tangential_speed', 'contact_tangent_damping', 'recovery_enter_gap',
                     'recovery_ready_seconds', 'recovery_timeout', 'recovery_approach_speed')
_ROTATE_PARAM_NAMES = ('approach_normal_speed', 'approach_tangential_speed', 'rotation_trim_speed',
                       'rotation_engage_speed', 'cooperative_rotation_speed', 'rotation_brake_seconds',
                       'settle_angular_speed')
_PUSH_PARAMS = _params_from_config(C.CONFIG, _PUSH_PARAM_NAMES, 'PUSH') + (
    ParamSpec('rotation_feedback_gain', 0.0, -2.0, 2.0, '',
              'contact-point tangent gain from face-normal rotation; + = CCW->left, CW->right'),
)
_ROTATE_PARAMS = _params_from_config(C.ROTATION_CONFIG, _ROTATE_PARAM_NAMES, 'ROTATE')
_REPO_PARAMS = ()  # Single fixed speed; do not show the old, ineffective override.
_FOLLOW_PARAMS = (ParamSpec('target_gap', .02, 0., .2, 'm', 'follow target gap'),
                  ParamSpec('speed_limit', .32, 0., .5, 'm/s', 'follow speed limit'),
                  ParamSpec('follow_normal_force', .75, 0., 3., 'N', 'follow normal force'),
                  ParamSpec('follow_force', 3.0, 0., 8., 'N', 'follow force'),
                  ParamSpec('follow_max_seconds', 8.0, .5, 20., 's', 'follow timeout'))


def _make_spec(action):
    if action.kind == 'push' and getattr(action, 'effect_kind', None) == 'rotate':
        family, params, obs, states, stages, joint = (
            'ROTATE', _ROTATE_PARAMS, ('own', 'patch', 'scan', 'contacting_object'),
            _PUSH_STATES, ROTATE_STAGES, ROTATE_JOINT)
    elif action.kind == 'push':
        family, params, obs, states, stages, joint = (
            'PUSH', _PUSH_PARAMS, ('own', 'patch', 'scan', 'contacting_object'),
            _PUSH_STATES, PUSH_STAGES, PUSH_JOINT)
    elif action.kind == 'reposition':
        family, params, obs, states, stages, joint = (
            'REPOSITION', _REPO_PARAMS, ('own', 'scan'), _MOVE_STATES, MOVE_STAGES, ())
    elif action.kind == 'follow':
        family, params, obs, states, stages, joint = (
            'FOLLOW', _FOLLOW_PARAMS, ('own', 'scan', 'peer'), _FOLLOW_STATES, FOLLOW_STAGES, FOLLOW_JOINT)
    else:
        family, params, obs, states, stages, joint = (
            'WAIT', (), ('own',), _WAIT_STATES, WAIT_STAGES, ())
    return OptionSpec(key=f'{family}:{action.index}:{action.label}', family=family,
                      action_index=action.index, label=action.label, action=action,
                      params=params, observations=obs, states=states, stages=stages,
                      joint_requirements=joint, source='experiments/option_rl_v19/spec.py')


SPECS = tuple(_make_spec(a) for a in ACTIONS)
ACTIVE_SPECS = tuple(s for s in SPECS if s.action_index not in ACTION_ALIASES)
SPEC_BY_INDEX = {s.action_index: s for s in SPECS}
SPEC_BY_KEY = {s.key: s for s in SPECS}

COORDINATOR_STATES = (
    ('pair_phase', 'IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT'),
    ('pair_identity', 'stable ids of the two skills'),
    ('both_ready', 'AND of both option ready'),
    ('both_contact', 'AND of both option contact_ready'),
    ('both_active', 'AND of both joint_active'),
    ('force_enable_0/1', 'normal force allowed only in PUSH'),
    ('pair_start_tick/engage_tick/push_tick', 'phase timestamps'),
    ('contact_loss_latch', 'cleared when both contact is lost'),
    ('stage_fallback', 'why a push stage was reverted'),
)


def make_skill(action, overrides=None):
    """Thin factory: v13 behavior, optional per-family parameter overrides."""
    action = ACTIONS[canonical_index(action.index)]
    base = C.make_skill(action)
    ov = (overrides or {}).get(SPEC_BY_INDEX[action.index].family, {})
    if ov and isinstance(base, ContinuousPush):
        allowed = {p.name for p in SPEC_BY_INDEX[action.index].params}
        cfg = {k: v for k, v in ov.items() if k in allowed and hasattr(base.config, k)}
        if cfg:
            base.config = replace(base.config, **cfg)
    elif ov and isinstance(base, RepositionSkill):
        for k in ('boundary_mode', 'peer_detour'):
            if k in ov:
                setattr(base, k, ov[k])
    elif ov and isinstance(base, SupportFollowSkill):
        for k in ('target_gap', 'speed_limit', 'follow_normal_force', 'follow_force', 'follow_max_seconds'):
            if k in ov and hasattr(base, k):
                setattr(base, k, float(ov[k]))
    return base


def make_env(instance, context='extended', overrides=None):
    from experiments.option_rl_v19.system import OptionSystemEnv
    return OptionSystemEnv(instance, context=context, overrides=overrides)


def _write(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def write_tables():
    option_rows = []
    for s in ACTIVE_SPECS:
        option_rows.append(dict(
            key=s.key, family=s.family, action_index=s.action_index, label=s.label,
            kind=s.kind, effect_kind=getattr(s.action, 'effect_kind', ''),
            speed=s.speed, params=';'.join(p.name for p in s.params),
            observations=';'.join(s.observations), states=';'.join(s.states),
            stages=';'.join(s.stages), joint_requirements=';'.join(s.joint_requirements),
            source=s.source))
    _write(OPTION_TABLE, option_rows,
           ['key', 'family', 'action_index', 'label', 'kind', 'effect_kind', 'speed', 'params',
            'observations', 'states', 'stages', 'joint_requirements', 'source'])
    seen = {}
    for s in SPECS:
        for name in s.states:
            row = seen.setdefault(name, dict(field=name, options=[]))
            row['options'].append(s.key)
    state_rows = []
    for name, row in sorted(seen.items()):
        state_rows.append(dict(field=name, scope='option', options=';'.join(row['options'][:8]),
                               persistence='live' if name in ('stage', 'ready', 'contact_ready',
                                                              'release', 'joint_active') else 'live_or_latched',
                               invariant='see option table'))
    for name, meaning in COORDINATOR_STATES:
        state_rows.append(dict(field=name, scope='coordinator', options='', persistence='live',
                               invariant=meaning))
    _write(STATE_TABLE, state_rows, ['field', 'scope', 'options', 'persistence', 'invariant'])
    param_rows = []
    for s in SPECS:
        for p in s.params:
            param_rows.append(dict(family=s.family, name=p.name, default=p.default,
                                   low=p.low, high=p.high, unit=p.unit, description=p.description))
    seen = set(); unique = []
    for r in param_rows:
        k = (r['family'], r['name'])
        if k not in seen:
            seen.add(k); unique.append(r)
    _write(PARAM_TABLE, unique, ['family', 'name', 'default', 'low', 'high', 'unit', 'description'])
    return OPTION_TABLE, STATE_TABLE, PARAM_TABLE
