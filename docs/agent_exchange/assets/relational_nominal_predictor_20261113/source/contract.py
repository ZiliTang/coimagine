"""Explicit public packets, bounded causal history, and entity-bound commands."""
from copy import deepcopy
import hashlib
import json
import math


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def public_observation(raw):
    """Project an already sensor-produced packet; never accept truth extensions."""
    from experiments.planar_tasks.semantic_planning.runtime import visible
    base = visible(raw)
    tick = raw['tick']
    if type(tick) is not int or tick < 0:
        raise ValueError('nonnegative integer tick required')
    base['tick'] = tick
    if 'active_target' in raw:
        if not isinstance(raw['active_target'],str):
            raise ValueError('public committed control target required')
        base['active_target']=raw['active_target']
    if 'source' in raw:
        if raw['source'] not in ('real_observation','nominal_prediction'):
            raise ValueError('unknown observation source')
        base['source']=raw['source']
    base['goal'] = deepcopy(raw['goal'])
    if set(base['goal']) != {'target_id', 'xy', 'radius'}:
        raise ValueError('public goal schema')
    goal = base['goal']
    if (not isinstance(goal['target_id'], str) or len(goal['xy']) != 2 or
            not all(finite(v) for v in goal['xy']) or not finite(goal['radius']) or goal['radius'] <= 0):
        raise ValueError('finite public goal required')
    for robot, projected in zip(raw['robots'], base['robots']):
        labels = robot.get('scan', {}).get('entity_ids')
        if labels is not None:
            if (not isinstance(labels, list) or len(labels) != len(projected['scan'].get('valid', [])) or
                    any(v is not None and not isinstance(v, str) for v in labels)):
                raise ValueError('one instance label per ray required')
            if any(label is not None and not valid for label, valid in zip(labels, projected['scan']['valid'])):
                raise ValueError('no entity label without a ray hit')
            projected['scan']['entity_ids'] = deepcopy(labels)
    base['entities'] = []
    for entity in raw['entities']:
        if not isinstance(entity['id'], str) or not entity['id']:
            raise ValueError('opaque entity ID required')
        points = entity['points']
        if not isinstance(points, list) or len(points) > 512:
            raise ValueError('bounded observed point list required')
        if any(len(p) != 2 or not all(finite(v) for v in p) for p in points):
            raise ValueError('finite observed points required')
        stamp = entity['last_seen_tick']
        if type(stamp) is not int or not 0 <= stamp <= tick:
            raise ValueError('future entity evidence')
        base['entities'].append(dict(id=entity['id'], points=deepcopy(points),
                                     last_seen_tick=stamp))
    ids = [e['id'] for e in base['entities']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate entity ID')
    if 'entity_legal_pairs' in raw:
        pairs=raw['entity_legal_pairs']
        if set(pairs) != set(ids):
            raise ValueError('per-target menus must cover exactly observed entities')
        for menu in pairs.values():
            if any(not isinstance(p,list) or len(p)!=2 or any(type(a) is not int for a in p) or p not in base['legal_pairs'] for p in menu):
                raise ValueError('per-target menu outside public union')
        base['entity_legal_pairs']=deepcopy(pairs)
    canonical(base)
    return base


def validate_action(action, observation):
    if not isinstance(action, dict) or set(action) != {'target_id', 'a0', 'a1', 'ticks'}:
        raise ValueError('exact target_id/a0/a1/ticks command required')
    if any(type(action[k]) is not int for k in ('a0', 'a1', 'ticks')):
        raise ValueError('integer command fields required')
    if not 1 <= action['ticks'] <= 40:
        raise ValueError('ticks outside 1..40')
    ids = {e['id'] for e in observation['entities']
           if e['last_seen_tick'] == observation['tick'] and e['points']}
    global_brake = action['target_id'] is None and action['a0'] == action['a1'] == 44
    if not global_brake and action['target_id'] not in ids:
        raise ValueError('target requires current observed support')
    if [action['a0'], action['a1']] not in observation['legal_pairs']:
        raise ValueError('joint pair outside public menu')
    if not global_brake and 'entity_legal_pairs' in observation and [action['a0'],action['a1']] not in observation['entity_legal_pairs'][action['target_id']]:
        raise ValueError('joint pair unavailable for this observed target')
    return deepcopy(action)


def action_space(observation):
    # Observation order follows sensor encounter order, never the value of an ID.
    ids = [e['id'] for e in observation['entities']
           if e['points'] and e['last_seen_tick'] == observation['tick']]
    durations = (40, 20, 10, 1) + tuple(x for x in range(39, 0, -1) if x not in (40, 20, 10, 1))
    pairs = list(dict.fromkeys(tuple(p) for p in observation['legal_pairs']))
    for ticks in durations:
        for target in ids:
            for a0, a1 in pairs:
                if 'entity_legal_pairs' not in observation or [a0,a1] in observation['entity_legal_pairs'][target]:
                    yield dict(target_id=target, a0=a0, a1=a1, ticks=ticks)
        if (44, 44) in pairs:
            yield dict(target_id=None, a0=44, a1=44, ticks=ticks)


class History:
    """Store at most 64 causal boundary samples; forks do not modify real history."""
    def __init__(self, events=()):
        self.events = deepcopy(list(events))

    def append(self, observation, action=None):
        packet = public_observation(observation)
        if self.events:
            previous = self.events[-1]['observation']
            if packet['tick'] <= previous['tick']:
                raise ValueError('history boundaries must advance; duplicate/future update')
            validate_action(action, previous)
            if packet['tick'] - previous['tick'] > action['ticks']:
                raise ValueError('feedback exceeds command time')
        elif action is not None:
            raise ValueError('initial boundary has no preceding action')
        self.events.append(dict(observation=packet, action=deepcopy(action)))
        self.events = self.events[-64:]

    def fork(self):
        return History(self.events)

    @property
    def sha256(self):
        return digest(self.events)
