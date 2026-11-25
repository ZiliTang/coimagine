"""Explicit public packets, bounded causal history, and entity-bound commands."""
from copy import deepcopy
import hashlib
import json
import math
import marshal
import pickle
from itertools import zip_longest
from .motion import ACTIONS, MOVE_IDS, VERSION, is_global_pair
from .action_scale import MIN_TICKS, MAX_TICKS
from experiments.planar_tasks.semantic_planning.runtime import public_copy


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def content_key(value):
    """Internal mutation detector, never a public/serialized evidence digest.

    Public packets are plain trees; marshal avoids pickle's Python-object memo
    overhead. Non-packet extensions retain the old fallback. Canonical JSON
    remains authoritative for history IDs and all frozen artifact hashes.
    """
    try:
        data = marshal.dumps(value)
    except (ValueError, TypeError):
        data = pickle.dumps(value, protocol=4)
    return hashlib.sha256(data).digest()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def public_observation(raw):
    """Public entry: validate/whitelist arbitrary packets and copy their values."""
    return _project_observation(raw, public_copy)


def _owned_public_observation(raw):
    """Host-only ownership transfer of a fresh environment packet.

    The caller must exclusively own every nested value and discard raw after
    this call. Whitelist, validation, and canonical check are identical to the
    public copying entry. Never use for a borrowed sensor/history/author input.
    """
    return _project_observation(raw, lambda value: value)


def _project_observation(raw, copy_value):
    """Project an already sensor-produced packet; never accept truth extensions."""
    from experiments.planar_tasks.semantic_planning.runtime import _visible
    base = _visible(raw, copy_value)
    tick = raw['tick']
    if type(tick) is not int or tick < 0:
        raise ValueError('nonnegative integer tick required')
    base['tick'] = tick
    for key in ('action_version', 'action_codebook_hash'):
        if key in raw:
            if not isinstance(raw[key], str):
                raise ValueError('string action version/hash required')
            base[key] = raw[key]
    if 'active_target' in raw:
        if not isinstance(raw['active_target'],str):
            raise ValueError('public committed control target required')
        base['active_target']=raw['active_target']
    if 'source' in raw:
        if raw['source'] not in ('real_observation','nominal_prediction'):
            raise ValueError('unknown observation source')
        base['source']=raw['source']
    base['goal'] = copy_value(raw['goal'])
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
            projected['scan']['entity_ids'] = copy_value(labels)
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
        base['entities'].append(dict(id=entity['id'], points=copy_value(points),
                                     last_seen_tick=stamp))
    ids = [e['id'] for e in base['entities']]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate entity ID')
    if 'entity_legal_pairs' in raw:
        pairs=raw['entity_legal_pairs']
        if set(pairs) != set(ids):
            raise ValueError('per-target menus must cover exactly observed entities')
        public_union = {tuple(p) for p in base['legal_pairs']}
        for menu in pairs.values():
            if any(not isinstance(p,list) or len(p)!=2 or any(type(a) is not int for a in p) or tuple(p) not in public_union for p in menu):
                raise ValueError('per-target menu outside public union')
        base['entity_legal_pairs']=copy_value(pairs)
    if 'global_legal_pairs' in raw:
        menu = raw['global_legal_pairs']
        union = {tuple(p) for p in base['legal_pairs']}
        if not isinstance(menu, list) or any(
                not isinstance(p, list) or len(p) != 2 or any(type(a) is not int for a in p)
                or not is_global_pair(*p) or tuple(p) not in union for p in menu):
            raise ValueError('invalid target-free action menu')
        base['global_legal_pairs'] = copy_value(menu)
    canonical(base)
    return base


class NoLegalPushFace(ValueError):
    """The named pusher has no offered translating action in this direction."""

    def __init__(self, robot):
        super().__init__('push_no_legal_face')
        self.robot = robot


def validate_action(action, observation):
    if not isinstance(action, dict) or set(action) != {'target_id', 'a0', 'a1', 'ticks'}:
        raise ValueError('exact target_id/a0/a1/ticks command required')
    if any(type(action[k]) is not int for k in ('a0', 'a1', 'ticks')):
        raise ValueError('integer command fields required')
    if not MIN_TICKS <= action['ticks'] <= MAX_TICKS:
        raise ValueError('ticks outside 1..40')
    ids = {e['id'] for e in observation['entities']
           if e['last_seen_tick'] == observation['tick'] and e['points']}
    global_action = action['target_id'] is None
    pair = [action['a0'], action['a1']]
    if global_action and (not is_global_pair(*pair) or pair not in observation.get('global_legal_pairs', [[44,44]])):
        raise ValueError('target-free command outside global menu')
    if not global_action and (action['a0'] in MOVE_IDS or action['a1'] in MOVE_IDS):
        raise ValueError('MOVE is target-free; use target_id=null')
    if not global_action and action['target_id'] not in ids:
        raise ValueError('target requires current observed support')
    menu = observation.get('entity_legal_pairs', {}).get(action['target_id'], ())
    if not global_action and observation.get('action_version') == VERSION and pair not in menu:
        # Each individually available translation is offered with a waiting peer.
        for robot, code in enumerate(pair):
            if (0 <= code < len(ACTIONS) and getattr(ACTIONS[code], 'effect_kind', None) == 'translate'
                    and not any(offered[robot] == code for offered in menu)):
                raise NoLegalPushFace(robot)
    if [action['a0'], action['a1']] not in observation['legal_pairs']:
        raise ValueError('joint pair outside public menu')
    if not global_action and 'entity_legal_pairs' in observation and [action['a0'],action['a1']] not in observation['entity_legal_pairs'][action['target_id']]:
        raise ValueError('joint pair unavailable for this observed target')
    return deepcopy(action)


def action_space(observation):
    # Observation order follows sensor encounter order, never the value of an ID.
    ids = [e['id'] for e in observation['entities']
           if e['points'] and e['last_seen_tick'] == observation['tick']]
    durations = (40, 20, 10, 1) + tuple(x for x in range(39, 0, -1) if x not in (40, 20, 10, 1))
    pairs = list(dict.fromkeys(tuple(p) for p in observation['legal_pairs']))
    for ticks in durations:
        def manipulation():
            for target in ids:
                for a0, a1 in pairs:
                    if a0 in MOVE_IDS or a1 in MOVE_IDS:
                        continue
                    if 'entity_legal_pairs' not in observation or [a0,a1] in observation['entity_legal_pairs'][target]:
                        yield dict(target_id=target, a0=a0, a1=a1, ticks=ticks)
        global_menu = observation.get('global_legal_pairs', [[44,44]] if (44,44) in pairs else [])
        navigation = (dict(target_id=None, a0=a0, a1=a1, ticks=ticks) for a0,a1 in global_menu)
        # Both families appear in early batches; no legal combination is pruned.
        if 'global_legal_pairs' in observation:
            for pair in zip_longest(manipulation(), navigation):
                yield from (a for a in pair if a is not None)
        else:
            yield from manipulation()
            yield from navigation


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
        # Content-based cache: mutations invalidate it; canonical digest is unchanged.
        previous = getattr(self, '_canonical_events', {})
        keys = []; current = {}
        for event in self.events:
            key = content_key(event)
            keys.append(key)
            current[key] = previous[key] if key in previous else canonical(event)
        if keys != getattr(self, '_digest_keys', None):
            self._digest_value = hashlib.sha256(('['+','.join(current[k] for k in keys)+']').encode()).hexdigest()
            self._digest_keys = keys
        self._canonical_events = current
        return self._digest_value
