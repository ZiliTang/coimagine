"""Exact prefix support of the existing public joint-action protocol."""
from copy import deepcopy
from experiments.planar_tasks.relational_search.contract import public_observation, validate_action
from experiments.planar_tasks.relational_search.motion import MOVE_IDS

from experiments.planar_tasks.relational_search.action_scale import MIN_TICKS, MAX_TICKS

FIELDS = ('target_id', 'a0', 'a1', 'ticks')


class ActionTree:
    def __init__(self, observation, *, _projected=False, _action_index=None):
        # Internal synchronous training path borrows a host-owned projection.
        self.observation = observation if _projected else public_observation(observation)
        o = self.observation
        self._action_index=_action_index
        union = {tuple(p) for p in o['legal_pairs']}
        self.menus = {}
        global_menu = o.get('global_legal_pairs', [[44, 44]] if (44, 44) in union else [])
        if global_menu:
            self.menus[None] = sorted(set(map(tuple, global_menu)))
        for entity in o['entities']:
            if not entity['points'] or entity['last_seen_tick'] != o['tick']:
                continue
            pairs = o.get('entity_legal_pairs', {}).get(entity['id'], o['legal_pairs'])
            menu = sorted({tuple(p) for p in pairs if not any(a in MOVE_IDS for a in p)})
            if menu:
                self.menus[entity['id']] = menu
        if not self.menus:
            raise ValueError('empty legal action space')
        self._pairs={k:frozenset(v) for k,v in self.menus.items()}
        self._first={k:sorted({a for a,_ in v}) for k,v in self.menus.items()}
        self._second={}
        for target,menu in self.menus.items():
            by_first={}
            for a,b in menu:by_first.setdefault(a,[]).append(b)
            self._second[target]=by_first

    def choices(self, prefix):
        if tuple(prefix) != FIELDS[:len(prefix)] or len(prefix) >= len(FIELDS):
            raise ValueError('ordered incomplete action prefix required')
        depth = len(prefix)
        if depth == 0:
            values = list(self.menus)
        else:
            menu = self.menus[prefix['target_id']]
            if depth == 1:
                values = self._first[prefix['target_id']]
            elif depth == 2:
                values = self._second[prefix['target_id']].get(prefix['a0'],[])
            else:
                if (prefix['a0'], prefix['a1']) not in self._pairs[prefix['target_id']]:
                    raise ValueError('prefix outside legal menu')
                values = list(range(MIN_TICKS, MAX_TICKS+1))
        return [dict(prefix, **{FIELDS[depth]: v}) for v in values]

    def complete(self, prefix):
        """Deterministic legal representative, NOT a predicted best continuation."""
        out = deepcopy(prefix)
        while len(out) < 4:
            if len(out) == 3:
                out['ticks'] = 4
            else:
                out = self.choices(out)[0]
        if self._action_index is not None:self._action_index.validate(out,self.observation)
        else:validate_action(out, self.observation)
        return out

    @property
    def count(self):
        return (MAX_TICKS-MIN_TICKS+1) * sum(map(len, self.menus.values()))
