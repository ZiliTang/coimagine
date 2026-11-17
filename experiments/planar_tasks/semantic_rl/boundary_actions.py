"""Private lookup view of one owned, validated decision-boundary packet.

The command validator is unchanged. Only its list membership queries use sets.
Never reuse this index for another packet, even at the same simulation tick.
"""
from experiments.planar_tasks.relational_search.contract import validate_action


class _Pairs:
    def __init__(self,pairs):self._values=frozenset(map(tuple,pairs))
    def __contains__(self,pair):return tuple(pair) in self._values


class BoundaryActions:
    def __init__(self,observation):
        self._owner=observation
        self._stamp=(observation['tick'],observation.get('active_target'))
        self._view=dict(observation)
        for key in ('legal_pairs','global_legal_pairs'):
            if key in observation:self._view[key]=_Pairs(observation[key])
        if 'entity_legal_pairs' in observation:
            self._view['entity_legal_pairs']={k:_Pairs(v) for k,v in observation['entity_legal_pairs'].items()}

    def validate(self,action,observation):
        if observation is not self._owner or (observation['tick'],observation.get('active_target'))!=self._stamp:
            raise ValueError('action index belongs to a different boundary packet')
        return validate_action(action,self._view)
