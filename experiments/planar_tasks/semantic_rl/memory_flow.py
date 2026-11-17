"""Training-only, episode-local cursors and compact in-memory PPO buffers.

Cursor tokens are NOT evidence hashes. Author/exported histories retain the
canonical protocol. Host reads are borrowed only within synchronous execution;
authored callbacks still receive independent copies at the runtime boundary.
"""
from copy import deepcopy
import marshal
import uuid
import numpy as np
from .stream_history import StreamHistory
from experiments.planar_tasks.relational_search.contract import public_observation, validate_action


def _json_order(value):
    """Match canonical JSON's key order and tuple normalization, without text.

    Input is already a validated, independently owned public projection.
    """
    if isinstance(value,dict):return {k:_json_order(value[k]) for k in sorted(value)}
    if isinstance(value,(list,tuple)):return [_json_order(v) for v in value]
    return value


class MemoryHistory(StreamHistory):
    def __init__(self):
        self.episode=uuid.uuid4().hex
        self.index=-1;self._event=None
        self._chain=self.episode+':-1'
        self.previous_chain=None;self.previous_tick=None
        self.action_index=None

    @property
    def events(self):
        # Compatibility readers get ownership, never a mutable host reference.
        return [] if self._event is None else [marshal.loads(marshal.dumps(self._event))]

    def append(self,observation,action=None):
        self._append_projected(public_observation(observation),action)

    def _append_projected(self,observation,action=None):
        # Private ownership transfer from Environment: its freshly emitted
        # public() packet has already passed the full whitelist and validation.
        # Public append() above still validates/copies arbitrary caller inputs.
        from .boundary_actions import BoundaryActions
        packet=_json_order(observation)
        old=None if self._event is None else self._event['observation']
        if old is not None:
            validate_action(action,old)
            if not 0<packet['tick']-old['tick']<=action['ticks']:
                raise ValueError('stream boundaries must advance within preceding action')
        elif action is not None:raise ValueError('initial boundary has no preceding action')
        self.previous_tick=None if old is None else old['tick']
        self.previous_chain=self._chain;self.index+=1
        self._chain=self.episode+':'+str(self.index)
        self._event=dict(action=_json_order(deepcopy(action)),observation=packet)
        self.action_index=BoundaryActions(packet)

    def snapshot(self):
        raise ValueError('training cursor is not an auditable history snapshot')

    @classmethod
    def restore(cls,value):
        raise ValueError('training cursor cannot restore evidence snapshots')


def current_event(history):
    """Repository-host borrow; never hand this reference to an author callback."""
    return history._event if isinstance(history,MemoryHistory) else history.events[-1]


def compact_step(step):
    """All conditional candidates retained; only non-training metadata omitted."""
    out={k:deepcopy(step[k]) for k in ('action','indices','temperature','logp','value',
          'periods','reward','terminal','action_protocol') if k in step}
    out['panels']=[]
    for p in step['panels']:
        panel={k:deepcopy(p[k]) for k in ('semantic_sha256','schema','semantic_protocol','action_protocol') if k in p}
        # Matches Scale.forward's dtype/None handling exactly; detached copies.
        for k in ('semantic','raw'):panel[k]=np.array(p[k],dtype=np.float32,copy=True)
        out['panels'].append(panel)
    return out
