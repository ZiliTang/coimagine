"""Immutable append-only boundary cursor; raw trajectories belong to the recorder.

Only the most recent event is retained. Readers receive independent decoded data;
mutating it cannot change the committed event or its chain identity.
"""
import json
import uuid
from experiments.planar_tasks.relational_search.contract import public_observation, validate_action, canonical, digest

VERSION = 'semantic-event-stream-v1'


class StreamHistory:
    def __init__(self):
        self.episode = uuid.uuid4().hex
        self.index = -1
        self._encoded = None
        self._chain = digest(dict(version=VERSION))
        self.previous_chain = None
        self.previous_tick = None

    @property
    def events(self):
        return [] if self._encoded is None else [json.loads(self._encoded)]

    def append(self, observation, action=None):
        packet = public_observation(observation)
        previous = self.events
        if previous:
            old = previous[0]['observation']
            validate_action(action, old)
            if not 0 < packet['tick']-old['tick'] <= action['ticks']:
                raise ValueError('stream boundaries must advance within preceding action')
        elif action is not None:
            raise ValueError('initial boundary has no preceding action')
        self.previous_tick = previous[0]['observation']['tick'] if previous else None
        self.previous_chain = self._chain
        self._encoded = canonical(dict(observation=packet, action=action))
        self.index += 1
        self._chain = digest(dict(version=VERSION, previous=self.previous_chain,
                                  index=self.index, previous_tick=self.previous_tick, event=json.loads(self._encoded)))

    @property
    def sha256(self):return self._chain

    def snapshot(self):
        return dict(version=VERSION, episode=self.episode, index=self.index, event=self._encoded,
                    chain=self._chain, previous_chain=self.previous_chain, previous_tick=self.previous_tick)

    @classmethod
    def restore(cls, value):
        if value.get('version') != VERSION:raise ValueError('stream version mismatch')
        h=cls()
        if type(value['index']) is not int or value['index'] < 0:raise ValueError('nonempty stream snapshot required')
        event=json.loads(value['event'])
        if set(event) != {'observation','action'}:raise ValueError('event fields')
        if canonical(public_observation(event['observation'])) != canonical(event['observation']):
            raise ValueError('non-public snapshot')
        tick=event['observation']['tick'];previous=value['previous_tick']
        if value['index']==0:
            if previous is not None or event['action'] is not None:raise ValueError('initial stream snapshot')
        elif type(previous) is not int or not 0<=previous<tick or not isinstance(event['action'],dict) or not 0<tick-previous<=event['action'].get('ticks',0):
            raise ValueError('invalid stream snapshot timing')
        expected=digest(dict(version=VERSION,previous=value['previous_chain'],index=value['index'],
                             previous_tick=previous,event=event))
        if expected != value['chain']:raise ValueError('stream checksum')
        h.episode=value['episode'];h.index=value['index'];h._encoded=canonical(event)
        h._chain=expected;h.previous_chain=value['previous_chain'];h.previous_tick=value['previous_tick']
        return h

    def fork(self):
        return self.restore(self.snapshot())
