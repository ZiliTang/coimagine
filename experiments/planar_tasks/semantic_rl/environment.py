"""Public-feedback event boundaries over the unchanged multi-object controller."""
from dataclasses import dataclass, asdict
from copy import deepcopy
import math
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import History, validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS


@dataclass(frozen=True)
class EventConfig:
    mode: str = 'progress'
    distance_m: float = .20
    contact_ticks: int = 3
    loss_ticks: int = 5
    blocked_ticks: int = 5

    def __post_init__(self):
        if self.mode not in ('time', 'progress', 'duration') or not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError('invalid event mode/distance')
        if any(type(n) is not int or n < 1 for n in (self.contact_ticks, self.loss_ticks, self.blocked_ticks)):
            raise ValueError('positive integer event windows required')


class Detector:
    """Persistent contact latch; per-decision signed travel, no object truth."""
    def __init__(self, config):
        self.config = config
        self.signature = None

    def begin(self, action, observation):
        signature = (action['target_id'], action['a0'], action['a1'])
        if signature != self.signature:
            self.signature = signature
            self.contact_emitted = self.loss_emitted = False
            self.contact_n = self.loss_n = 0
        self.previous = observation
        self.action = action
        self.progress = [0., 0.]
        self.blocked = [0, 0]
        self.initial_done = [r['execution'].get('done', False) for r in observation['robots']]

    def update(self, observation):
        old = self.previous; self.previous = observation
        active = []
        for i, key in enumerate(('a0', 'a1')):
            spec = ACTIONS[self.action[key]]
            r, p = observation['robots'][i], old['robots'][i]
            ex = r['execution']
            committed = ex.get('stage') not in ('brake', 'switch_brake')
            dx, dy = r['own']['x']-p['own']['x'], r['own']['y']-p['own']['y']
            if spec.kind == 'push':
                active.append(i)
                if committed and r['contacting_object'] and p['contacting_object'] and ex.get('direction_valid', False):
                    direction = getattr(spec, 'effect_direction', None)
                    if direction is not None:
                        self.progress[i] += dx*direction[0]+dy*direction[1]
            elif spec.kind == 'move':
                angle = ('E','NE','N','NW','W','SW','S','SE').index(spec.direction)*math.pi/4
                self.progress[i] += dx*math.cos(angle)+dy*math.sin(angle)
            # Reposition/rotation use controller completion or the duration cap;
            # do not pretend translation odometry measures their effect.
            bad = ex.get('direction_unavailable', False) or ex.get('stage') in ('move_blocked', 'yield_blocked')
            self.blocked[i] = self.blocked[i]+1 if bad else 0
            if ex.get('done', False) and not self.initial_done[i] and spec.kind not in ('wait',):
                return 'controller_end'
        contact = bool(active) and all(observation['robots'][i]['contacting_object'] and
                observation['robots'][i]['execution'].get('direction_valid', False) for i in active)
        self.contact_n = self.contact_n+1 if contact else 0
        self.loss_n = 0 if contact else self.loss_n+1
        if max(self.blocked) >= self.config.blocked_ticks:
            return 'blocked'
        if active and not self.contact_emitted and self.contact_n >= self.config.contact_ticks:
            self.contact_emitted = True; self.loss_emitted = False
            return 'contact'
        if active and self.contact_emitted and not self.loss_emitted and self.loss_n >= self.config.loss_ticks:
            self.loss_emitted = True
            return 'contact_lost'
        if contact and self.contact_n >= self.config.contact_ticks:
            self.loss_emitted = False
        if max(self.progress) >= self.config.distance_m:
            return 'progress'
        return None


class Environment:
    def __init__(self, fixture, config=EventConfig(), *, fast_feedback=False, history_factory=History, record_details=True):
        scene, boxes = X.fixture(fixture)
        self.env = X.RecordedEnv(scene, boxes, overrides=fixture['overrides'], record_frames=record_details)
        self.record_details=record_details
        self.decision_count=0
        self.scene, self.boxes = scene, boxes
        self.config = config
        self.fast_feedback = fast_feedback
        self.env.cache_public_sensors = fast_feedback
        self.detector = Detector(config)
        self.history = history_factory(); self._append_public(self.env.public())
        self.events = []

    def _append_public(self,packet,action=None):
        """Only fresh outputs of this environment's public() enter this path."""
        from .memory_flow import MemoryHistory
        if isinstance(self.history,MemoryHistory):self.history._append_projected(packet,action)
        else:self.history.append(packet,action)

    def feedback(self):
        # Detector reads only these legal sensor fields, never menus or geometry.
        # Build the full public packet only at actual policy boundaries.
        from experiments.planar_tasks.interface import json_safe, FEEDBACK_FIELDS
        states = self.env._legal_robot_states()
        contacts = self.env.contacts()
        return json_safe(dict(robots=[dict(
            own=dict(x=states[i][0], y=states[i][1]),
            contacting_object=any(c[0]=='object' for c in contacts[str(i)]),
            execution={k:slot.last_info.get(k) for k in FEEDBACK_FIELDS})
            for i,slot in enumerate(self.env.slots)]))

    def step(self, requested, remaining_ticks=None):
        if self.config.mode == 'duration':
            from .duration_execution import step_duration
            return step_duration(self, requested, remaining_ticks)
        env = self.env
        if env.termination is not None:
            raise ValueError('terminal environment')
        if remaining_ticks is not None and remaining_ticks < 1:
            raise ValueError('no execution budget')
        from .memory_flow import current_event
        validate_action(requested, current_event(self.history)['observation'])
        if self.config.mode == 'time':
            # Compatibility mode is byte-for-byte the old command adapter.
            if remaining_ticks is not None and requested['ticks'] > remaining_ticks:
                raise ValueError('time-mode command exceeds remaining budget')
            env.command_index = self.decision_count
            event = env.step_command(requested)
            reason = 'terminal' if env.termination else 'duration'
            public = env.public()
        else:
            before = current_event(self.history)['observation']
            handoff = env.binding.select(requested, before)
            if handoff['commit']:
                env._commit_target(handoff['target'])
            executed = handoff['command']
            self.detector.begin(executed, before)
            start = env.total_periods
            limit = min(requested['ticks'], remaining_ticks or requested['ticks'])
            env.command_index = self.decision_count
            reason = 'duration'
            for _ in range(limit):
                env.joint_step(executed['a0'], executed['a1'])
                public = self.feedback() if self.fast_feedback else env.public()
                detected = self.detector.update(public)
                if env.termination:
                    reason = 'terminal'; break
                if handoff['phase'] == 'switch_brake':
                    reason = 'target_handoff'; break
                if detected:
                    reason = detected; break
            if env.total_periods-start == limit and limit < requested['ticks'] and reason == 'duration':
                reason = 'collector_cut'
            if self.fast_feedback:
                public = env.public()
            env.command_prefix.append(deepcopy(requested))
            event = dict(start_tick=start, end_tick=env.total_periods, requested=deepcopy(requested),
                         executed=deepcopy(executed), handoff=handoff, termination=env.termination)
        self._append_public(public, requested)
        event.update(reason=reason, periods=event['end_tick']-event['start_tick'],
                     success=bool(env.success), config=asdict(self.config))
        self.decision_count+=1
        if self.record_details:self.events.append(event)
        return event

    def replay(self):
        return dict(scope='privileged human visualization only', scene=X.scene_metadata(self.scene, self.boxes), frames=self.env.frames)
