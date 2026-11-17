"""Bounded nominal predictor: explicit hypotheses only, no execution-world access.

Forks reconstruct this predictor's own history, including persistent controllers.
Replay is charged like forward simulation. Creating a hypothesis cold-starts
controllers, contacts and success counters; it is NOT a live-state clone.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import time

from coimagine.continuous2d.scenes import Box, Pose
from coimagine.random_start.layout import InstanceSpec
from ..interface import TaskEnvironment, validate_action

LIMITS = dict(calls=2000, ticks=12000, models=512, cpu_seconds=30.,
              decision_calls=64, decision_ticks=600, decision_cpu_seconds=1., handles=16)
DEVELOPMENT_LIMITS = dict(LIMITS, calls=12000, ticks=120000, models=4096, cpu_seconds=300.)


def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != {'version', 'runtime'}:
        raise ValueError('prediction requires version and runtime')
    if profile['version'] != 'nominal-v1' or type(profile['runtime']) is not bool:
        raise ValueError('only nominal-v1 and boolean runtime supported')


def vector(value, size, bound=30.):
    if not isinstance(value, list) or len(value) != size:
        raise ValueError(f'expected list of {size} finite numbers')
    if any(type(v) not in (int, float) or not math.isfinite(v) or abs(v) > bound for v in value):
        raise ValueError('nonfinite or out-of-range hypothesis')
    return tuple(float(v) for v in value)


def validate_hypothesis(h):
    if not isinstance(h, dict) or set(h) != {'arena', 'goal', 'goal_radius', 'object', 'robots', 'obstacles'}:
        raise ValueError('hypothesis requires arena, goal, goal_radius, object, robots, obstacles only')
    arena = vector(h['arena'], 2, 10.)
    if min(arena) < 1.:
        raise ValueError('arena dimensions must be 1..10 m')
    goal = vector(h['goal'], 2, 10.)
    radius = vector([h['goal_radius']], 1, 1.)[0]
    if radius <= 0:
        raise ValueError('goal_radius must be positive')
    obj = vector(h['object'], 6)
    if not isinstance(h['robots'], list) or len(h['robots']) != 2:
        raise ValueError('two robot states required')
    robots = [vector(r, 6) for r in h['robots']]
    for state in [obj, *robots]:
        if not (0 <= state[0] <= arena[0] and 0 <= state[1] <= arena[1]):
            raise ValueError('body centre outside arena')
        if max(abs(v) for v in state[3:]) > 3.:
            raise ValueError('initial speed exceeds hypothesis bound 3')
    if not isinstance(h['obstacles'], list) or len(h['obstacles']) > 16:
        raise ValueError('at most 16 obstacle boxes')
    boxes = []
    for item in h['obstacles']:
        cx, cy, hw, hh, angle = vector(item, 5, 10.)
        if min(hw, hh) <= 0:
            raise ValueError('box half sizes must be positive')
        boxes.append(Box(cx, cy, hw, hh, angle))
    return InstanceSpec('hypothesis', 'prediction', 'synthetic', 0, 0, arena, tuple(boxes),
                        Pose(*obj[:3]), tuple(Pose(*r[:3]) for r in robots), goal, radius)


class PredictionLedger:
    def __init__(self, limits=None, usage=None, persist=None):
        self.limits = dict(LIMITS if limits is None else limits)
        self.usage = usage if usage is not None else {}
        self.persist = persist or (lambda: None)

    def charge(self, key, amount=1):
        if self.usage.get(key, 0) + amount > self.limits[key]:
            raise RuntimeError('prediction ' + key + ' budget exhausted')
        self.usage[key] = self.usage.get(key, 0) + amount


class PredictionService:
    def __init__(self, *, ledger=None, log_path=None, local_limits=None):
        self.ledger = ledger or PredictionLedger()
        self.worlds = {}
        self.serial = 0
        self.log_path = log_path
        self.local_usage = {}
        self.local_limits = local_limits
        self.begin_decision()

    def begin_decision(self):
        self.decision = dict(calls=0, ticks=0, cpu_seconds=0.)

    def charge(self, key, amount=1):
        if self.local_limits and self.local_usage.get(key, 0) + amount > self.local_limits[key]:
            raise RuntimeError('prediction episode ' + key + ' budget exhausted')
        limit = self.ledger.limits.get('decision_' + key)
        if limit is not None and self.decision.get(key, 0) + amount > limit:
            raise RuntimeError('prediction per-decision ' + key + ' budget exhausted')
        self.ledger.charge(key, amount)
        self.local_usage[key] = self.local_usage.get(key, 0) + amount
        self.decision[key] = self.decision.get(key, 0) + amount

    def check_time(self, deadline, cpu_start):
        if time.monotonic() >= deadline:
            raise TimeoutError('prediction caller deadline exceeded')
        delta = time.process_time() - cpu_start
        if self.local_limits and self.local_usage.get('cpu_seconds', 0) + delta >= self.local_limits['cpu_seconds']:
            raise TimeoutError('prediction episode CPU budget exhausted')
        if self.decision['cpu_seconds'] + delta >= self.ledger.limits['decision_cpu_seconds']:
            raise TimeoutError('prediction per-decision CPU budget exhausted')
        if self.ledger.usage.get('cpu_seconds', 0) + delta >= self.ledger.limits['cpu_seconds']:
            raise TimeoutError('prediction total CPU budget exhausted')

    def build(self, h):
        instance = validate_hypothesis(h)
        if len(self.worlds) >= self.ledger.limits['handles']:
            raise RuntimeError('prediction live handle budget exhausted')
        self.charge('models')
        world = TaskEnvironment(instance)
        backend = world._env.env.backend
        for body, state in zip([backend.object_body, *backend.robot_bodies], [h['object'], *h['robots']]):
            body.velocity = tuple(state[3:5])
            body.angular_velocity = state[5]
        world._env.heading_rate = [r[5] for r in h['robots']]
        return dict(world=world, hypothesis=deepcopy(h), history=[])

    def advance(self, branch, action, deadline, cpu_start):
        world = branch['world']; env = world._env
        if env.termination is not None:
            raise ValueError('prediction terminated')
        a0, a1, ticks = validate_action(action, world.observe())
        for _ in range(ticks):
            self.check_time(deadline, cpu_start)
            self.charge('ticks')
            # Recording is required by the same option controller history. It
            # remains private to this synthetic branch and bounded by 1800 ticks.
            env.joint_step(a0, a1, record=True)
            if env.termination is not None:
                break
        world.decisions += 1; env.skills_used = world.decisions
        branch['history'].append(dict(action))

    def result(self, handle):
        world = self.worlds[handle]['world']; env = world._env
        return dict(handle=handle, source='synthetic_hypothesis_only',
                    state=dict(object=asdict(env.object_state()), robots=[asdict(r) for r in env.robot_states()]),
                    observation=world.observe(), ticks=env.total_periods,
                    success=bool(env.success), termination=env.termination,
                    usage=dict(self.ledger.usage))

    def rpc(self, request, deadline):
        started = time.monotonic(); cpu_start = time.process_time()
        result = None; error = None; invalidated = None
        try:
            self.charge('calls'); self.check_time(deadline, cpu_start)
            schemas = dict(create={'hypothesis'}, step={'handle', 'action'}, fork={'handle'},
                           close={'handle'}, status=set())
            op = request.get('operation')
            if request.get('method') != 'predict' or op not in schemas or set(request) != schemas[op] | {'method', 'operation'}:
                raise ValueError('invalid prediction RPC')
            if op == 'status':
                result = dict(usage=dict(self.ledger.usage), limits=self.ledger.limits,
                              handles=list(self.worlds), decision=dict(self.decision))
            elif op == 'create':
                branch = self.build(request['hypothesis'])
                self.check_time(deadline, cpu_start)
                self.serial += 1; handle = str(self.serial); self.worlds[handle] = branch
                invalidated = handle
                result = self.result(handle)
            else:
                handle = request['handle']
                if not isinstance(handle, str) or handle not in self.worlds:
                    raise ValueError('unknown prediction handle; no live-world handles exist')
                if op == 'close':
                    del self.worlds[handle]; result = dict(closed=handle)
                elif op == 'fork':
                    original = self.worlds[handle]
                    branch = self.build(original['hypothesis'])
                    for action in original['history']:
                        self.advance(branch, action, deadline, cpu_start)
                    self.serial += 1; new_handle = str(self.serial); self.worlds[new_handle] = branch
                    invalidated = new_handle
                    result = self.result(new_handle)
                else:
                    # Any partial failure discards only this branch; no ambiguous
                    # partially executed state can be reused after a timeout.
                    invalidated = handle
                    self.advance(self.worlds[handle], request['action'], deadline, cpu_start)
                    result = self.result(handle)
            self.check_time(deadline, cpu_start)
            return result
        except Exception as exc:
            if invalidated is not None:
                self.worlds.pop(invalidated, None)
            error = dict(type=type(exc).__name__, message=str(exc))
            raise
        finally:
            cpu = time.process_time() - cpu_start
            # Actual cost, including rejected/error calls, may cross the bound by
            # at most one bounded engine operation. Next call is rejected.
            for usage in (self.ledger.usage, self.local_usage, self.decision):
                usage['cpu_seconds'] = usage.get('cpu_seconds', 0) + cpu
            self.ledger.persist()
            if self.log_path is not None:
                with self.log_path.open('a') as stream:
                    stream.write(json.dumps(dict(request=request, result=result, error=error,
                        cpu_seconds=cpu, wall_seconds=time.monotonic()-started,
                        usage=dict(self.ledger.usage)), allow_nan=False) + '\n')
