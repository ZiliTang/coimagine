"""Host-only fixture/physics adapter. Replay geometry never enters author packets."""
from copy import deepcopy
from dataclasses import asdict, replace
import time
import math

from coimagine.continuous2d.scenes import Pose
from . import testbed as T
from .contract import History, public_observation, validate_action
from .environment import MultiObjectOptionEnv


def seed_fixture():
    s = T.scene()
    return dict(object=asdict(s.object_pose), robots=[asdict(p) for p in s.robot_poses],
                goal=dict(x=s.goal_centre[0], y=s.goal_centre[1], radius=s.goal_radius),
                movable_objects=T.default_boxes(), overrides={})


def fixture(value):
    required = {'object', 'robots', 'goal', 'movable_objects', 'overrides'}
    random_room = value.get('scene_id') == 'random_room_v1'
    if set(value) not in (required, required | {'scene_id'}, required | {'scene_id', 'room'}):
        raise ValueError('exact fixture schema required')
    if ('room' in value) != random_room:
        raise ValueError('room geometry requires random_room_v1')
    # Host-only layout selection; never forwarded to deployed policy inputs.
    # Omission preserves the historical clear-transport geometry exactly.
    scene_id = value.get('scene_id', T.SCENE_ID)
    if random_room:
        from coimagine.continuous2d.scenes import Box
        room = value['room']
        if set(room) != {'arena', 'obstacles'} or tuple(room['arena']) != (6., 4.8):
            raise ValueError('random_room_v1 fixes a 6 x 4.8 m arena')
        if not isinstance(room['obstacles'], list) or len(room['obstacles']) > 4:
            raise ValueError('at most four static boxes')
        obstacles=[]
        for b in room['obstacles']:
            if set(b) != {'cx','cy','half_w','half_h','angle'} or any(type(v) not in (int,float) or not math.isfinite(v) for v in b.values()) or min(b['half_w'],b['half_h']) <= 0:
                raise ValueError('finite positive static boxes required')
            obstacles.append(Box(**b))
        base=replace(T.scene(), name='random_room_v1', obstacles=tuple(obstacles))
    elif scene_id == T.SCENE_ID:
        base = T.scene()
    elif scene_id == 'planar_wall_anchor':
        base = T.task_catalog()[scene_id].instance
    else:
        raise ValueError('unsupported fixture scene_id')
    if len(value['robots']) != 2:
        raise ValueError('two robots required')
    numbers=[v for p in [value['object'],*value['robots'],value['goal']] for v in p.values()]
    if any(type(v) not in (float,int) or not math.isfinite(v) for v in numbers):
        raise ValueError('finite numeric poses and public goal required')
    s = replace(base, object_pose=Pose(**value['object']),
                robot_poses=tuple(Pose(**p) for p in value['robots']),
                goal_centre=(value['goal']['x'], value['goal']['y']), goal_radius=value['goal']['radius'])
    if value['goal']['radius'] != base.goal_radius:
        raise ValueError('do not change the success radius')
    boxes = T.parse_boxes(value['movable_objects'])
    T.validate_geometry(s, boxes)
    return s, boxes


def commands(segments):
    """Preserve the Workbench's 0.2 s boundaries, including handoff timing."""
    labels = {T.PRIMARY_ID} | {s.get('target_id') for s in segments if s.get('target_id')}
    return [dict(target_id=s['target_id'], a0=s['a0'], a1=s['a1'], ticks=4)
            for s in T.normalize_sequence(segments, labels) for _ in range(s['decisions'])]


def validate_commands(rows, allow_empty=False):
    if not isinstance(rows, list) or not (0 if allow_empty else 1) <= len(rows) <= 2000:
        raise ValueError('bounded command list required')
    for a in rows:
        if (not isinstance(a, dict) or set(a) != {'target_id', 'a0', 'a1', 'ticks'} or
                a['target_id'] is not None and not isinstance(a['target_id'], str) or
                any(type(a[k]) is not int for k in ('a0', 'a1', 'ticks')) or not 1 <= a['ticks'] <= 40):
            raise ValueError('exact integer command schema, ticks 1..40')
    return deepcopy(rows)


def scene_metadata(s, boxes):
    polygon = s.object_polygon
    if polygon is None:
        from coimagine.random_start import spec as R
        w, h = R.OBJECT_WIDTH/2, R.OBJECT_HEIGHT/2
        polygon = ((-w,-h),(w,-h),(w,h),(-w,h))
    return dict(arena=s.arena, object_polygon=polygon, robot_radius=.18,
                goal=dict(x=s.goal_centre[0], y=s.goal_centre[1], radius=s.goal_radius),
                primary_id=T.PRIMARY_ID, obstacles=[asdict(b) for b in s.obstacles],
                movable_objects=[asdict(b) for b in boxes])


class RecordedEnv(MultiObjectOptionEnv):
    def __init__(self, *args, **kwargs):
        self.record_frames = kwargs.pop('record_frames', True)
        super().__init__(*args, **kwargs)
        self.command_index = -1
        self.frames = []
        self.capture()

    def capture(self):
        if not self.record_frames:return
        a = self.offline_audit()
        self.frames.append(dict(tick=self.total_periods, t=self.total_periods*.05,
            entities=a['entities'], robots=a['robots'], active_target=self.active_target,
            command_index=self.command_index))

    def joint_step(self, a0, a1, record=False):
        result = super().joint_step(a0, a1, record=record)
        self.capture()
        return result


def execute(value, sequence, *, max_ticks=1800, wall_seconds=120., on_boundary=None):
    """All starts are fresh resets; all prefixes cost real ticks. No live clone."""
    validate_commands(sequence)
    if sum(a['ticks'] for a in sequence) > max_ticks:
        raise ValueError('sequence exceeds declared episode tick cap; no silent truncation')
    s, boxes = fixture(value)
    env = RecordedEnv(s, boxes, overrides=value['overrides'])
    h = History(); h.append(env.public())
    rows = [dict(command_index=0, observation=h.events[-1]['observation'], action=None,
                 history_hash=h.sha256)]
    events = []; error = None; started = time.monotonic()
    for index, action in enumerate(sequence):
        if env.termination is not None:
            break
        if time.monotonic()-started > wall_seconds:
            error = dict(kind='wall_timeout', command_index=index, message='episode wall limit')
            break
        env.command_index = index
        try:
            validate_action(action, h.events[-1]['observation'])
        except ValueError as exc:
            error = dict(kind='invalid_action', command_index=index, message=str(exc))
            break
        event = env.step_command(action)
        events.append(event)
        h.append(env.public(), action)
        rows.append(dict(command_index=index+1, observation=h.events[-1]['observation'],
                         action=deepcopy(action), history_hash=h.sha256))
        if on_boundary:
            on_boundary(index+1, env.total_periods)
    status = ('tool_error' if error else 'success' if env.success else 'task_failure')
    # Same task-cost metric as planar runner: success time, otherwise 2*T.
    cost = None if error else env.total_periods*.05 if env.success else 2*env.config.time_budget
    return dict(summary=dict(status=status, success=bool(env.success), ticks=env.total_periods,
                    task_cost_seconds=cost, termination=env.termination or 'sequence_exhausted',
                    error=error, boundaries=len(rows), wall_seconds=time.monotonic()-started),
                public_rows=rows, events=events,
                replay=dict(scope='privileged human visualization only', scene=scene_metadata(s, boxes),
                            frames=env.frames))


def history_at(rows, boundary):
    if type(boundary) is not int or not 0 <= boundary < len(rows):
        raise ValueError('boundary unavailable')
    h = History()
    for row in rows[:boundary+1]:
        h.append(public_observation(row['observation']), row['action'])
        if h.sha256 != row['history_hash']:
            raise ValueError('history integrity mismatch')
    return h
