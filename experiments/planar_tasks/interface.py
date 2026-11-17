"""Centralized legal sensor sharing, including rotation; independent of V2 RL.

Only JSON messages cross the policy boundary. No task ID, scene truth, reward,
full obstacle map, environment object or simulator snapshot is part of that API.
"""
from dataclasses import asdict
import math

from coimagine.random_start import spec as R
from coimagine.random_start.skills import SkillObservation
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.event_env import legal_pair as rl_legal_pair
from experiments.option_rl_v21.system import OptionSystemEnv

VERSION = 'planar-option-json-v1.1'
# Explicit whitelist: additions to SkillObservation do not silently expand the API.
OBS_FIELDS = ('robot_id', 'tick', 'time', 'own', 'teammate', 'scan', 'patches', 'patch',
              'contacting_object', 'contact_point_robot', 'contacting_static',
              'contacting_teammate', 'goal_xy', 'goal_radius', 'arena', 'force_limit',
              'remaining_seconds', 'elapsed_seconds', 'static_nearest', 'teammate_nearest')
FEEDBACK_FIELDS = ('stage', 'done', 'exit_reason', 'pause_reason', 'direction_valid',
                   'direction_unavailable', 'reference_velocity', 'desired_velocity', 'force')
KINEMATICS_FIELDS = ('x', 'y', 'angle', 'vx', 'vy', 'omega')
SCAN_FIELDS = ('robot_id', 'tick', 'sample_time', 'origin', 'heading', 'ray_angles',
               'distances', 'valid', 'category')
PATCH_FIELDS = ('points_robot', 'points_world', 'distance', 'centroid_distance',
                'nearest_distance', 'normal_world', 'tangent_world', 'confidence',
                'arc_rays', 'span_m', 'rms_m', 'valid')


def sensor_fields(value, names):
    """Project nested sensors explicitly; diagnostic additions stay host-only."""
    return {name: getattr(value, name) for name in names}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def legal_pair(a, b):
    # Preserve V2's existing translation/reposition/wait/yield combinations.
    if rl_legal_pair(a, b):
        return True
    # Rotation uses the existing same-family coordinator. Do not introduce
    # opposite rotation intents or unvalidated mixed rotation controllers.
    return (a.kind == b.kind == 'push' and a.effect_kind == b.effect_kind == 'rotate'
            and a.rotation_sign == b.rotation_sign)


def policy_observation(observations, feedback, decision):
    masks = [S.mask(o) for o in observations]
    pairs = [[a.index, b.index] for a in S.ACTIVE_ACTIONS for b in S.ACTIVE_ACTIONS
             if masks[0][a.index] and masks[1][b.index] and legal_pair(a, b)]
    robots = []
    for observation, execution in zip(observations, feedback):
        if not isinstance(observation, SkillObservation):
            raise TypeError('SkillObservation required')
        robot = {key: getattr(observation, key) for key in OBS_FIELDS}
        for key in ('own', 'teammate'):
            robot[key] = sensor_fields(getattr(observation, key), KINEMATICS_FIELDS)
        robot['scan'] = sensor_fields(observation.scan, SCAN_FIELDS)
        robot['patches'] = [sensor_fields(patch, PATCH_FIELDS) for patch in observation.patches]
        robot['patch'] = None if observation.patch is None else sensor_fields(observation.patch, PATCH_FIELDS)
        robot['execution'] = {key: execution.get(key) for key in FEEDBACK_FIELDS}
        robots.append(robot)
    return json_safe(dict(protocol=VERSION, decision=decision, robots=robots,
                          legal_pairs=pairs))


def contract():
    return dict(protocol=VERSION, information='centralized_shared_sensors',
        action_version=S.VERSION, codebook_sha256=S.CODEBOOK_HASH,
        actions=[asdict(a) for a in S.ACTIVE_ACTIONS if a.kind != 'follow'],
        response=dict(a0='integer action ID', a1='integer action ID',
                      ticks='integer 1..40; each tick is 0.05 simulated seconds'),
        semantics=dict(persistent=True, repeated_action='continues current controller; does not reset it',
            rotate='both robots, same rotation sign',
            brake='active braking, not passive absence of force',
            nonfinite_sensor='null', history='client may retain all earlier observations',
            termination='runner stops; terminal truth is evaluation-only',
            coordinates='world x right, y up, radians CCW; scans include ray angles and origin'),
        control_dt=R.CONTROL_DT, max_ticks=40)


def validate_action(response, observation):
    if not isinstance(response, dict) or set(response) != {'a0', 'a1', 'ticks'}:
        raise ValueError('response must contain exactly a0, a1, ticks')
    if any(type(response[k]) is not int for k in response):
        raise ValueError('action IDs and ticks must be integers (not booleans)')
    if not 1 <= response['ticks'] <= 40:
        raise ValueError('ticks outside 1..40')
    if [response['a0'], response['a1']] not in observation['legal_pairs']:
        raise ValueError('joint action is not in the observable legal menu')
    return response['a0'], response['a1'], response['ticks']


class TaskEnvironment:
    def __init__(self, instance):
        self._env = OptionSystemEnv(instance)
        self._env.decision_ticks = 1
        self.decisions = 0

    def observe(self):
        return policy_observation(self._env.boundary_observations(),
                                  [s.last_info for s in self._env.slots], self.decisions)

    def step(self, response):
        a0, a1, ticks = validate_action(response, self.observe())
        start = self._env.total_periods
        for _ in range(ticks):
            if self._env.termination is not None:
                break
            self._env.joint_step(a0, a1, record=True)
        self.decisions += 1
        self._env.skills_used = self.decisions
        return dict(start_tick=start, end_tick=self._env.total_periods,
                    requested_ticks=ticks, actions=[a0, a1])
