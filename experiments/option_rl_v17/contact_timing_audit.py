"""D003: read-only contact timing audit (v1).

Wraps the existing v13/v14 controller call sites without changing the control
loop: P0 Slot.prepare, P1/P2 _coordinate, P3 Slot.step, P4 env.step_control, plus
clearance.step.  Baseline and instrumented replays are compared for non-interference.
No training, no optimizer, no controller/reward/observation change.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coimagine.random_start import spec, layout
from coimagine.random_start import geometry as G
from coimagine.continuous2d.scenes import Pose
from experiments.option_rl_v14.controller import ACTIONS, make_env, mask
from experiments.option_rl_v14.scenes import geometry_audit
from experiments.option_rl_v15.render_episodes import instance_from_dict
from experiments.option_rl_v16.learnability import load_panel
from experiments.option_rl_v16.live import atomic_json

VERSION = 'd003-contact-timing-v1'
TOL = 1e-9
SOURCE_TOL = 1e-5
OLD_TOL = 1e-5
MAX_WORKERS = 4


# ---------------------------------------------------------------------------
# basic helpers
# ---------------------------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return repr(value)


def _pose_error(a, b):
    return float(max(abs(v) for v in a + b))


def _object_values(env):
    o = env.object_state()
    return [float(o.x), float(o.y), float(o.angle), float(o.vx), float(o.vy), float(o.omega)]


def _robot_values(env):
    out = []
    for r in env.robot_states():
        out.append([float(r.x), float(r.y), float(r.angle), float(r.vx), float(r.vy), float(r.omega)])
    return out


# ---------------------------------------------------------------------------
# instrumentation
# ---------------------------------------------------------------------------

def _true_edge_gap(env):
    obj = env.object_state()
    poly = G.box_polygon(obj.x, obj.y, obj.angle, spec.OBJECT_WIDTH / 2, spec.OBJECT_HEIGHT / 2)
    return {str(i): float(G.circle_polygon_gap((r.x, r.y), spec.ROBOT_RADIUS, poly))
            for i, r in enumerate(env.robot_states())}


class Recorder:
    def __init__(self, case_id, run_kind):
        self.case_id = case_id
        self.run_kind = run_kind
        self.skill_uids = {}
        self.uid_counter = 0
        self.ticks = []
        self.current = None
        self.previous_pair = None
        self.p3_skipped = False

    def skill_uid(self, skill):
        key = id(skill)
        if key not in self.skill_uids:
            self.skill_uids[key] = self.uid_counter
            self.uid_counter += 1
        return self.skill_uids[key]

    def flags(self, slot):
        skill = slot.skill
        action = slot.action
        ref = (slot.brake or slot.pause_brake or skill).reference.value
        return dict(
            action_label=action.label, action_index=int(getattr(action, 'index', -1)),
            action_type=int(getattr(action, 'type_index', -1)),
            action_kind=action.kind, action_effect_kind=getattr(action, 'effect_kind', None),
            action_speed=float(getattr(action, 'speed', 0.)),
            pending_label=slot.pending.label if slot.pending is not None else None,
            pending_type=int(getattr(slot.pending, 'type_index', -1)) if slot.pending is not None else None,
            pending_family=None,
            skill_uid=self.skill_uid(skill),
            skill_stage=getattr(skill, 'stage', None),
            ready=bool(getattr(skill, 'ready', False)),
            contact_ready=bool(getattr(skill, 'contact_ready', False)),
            release=bool(getattr(skill, 'release', False)),
            joint_active=bool(getattr(skill, 'joint_active', False)),
            joint_pair=bool(getattr(skill, 'joint_pair', False)),
            cooperative=bool(getattr(skill, 'cooperative', False)),
            done=bool(getattr(skill, 'done', False)),
            exit_reason=getattr(skill, 'exit_reason', None),
            pause_reason=slot.pause_reason,
            recovery_requested=bool(getattr(skill, 'recovery_requested', False)),
            recovery_ready=bool(getattr(skill, 'recovery_ready', False)),
            skill_time=float(getattr(skill, 'skill_time', 0.)),
            reference=[float(ref[0]), float(ref[1])] if ref is not None else None,
        )

    def obs(self, obs):
        own = obs.own
        patch = obs.patch
        scan = getattr(obs, 'scan', None)
        scan_info = None
        if scan is not None:
            scan_info = dict(origin=[float(scan.origin[0]), float(scan.origin[1])],
                             heading=float(scan.heading),
                             angles=[float(a) for a in scan.ray_angles],
                             distances=[float(d) for d in scan.distances],
                             valid=[bool(v) for v in scan.valid],
                             category=[int(c) for c in scan.category])
        return dict(
            time=float(getattr(obs, 'time', 0.)), period=int(getattr(obs, 'period', -1)),
            own=[float(own.x), float(own.y), float(own.angle), float(own.speed)],
            contacting_object=bool(obs.contacting_object),
            contacting_static=bool(obs.contacting_static),
            contacting_teammate=bool(obs.contacting_teammate),
            contact_point=list(obs.contact_point_robot) if obs.contact_point_robot is not None else None,
            patch_valid=bool(patch is not None and patch.valid),
            patch_distance=float(patch.distance) if patch is not None else None,
            patch_nearest=float(patch.nearest_distance) if patch is not None else None,
            normal=[float(patch.normal_world[0]), float(patch.normal_world[1])] if patch is not None else None,
            tangent=[float(patch.tangent_world[0]), float(patch.tangent_world[1])] if patch is not None else None,
            scan=scan_info,
        )

    def _tick(self, env):
        tick = int(env.total_periods)
        if self.current is None or self.current['tick'] != tick:
            if self.current is not None:
                self.ticks.append(self.current)
            self.current = dict(
                case=self.case_id, run=self.run_kind, tick=tick,
                t_start=tick * spec.CONTROL_DT, t_end=(tick + 1) * spec.CONTROL_DT,
                p0={}, p1={}, p2={}, p3={}, p4={}, clearance={}, p3_skipped=False,
            )
            self.p3_skipped = False
        return self.current

    def p0_entry(self, env, side, obs):
        tick = self._tick(env)
        flags = self.flags(env.slots[side])
        m = mask(obs)
        tick['p0'][str(side)] = dict(
            phase='entry', obs=self.obs(obs), flags=flags,
            local_mask_legal=[int(i) for i in np.flatnonzero(m)],
        )

    def p0_exit(self, env, side, obs):
        tick = self._tick(env)
        tick['p0'][str(side)]['post'] = self.flags(env.slots[side])

    def p1_entry(self, env, now, observations):
        tick = self._tick(env)
        tick['p1'] = dict(
            now=float(now),
            sides={str(i): self.flags(env.slots[i]) for i in (0, 1)},
            pair=[self.skill_uid(env.slots[i].skill) for i in (0, 1)],
            recovery=dict(started=env.recovery_started is not None, time=float(env.recovery_time),
                          attempts=int(env.recovery_attempts)),
            clearance=dict(started=bool(env.clearance.started), finished=bool(env.clearance.finished),
                           failed=bool(env.clearance.failed)),
            pair_changed=self.previous_pair is not None and
            [self.skill_uid(env.slots[i].skill) for i in (0, 1)] != self.previous_pair,
        )

    def p2_exit(self, env):
        tick = self._tick(env)
        tick['p2'] = dict(
            sides={str(i): self.flags(env.slots[i]) for i in (0, 1)},
            pair=[self.skill_uid(env.slots[i].skill) for i in (0, 1)],
            recovery=dict(started=env.recovery_started is not None, time=float(env.recovery_time),
                          attempts=int(env.recovery_attempts)),
            clearance=dict(started=bool(env.clearance.started), finished=bool(env.clearance.finished),
                           failed=bool(env.clearance.failed)),
        )
        self.previous_pair = [self.skill_uid(env.slots[i].skill) for i in (0, 1)]

    def clearance_step(self, env, result):
        tick = self._tick(env)
        tick['clearance'] = dict(
            started=bool(env.clearance.started), finished=bool(env.clearance.finished),
            failed=bool(env.clearance.failed), active=result is not None,
        )
        self.p3_skipped = result is not None

    def p3_entry(self, env, side, obs):
        tick = self._tick(env)
        skill = env.slots[side].skill
        tick['p3'][str(side)] = dict(
            entry=self.flags(env.slots[side]), obs=self.obs(obs),
            debug=_jsonable(dict(
                contact_target=getattr(skill, 'contact_target', None),
                reference_centre=getattr(skill, 'reference_centre', None),
                skill_stage=getattr(skill, 'stage', None),
            )),
        )

    def p3_exit(self, env, side, obs, result):
        tick = self._tick(env)
        force, info = result
        tick['p3'][str(side)].update(
            exit=self.flags(env.slots[side]), force=[float(force[0]), float(force[1])],
            info=_jsonable({k: v for k, v in (info or {}).items() if k in (
                'stage', 'exit_reason', 'done', 'ready', 'contact_ready', 'joint_active', 'gap',
                'centre_gap', 'gap_target', 'tangent_error', 'align_error_deg', 'desired_heading',
                'reference_velocity', 'desired_velocity', 'force_norm', 'force_saturated',
                'progress_guard', 'peer_clearance', 'requested_kind', 'declared_direction',
                'patch_valid', 'normal_valid', 'contact_target', 'reference_centre', 'face_midpoint',
                'face_normal', 'heading_error', 'lateral_error', 'rotation_feedback', 'n', 't',
                'heading', 'omega_est', 'anchor', 'anchor_side', 'endpoint_status',
                'point_offset', 'role_offset', 'q_rot', 'diagnostic_tangent_offset',
                'target_before_visibility', 'target_before_reservation', 'target_final',
                'e_t', 'v_t', 'first_clip', 'c_v_t', 'lateral', 'radial',
                'raw_before_avoid', 'raw_after_avoid', 'raw_before_norm_cap',
                'raw_after_norm_cap', 'v_ref', 'F_pre_norm_cap', 'F_post_norm_cap',
                'effect', 'hold',
                'option_timeout_seconds', 'safety_trigger')}),
        )

    def p4(self, env, forces, headings, before, after, result, row, entry_flags=None):
        tick = self._tick(env)
        _, d = result
        tick['p4'] = dict(
            entry_flags=_jsonable(entry_flags or {}),
            forces=[[float(v) for v in forces[0]], [float(v) for v in forces[1]]],
            headings=[[float(headings[0][0]), float(headings[0][1])],
                      [float(headings[1][0]), float(headings[1][1])]],
            before=before, after=after,
            finite=bool(d.get('finite', True)), penetration=float(d.get('penetration', 0.)),
            in_bounds=bool(d.get('in_bounds', True)),
            contacts=_jsonable(env.env.backend.contacts()) if hasattr(env.env, 'backend') else 'unavailable',
            gate_pre=_jsonable(getattr(env, 'gate_pre', None)),
            gate_post=_jsonable(getattr(env, 'gate_post', None)),
            true_edge_gap=_true_edge_gap(env),
            success_conditions=_jsonable(env.success_conditions()),
            true_goal_distance=float(env.object_distance()),
            period=int(env.total_periods),
        )
        tick['p3_skipped'] = self.p3_skipped
        tick['termination'] = env.termination
        self.ticks.append(tick)
        self.current = None

    def finish(self, env, success, termination, reward):
        if self.current is not None:
            self.ticks.append(self.current)
            self.current = None
        return dict(case=self.case_id, run=self.run_kind, ticks=self.ticks,
                    success=bool(success), termination=termination, reward=float(reward),
                    periods=int(env.total_periods))


class Instrumentation:
    def __init__(self, env, recorder):
        self.env = env
        self.rec = recorder
        self.original = []

    def install(self):
        env = self.env
        for i, slot in enumerate(env.slots):
            orig_prepare = slot.prepare
            orig_step = slot.step

            def prepare(obs, i=i, orig=orig_prepare):
                self.rec.p0_entry(env, i, obs)
                result = orig(obs)
                self.rec.p0_exit(env, i, obs)
                return result

            def step(obs, i=i, orig=orig_step):
                self.rec.p3_entry(env, i, obs)
                result = orig(obs)
                self.rec.p3_exit(env, i, obs, result)
                return result

            self.original.append((slot, 'prepare', slot.prepare))
            self.original.append((slot, 'step', slot.step))
            slot.prepare = prepare
            slot.step = step
        orig_coord = env._coordinate
        orig_clear = env.clearance.step
        orig_step_control = env.env.step_control

        def coord(now, observations=None):
            self.rec.p1_entry(env, now, observations)
            result = orig_coord(now, observations)
            self.rec.p2_exit(env)
            return result

        def clear(observations, dt):
            result = orig_clear(observations, dt)
            self.rec.clearance_step(env, result)
            return result

        def step_control(forces, headings, **kwargs):
            before = dict(object=_object_values(env), robots=_robot_values(env))
            entry_flags = {str(i): self.rec.flags(env.slots[i]) for i in (0, 1)}
            result = orig_step_control(forces, headings, **kwargs)
            after = dict(object=_object_values(env), robots=_robot_values(env))
            row = env.rows[-1] if env.rows else None
            self.rec.p4(env, forces, headings, before, after, result, row, entry_flags=entry_flags)
            return result

        self.original += [(env, '_coordinate', env._coordinate),
                          (env.clearance, 'step', env.clearance.step),
                          (env.env, 'step_control', env.env.step_control)]
        env._coordinate = coord
        env.clearance.step = clear
        env.env.step_control = step_control
        return self

    def restore(self):
        for obj, name, value in reversed(self.original):
            setattr(obj, name, value)
        self.original = []


# ---------------------------------------------------------------------------
# replay
# ---------------------------------------------------------------------------

def _run_commands(instance, commands, *, record, instrumented, recorder=None):
    env = make_env(instance, context='extended')
    inst = None
    if instrumented:
        inst = Instrumentation(env, recorder).install()
    try:
        commands = list(commands)
        for a0, a1 in commands:
            env.joint_step(int(a0), int(a1), record=record)
            if env.termination is not None:
                break
        rows = list(env.rows)
        success = bool(env.success)
        termination = env.termination
        reward = float(env.reward_total)
        periods = int(env.total_periods)
        ticks = recorder.finish(env, success, termination, reward) if instrumented else None
    finally:
        if inst is not None:
            inst.restore()
    return dict(env=env, rows=rows, success=success, termination=termination,
                reward=reward, periods=periods, ticks=ticks)


def _row_values(row):
    o = row['true']['object'][:3]
    r = [rr[:3] for rr in row['true']['robots']]
    forces = [[float(v) for v in f] for f in row.get('forces', [])]
    return o, r, forces


def _compare_rows(a, b, tol=TOL):
    if len(a) != len(b):
        return None, f'row count {len(a)} != {len(b)}'
    max_err = 0.0
    for i, (ra, rb) in enumerate(zip(a, b)):
        oa, rra, fa = _row_values(ra)
        ob, rrb, fb = _row_values(rb)
        values = [abs(oa[k] - ob[k]) for k in (0, 1)] + [abs(_wrap(oa[2] - ob[2]))]
        for j in range(2):
            values += [abs(rra[j][k] - rrb[j][k]) for k in (0, 1)] + [abs(_wrap(rra[j][2] - rrb[j][2]))]
        for f0, f1 in zip(fa, fb):
            values += [abs(f0[0] - f1[0]), abs(f0[1] - f1[1])]
        max_err = max(max_err, max(values) if values else 0.)
        if max_err > tol:
            return float(max_err), f'row {i} exceeds {tol}'
    return float(max_err), None


def _source_frame(npz, t_start):
    times = np.asarray(npz['trace_t'], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - t_start)))
    return idx, dict(object=np.asarray(npz['trace_object'][idx], dtype=np.float64),
                     robots=np.asarray(npz['trace_robots'][idx], dtype=np.float64), time=float(times[idx]))


def _pose_error_to_frame(env, frame):
    obj = env.object_state()
    robots = env.robot_states()
    values = [obj.x - frame['object'][0], obj.y - frame['object'][1], _wrap(obj.angle - frame['object'][2])]
    for i in (0, 1):
        values += [robots[i].x - frame['robots'][i][0], robots[i].y - frame['robots'][i][1],
                   _wrap(robots[i].angle - frame['robots'][i][2])]
    return float(max(abs(v) for v in values))


def _witness_instance(log):
    data = json.loads(Path(log).read_text())
    return instance_from_dict(data['instance']), data


def _failure_instance(cfg, sample, panel):
    train, _ = load_panel(panel)
    by_name = {i.name: i for i in train}
    return by_name[sample['root']]


def run_witness(cfg, sample, kind):
    instance, data = _witness_instance(sample['log'])
    commands = [tuple(c) for c in data['commands']]
    base = _run_commands(instance, commands, record=True, instrumented=False)
    rec = Recorder(sample['id'], kind)
    ins = _run_commands(instance, commands, record=True, instrumented=True, recorder=rec)
    err, why = _compare_rows(base['rows'], ins['rows'], TOL)
    # old float32 log comparison
    old = data.get('rows', [])
    old_err, old_why = _compare_rows(old, base['rows'], OLD_TOL) if old else (None, 'old rows missing')
    old_success = bool(data.get('metric', {}).get('success'))
    old_seconds = float(data.get('metric', {}).get('seconds', 0.))
    old_rows = len(old)
    result = dict(case=sample['id'], kind='witness', root=sample['root'],
                  source=dict(log=sample['log'], commands=len(commands), old_success=old_success,
                              old_seconds=old_seconds, old_rows=old_rows),
                  baseline=dict(success=base['success'], termination=base['termination'],
                                periods=base['periods'], reward=base['reward'],
                                rows=len(base['rows'])),
                  instrumented=dict(success=ins['success'], termination=ins['termination'],
                                    periods=ins['periods'], reward=ins['reward'],
                                    rows=len(ins['rows'])),
                  noninterference_error=err, noninterference_reason=why,
                  old_error=old_err, old_reason=old_why,
                  ticks=ins['ticks'], rows=base['rows'],
                  status='ok' if (err is not None and err <= TOL and old_err is not None
                                  and old_err <= OLD_TOL and base['success'] and ins['success']
                                  and base['periods'] == ins['periods'] and base['success'] and ins['success']
                                  and old_success and abs(base['reward'] - ins['reward']) <= TOL
                                  and old_rows == len(base['rows'])) else 'reproduction_failed')
    return result


def run_failure(cfg, sample, kind, d001, d002, panel):
    episode_id = f"joint_types__{sample['root']}__random__{sample['seed']}"
    ep = json.loads((d001 / 'episodes' / f'{episode_id}.json').read_text())
    npz = np.load(d001 / 'episodes' / f'{episode_id}.npz', allow_pickle=False)
    instance = _failure_instance(cfg, sample, panel)
    t_start = float(sample['t_start'])
    di = int(sample['decision_index'])
    prefix = [tuple((int(s['a0']), int(s['a1']))) for s in ep['execution_samples']
              if float(s['time']) <= t_start + 1e-9]
    if 'source_action' in sample:
        source_action = (int(sample['source_action'][0]), int(sample['source_action'][1]))
    else:
        source_action = (int(sample['a0']), int(sample['a1']))
    hold_branch = json.loads(Path(sample['d002_hold_branch']).read_text())
    d002_rows = hold_branch.get('rows', [])[:64]

    def run(instrumented):
        env = make_env(instance, context='extended')
        inst = None
        rec = None
        try:
            # Both prefixes use the original env and record=False.  Wrappers are
            # installed only for the target 24-step window.  A PairGateEnv is
            # disabled during the prefix so source reconstruction is unchanged.
            if hasattr(env, 'set_gate_enabled'):
                env.set_gate_enabled(False)
            for a0, a1 in prefix:
                env.joint_step(a0, a1, record=False)
                if env.termination is not None:
                    raise RuntimeError('termination during prefix')
            obs, m0, m1 = _observe(env)
            raw_err = float(np.max(np.abs(obs.astype(np.float64) - np.asarray(npz['raw_obs'][di], dtype=np.float64))))
            mask_err = int(np.sum(m0 != npz['mask0'][di]) + np.sum(m1 != npz['mask1'][di]))
            frame_idx, frame = _source_frame(npz, t_start)
            pose_err = _pose_error_to_frame(env, frame)
            expected_periods = int(round(t_start / spec.CONTROL_DT))
            source_ok = raw_err <= SOURCE_TOL and mask_err == 0 and pose_err <= SOURCE_TOL \
                and int(env.total_periods) == expected_periods
            if hasattr(env, 'set_gate_enabled'):
                env.set_gate_enabled(True)
            if instrumented:
                rec = Recorder(sample['id'], kind)
                inst = Instrumentation(env, rec).install()
            for _ in range(24):
                env.joint_step(source_action[0], source_action[1], record=True)
                if env.termination is not None:
                    break
            rows = list(env.rows)
            return dict(env=env, rows=rows, success=bool(env.success), termination=env.termination,
                        reward=float(env.reward_total), periods=int(env.total_periods),
                        ticks=rec.finish(env, bool(env.success), env.termination, float(env.reward_total)) if instrumented else None,
                        source=dict(raw_error=raw_err, mask_error=mask_err, pose_error=pose_err,
                                    expected_periods=expected_periods, total_periods=int(env.total_periods),
                                    frame_time=frame['time'], source_frame_index=frame_idx, ok=bool(source_ok)),
                        prefix=len(prefix))
        finally:
            if inst is not None:
                inst.restore()

    base = run(False)
    ins = run(True)
    err, why = _compare_rows(base['rows'], ins['rows'], TOL)
    old_err, old_why = _compare_rows(d002_rows, ins['rows'][:64], OLD_TOL)
    result = dict(case=sample['id'], kind='failure', root=sample['root'], seed=sample['seed'],
                  decision_index=di, t_start=t_start, source_action=list(source_action),
                  role=sample.get('role'), source=ins['source'], baseline_source=base['source'],
                  baseline=dict(success=base['success'], termination=base['termination'],
                                periods=base['periods'], reward=base['reward'], rows=len(base['rows'])),
                  instrumented=dict(success=ins['success'], termination=ins['termination'],
                                    periods=ins['periods'], reward=ins['reward'], rows=len(ins['rows'])),
                  noninterference_error=err, noninterference_reason=why,
                  d002_overlap_error=old_err, d002_overlap_reason=old_why,
                  ticks=ins['ticks'], rows=base['rows'],
                  status='ok' if (err is not None and err <= TOL and ins['source']['ok'] and base['source']['ok']
                                  and old_err is not None and old_err <= OLD_TOL
                                  and base['periods'] == ins['periods']
                                  and abs(base['reward'] - ins['reward']) <= TOL) else 'reproduction_failed')
    return result


def _observe(env):
    from experiments.option_rl_v14.interface import observe
    return observe(env)


# ---------------------------------------------------------------------------
# event / label analysis
# ---------------------------------------------------------------------------

def _side_stage(tick, stage, side):
    node = tick.get(stage, {}) or {}
    if str(side) in node:
        return node[str(side)]
    return (node.get('sides', {}) or {}).get(str(side), {})


def _contact(tick, side):
    return bool(tick.get('p0', {}).get(str(side), {}).get('obs', {}).get('contacting_object', False))


def _ready(tick, side):
    p3 = tick.get('p3', {}).get(str(side))
    if p3 and 'exit' in p3:
        return bool(p3['exit'].get('contact_ready', False))
    p4 = tick.get('p4', {}).get('entry_flags', {}).get(str(side))
    if p4:
        return bool(p4.get('contact_ready', False))
    return bool(_side_stage(tick, 'p2', side).get('contact_ready', False))


def _active(tick, side):
    return bool(_side_stage(tick, 'p2', side).get('joint_active', False))


def _pre_active(tick, side):
    return bool(_side_stage(tick, 'p1', side).get('joint_active', False))


def _action(tick, side):
    p3 = tick.get('p3', {}).get(str(side))
    if p3 and 'exit' in p3:
        return p3['exit']
    p4 = tick.get('p4', {}).get('entry_flags', {}).get(str(side))
    if p4:
        return p4
    return _side_stage(tick, 'p2', side)


def _p3_info(tick, side):
    return (tick.get('p3', {}).get(str(side), {}) or {}).get('info') or {}


def _true_object(tick):
    return (tick.get('p4', {}) or {}).get('after', {}).get('object')


def _true_robots(tick):
    return (tick.get('p4', {}) or {}).get('after', {}).get('robots')


def _row_tick_distance(tick):
    return (tick.get('p4', {}) or {}).get('true_goal_distance')


def derive_events(ticks):
    n = len(ticks)
    both_contact = [bool(_contact(t, 0) and _contact(t, 1)) for t in ticks]
    both_ready = [bool(_ready(t, 0) and _ready(t, 1)) for t in ticks]
    both_active = [bool(_active(t, 0) and _active(t, 1)) for t in ticks]
    any_active = [bool(_active(t, 0) or _active(t, 1)) for t in ticks]
    stable_runs = []
    run_start = None
    for i, value in enumerate(both_contact):
        if value and run_start is None:
            run_start = i
        if not value and run_start is not None:
            stable_runs.append((run_start, i - 1))
            run_start = None
    if run_start is not None:
        stable_runs.append((run_start, n - 1))
    stable = max(stable_runs, key=lambda x: x[1] - x[0]) if stable_runs else None
    stable_ticks = (stable[1] - stable[0] + 1) if stable else 0
    onsets = [i for i in range(1, n) if both_contact[i] and not both_contact[i - 1]]
    losses = [i for i in range(1, n) if not both_contact[i] and both_contact[i - 1]]
    first_active = next((i for i, v in enumerate(both_active) if v), None)
    ready_updates = []
    for i, tick in enumerate(ticks):
        for side in (0, 1):
            p3 = tick.get('p3', {}).get(str(side))
            if p3 and 'entry' in p3 and 'exit' in p3:
                if not p3['entry'].get('contact_ready') and p3['exit'].get('contact_ready'):
                    ready_updates.append((i, side))
    activation_ticks = [i for i, tick in enumerate(ticks)
                        if any((not _pre_active(tick, s)) and _active(tick, s) for s in (0, 1))]
    initial_both = bool(both_contact and both_contact[0])
    last_both = bool(both_contact and both_contact[-1])
    right_censored = bool(last_both and not losses)
    left_censored = bool(initial_both and not onsets)
    return dict(
        n=n, both_contact=both_contact, both_ready=both_ready, both_active=both_active,
        any_active=any_active, stable_runs=stable_runs, stable=stable, stable_ticks=stable_ticks,
        onsets=onsets, losses=losses, first_active=first_active,
        ready_updates=ready_updates, activation_ticks=activation_ticks,
        initial_both=initial_both, last_both=last_both,
        right_censored=right_censored, left_censored=left_censored,
        both_contact_ticks=sum(both_contact), both_ready_ticks=sum(both_ready),
        both_active_ticks=sum(both_active), any_active_ticks=sum(any_active),
        contact_seconds=sum(both_contact) * spec.CONTROL_DT,
        ready_seconds=sum(both_ready) * spec.CONTROL_DT,
        active_seconds=sum(both_active) * spec.CONTROL_DT,
        stable_seconds=stable_ticks * spec.CONTROL_DT,
    )


def _goal_vector(case_goal, object_values):
    if not object_values or case_goal is None:
        return None
    dx = float(case_goal[0]) - float(object_values[0])
    dy = float(case_goal[1]) - float(object_values[1])
    norm = math.hypot(dx, dy)
    return (dx / norm, dy / norm) if norm > 1e-9 else None


def _push_direction(action):
    # Use the actual committed action flags.  BRAKE_HOLD/ROTATE/REPOSITION
    # must not be classified as a push merely because ACTIONS[type] exists.
    if action.get('action_kind') != 'push' or action.get('action_effect_kind') != 'translate':
        return None
    try:
        t = int(action.get('action_type'))
        a = ACTIONS[t]
        return (float(a.effect_direction[0]), float(a.effect_direction[1]))
    except Exception:
        return None


def _blockers(tick, side):
    info = _p3_info(tick, side)
    flags = _action(tick, side)
    out = set()
    if flags.get('pending_label') is not None:
        out.add('pending')
    if flags.get('pause_reason'):
        out.add('pause:' + str(flags['pause_reason']))
    if flags.get('done') or flags.get('exit_reason') not in (None, 'continuing'):
        out.add('local_failure')
    if flags.get('recovery_requested'):
        out.add('recovery')
    if flags.get('contact_ready'):
        out.add('contact_ready')
    if not flags.get('ready'):
        out.add('ready_false')
    if not flags.get('patch_valid', True):
        out.add('face_invalid')
    gap = info.get('gap')
    gap_target = info.get('gap_target')
    if gap is None:
        out.add('gap_unknown')
    else:
        limit = float(gap_target) if gap_target is not None else 0.03
        if float(gap) > limit + 0.01:
            out.add('gap_large')
    tangent = info.get('tangent_error')
    if tangent is None:
        out.add('tangent_unknown')
    elif abs(float(tangent)) > 0.025:
        out.add('tangent_large')
    align = info.get('align_error_deg')
    if align is None:
        out.add('alignment_unknown')
    elif abs(float(align)) > 20.0:
        out.add('alignment_large')
    if info.get('normal_valid') is False:
        out.add('normal_invalid')
    if tick.get('clearance', {}).get('active'):
        out.add('clearance_active')
    return sorted(out)


def _goal_opposed(tick, case_goal):
    if not case_goal:
        return None
    object_values = _true_object(tick)
    unit = _goal_vector(case_goal, object_values)
    if unit is None:
        return None
    sides = []
    for side in (0, 1):
        action = _action(tick, side)
        direction = _push_direction(action)
        if direction is None:
            continue
        dot = direction[0] * unit[0] + direction[1] * unit[1]
        if dot <= -0.25:
            sides.append(dict(side=side, dot=float(dot), action=action.get('action_label'),
                              action_index=action.get('action_index'),
                              face_normal=_side_stage(tick, 'p0', side).get('obs', {}).get('normal'),
                              local_mask_legal=_side_stage(tick, 'p0', side).get('local_mask_legal')))
    return sides or None


def _contact_run_labels(events, ticks, case_goal):
    labels = []
    for start, end in events['stable_runs']:
        if end - start + 1 >= 10:
            continue
        # A short contact run that was lost (not the window-end run) is a candidate.
        if end < events['n'] - 1:
            loss_tick = end + 1
            changed = False
            brake = False
            safety = False
            for i in range(max(start, loss_tick - 8), min(events['n'], loss_tick + 1)):
                for side in (0, 1):
                    action = _action(ticks[i], side)
                    brake |= action.get('action_kind') == 'wait'
                    safety |= bool((_p3_info(ticks[i], side) or {}).get('safety_trigger'))
                prev = _action(ticks[start], 0).get('action_label')
                now = _action(ticks[i], 0).get('action_label')
                changed |= now != prev
            labels.append(dict(kind='contact_not_maintained', start=start, end=end,
                               loss_tick=loss_tick, duration_ticks=end - start + 1,
                               changed_command=bool(changed), brake=bool(brake),
                               safety=bool(safety), goal_opposed=bool(_goal_opposed(ticks[start], case_goal))))
    return labels


def _coordination_labels(events, ticks):
    labels = []
    for i, tick in enumerate(ticks):
        p1 = tick.get('p1', {})
        p2 = tick.get('p2', {})
        s0 = p1.get('sides', {}).get('0', {})
        s1 = p1.get('sides', {}).get('1', {})
        if not (s0.get('contact_ready') and s1.get('contact_ready')):
            continue
        if s0.get('done') or s1.get('done') or s0.get('pending_label') or s1.get('pending_label'):
            continue
        if p1.get('pair_changed') or p1.get('recovery', {}).get('started'):
            continue
        a0 = p2.get('sides', {}).get('0', {}).get('joint_active')
        a1 = p2.get('sides', {}).get('1', {}).get('joint_active')
        if not (a0 and a1):
            labels.append(dict(kind='coordination_contract_violation_candidate', tick=i,
                               p1_contact_ready=[True, True], p2_joint_active=[bool(a0), bool(a1)],
                               pair=list(p1.get('pair', [])), recovery=p1.get('recovery'),
                               note='P1 both contact_ready, pair stable, no pending/done/recovery; P2 not both active'))
    return labels


def analyse_case(case, result, goal):
    ticks = result.get('ticks', {}).get('ticks', []) if result.get('ticks') else []
    events = derive_events(ticks)
    labels = []
    # Goal-opposed push command, at least 10 consecutive ticks.
    runs = []
    current = None
    for i, tick in enumerate(ticks):
        opposed = _goal_opposed(tick, goal)
        if opposed:
            if current is None:
                current = dict(start=i, end=i, sides=opposed)
            else:
                current['end'] = i
                current['sides'] = opposed
        else:
            if current is not None and current['end'] - current['start'] + 1 >= 10:
                runs.append(current)
            current = None
    if current is not None and current['end'] - current['start'] + 1 >= 10:
        runs.append(current)
    for run in runs:
        labels.append(dict(kind='goal_opposed_command', start=run['start'], end=run['end'],
                           duration_ticks=run['end'] - run['start'] + 1, sides=run['sides']))
    # Preparation blockers on no-contact segments of at least 20 ticks.
    blockers = []
    start = None
    for i, tick in enumerate(ticks):
        ok = _contact(tick, 0) and _contact(tick, 1)
        if not ok and start is None:
            start = i
        if ok and start is not None:
            if i - start >= 20:
                blockers.append(dict(start=start, end=i - 1, duration_ticks=i - start,
                                     blockers=sorted({b for j in range(start, i) for b in _blockers(ticks[j], 0)})))
                labels.append(dict(kind='preparation_not_ready', start=start, end=i - 1,
                                   duration_ticks=i - start,
                                   blockers=blockers[-1]['blockers']))
            start = None
    if start is not None and events['n'] - start >= 20:
        blockers.append(dict(start=start, end=events['n'] - 1, duration_ticks=events['n'] - start,
                             blockers=sorted({b for j in range(start, events['n']) for b in _blockers(ticks[j], 0)})))
        labels.append(dict(kind='preparation_not_ready', start=start, end=events['n'] - 1,
                           duration_ticks=events['n'] - start, blockers=blockers[-1]['blockers']))
    # Approach-response windows: 1 s of no pending/pause/clearance/failure, reference inward,
    # gap >= 2 cm and closing by <5 mm, stable face normal.
    for start in range(0, max(0, events['n'] - 20) + 1):
        window = ticks[start:start + 20]
        if any(not (_contact(t, 0) and _contact(t, 1)) for t in window):
            continue
        if any(t.get('clearance', {}).get('active') for t in window):
            continue
        if any(_action(t, s).get('pending_label') or _action(t, s).get('pause_reason') or
               _action(t, s).get('done') or _action(t, s).get('exit_reason') not in (None, 'continuing')
               for t in window for s in (0, 1)):
            continue
        norms = [_side_stage(t, 'p0', 0).get('obs', {}).get('normal') for t in window]
        if not norms or any(v is None for v in norms):
            continue
        spread = max(math.degrees(abs(_wrap(math.atan2(a[1], a[0]) - math.atan2(b[1], b[0]))))
                     for a in norms for b in norms)
        if spread > 10.0:
            continue
        refs = []
        gaps = []
        for t in window:
            info = _p3_info(t, 0)
            rv = info.get('reference_velocity')
            normal = _side_stage(t, 'p0', 0).get('obs', {}).get('normal')
            if rv is None or normal is None:
                refs = []
                break
            refs.append(float(rv[0]) * float(normal[0]) + float(rv[1]) * float(normal[1]))
            true_gap = ((t.get('p4', {}) or {}).get('true_edge_gap') or {}).get('0')
            gaps.append(float(true_gap) if true_gap is not None else None)
        if not refs or not gaps or any(g is None for g in gaps):
            continue
        if min(refs) >= 0.02 and min(gaps) >= 0.02 and (gaps[0] - gaps[-1]) < 0.005:
            labels.append(dict(kind='approach_response_suspect', start=start, end=start + 19,
                               normal_spread_deg=float(spread), min_reference_inward=float(min(refs)),
                               gap_start=float(gaps[0]), gap_end=float(gaps[-1]),
                               gap_change=float(gaps[-1] - gaps[0])))
            break
    labels.extend(_contact_run_labels(events, ticks, goal))
    labels.extend(_coordination_labels(events, ticks))
    if events['right_censored']:
        labels.append(dict(kind='right_censored', tick=events['n'] - 1,
                           note='both contact at the final tick; no following tick to decide activation'))
    if events['left_censored']:
        labels.append(dict(kind='left_censored', tick=0,
                           note='both contact at the first tick; onset not observed'))
    labels = [l for l in labels if l.get('kind') in (
        'goal_opposed_command', 'preparation_not_ready', 'approach_response_suspect',
        'contact_not_maintained', 'coordination_contract_violation_candidate',
        'right_censored', 'left_censored')]
    labels = [l for l in labels if not (l['kind'] == 'right_censored' and not events['right_censored'])]
    metrics = dict(
        case=case['id'], role=case.get('role'), kind=case['kind'], root=case['root'],
        success=bool(result.get('instrumented', {}).get('success')),
        termination=result.get('instrumented', {}).get('termination'),
        periods=int(result.get('instrumented', {}).get('periods', 0)),
        baseline_success=bool(result.get('baseline', {}).get('success')),
        status=result.get('status'), source_ok=bool((result.get('source') or {}).get('ok', True)),
        noninterference_error=result.get('noninterference_error'),
        d002_overlap_error=result.get('d002_overlap_error'),
        old_error=result.get('old_error'),
        both_contact_ticks=events['both_contact_ticks'], both_ready_ticks=events['both_ready_ticks'],
        both_active_ticks=events['both_active_ticks'], any_active_ticks=events['any_active_ticks'],
        stable_ticks=events['stable_ticks'], stable_seconds=events['stable_seconds'],
        first_active=events['first_active'], activation_ticks=events['activation_ticks'],
        onsets=events['onsets'], losses=events['losses'], right_censored=events['right_censored'],
        left_censored=events['left_censored'], labels=labels,
        n_ticks=events['n'],
        tick_offset=int(ticks[0]['tick']) if ticks else None,
        onsets_global=[int(ticks[i]['tick']) for i in events['onsets']],
        losses_global=[int(ticks[i]['tick']) for i in events['losses']],
        activation_global=[int(ticks[i]['tick']) for i in events['activation_ticks']],
    )
    return dict(events=events, labels=labels, blockers=blockers, metrics=metrics)


def _minimal_candidate(all_metrics):
    kinds = [l['kind'] for m in all_metrics for l in m['labels']]
    for kind, text in [
        ('coordination_contract_violation_candidate', 'coordination_contract_timing'),
        ('approach_response_suspect', 'approach_or_reference_control'),
        ('contact_not_maintained', 'contact_maintenance_or_recovery'),
        ('goal_opposed_command', 'policy_selection_alignment'),
        ('preparation_not_ready', 'preparation_condition'),
    ]:
        if kind in kinds:
            return dict(kind=text, evidence=kind, count=kinds.count(kind))
    return dict(kind='insufficient_evidence', evidence=None, count=0)


# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------

def _write_csv(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = list(rows[0].keys())
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v
                             for k, v in row.items()})


def _write_raw(path, cases):
    with Path(path).open('w') as f:
        for case in cases:
            for tick in (case.get('ticks') or {}).get('ticks', []):
                f.write(json.dumps(tick, ensure_ascii=False, allow_nan=False) + '\n')


def _write_baseline(out, cases):
    folder = out / 'baseline'
    folder.mkdir(exist_ok=True)
    for case in cases:
        if case.get('rows'):
            atomic_json(folder / f"{case['case']}.json", dict(case=case['case'], rows=case['rows']))


def _timelines(out, cases, analyses):
    parts = ['<!doctype html><html><meta charset="utf-8"><title>D003 timelines</title>',
             '<style>body{font-family:system-ui;margin:20px}table{border-collapse:collapse;margin:10px 0}'
             'td,th{border:1px solid #ccc;padding:2px 5px;font:12px monospace}h2{margin-top:28px}</style>']
    for case, ana in zip(cases, analyses):
        m = ana['metrics']
        parts.append(f"<h2>{case['id']} · {case['role']} · success={m['success']} · "
                     f"both contact={m['both_contact_ticks']} · active={m['both_active_ticks']} · "
                     f"stable={m['stable_ticks']} · labels={[l['kind'] for l in m['labels']]}</h2>")
        parts.append('<table><tr><th>tick</th><th>t</th><th>contact0</th><th>contact1</th>'
                     '<th>ready0</th><th>ready1</th><th>active0</th><th>active1</th>'
                     '<th>a0</th><th>a1</th><th>gap0</th><th>gap1</th><th>obj dist</th></tr>')
        for t in (case.get('ticks') or {}).get('ticks', []):
            gap0 = (_p3_info(t, 0) or {}).get('gap')
            gap1 = (_p3_info(t, 1) or {}).get('gap')
            parts.append(
                f"<tr><td>{t['tick']}</td><td>{t['t_start']:.2f}</td>"
                f"<td>{int(_contact(t,0))}</td><td>{int(_contact(t,1))}</td>"
                f"<td>{int(_ready(t,0))}</td><td>{int(_ready(t,1))}</td>"
                f"<td>{int(_active(t,0))}</td><td>{int(_active(t,1))}</td>"
                f"<td>{_action(t,0).get('action_label')}</td><td>{_action(t,1).get('action_label')}</td>"
                f"<td>{'—' if gap0 is None else round(float(gap0),4)}</td>"
                f"<td>{'—' if gap1 is None else round(float(gap1),4)}</td>"
                f"<td>{'—' if _row_tick_distance(t) is None else round(float(_row_tick_distance(t)),4)}</td></tr>")
        parts.append('</table>')
    (out / 'timelines.html').write_text('\n'.join(parts))


def _replay(out, cases, goals):
    from experiments.option_rl_v45.render_episodes import TEMPLATE
    episodes = []
    for case in cases:
        ticks = (case.get('ticks') or {}).get('ticks', [])
        frames = []
        decisions = []
        for t in ticks:
            obj = _true_object(t)
            robots = _true_robots(t)
            if obj is None or robots is None:
                continue
            frames.append(dict(t=float(t['t_start']),
                               o=[float(v) for v in obj[:3]],
                               r=[[float(v) for v in r[:3]] for r in robots],
                               ct=[_contact(t, 0), _contact(t, 1)],
                               st=[str(_action(t, 0).get('skill_stage')),
                                   str(_action(t, 1).get('skill_stage'))],
                               fo=[float(t.get('p3', {}).get('0', {}).get('force', [0, 0])[0]),
                                   float(t.get('p3', {}).get('1', {}).get('force', [0, 0])[0])]))
            if t['tick'] % 4 == 0:
                decisions.append(dict(t=float(t['t_start']),
                                      a0=str(_action(t, 0).get('action_label')),
                                      a1=str(_action(t, 1).get('action_label'))))
        if not frames:
            continue
        goal = goals.get(case['case'])
        episodes.append(dict(name=case['case'], goal=list(goal) if goal else None,
                             radius=float(spec.GOAL_RADIUS),
                             object_size=[spec.OBJECT_WIDTH, spec.OBJECT_HEIGHT],
                             start_goal_distance=0.,
                             frames=frames, decisions=decisions,
                             success=bool(case.get('instrumented', {}).get('success')),
                             termination=case.get('instrumented', {}).get('termination'),
                             summary=dict(role=case.get('role'), labels=[l['kind'] for l in case.get('labels', [])])))
    payload = dict(checkpoint='D003 contact timing audit | instrumented true-state replay',
                   arena=[5.0, 4.0], episodes=episodes)
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    html = html.replace('ARENA[1]', 'DATA.arena[1]').replace('ARENA[0]', 'DATA.arena[0]')
    (out / 'replay.html').write_text(html)
    (out / 'replay_payload.json').write_text(json.dumps(payload, indent=1))


# ---------------------------------------------------------------------------
# worker / main
# ---------------------------------------------------------------------------

def _case_instance_goal(cfg, case, panel):
    if case['kind'] == 'witness':
        inst = instance_from_dict(json.loads(Path(case['log']).read_text())['instance'])
    else:
        train, _ = load_panel(panel)
        inst = {i.name: i for i in train}[case['root']]
    return inst


def _run_case(job):
    cfg = job['config']
    case = job['case']
    if case['kind'] == 'witness':
        return run_witness(cfg, case, 'witness')
    return run_failure(cfg, case, 'failure', Path(job['d001']), Path(job['d002']), Path(job['panel']))


def _source_snapshot(out, cfg):
    files = [Path(__file__), ROOT / 'experiments/option_rl_v17/__init__.py',
             ROOT / 'experiments/option_rl_v14/controller.py', ROOT / 'experiments/option_rl_v14/interface.py',
             ROOT / 'experiments/option_rl_v14/mode_env.py', ROOT / 'experiments/option_rl_v14/scenes.py',
             ROOT / 'experiments/option_rl_v13/controller.py', ROOT / 'experiments/option_rl_v13/mode_env.py',
             ROOT / 'experiments/option_rl_v13/push_controller.py',
             ROOT / 'experiments/option_rl_v15/render_episodes.py',
             ROOT / 'experiments/option_rl_v16/learnability.py', ROOT / 'experiments/option_rl_v16/live.py',
             ROOT / 'experiments/option_rl_v45/render_episodes.py',
             ROOT / 'src/coimagine/random_start/env.py', ROOT / 'src/coimagine/random_start/spec.py',
             ROOT / 'src/coimagine/random_start/skills.py', ROOT / 'src/coimagine/random_start/effect_push.py',
             ROOT / 'docs/agent_exchange/tasks/D003_Contact_Timing_Audit.md',
             ROOT / 'docs/agent_exchange/tasks/D003_config.json']
    hashes = {str(p.relative_to(ROOT)): _sha256(p) for p in files if p.exists()}
    snap = out / 'source_snapshot'
    snap.mkdir(exist_ok=True)
    for p in files:
        if p.exists():
            dest = snap / str(p.relative_to(ROOT))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    (out / 'source_hashes.json').write_text(json.dumps(hashes, indent=1))
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--d001', type=Path, required=True)
    parser.add_argument('--d002', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--reuse-witness', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and not args.reuse_witness and any(p.name != 'supervisor.json' for p in out.iterdir()):
        raise SystemExit(f'output directory must be new/empty (except supervisor.json): {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'baseline').mkdir(exist_ok=True)
    cfg = json.loads(args.config.read_text())
    d001 = args.d001.resolve(); d002 = args.d002.resolve(); panel = args.panel.resolve()
    path_map = {
        'panel': panel,
        'd001_manifest': d001 / 'manifest.json',
        'd002_manifest': d002 / 'manifest.json',
        'witness_dual_2': ROOT / 'outputs/option_rl_v14/learnability_002/logs/dual_2.json',
        'witness_dual_0': ROOT / 'outputs/option_rl_v14/learnability_002/logs/dual_0.json',
        'task': ROOT / 'docs/agent_exchange/tasks/D003_Contact_Timing_Audit.md',
        'brief': ROOT / 'docs/agent_exchange/briefs/D003_Why_This_Audit.md',
    }
    for sample in cfg['samples']:
        if sample['kind'] == 'failure':
            path_map[f"{sample['id']}_d001_json"] = Path(sample['d001_episode_json'])
            path_map[f"{sample['id']}_d001_npz"] = Path(sample['d001_episode_npz'])
            path_map[f"{sample['id']}_d002_hold"] = Path(sample['d002_hold_branch'])
    input_hashes = dict(cfg['input_sha256'])
    for key, digest in input_hashes.items():
        path = path_map.get(key)
        if path is None or not Path(path).exists():
            raise SystemExit(f'input path missing for hash: {key}')
        if _sha256(path) != digest:
            raise SystemExit(f'input hash mismatch: {key}')
    witness_cases = [c for c in cfg['samples'] if c['kind'] == 'witness']
    failure_cases = [c for c in cfg['samples'] if c['kind'] == 'failure']
    if len(witness_cases) != 2 or len(failure_cases) != 6:
        raise SystemExit('D003 expects 2 witness + 6 failure cases')
    if args.reuse_witness:
        witness_saved = json.loads((out / 'witness.json').read_text())
        if not witness_saved.get('ok'):
            raise SystemExit('reuse-witness requested but saved witness did not pass')
    preflight = dict(version=VERSION, cases=len(cfg['samples']), witness=len(witness_cases),
                     failure=len(failure_cases), input_hashes=cfg['input_sha256'],
                     workers=args.workers, budget=cfg['budget'])
    atomic_json(out / 'preflight.json', preflight)
    if args.preflight_only:
        print(json.dumps(preflight, indent=1))
        return
    started = time.monotonic()
    results = []
    errors = []
    # Witness first, 2 jobs (each runs baseline + instrumented), unless reusing
    # the already completed witness stage from the same output directory.
    if args.reuse_witness:
        witness_results = witness_saved['results']
        witness_analysis = []
        witness_ok = bool(witness_saved.get('ok'))
        for r in witness_results:
            case = next(c for c in witness_cases if c['id'] == r['case'])
            inst = _case_instance_goal(cfg, case, panel)
            ana = analyse_case(case, r, inst.goal_centre)
            witness_analysis.append(ana)
            if ana['events']['stable_ticks'] < 10 or ana['events']['both_active_ticks'] < 1:
                witness_ok = False
            if r.get('status') != 'ok':
                witness_ok = False
        errors.extend(witness_saved.get('errors', []))
    else:
        witness_jobs = [dict(case=c, config=cfg, d001=str(d001), d002=str(d002), panel=str(panel))
                        for c in witness_cases]
        with ProcessPoolExecutor(max_workers=min(2, args.workers),
                                 mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
            futures = [ex.submit(_run_case, j) for j in witness_jobs]
            for f in futures:
                try:
                    results.append(f.result())
                except Exception as exc:
                    import traceback
                    errors.append(dict(stage='witness', error=repr(exc), traceback=traceback.format_exc()))
        witness_results = [r for r in results if r.get('kind') == 'witness']
        witness_ok = len(witness_results) == 2 and not errors
        witness_analysis = []
        for r in witness_results:
            inst = _case_instance_goal(cfg, next(c for c in witness_cases if c['id'] == r['case']), panel)
            ana = analyse_case(next(c for c in witness_cases if c['id'] == r['case']), r, inst.goal_centre)
            witness_analysis.append(ana)
            # Positive control: both contact and activation must be observable.
            if ana['events']['stable_ticks'] < 10 or ana['events']['both_active_ticks'] < 1:
                witness_ok = False
            if r.get('status') != 'ok':
                witness_ok = False
        atomic_json(out / 'witness.json', dict(ok=bool(witness_ok), results=witness_results,
                                               analysis=[a['metrics'] for a in witness_analysis], errors=errors))
    if not witness_ok:
        atomic_json(out / 'manifest.json', dict(version=VERSION, status='incomplete',
                                                reason='instrumentation_or_contract_unresolved',
                                                witness_ok=False, errors=errors))
        print(json.dumps(dict(status='incomplete', reason='instrumentation_or_contract_unresolved'), indent=1))
        return
    # Failure cases only after the witness passes.
    failure_jobs = [dict(case=c, config=cfg, d001=str(d001), d002=str(d002), panel=str(panel))
                    for c in failure_cases]
    failure_results = []
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        futures = [ex.submit(_run_case, j) for j in failure_jobs]
        pending = set(futures)
        while pending:
            done, pending = wait(pending, timeout=20, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    failure_results.append(f.result())
                except Exception as exc:
                    import traceback
                    errors.append(dict(stage='failure', error=repr(exc), traceback=traceback.format_exc()))
    atomic_json(out / 'failure_errors.json', errors)
    # Any source / non-interference failure is incomplete; do not auto-retry.
    failure_ok = len(failure_results) == 6 and all(r.get('status') == 'ok' for r in failure_results)
    failure_analysis = []
    goals = {}
    for r in failure_results:
        case = next(c for c in failure_cases if c['id'] == r['case'])
        inst = _case_instance_goal(cfg, case, panel)
        goals[case['id']] = inst.goal_centre
        failure_analysis.append(analyse_case(case, r, inst.goal_centre))
    witness_result_cases = [next(c for c in witness_cases if c['id'] == r['case']) for r in witness_results]
    failure_result_cases = [next(c for c in failure_cases if c['id'] == r['case']) for r in failure_results]
    for case in witness_result_cases + failure_result_cases:
        inst = _case_instance_goal(cfg, case, panel)
        goals[case['id']] = inst.goal_centre
    # Save raw ticks, baseline rows, events, metrics, blockers (order aligned with results).
    all_results = witness_results + failure_results
    all_analyses = witness_analysis + failure_analysis
    all_cases = witness_result_cases + failure_result_cases
    for case, r in zip(all_cases, all_results):
        if r.get('ticks'):
            atomic_json(out / 'baseline' / f"{case['id']}__instrumented_ticks.json", r['ticks'])
    _write_raw(out / 'raw_ticks.jsonl', all_results)
    _write_baseline(out, all_results)
    metrics = [a['metrics'] for a in all_analyses]
    events_rows = []
    blocker_rows = []
    for ana, case, result in zip(all_analyses, all_cases, all_results):
        events = ana.get('events') or {}
        labels = ana.get('labels') or ana.get('metrics', {}).get('labels') or []
        ticks = (result.get('ticks') or {}).get('ticks', [])
        def tick_at(i):
            return int(ticks[i]['tick']) if ticks and i is not None and 0 <= i < len(ticks) else None
        def time_at(i):
            return float(ticks[i]['t_start']) if ticks and i is not None and 0 <= i < len(ticks) else None
        for i in events.get('onsets', []):
            events_rows.append(dict(case=case['id'], event='both_contact_onset', tick=tick_at(i),
                                    time=time_at(i)))
        for i in events.get('losses', []):
            events_rows.append(dict(case=case['id'], event='both_contact_loss', tick=tick_at(i),
                                    time=time_at(i)))
        for i, side in events.get('ready_updates', []):
            events_rows.append(dict(case=case['id'], event='contact_ready_update', tick=tick_at(i),
                                    time=time_at(i), side=side))
        for i in events.get('activation_ticks', []):
            events_rows.append(dict(case=case['id'], event='joint_active_transition', tick=tick_at(i),
                                    time=time_at(i)))
        for label in labels:
            i = label.get('tick', label.get('start'))
            events_rows.append(dict(case=case['id'], event=label['kind'], tick=tick_at(i),
                                    time=time_at(i), details=label))
        for b in ana.get('blockers') or []:
            blocker_rows.append(dict(case=case['id'], start_tick=tick_at(b['start']), end_tick=tick_at(b['end']),
                                     start_time=time_at(b['start']), end_time=time_at(b['end']),
                                     duration_seconds=(b['end'] - b['start'] + 1) * spec.CONTROL_DT,
                                     blockers=b['blockers']))
    _write_csv(out / 'case_metrics.csv', metrics)
    _write_csv(out / 'events.csv', events_rows)
    _write_csv(out / 'blocker_intervals.csv', blocker_rows)
    candidate = _minimal_candidate(metrics)
    reproduction = dict(
        witness_ok=bool(witness_ok), failure_ok=bool(failure_ok), errors=errors,
        witness=[dict(case=r['case'], status=r.get('status'), old_error=r.get('old_error'),
                      noninterference_error=r.get('noninterference_error'),
                      old_success=(r.get('source') or {}).get('old_success'),
                      old_rows=(r.get('source') or {}).get('old_rows'),
                      stable_ticks=a['metrics']['stable_ticks'],
                      both_active_ticks=a['metrics']['both_active_ticks'])
                 for r, a in zip(witness_results, witness_analysis)],
        failures=[dict(case=r['case'], status=r.get('status'), source=r.get('source'),
                       noninterference_error=r.get('noninterference_error'),
                       d002_overlap_error=r.get('d002_overlap_error'))
                  for r in failure_results],
    )
    atomic_json(out / 'reproduction.json', reproduction)
    display_cases = [{**case, **result} for case, result in zip(all_cases, all_results)]
    _timelines(out, display_cases, all_analyses)
    _replay(out, display_cases, goals)
    source_hashes = _source_snapshot(out, cfg)
    (out / 'task.md').write_text(ROOT.joinpath('docs/agent_exchange/tasks/D003_Contact_Timing_Audit.md').read_text())
    (out / 'config.json').write_text(args.config.read_text())
    status = 'complete' if (witness_ok and failure_ok and not errors) else 'incomplete'
    manifest = dict(version=VERSION, status=status, witness_ok=bool(witness_ok),
                    failure_ok=bool(failure_ok), errors=errors,
                    cases=len(all_results), replays=16, elapsed_seconds=time.monotonic() - started,
                    workers=args.workers, input_hashes=cfg['input_sha256'],
                    source_hashes=source_hashes, head=_git('rev-parse', 'HEAD'),
                    dirty_files=len(_git('status', '--porcelain').splitlines()),
                    candidate=candidate, command=sys.argv)
    atomic_json(out / 'manifest.json', manifest)
    summary = dict(version=VERSION, status=status, candidate=candidate,
                   reproduction=reproduction, metrics=metrics, errors=errors,
                   manifest=manifest)
    atomic_json(out / 'summary.json', summary)
    print(json.dumps(dict(status=status, witness_ok=witness_ok, failure_ok=failure_ok,
                          candidate=candidate, elapsed=round(manifest['elapsed_seconds'], 1)), indent=1))
    if status != 'complete':
        raise SystemExit(1)


def _git(*args):
    import subprocess
    return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


if __name__ == '__main__':
    main()
