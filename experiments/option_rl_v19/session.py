"""Headless testbed session: manual 5 Hz option sequence on the same Pymunk env.

No UI yet.  A run is a JSON-serializable scene/sequence/override/frame record;
the Web testbed (D006) can replay these runs and edit the scene/sequence.
"""
from __future__ import annotations
import json, hashlib, math, time
from dataclasses import dataclass, asdict
from pathlib import Path
from experiments.option_rl_v17 import contact_timing_audit as A
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.unknown_shape import UnknownShapeEnvMixin
from experiments.option_rl_v19 import unknown_shape as _controller
from experiments.option_rl_v19 import spec as S

# Freeze the source identity when this process loads the controller. Reading
# it only after a run mislabels old in-memory code after an on-disk edit.
CONTROLLER_SOURCE_PATH = Path(_controller.__file__)
CONTROLLER_SOURCE_SHA256 = hashlib.sha256(CONTROLLER_SOURCE_PATH.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Segment:
    a0: int
    a1: int
    decisions: int = 1
    budget_m: float | None = None
    time_budget_s: float | None = None
    progress: str = 'decisions'  # decisions | object_normal | robot_path


class TestbedSession:
    __test__ = False
    def __init__(self, instance, *, overrides=None, controller_version='m3b'):
        self.instance = instance
        self.overrides = overrides or {}
        self.controller_version = 'm3b'
        self.env = OptionSystemEnv(instance, overrides=self.overrides)
        UnknownShapeEnvMixin.install_prior_free(self.env)
        self.sequence = []
        self.last = None
        self.runs = []

    def add(self, a0, a1, decisions=1, budget_m=None, time_budget_s=None, progress='decisions'):
        self.sequence.append(Segment(S.canonical_index(a0), S.canonical_index(a1), int(decisions),
                                     None if budget_m is None else float(budget_m),
                                     None if time_budget_s is None else float(time_budget_s),
                                     str(progress)))
        return self

    def clear(self):
        self.sequence = []
        return self

    def reset(self):
        self.env = OptionSystemEnv(self.instance, overrides=self.overrides)
        UnknownShapeEnvMixin.install_prior_free(self.env)
        self.last = None
        return self

    def run(self):
        session_started = time.perf_counter()
        env = self.env
        rec = A.Recorder('session', 'session')
        inst = A.Instrumentation(env, rec).install()
        try:
            budget_trace = {}
            execution_started = time.perf_counter()
            for seg in self.sequence:
                budget_applied = False
                for _ in range(max(1, seg.decisions)):
                    if env.termination is not None:
                        break
                    env.joint_step(seg.a0, seg.a1, record=True)
                    budget_trace[int(env.total_periods)] = [
                        env.budget_trackers[i].info() for i in (0, 1)]
                    # Budgets live in the option controllers; the testbed only
                    # installs the request and reads controller-progress.
                    if not budget_applied and (seg.budget_m is not None or seg.time_budget_s is not None):
                        now = float(env.total_periods) * 0.2
                        for i in (0, 1):
                            env.budget_trackers[i].configure(seg.budget_m, seg.time_budget_s, seg.progress, now)
                        budget_applied = True
                    if budget_applied:
                        reached = any(env.budget_trackers[i].reached for i in (0, 1))
                        if reached:
                            break
            execution_seconds = time.perf_counter() - execution_started
            finalize_started = time.perf_counter()
            ticks = rec.finish(env, bool(env.success), env.termination, float(env.reward_total))
            finalize_seconds = time.perf_counter() - finalize_started
        finally:
            inst.restore()
        raw = ticks['ticks']
        frames_started = time.perf_counter()
        coord = {r['tick']: r for r in env.coordinator.rows}
        frames = []
        strip = []

        def _scan(t, side):
            return ((t.get('p0', {}) or {}).get(str(side), {}) or {}).get('obs', {}).get('scan')

        def _info(t, side):
            return ((t.get('p3', {}) or {}).get(str(side), {}) or {}).get('info') or {}

        def _debug(t, side):
            return ((t.get('p3', {}) or {}).get(str(side), {}) or {}).get('debug') or {}

        def _target(t, side):
            d = _debug(t, side)
            return d.get('contact_target') or d.get('reference_centre') or _info(t, side).get('contact_target')

        def _ref_centre(t, side):
            return _debug(t, side).get('reference_centre') or _info(t, side).get('reference_centre')

        def _mb(t, side):
            info=_info(t,side); eff=info.get('effect') or {}; obs=((t.get('p0',{}) or {}).get(str(side),{}) or {}).get('obs',{}) or {}
            own=obs.get('own') or {}
            return dict(action=info.get('skill'), stage=info.get('stage'), contact_phase=eff.get('contact_phase'),
                        candidate=eff.get('candidate'), candidate_err_t=eff.get('candidate_err_t'),
                        contact=bool(info.get('contact_ready') if info.get('contact_ready') is not None else info.get('contacting_object')), contact_point_robot=obs.get('contact_point_robot'),
                        contact_point_world=eff.get('contact_point_world'), own=eff.get('own'), own_v=eff.get('own_v'),
                        desired=info.get('desired_velocity'), reference=info.get('reference_velocity'), force=info.get('force'),
                        ff_t=eff.get('ff_t'), u_adjust=eff.get('u_adjust'), contact_err_t=eff.get('contact_err_t'), normal_angle=(math.atan2(eff.get('face_normal',[0,1])[1],eff.get('face_normal',[0,1])[0]) if eff.get('face_normal') else None),
                        face_points=eff.get('face_points'), face_normal=eff.get('face_normal'), v_est=eff.get('v_est'), v_cov=eff.get('v_cov'), estimator_debug=eff.get('estimator_debug'),
                        peer_contact=info.get('peer_contact'), pair_push_ready=info.get('pair_push_ready'), gap=info.get('gap'), ref_center=info.get('ref_center'), contact_ready=info.get('contact_ready'), selector_debug=(info.get('selector_debug') if info.get('selector_debug') is not None else eff.get('selector_debug')), q_prime=(info.get('q_prime') if info.get('q_prime') is not None else eff.get('q_prime')), contact_offset=(info.get('contact_offset') if info.get('contact_offset') is not None else eff.get('contact_offset')),
                        desired_pre=(info.get('desired_pre') if info.get('desired_pre') is not None else eff.get('desired_pre')), avoidance_active=(info.get('avoidance_active') if info.get('avoidance_active') is not None else eff.get('avoidance_active')), avoidance_direction=(info.get('avoidance_direction') or eff.get('avoidance_direction')),
                        gate_pre=(t.get('p4') or {}).get('gate_pre'), gate_post=(t.get('p4') or {}).get('gate_post'),
                        progress_m=(budget_trace.get(int(t['tick']), [{}]*2)[side].get('progress_m') if side < len(budget_trace.get(int(t['tick']), [])) else None),
                        budget_m=(budget_trace.get(int(t['tick']), [{}]*2)[side].get('budget_m') if side < len(budget_trace.get(int(t['tick']), [])) else None),
                        budget_reached=(budget_trace.get(int(t['tick']), [{}]*2)[side].get('budget_reached') if side < len(budget_trace.get(int(t['tick']), [])) else None),
                        time_budget_s=(budget_trace.get(int(t['tick']), [{}]*2)[side].get('time_budget_s') if side < len(budget_trace.get(int(t['tick']), [])) else None))

        for t in raw:
            p4 = t.get('p4') or {}
            bt = budget_trace.get(int(t['tick']), [])
            after = p4.get('after') or {}
            obj = after.get('object')
            robots = after.get('robots')
            stage0 = (t.get('p3', {}).get('0', {}) or {}).get('exit', {}).get('skill_stage') \
                or (p4.get('entry_flags', {}).get('0', {}) or {}).get('skill_stage')
            stage1 = (t.get('p3', {}).get('1', {}) or {}).get('exit', {}).get('skill_stage') \
                or (p4.get('entry_flags', {}).get('1', {}) or {}).get('skill_stage')
            c = coord.get(t['tick'], {})
            frames.append(dict(t=float(t['t_start']), object=obj, robots=robots,
                               stage=[stage0, stage1], phase=c.get('phase'),
                               scan=[_scan(t, 0), _scan(t, 1)],
                               target=[_target(t, 0), _target(t, 1)],
                               ref_centre=[_ref_centre(t, 0), _ref_centre(t, 1)],
                               mb=[_mb(t, 0), _mb(t, 1)],
                               progress=[(bt[i] if i < len(bt) else {}) for i in (0, 1)]))
            strip.append(dict(t=float(t['t_start']), tick=int(t['tick']),
                              stage=[stage0, stage1], phase=c.get('phase'),
                              contact=[bool(A._contact(t, 0)), bool(A._contact(t, 1))],
                              joint_active=bool(A._active(t, 0) or A._active(t, 1)),
                              forces=p4.get('forces')))
        disk_sha = hashlib.sha256(CONTROLLER_SOURCE_PATH.read_bytes()).hexdigest()
        run = dict(
            case='session', controller_version=self.controller_version,
            action_version=S.VERSION, action_codebook_hash=S.CODEBOOK_HASH,
            action_labels=[a.label for a in S.ACTIONS],
            support_memory_enabled=False, q_acquisition_mode='current_visible_support',
            controller_source_sha256=CONTROLLER_SOURCE_SHA256,
            controller_disk_sha256=disk_sha,
            controller_restart_required=(disk_sha != CONTROLLER_SOURCE_SHA256),
            sequence=[asdict(s) for s in self.sequence],
            overrides=self.overrides, frames=frames, strip=strip,
            success=bool(env.success), termination=env.termination,
            periods=int(env.total_periods), reward=float(env.reward_total),
            coordinator=[dict(r) for r in env.coordinator.rows],
            timing=dict(execution_and_recording_seconds=execution_seconds,
                        recording_finalize_seconds=finalize_seconds,
                        frame_build_seconds=time.perf_counter()-frames_started,
                        session_seconds=time.perf_counter()-session_started,
                        simulated_seconds=len(frames)*.05),
        )
        self.last = run
        self.runs.append(run)
        return run

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.last, indent=1))
        return path

    def save_overrides(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.overrides, indent=1))
        return path

    def list_runs(self):
        return list(self.runs)

    def save_runs(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, run in enumerate(self.runs):
            path = directory / f'run_{i:03d}.json'
            path.write_text(json.dumps(run, indent=1))
            paths.append(path)
        return paths

    @staticmethod
    def load(path):
        return json.loads(Path(path).read_text())
