"""Focused, non-physical tests for the D003 contact-timing audit."""
import os, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.option_rl_v17 import contact_timing_audit as A
from experiments.option_rl_v17.run_d003 import run_supervisor


class FakeSkill:
    def __init__(self):
        self.stage = 'push'; self.ready = True; self.contact_ready = False
        self.release = False; self.joint_active = False; self.joint_pair = False
        self.cooperative = False; self.done = False; self.exit_reason = None
        self.recovery_requested = False; self.recovery_ready = False
        self.skill_time = 1.0; self.reference = SimpleNamespace(value=(0.1, 0.0))


class FakeAction:
    label = 'PUSH_E[v=0.08m/s]'; index = 1; type_index = 0
    kind = 'push'; effect_kind = 'translate'; speed = 0.08


class FakeSlot:
    def __init__(self, calls, side):
        self.calls = calls; self.side = side
        self.action = FakeAction(); self.pending = None; self.pause_reason = None
        self.brake = None; self.pause_brake = None; self.skill = FakeSkill()

    def prepare(self, obs):
        self.calls.append(f'prepare{self.side}')
        return None

    def step(self, obs):
        self.calls.append(f'step{self.side}')
        return ((0., 0.), {'stage': 'push', 'contact_ready': True, 'ready': True})


class FakeBackend:
    def contacts(self):
        return []


class FakeInner:
    def __init__(self, calls):
        self.calls = calls; self.backend = FakeBackend()

    def step_control(self, forces, headings, **kwargs):
        self.calls.append('step_control')
        return None, {'finite': True, 'penetration': 0., 'in_bounds': True}


class FakeClearance:
    def __init__(self, calls):
        self.calls = calls; self.started = False; self.finished = False; self.failed = False
        self.tick = 0

    def step(self, observations, dt):
        self.calls.append('clearance')
        self.tick += 1
        return None if self.tick == 1 else [((0., 0.), {}), ((0., 0.), {})]


class FakeEnv:
    def __init__(self):
        self.calls = []
        self.slots = [FakeSlot(self.calls, 0), FakeSlot(self.calls, 1)]
        self.clearance = FakeClearance(self.calls)
        self.env = FakeInner(self.calls)
        self.rows = []
        self.total_periods = 0
        self.recovery_started = None
        self.recovery_time = 0.
        self.recovery_attempts = 0
        self.termination = None
        self.success = False
        self.reward_total = 0.
        self.obs = SimpleNamespace(
            own=SimpleNamespace(x=2.5, y=1.7, angle=1.57, speed=0.),
            contacting_object=True, contacting_static=False, contacting_teammate=False,
            contact_point_robot=None, patch=None, time=0., period=0)

    def _coordinate(self, now, observations=None):
        self.calls.append('coordinate')
        return None

    def object_state(self):
        return SimpleNamespace(x=2.5, y=2.0, angle=0., vx=0., vy=0., omega=0.)

    def robot_states(self):
        return [SimpleNamespace(x=2.5, y=1.7, angle=0., vx=0., vy=0., omega=0.),
                SimpleNamespace(x=2.25, y=1.7, angle=0., vx=0., vy=0., omega=0.)]

    def success_conditions(self):
        return dict(distance=0.45, linear=0., angular=0.)

    def object_distance(self):
        return 0.45

    def joint_step(self, a0, a1, record=False):
        for slot in self.slots:
            slot.prepare(self.obs)
        self._coordinate(0., self.obs)
        sep = self.clearance.step(self.obs, 0.05)
        if sep is None:
            for slot in self.slots:
                slot.step(self.obs)
        self.env.step_control([(0., 0.), (0., 0.)], [(0., 1.), (0., 1.)])
        self.total_periods += 1
        self.rows.append(dict(true=dict(object=[2.5, 2.0, 0.], robots=[[2.5, 1.7, 0.], [2.25, 1.7, 0.]]),
                              forces=[[0., 0.], [0., 0.]]))


def _obs():
    return FakeEnv().obs


def test_wrapper_order_once_and_clearance_bypass(monkeypatch):
    monkeypatch.setattr(A, 'mask', lambda obs: np.ones(46, dtype=bool))
    env = FakeEnv()
    rec = A.Recorder('case', 'instrumented')
    inst = A.Instrumentation(env, rec).install()
    env.joint_step(1, 2, record=True)
    env.joint_step(1, 2, record=True)
    inst.restore()
    assert env.calls[:7] == ['prepare0', 'prepare1', 'coordinate', 'clearance', 'step0', 'step1', 'step_control']
    assert env.calls.count('prepare0') == 2 and env.calls.count('step0') == 1
    assert env.calls.count('coordinate') == 2 and env.calls.count('clearance') == 2
    ticks = rec.finish(env, False, None, 0.)['ticks']
    assert len(ticks) == 2
    assert set(ticks[0]['p3']) == {'0', '1'} and not ticks[0]['p3_skipped']
    assert ticks[1]['p3_skipped'] is True and ticks[1]['p3'] == {}
    assert ticks[0]['p4']['entry_flags']['0']['action_label'] == 'PUSH_E[v=0.08m/s]'


def _tick(contact, ready, active, *, pair_changed=False, recovery=False):
    return {
        'p0': {'0': {'obs': {'contacting_object': contact}}, '1': {'obs': {'contacting_object': contact}}},
        'p1': {'sides': {'0': {'contact_ready': ready, 'joint_active': False, 'done': False, 'pending_label': None},
                          '1': {'contact_ready': ready, 'joint_active': False, 'done': False, 'pending_label': None}},
               'pair': [0, 1], 'pair_changed': pair_changed, 'recovery': {'started': recovery}, 'clearance': {}},
        'p2': {'sides': {'0': {'contact_ready': ready, 'joint_active': active},
                          '1': {'contact_ready': ready, 'joint_active': active}},
               'pair': [0, 1], 'recovery': {'started': recovery}, 'clearance': {}},
        'p3': {'0': {'entry': {'contact_ready': False}, 'exit': {'contact_ready': ready},
                     'info': {'gap': 0.01, 'gap_target': 0.02, 'tangent_error': 0.0, 'align_error_deg': 0.,
                              'normal_valid': True}},
               '1': {'entry': {'contact_ready': False}, 'exit': {'contact_ready': ready},
                     'info': {'gap': 0.01, 'gap_target': 0.02, 'tangent_error': 0.0, 'align_error_deg': 0.,
                              'normal_valid': True}}},
        'p4': {'entry_flags': {'0': {'contact_ready': ready, 'joint_active': active},
                                '1': {'contact_ready': ready, 'joint_active': active}},
               'after': {'object': [2.5, 2.0, 0.]}, 'true_goal_distance': 0.45,
               'true_edge_gap': {'0': 0.01, '1': 0.01}, 'contacts': []},
        'clearance': {}, 'tick': 0,
    }


def test_activation_not_reported_as_never_and_censors():
    t0 = _tick(True, True, False)
    t1 = _tick(False, False, True)
    t1['tick'] = 1
    ev = A.derive_events([t0, t1])
    assert ev['both_contact_ticks'] == 1
    assert ev['activation_ticks'] == [1] and ev['both_active_ticks'] == 1
    assert ev['stable_ticks'] == 1
    labels = A._contact_run_labels(ev, [t0, t1], None)
    assert any(l['kind'] == 'contact_not_maintained' for l in labels)
    # Last-tick contact is right-censored, not a contact-loss event.
    ev2 = A.derive_events([t0, _tick(True, True, False)])
    assert ev2['right_censored'] and ev2['losses'] == []
    # First-tick contact is left-censored.
    assert A.derive_events([t0])['left_censored']
    # A new pair or recovery invalidates the old ready state for coordination labels.
    t = _tick(True, True, False, pair_changed=True)
    assert A._coordination_labels(A.derive_events([t]), [t]) == []


def test_coordination_candidate_requires_stable_pair_and_no_recovery():
    t = _tick(True, True, False)
    labels = A._coordination_labels(A.derive_events([t]), [t])
    assert labels and labels[0]['kind'] == 'coordination_contract_violation_candidate'


def test_outer_supervisor_kills_process_group(tmp_path):
    pidfile = tmp_path / 'pids'
    script = (
        "import os,subprocess,sys,time\n"
        "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])\n"
        "open(sys.argv[1],'w').write(f'{os.getpid()} {child.pid}')\n"
        "time.sleep(30)\n"
    )
    result = run_supervisor([sys.executable, '-c', script, str(pidfile)], tmp_path / 'out',
                            deadline=1., grace=2.)
    assert result['status'] == 'timeout'
    pids = [int(x) for x in pidfile.read_text().split()]
    for pid in pids:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        else:
            os.kill(pid, 9)
            pytest.fail(f'process {pid} survived supervisor timeout')
