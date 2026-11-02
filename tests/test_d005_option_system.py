"""Focused tests for D005 registry/coordinator/session."""
import json
from pathlib import Path
from types import SimpleNamespace
from experiments.option_rl_v19 import spec as S
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v19.session import TestbedSession
from experiments.option_rl_v13.controller import ACTIONS, make_skill
from experiments.option_rl_v16.learnability import load_panel

ROOT = Path(__file__).resolve().parents[1]
PUSH = [a for a in ACTIONS if a.kind == 'push' and getattr(a, 'effect_kind', None) == 'translate'][0]


def _env():
    train, _ = load_panel(ROOT / 'outputs/option_rl_v14/learnability_002/panel.json')
    env = OptionSystemEnv(train[0])
    for slot in env.slots:
        slot.action = PUSH
        slot.skill = make_skill(PUSH)
    for s in env.slots:
        s.pending = None; s.pause_reason = None
    env.recovery_started = None; env.clearance.started = False; env.clearance.finished = True
    env.total_periods = 0
    return env


def test_registry_and_tables_are_generated():
    S.write_tables()
    assert len(S.SPECS) == 46
    assert len({s.action_index for s in S.SPECS}) == 46
    assert {s.family for s in S.SPECS} == {'PUSH', 'ROTATE', 'REPOSITION', 'WAIT', 'FOLLOW'}
    opt = S.OPTION_TABLE.read_text().splitlines()
    state = S.STATE_TABLE.read_text().splitlines()
    param = S.PARAM_TABLE.read_text().splitlines()
    assert len(opt) == len(S.ACTIVE_SPECS) + 1 and len(state) > 40 and len(param) > 10
    assert 'same_family' in S.SPEC_BY_INDEX[0].joint_requirements


def test_coordinator_phases_are_live():
    env = _env(); s0, s1 = env.slots[0].skill, env.slots[1].skill
    s0.stage = s1.stage = 'approach'; s0.ready = s1.ready = False
    s0.contact_ready = s1.contact_ready = False
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'APPROACH'
    s0.stage = s1.stage = 'push'; s0.ready = s1.ready = True
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'READY' and s0.release and not s0.joint_active
    s0.contact_ready = True
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'ENGAGE' and not s0.joint_active
    s1.contact_ready = True
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'PUSH' and s0.joint_active and s1.joint_active
    s1.contact_ready = False
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'ENGAGE' and not s0.joint_active
    # not ready -> stage fallback, release live
    s1.ready = False
    s0.stage = s1.stage = 'push'
    env.coordinator.update(env, 0.0)
    assert env.coordinator.phase == 'APPROACH'
    assert s0.stage == 'approach' and not s0.release and env.coordinator.stage_fallback


def test_force_gate_passes_ready_engage():
    env = _env(); s0, s1 = env.slots[0].skill, env.slots[1].skill
    s0.stage = s1.stage = 'push'; s0.ready = s1.ready = True
    s0.contact_ready = s1.contact_ready = False
    env.coordinator.update(env, 0.0)
    seen = {}
    env.coordinator.gate(env, ((2., 0.), (1., 0.)), lambda f: seen.setdefault('f', f))
    assert seen['f'] == ((2., 0.), (1., 0.))
    assert env.coordinator.phase == 'READY'


def test_session_run_roundtrip(tmp_path):
    log = json.loads((ROOT / 'outputs/option_rl_v14/learnability_002/logs/dual_2.json').read_text())
    from experiments.option_rl_v15.render_episodes import instance_from_dict
    session = TestbedSession(instance_from_dict(log['instance']))
    for a0, a1 in log['commands']:
        session.add(a0, a1, 1)
    run = session.run()
    assert run['success'] and run['periods'] == 129
    assert len(run['frames']) == len(run['strip']) == 129
    assert session.save(tmp_path / 'run.json').exists()
    loaded = TestbedSession.load(tmp_path / 'run.json')
    assert loaded['periods'] == 129 and loaded['sequence'] == run['sequence']


def test_param_override_reaches_skill():
    base = S.make_skill(PUSH)
    over = S.make_skill(PUSH, {'PUSH': {'joint_engage_speed': 0.05}})
    assert base.config.joint_engage_speed != over.config.joint_engage_speed
    assert abs(over.config.joint_engage_speed - 0.05) < 1e-12


def test_session_multiple_runs_roundtrip(tmp_path):
    log = json.loads((ROOT / 'outputs/option_rl_v14/learnability_002/logs/dual_2.json').read_text())
    from experiments.option_rl_v15.render_episodes import instance_from_dict
    session = TestbedSession(instance_from_dict(log['instance']))
    for a0, a1 in log['commands']:
        session.add(a0, a1, 1)
    session.run(); session.run()
    assert len(session.list_runs()) == 2
    paths = session.save_runs(tmp_path / 'runs')
    assert len(paths) == 2 and all(p.exists() for p in paths)
