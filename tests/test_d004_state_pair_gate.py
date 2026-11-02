"""Focused, non-physical tests for D004 state registry + pair gate."""
from pathlib import Path
from experiments.option_rl_v18 import state_schema as S
from experiments.option_rl_v18.pair_gate import PairGateEnv, make_pair_env
from experiments.option_rl_v13.controller import ACTIONS, make_skill
from experiments.option_rl_v16.learnability import load_panel

ROOT = Path(__file__).resolve().parents[1]
PUSH = [a for a in ACTIONS if a.kind == 'push' and getattr(a, 'effect_kind', None) == 'translate'][0]


def _env():
    train, _ = load_panel(ROOT / 'outputs/option_rl_v14/learnability_002/panel.json')
    env = make_pair_env(train[0], context='extended')
    env.slots[0].skill = make_skill(PUSH)
    env.slots[1].skill = make_skill(PUSH)
    for s in env.slots:
        s.pending = None; s.pause_reason = None
    env.recovery_started = None; env.clearance.started = False; env.clearance.finished = False
    env.total_periods = 0
    return env


def test_state_table_and_inventory_are_synced():
    S.write_tables()
    rows = S.classify_runtime_fields()
    unknown = [r for r in rows if r['kind'] == 'unclassified']
    assert not unknown, unknown[:5]
    state_fields = {f['field'] for f in S.FIELD_SPECS}
    assert len(state_fields) == len(S.FIELD_SPECS)
    # Every runtime field has an inventory row; every semantic state field is in the table.
    assert len(rows) > 150
    for r in rows:
        if r['kind'] == 'state':
            assert r['field'] in state_fields, r


def test_pair_phase_transitions_and_live_flags():
    env = _env(); s0, s1 = env.slots[0].skill, env.slots[1].skill
    # APPROACH: not both ready
    s0.stage = 'approach'; s1.stage = 'approach'; s0.ready = s1.ready = False
    s0.contact_ready = s1.contact_ready = False
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'APPROACH' and not s0.release and not s0.joint_active
    # READY: both ready, no contact
    s0.stage = s1.stage = 'push'; s0.ready = s1.ready = True
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'READY' and s0.release and not s0.joint_active
    # ENGAGE: one contact
    s0.contact_ready = True; s1.contact_ready = False
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'ENGAGE' and s0.release and not s0.joint_active
    # PUSH: both contact
    s1.contact_ready = True
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'PUSH' and s0.joint_active and s1.joint_active
    # contact loss is live and clears joint_active in the same coordinator call
    s1.contact_ready = False
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'ENGAGE' and not s0.joint_active and not s1.joint_active
    assert env.contact_loss_latch


def test_stage_fallback_clears_unilateral_push_stage():
    env = _env(); s0, s1 = env.slots[0].skill, env.slots[1].skill
    s0.stage = 'push'; s1.stage = 'push'; s0.ready = True; s1.ready = False
    s0.contact_ready = True; s1.contact_ready = False
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'APPROACH'
    assert s0.stage == 'approach' and s1.stage == 'approach'
    assert not s0.release and not s0.joint_active
    assert env.stage_fallback_reason


def test_pending_or_fault_breaks_pair():
    env = _env()
    env.slots[0].pending = object()
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'IDLE'
    env.slots[0].pending = None
    env.recovery_started = 0.0
    env._apply_pair_gate(0.0)
    assert env.pair_phase == 'RECOVER'
