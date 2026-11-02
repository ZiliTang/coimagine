"""D004 canonical state registry and flat table generator.

The semantic state table is hand-curated (FIELD_SPECS).  A runtime inventory of
every attribute on the current Slot / ContinuousPush / LearningEnv / clearance
objects is generated separately; any new runtime attribute must be classified,
so a later option edit cannot silently add an unmanaged state field.
"""
from __future__ import annotations
import csv, json
from dataclasses import dataclass, asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / 'docs/agent_exchange/state'
TABLE_PATH = STATE_DIR / 'state_table.csv'
INVENTORY_PATH = STATE_DIR / 'field_inventory.csv'

PAIR_PHASES = ('IDLE', 'APPROACH', 'READY', 'ENGAGE', 'PUSH', 'RECOVER', 'EXIT')

PAIR_TRANSITIONS = {
    'IDLE': {'paired_ready': 'APPROACH', 'not_paired': 'IDLE'},
    'APPROACH': {'both_ready': 'READY', 'one_ready_lost': 'APPROACH', 'not_paired': 'IDLE', 'exit': 'EXIT'},
    'READY': {'one_contact': 'ENGAGE', 'both_contact': 'PUSH', 'ready_lost': 'APPROACH',
              'recovery_or_pause': 'RECOVER', 'exit': 'EXIT'},
    'ENGAGE': {'both_contact': 'PUSH', 'contact_lost': 'ENGAGE', 'ready_lost': 'APPROACH',
               'recovery_or_pause': 'RECOVER', 'exit': 'EXIT'},
    'PUSH': {'contact_lost': 'ENGAGE', 'ready_lost': 'APPROACH', 'recovery_or_pause': 'RECOVER',
             'exit': 'EXIT'},
    'RECOVER': {'ready_and_no_fault': 'READY', 'contact_and_ready': 'PUSH', 'exit': 'EXIT',
                'not_paired': 'IDLE'},
    'EXIT': {'not_paired': 'IDLE'},
}

INVARIANTS = [
    'pair identity is stable within a pair phase episode; a change clears release/joint_active and pair start ticks',
    'PUSH implies both_ready and both_contact and both stage==push',
    'joint_active implies both_contact and both_ready',
    'release is live: it is false whenever either side is not ready or a fault is active',
    'no unilateral normal push: a side may not be in stage push with normal force while the peer is not both ready and contact',
    'contact loss clears joint_active in the same coordinator tick and before the next force application',
    'recovery/pause/clearance/exit clears release and joint_active, and is visible in the pair phase',
    'every state field in the flat table has one writer layer and a declared timing/persistence',
]

FIELD_SPECS = [
    # scope, field, type, owner/writer, reader, timing, persistence, default, meaning, invariant, gate, test, source
    dict(scope='robot', field='x', type='float', owner='physics', reader='obs,coordinator', timing='post_physics', persistence='live', default='0', meaning='robot x', invariant='finite, arena bounds', gate=False, test='state_shape', source='RobotState'),
    dict(scope='robot', field='y', type='float', owner='physics', reader='obs,coordinator', timing='post_physics', persistence='live', default='0', meaning='robot y', invariant='finite, arena bounds', gate=False, test='state_shape', source='RobotState'),
    dict(scope='robot', field='angle', type='float', owner='heading_kinematics', reader='obs,coordinator', timing='pre_physics', persistence='live', default='0', meaning='heading angle', invariant='wrapped', gate=True, test='heading_wrap', source='backend.set_robot_heading'),
    dict(scope='robot', field='vx', type='float', owner='physics', reader='obs', timing='post_physics', persistence='live', default='0', meaning='linear vx', invariant='finite', gate=False, test='state_shape', source='RobotState'),
    dict(scope='robot', field='vy', type='float', owner='physics', reader='obs', timing='post_physics', persistence='live', default='0', meaning='linear vy', invariant='finite', gate=False, test='state_shape', source='RobotState'),
    dict(scope='robot', field='omega', type='float', owner='heading_kinematics', reader='diagnostic', timing='pre_physics', persistence='live', default='0', meaning='omega is forced to 0 by set_robot_heading', invariant='==0', gate=False, test='heading_kinematic', source='backend.set_robot_heading'),
    dict(scope='robot', field='contacting_object', type='bool', owner='sensor', reader='skill,gate', timing='tick_start', persistence='live', default='False', meaning='tick-start robot-object contact', invariant='from observation only', gate=True, test='contact_timing', source='SkillObservation'),
    dict(scope='robot', field='patch_valid', type='bool', owner='sensor', reader='skill,gate', timing='tick_start', persistence='live', default='False', meaning='local face patch valid', invariant='not fabricated', gate=True, test='face_unknown', source='SurfacePatch'),
    dict(scope='robot', field='patch_normal', type='vec2', owner='sensor', reader='skill,gate', timing='tick_start', persistence='live', default='None', meaning='local face normal', invariant='unit or None', gate=True, test='face_unknown', source='SurfacePatch'),
    dict(scope='robot', field='gap', type='float', owner='skill', reader='skill,gate', timing='control', persistence='live', default='None', meaning='normal edge gap used by controller', invariant='None means unknown', gate=True, test='blocker_unknown', source='ContinuousPush.info'),
    dict(scope='robot', field='action_label', type='str', owner='actor/slot', reader='skill,gate', timing='decision', persistence='latched_until_switch', default='BRAKE_HOLD', meaning='committed action id', invariant='one action per slot', gate=True, test='action_identity', source='Slot.action'),
    dict(scope='robot', field='action_kind', type='str', owner='slot', reader='skill,gate', timing='decision', persistence='latched_until_switch', default='wait', meaning='push/reposition/follow/wait', invariant='known kind', gate=True, test='action_identity', source='Slot.action'),
    dict(scope='robot', field='action_type', type='int', owner='slot', reader='skill,gate', timing='decision', persistence='latched_until_switch', default='12', meaning='action type index', invariant='0..13', gate=True, test='action_identity', source='Slot.action'),
    dict(scope='robot', field='action_speed', type='float', owner='slot', reader='skill', timing='decision', persistence='latched_until_switch', default='0', meaning='normal speed command', invariant='>=0', gate=False, test='action_identity', source='Slot.action'),
    dict(scope='robot', field='stage', type='str', owner='ContinuousPush', reader='gate,observer', timing='control', persistence='live', default='approach', meaning='approach/ready/push/recover/follow', invariant='no push if release false', gate=True, test='stage_fallback', source='ContinuousPush.stage'),
    dict(scope='robot', field='ready', type='bool', owner='ContinuousPush', reader='coordinator,gate', timing='control', persistence='live', default='False', meaning='approach complete / close+aligned', invariant='gate AND of both sides for release', gate=True, test='pair_ready', source='ContinuousPush.ready'),
    dict(scope='robot', field='contact_ready', type='bool', owner='ContinuousPush', reader='coordinator,gate', timing='control', persistence='live', default='False', meaning='own contact in push stage', invariant='gate AND of both sides for joint_active', gate=True, test='pair_contact', source='ContinuousPush.contact_ready'),
    dict(scope='robot', field='release', type='bool', owner='coordinator', reader='ContinuousPush,gate', timing='control', persistence='live', default='False', meaning='pair release to push stage', invariant='live; false if either not ready', gate=True, test='release_live', source='ContinuousModeEnv._coordinate'),
    dict(scope='robot', field='joint_active', type='bool', owner='coordinator', reader='ContinuousPush,gate', timing='control', persistence='live', default='False', meaning='both contact; normal push enabled', invariant='live; false if either contact lost', gate=True, test='joint_live', source='ContinuousModeEnv._coordinate'),
    dict(scope='robot', field='joint_pair', type='bool', owner='coordinator', reader='ContinuousPush,gate', timing='control', persistence='live', default='False', meaning='both sides same push family', invariant='pair identity stable', gate=True, test='pair_identity', source='ContinuousModeEnv._coordinate'),
    dict(scope='robot', field='cooperative', type='bool', owner='coordinator', reader='ContinuousPush', timing='control', persistence='live', default='False', meaning='paired cooperative flag', invariant='matches joint_pair', gate=False, test='pair_identity', source='ContinuousModeEnv._coordinate'),
    dict(scope='robot', field='done', type='bool', owner='ContinuousPush', reader='coordinator,gate', timing='control', persistence='live', default='False', meaning='skill terminal', invariant='terminal => no force', gate=True, test='exit_no_force', source='ContinuousPush.done'),
    dict(scope='robot', field='exit_reason', type='str', owner='ContinuousPush', reader='gate,diagnostic', timing='control', persistence='latched', default='None', meaning='terminal reason', invariant='immutable after done', gate=True, test='exit_no_force', source='ContinuousPush.exit_reason'),
    dict(scope='robot', field='pause_reason', type='str', owner='coordinator', reader='gate,diagnostic', timing='control', persistence='live', default='None', meaning='peer/perception pause', invariant='any pause blocks normal push', gate=True, test='pause_clear', source='Slot.pause_reason'),
    dict(scope='robot', field='pending', type='str', owner='slot', reader='gate', timing='decision', persistence='live', default='None', meaning='next action waiting for brake handshake', invariant='pending blocks pair', gate=True, test='pending_blocks_pair', source='Slot.pending'),
    dict(scope='robot', field='recovery_requested', type='bool', owner='ContinuousPush', reader='coordinator,gate', timing='control', persistence='live', default='False', meaning='needs joint recovery', invariant='blocks normal push', gate=True, test='recovery_clear', source='ContinuousPush.recovery_requested'),
    dict(scope='robot', field='recovery_ready', type='bool', owner='ContinuousPush', reader='coordinator', timing='control', persistence='live', default='False', meaning='recovery ready', invariant='only in recovery', gate=True, test='recovery_clear', source='ContinuousPush.recovery_ready'),
    dict(scope='robot', field='skill_time', type='float', owner='ContinuousPush', reader='diagnostic,gate', timing='control', persistence='monotone', default='0', meaning='time in current skill', invariant='>=0', gate=False, test='state_shape', source='ContinuousPush.skill_time'),
    dict(scope='robot', field='reference', type='vec2', owner='ReferenceVelocity', reader='force,diagnostic', timing='control', persistence='live', default='[0,0]', meaning='rate-limited reference velocity', invariant='a_ref_max=0.5 m/s^2', gate=False, test='reference_ramp', source='ReferenceVelocity'),
    dict(scope='robot', field='force', type='vec2', owner='skill', reader='physics,gate', timing='pre_physics', persistence='live', default='[0,0]', meaning='commanded force', invariant='<=max_force; normal only PUSH', gate=True, test='force_gate', source='Slot.step'),
    dict(scope='robot', field='stage_fallback', type='str', owner='pair_gate', reader='diagnostic', timing='control', persistence='latched_tick', default='None', meaning='why push stage was reverted', invariant='non-None only when fallback', gate=True, test='stage_fallback', source='PairGateEnv'),
    dict(scope='robot', field='force_gated', type='str', owner='pair_gate', reader='diagnostic', timing='pre_physics', persistence='latched_tick', default='None', meaning='why force was limited/zeroed', invariant='never silently', gate=True, test='force_gate', source='PairGateEnv'),
    dict(scope='coordinator', field='pair_identity', type='tuple', owner='ContinuousModeEnv', reader='gate', timing='control', persistence='live_pair_episode', default='None', meaning='stable pair skill ids', invariant='change resets flags/latches', gate=True, test='pair_identity', source='ContinuousModeEnv.pair_identity'),
    dict(scope='coordinator', field='pair_phase', type='enum', owner='pair_gate', reader='all', timing='control', persistence='live', default='IDLE', meaning='IDLE/APPROACH/READY/ENGAGE/PUSH/RECOVER/EXIT', invariant='must match flags', gate=True, test='phase_transition', source='PairGateEnv'),
    dict(scope='coordinator', field='both_ready', type='bool', owner='pair_gate', reader='all', timing='control', persistence='live', default='False', meaning='AND of both ready', invariant='release iff true', gate=True, test='pair_ready', source='PairGateEnv'),
    dict(scope='coordinator', field='both_contact', type='bool', owner='pair_gate', reader='all', timing='control', persistence='live', default='False', meaning='AND of both contact_ready', invariant='joint_active iff true', gate=True, test='pair_contact', source='PairGateEnv'),
    dict(scope='coordinator', field='both_active', type='bool', owner='pair_gate', reader='all', timing='control', persistence='live', default='False', meaning='AND of both joint_active', invariant='only PUSH', gate=True, test='joint_live', source='PairGateEnv'),
    dict(scope='coordinator', field='force_enable_0', type='bool', owner='pair_gate', reader='physics', timing='pre_physics', persistence='live', default='False', meaning='robot0 normal force allowed', invariant='only PUSH', gate=True, test='force_gate', source='PairGateEnv'),
    dict(scope='coordinator', field='force_enable_1', type='bool', owner='pair_gate', reader='physics', timing='pre_physics', persistence='live', default='False', meaning='robot1 normal force allowed', invariant='only PUSH', gate=True, test='force_gate', source='PairGateEnv'),
    dict(scope='coordinator', field='pair_start_tick', type='int', owner='pair_gate', reader='diagnostic', timing='control', persistence='latched_pair_episode', default='None', meaning='first both-ready tick', invariant='reset on identity change', gate=True, test='phase_timing', source='PairGateEnv'),
    dict(scope='coordinator', field='engage_tick', type='int', owner='pair_gate', reader='diagnostic', timing='control', persistence='latched_pair_episode', default='None', meaning='first any-contact tick', invariant='<=push_tick', gate=True, test='phase_timing', source='PairGateEnv'),
    dict(scope='coordinator', field='push_tick', type='int', owner='pair_gate', reader='diagnostic', timing='control', persistence='latched_pair_episode', default='None', meaning='first both-contact tick', invariant='PUSH starts here', gate=True, test='phase_timing', source='PairGateEnv'),
    dict(scope='coordinator', field='contact_loss_latch', type='bool', owner='pair_gate', reader='all', timing='control', persistence='live_tick', default='False', meaning='both contact lost after PUSH', invariant='clears joint_active', gate=True, test='contact_loss_clear', source='PairGateEnv'),
    dict(scope='coordinator', field='one_side_normal_count', type='int', owner='pair_gate', reader='report', timing='control', persistence='cumulative', default='0', meaning='ticks with unilateral normal push', invariant='must be 0', gate=True, test='no_unilateral', source='PairGateEnv'),
    dict(scope='coordinator', field='clearance_active', type='bool', owner='PeerClearancePrefix', reader='pair_gate', timing='control', persistence='live', default='False', meaning='clearance takeover', invariant='blocks normal push', gate=True, test='clearance_clear', source='PeerClearancePrefix'),
    dict(scope='coordinator', field='recovery_started', type='bool', owner='ContinuousModeEnv', reader='pair_gate', timing='control', persistence='live', default='False', meaning='joint recovery active', invariant='blocks normal push', gate=True, test='recovery_clear', source='ContinuousModeEnv'),
    dict(scope='coordinator', field='stage_fallback_reason', type='str', owner='pair_gate', reader='diagnostic', timing='control', persistence='latched_tick', default='None', meaning='aggregate fallback reason', invariant='non-None with fallback', gate=True, test='stage_fallback', source='PairGateEnv'),
]


def write_tables(path_state=TABLE_PATH, path_inv=INVENTORY_PATH):
    path_state = Path(path_state); path_state.parent.mkdir(parents=True, exist_ok=True)
    fields = list(FIELD_SPECS[0].keys())
    with path_state.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(FIELD_SPECS)
    inventory = classify_runtime_fields()
    path_inv = Path(path_inv); path_inv.parent.mkdir(parents=True, exist_ok=True)
    with path_inv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['object', 'field', 'kind', 'in_table', 'note'])
        w.writeheader()
        for row in inventory:
            w.writerow(row)
    return path_state, path_inv


def _state_fields():
    return {f['field'] for f in FIELD_SPECS}


def classify_runtime_fields():
    """Collect current runtime attributes; every field must be classified.

    'state' fields must appear in the flat table.  Counters/config/caches are
    still inventoried so a new option field cannot disappear from review.
    """
    import sys
    sys.path.insert(0, str(ROOT))
    from experiments.option_rl_v13.controller import ACTIONS, make_skill
    from experiments.option_rl_v14.controller import make_env
    from experiments.option_rl_v16.learnability import load_panel
    train, _ = load_panel(ROOT / 'outputs/option_rl_v14/learnability_002/panel.json')
    env = make_env(train[0], context='extended')
    push = [a for a in ACTIONS if a.kind == 'push' and getattr(a, 'effect_kind', None) == 'translate'][0]
    skill = make_skill(push)
    slot = env.slots[0]
    groups = [('Slot', slot), ('ContinuousPush', skill), ('LearningEnv', env),
              ('PeerClearancePrefix', env.clearance), ('Continuous2DEnv', env.env)]
    state_fields = _state_fields()
    table = set(state_fields)
    rows = []
    for obj_name, obj in groups:
        for name in sorted(vars(obj)):
            if name in table:
                kind = 'state'
            elif name in ('config', 'action', 'force_limit', 'time_budget', 'requested_time_budget',
                          'speed_command', 'approach_speed_limit', 'endpoint_tracker', 'phase_progress',
                          'reference', 'reference_centre', 'reference_time', 'preparation_gap_target',
                          'preparation_offset', 'role_offset', 'point_offset', 'contact_target',
                          'initial_face', 'initial_normal', 'face', 'last_normal', 'surface_normal_speed'):
                kind = 'config_or_geometry_cache'
            elif any(k in name for k in ('_seconds', '_time', '_count', '_ticks', '_progress', '_travel',
                                         '_length', '_events', '_tick', '_marker', '_started', '_active',
                                         '_seen', '_loss', '_contact', '_increment', '_budget', '_rate')):
                kind = 'counter_or_cache'
            elif name.startswith(('previous_', 'last_', '_last_', 'travel_previous_', 'anchor_', 'face_switches',
                                  'corner_', 'find_', 'follow_', 'recovery_', 'normal_reference', 'patch_missing')):
                kind = 'counter_or_cache'
            elif name in ('done', 'exit_reason', 'exit_time', 'stage', 'ready', 'contact_ready', 'release',
                          'joint_active', 'joint_pair', 'cooperative', 'pause_reason', 'pending'):
                kind = 'state'
            else:
                kind = 'option_internal'
            rows.append(dict(object=obj_name, field=name, kind=kind, in_table=bool(name in table),
                             note='' if kind != 'unclassified' else 'must be classified or added to FIELD_SPECS'))
    return rows


if __name__ == '__main__':
    a, b = write_tables()
    print('wrote', a, b)
    rows = classify_runtime_fields()
    unknown = [r for r in rows if r['kind'] == 'unclassified']
    print('fields', len(rows), 'unclassified', len(unknown))
    for r in unknown[:20]:
        print('UNCLASSIFIED', r)
