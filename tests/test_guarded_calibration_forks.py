from pathlib import Path

import pytest

from tools import cap_guarded_calibration_forks as forks


def test_preserve_pusher_and_peer_but_request_perpendicular_face():
    request = dict(a0=46, a1=31, ticks=12, target_id='box')
    follow = forks.new_face(request, robot=1)
    assert follow == dict(a0=46, a1=5, ticks=24, target_id='box')
    assert request['a1'] == 31
    assert request['ticks'] == 12


def test_particle_rejects_non_development_room_before_read(monkeypatch):
    monkeypatch.setattr(forks, 'development_rooms', lambda: {'dev1': {}})
    monkeypatch.setattr(forks.archive, 'read', lambda path: pytest.fail(str(path)))
    with pytest.raises(ValueError, match='development'):
        forks.simulate_development_particle('held1', 'early', .10)


def test_particle_reads_only_public_anchor_index_and_never_runs_suffix(monkeypatch):
    fixture = {'scene_id': 'dev1'}
    index = dict(collector_sha256='test', anchors=[dict(anchor_id='late')])
    paths = []
    def read(path):
        paths.append(path)
        return index
    monkeypatch.setattr(forks, 'development_rooms', lambda: {'dev1': fixture})
    monkeypatch.setattr(forks.archive, 'read', read)
    monkeypatch.setattr(forks, 'record_for', lambda split, room: (split, room))
    monkeypatch.setattr(forks, 'frozen_hash', lambda: 'test')
    monkeypatch.setattr(forks, 'OUT', Path('/results'))
    def evaluate(scene, record, anchor, progress_m, *, execute_followup):
        assert scene is fixture and record == ('dev', 'dev1')
        assert anchor is index['anchors'][0] and progress_m == .2
        assert execute_followup is False
        return {'followup_ticks': 0}
    monkeypatch.setattr(forks, 'evaluate', evaluate)
    assert forks.simulate_development_particle('dev1', 'late', .20) == {'followup_ticks': 0}
    assert paths == [Path('/results/particles/dev/dev1.json')]


def test_particle_does_not_execute_suffix_or_observe_its_result(monkeypatch):
    class FakeWorld:
        class env:
            total_periods = 107
    request = dict(a0=1, a1=46, ticks=40, target_id='box')
    FakeWorld.env.termination = None
    FakeWorld.decision_count = 0
    FakeWorld.history = type('History', (), {'events':[{'observation':{}}]})()
    monkeypatch.setattr(forks.push_phases, 'contact_state', lambda *args: 'ready')
    monkeypatch.setattr(forks.joint_face_change, 'next_step', lambda *args: 'orbit')
    assert forks.followup(FakeWorld(), request, execute=False) == dict(
        endpoint_followup_available=True, followup_ticks=0)


def test_terminal_endpoint_credits_neither_availability_nor_completion(monkeypatch):
    class FakeWorld:
        class env:
            total_periods = 107
    FakeWorld.env.termination = 'budget'
    request = dict(a0=1, a1=46, ticks=40, target_id='box')
    result = forks.followup(FakeWorld(), request, execute=True)
    assert not result['endpoint_followup_available']
    assert not result['followup_completed']
    assert result['followup_ticks'] == 0 and result['followup_reason'] == 'terminated'


def test_new_face_alignment_scopes_continuing_and_pending_pushes(monkeypatch):
    from types import SimpleNamespace
    from experiments.planar_tasks.semantic_rl import push_phases

    def original(*args):
        return None

    def factory(action):
        return SimpleNamespace(_allocated_navigate=original)

    slot = SimpleNamespace(factory=factory, skill=factory(None))
    old = slot.skill
    other = SimpleNamespace(factory=factory, skill=factory(None))
    world = SimpleNamespace(env=SimpleNamespace(slots=[slot, other]))

    def failed_contact(w, req, budget):
        assert old._allocated_navigate is not original
        slot.skill = slot.factory(None)
        assert slot.skill._allocated_navigate is not original
        assert other.skill._allocated_navigate is original and other.factory is factory
        raise ValueError('interrupted')

    monkeypatch.setattr(push_phases, 'contact', failed_contact)
    with pytest.raises(ValueError, match='interrupted'):
        push_phases.contact_on_new_face(world, dict(target_id='box', a0=7, a1=44, ticks=24), 24)
    assert slot.factory is factory
    assert old._allocated_navigate is original and slot.skill._allocated_navigate is original
