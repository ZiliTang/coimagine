"""Synthetic host tool tests: no model calls and no physics stepping."""
from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from experiments.planar_tasks.baseline import session as module
from experiments.planar_tasks.baseline.config import Budget, validate_config


class SyntheticWorld:
    def __init__(self, instance):
        self.instance = instance
        self.decisions = 0
        self.calls = []
        self._env = SimpleNamespace(termination=None, success=False, joint_step=self.tick,
                                    hidden_truth='PRIVATE_GEOMETRY', reward=999)

    def tick(self, a0, a1, record):
        self.calls.append((a0, a1, record))

    def observe(self):
        robot = dict(arena=[4, 3], own={'x': 1, 'y': 1}, goal_xy=[3, 2], goal_radius=.2,
                     scan=dict(origin=[1, 1], ray_angles=[0], distances=[.5], valid=[True], category=[1]))
        return {'robots': [deepcopy(robot), deepcopy(robot)], 'legal_pairs': [[1, 2]],
                'decision': self.decisions}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    release = tmp_path / 'release.json'
    release.write_text('{}')
    monkeypatch.setattr(module.seal, 'verify_release', lambda path: {'development_tasks': {'alpha': 'hash', 'beta': 'hash'}})
    monkeypatch.setattr(module, 'task_catalog', lambda: {
        'alpha': SimpleNamespace(instance='selected-alpha'), 'beta': SimpleNamespace(instance='selected-beta')})
    budget = {f.name: 100 for f in fields(Budget)}
    budget.update(wall_seconds=60., workspace_bytes=1000000, output_bytes=100000,
                  control_ticks=5, python_total_seconds=10., python_call_seconds=2.)
    config = {'model': {'provider': 'synthetic', 'name': 'unused', 'settings': {},
                       'source_access': 'hidden', 'accounting': 'reserved_output_tokens_and_input_bytes'},
              'budget': budget, 'development_tasks': ['alpha', 'beta'], 'purpose': 'infrastructure_smoke'}
    session = module.Session.create(tmp_path / 'session', config, release_path=release, world_factory=SyntheticWorld)
    yield session
    session.close()


def rpc(campaign, **request):
    return campaign.rpc(request, time.monotonic() + 5)


def test_agent_selects_reset_and_ticks_stop_exactly(campaign):
    reply = rpc(campaign, method='reset', task='beta')
    assert campaign.world.instance == 'selected-beta'
    assert reply['success'] is None
    with pytest.raises(ValueError, match='control_ticks budget'):
        rpc(campaign, method='step', action={'a0': 1, 'a1': 2, 'ticks': 7})
    assert len(campaign.world.calls) == 5
    assert campaign.state['usage']['control_ticks'] == 5
    assert campaign.state['episodes'][0]['ticks'] == 5
    assert all(call == (1, 2, True) for call in campaign.world.calls)


def test_illegal_action_has_no_controller_tick_or_fallback(campaign):
    rpc(campaign, method='reset', task='alpha')
    for action in ({'a0': 9, 'a1': 8, 'ticks': 1}, {'a0': True, 'a1': 2, 'ticks': 1},
                   {'a0': 1, 'a1': 2, 'ticks': 41}):
        with pytest.raises(ValueError):
            rpc(campaign, method='step', action=action)
    assert campaign.world.calls == []
    assert campaign.state['usage'].get('control_ticks', 0) == 0


def test_only_legal_observation_and_render(campaign):
    result = rpc(campaign, method='reset', task='alpha')
    svg = rpc(campaign, method='render')
    text = json.dumps(result) + svg
    assert 'PRIVATE_GEOMETRY' not in text and 'reward' not in text and 'hidden_truth' not in text
    assert '<svg' in svg and '1.5' in svg
    # Privileged changes cannot alter a rendering from fixed legal observations.
    campaign.world._env.hidden_truth = 'OTHER_SHAPE'
    assert rpc(campaign, method='render') == svg
    for method in ('truth', 'reward', 'environment'):
        with pytest.raises(ValueError, match='unknown RPC'):
            rpc(campaign, method=method)


def test_file_path_escape_and_symlink(campaign, tmp_path):
    outside = tmp_path / 'outside.txt'; outside.write_text('secret')
    (campaign.workspace / 'link').symlink_to(tmp_path, target_is_directory=True)
    for path in ('../outside.txt', str(outside), 'link/outside.txt', './policy.py', 'dir//file'):
        result = campaign.tool('read_file', {'path': path})
        assert result['ok'] is False
    result = campaign.tool('write_file', {'path': 'link/outside.txt', 'text': 'overwrite'})
    assert result['ok'] is False and outside.read_text() == 'secret'


def test_test_archive_does_not_choose_submission(campaign, monkeypatch):
    selected = 'class Policy:\n    def __init__(self, contract): pass\n    def act(self, obs): return dict(a0=1,a1=2,ticks=1)\n# explicitly chosen\n'
    campaign.write_file('policy.py', 'best-on-development')
    calls = []
    def fake_episode(task, command, destination, budget, **kwargs):
        calls.append(kwargs)
        assert kwargs['max_control_ticks'] == 3
        assert kwargs['security_sandbox']['development_rpc'] is False
        assert '/home/' not in ' '.join(command[command.index('--ro-bind') + 2:])
        return {'success': True, 'executed_control_ticks': 3, 'reward': 99, 'hidden_truth': 'secret'}
    monkeypatch.setattr(module, 'run_episode', fake_episode)
    result = campaign.tool('test_policy', {'files': ['policy.py'], 'task': 'alpha', 'max_ticks': 3})
    assert result['ok'], result
    assert 'reward' not in result['result']['summary']
    assert not (campaign.path / 'frozen').exists()
    archived = campaign.path / 'private/test_0000/package/policy.py'
    assert archived.read_text() == 'best-on-development'
    campaign.write_file('policy.py', selected)
    submitted = campaign.tool('submit', {'files': ['policy.py']})
    assert submitted['ok'], submitted
    frozen = campaign.path / 'frozen/package/policy.py'
    assert frozen.read_text() == selected
    manifest = json.loads((campaign.path / 'frozen/manifest.json').read_text())
    assert manifest['selection'] == 'agent_explicit_submission'
    assert manifest['files_sha256']['policy.py'] == module.seal.digest(frozen)
    campaign.write_file('policy.py', 'workspace changed after freeze')
    assert frozen.read_text() == selected
    assert archived.read_text() == 'best-on-development'
    for name, arguments in [('submit', {'files': ['policy.py']}), ('list_files', {}),
                            ('run_python', {'path': 'policy.py', 'seconds': 1})]:
        assert campaign.tool(name, arguments)['ok'] is False
    assert len(calls) == 1


def test_budget_persists_and_config_tampering_rejected(campaign):
    rpc(campaign, method='reset', task='alpha')
    rpc(campaign, method='step', action={'a0': 1, 'a1': 2, 'ticks': 2})
    before = deepcopy(campaign.state['usage'])
    campaign.close()
    reopened = module.Session(campaign.path, release_path=campaign.release_path, world_factory=SyntheticWorld)
    try:
        assert reopened.state['usage'] == before
        with pytest.raises(ValueError, match='reset required'):
            rpc(reopened, method='observe')
        config = json.loads((campaign.path / 'session.json').read_text())
        config['budget']['control_ticks'] = 999
        (campaign.path / 'session.json').write_text(json.dumps(config))
        with pytest.raises(ValueError, match='config changed'):
            reopened._active()
    finally:
        reopened.close()


def test_missing_budget_configuration_rejected(campaign):
    config = deepcopy(campaign.config)
    del config['budget']['control_ticks']
    with pytest.raises(TypeError):
        validate_config(config)
    del config['model']
    with pytest.raises(ValueError):
        validate_config(config)
