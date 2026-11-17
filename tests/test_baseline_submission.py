"""Atomic packaging and isolated interface checks, without physical rollouts."""
import json
from dataclasses import replace
import pytest
from test_baseline_session import campaign
from test_baseline_budget import POLICY,enable_reserve


def test_atomic_submission_uses_explicit_helpers_and_does_not_change_workspace(campaign):
    campaign.write_file('policy.py','old workspace version')
    campaign.write_file('helper.py','VALUE=44\n')
    source='from helper import VALUE\n'+POLICY.replace('a0=44','a0=VALUE')
    result=campaign.tool('submit',dict(files=['policy.py','helper.py'],contents={'policy.py':source}))
    assert result['ok'],result
    assert (campaign.workspace/'policy.py').read_text()=='old workspace version'
    assert (campaign.path/'frozen/package/policy.py').read_text()==source
    manifest=json.loads((campaign.path/'frozen/validation.json').read_text())
    assert manifest['status']=='passed' and manifest['physics_evaluated'] is False
    assert campaign.world is None


@pytest.mark.parametrize('source',[
 'class Policy(: pass',
 'from absent_helper import Policy',
 'class Policy: pass',
 'class Policy:\n def __init__(self,contract): pass\n def act(self,a,b): pass\n',
 'class Policy:\n def __init__(self,contract): raise RuntimeError("constructor failure")\n def act(self,obs): pass\n',
])
def test_invalid_program_is_not_frozen_and_can_be_corrected(campaign,source):
    failed=campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':source}))
    assert not failed['ok'] and not (campaign.path/'frozen').exists()
    assert campaign.state['status']=='developing'
    assert campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':POLICY}))['ok']
    assert campaign.state['submission_attempts']==2
    assert (campaign.path/'private/submission_000000/package/policy.py').read_text()==source


def test_validation_is_bounded_and_does_not_call_act(campaign):
    campaign.budget=replace(campaign.budget,response_seconds=.15)
    source='class Policy:\n def __init__(self,contract):\n  while True: pass\n def act(self,obs): pass\n'
    denied=campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':source}))
    assert not denied['ok'] and 'TimeoutError' in denied['error']['message']
    assert not (campaign.path/'frozen').exists()
    source='class Policy:\n def __init__(self,contract): pass\n def act(self,obs): raise RuntimeError("no physics feedback in validation")\n'
    assert campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':source}))['ok']


def test_inline_submission_cannot_add_unlisted_reserved_or_escape_files(campaign):
    for files,contents in [(['policy.py'],{'policy.py':POLICY,'extra.py':'pass'}),
        (['policy.py','../escape.py'],{'policy.py':POLICY,'../escape.py':'pass'}),
        (['policy.py','client.py'],{'policy.py':POLICY,'client.py':'pass'})]:
        assert not campaign.tool('submit',dict(files=files,contents=contents))['ok']
        assert not (campaign.path/'frozen').exists()
    assert not (campaign.path/'private/escape.py').exists()


def test_validation_cannot_read_host_or_use_development_rpc(campaign):
    source='class Policy:\n def __init__(self, contract): open("'+str(campaign.path/'session.json')+'")\n def act(self, obs): pass\n'
    result=campaign.tool('submit',dict(files=['policy.py'],contents={'policy.py':source}))
    assert not result['ok'] and 'FileNotFoundError' in result['error']['message']
    assert campaign.world is None
