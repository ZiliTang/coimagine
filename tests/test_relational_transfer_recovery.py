from experiments.planar_tasks.authoring_v2 import relational_transfer_recovery as R


def test_only_explicit_cap_legality_failures_are_task_failures():
    assert R.policy_failure(dict(error=dict(type='ValueError',message='CAP first recommendation missing/illegal')))
    assert not R.policy_failure(dict(error=dict(type='ValueError',message='feature protocol mismatch')))
    assert not R.policy_failure(dict(error=dict(type='TimeoutError',message='worker budget')))
    assert not R.policy_failure(dict(error=None))


def test_worker_saves_before_returning_to_collector(monkeypatch,tmp_path):
    result=dict(summary=dict(success=False,error=dict(type='ValueError',message='CAP first recommendation missing/illegal')),public_rows=[{'observed':1}])
    monkeypatch.setattr(R.R,'worker',lambda job:result)
    returned=R.durable_worker(dict(artifact_path=str(tmp_path)))
    assert R.S.unpack(tmp_path/'result.json.gz')==returned==result
