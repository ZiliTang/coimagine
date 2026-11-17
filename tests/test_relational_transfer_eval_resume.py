from experiments.planar_tasks.authoring_v2 import relational_transfer_eval_resume as R


def test_worker_returns_only_summary_after_persisting_complete_result(monkeypatch,tmp_path):
    result=dict(summary=dict(success=True,ticks=12),fixture_sha256='fixture',source_sha256='source',replay={'large':'artifact'},public_rows=[{'observation':1}])
    monkeypatch.setattr(R.R,'worker',lambda job:result)
    reply=R.durable_summary_worker(dict(artifact_path=str(tmp_path),case_id='case',condition='G'))
    assert set(reply)=={'summary','fixture_sha256','source_sha256'}
    assert R.S.unpack(tmp_path/'result.json.gz')==result
    assert (tmp_path/'worker_started.json').exists()
