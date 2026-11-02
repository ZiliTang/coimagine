import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from experiments.option_rl_v19 import testbed_server as server


@pytest.fixture
def store(tmp_path, monkeypatch):
    runs = tmp_path / 'runs'; runs.mkdir()
    monkeypatch.setattr(server, 'RUNS', runs)
    monkeypatch.setattr(server, 'RUN_INDEX', tmp_path / 'run_index.json')
    monkeypatch.setattr(server, '_RUN_INDEX_CACHE', None)
    return runs


def write_run(runs, name, periods=91):
    run = dict(id=name, name=name, scene='approach_0', success=False, periods=periods,
               created=1., sequence=[{}], frames=[{'large_data': 'x'*1000}])
    path = runs / f'{name}.json'; path.write_text(json.dumps(run))
    return path, run


def test_warm_and_persisted_index_do_not_decode_unchanged_trajectories(store, monkeypatch):
    path, _ = write_run(store, 'one')
    expected = server._run_list()
    original = Path.read_text
    def guarded(p, *args, **kwargs):
        assert p != path, 'history trajectory was reread'
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    assert server._run_list() == expected
    server._RUN_INDEX_CACHE = None
    assert server._run_list() == expected


def test_modified_or_deleted_run_invalidates_only_its_metadata(store):
    path, _ = write_run(store, 'one')
    assert server._run_list()[0]['periods'] == 91
    write_run(store, 'one', periods=12345)
    assert server._run_list()[0]['periods'] == 12345
    path.unlink()
    assert server._run_list() == []
    assert json.loads(server.RUN_INDEX.read_text())['runs'] == {}


def test_new_run_metadata_is_available_without_rereading_frames(store, monkeypatch):
    path, run = write_run(store, 'new')
    server._index_run(path, run)
    original = Path.read_text
    def guarded(p, *args, **kwargs):
        assert p != path
        return original(p, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', guarded)
    assert server._run_list()[0]['id'] == 'new'


def test_corrupt_index_rebuilds_and_invalid_run_is_skipped(store):
    write_run(store, 'good')
    server.RUN_INDEX.write_text('broken json')
    (store/'bad.json').write_text('[]')
    assert [r['id'] for r in server._run_list()] == ['good']


def test_concurrent_new_entries_survive_persistent_cache_reload(store):
    entries = [write_run(store, f'run{i}') for i in range(6)]
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(lambda pair: server._index_run(*pair), entries))
    server._RUN_INDEX_CACHE = None
    assert {r['id'] for r in server._run_list()} == {f'run{i}' for i in range(6)}
