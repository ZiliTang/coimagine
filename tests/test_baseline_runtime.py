"""Synthetic execution-boundary checks; no simulator or model API calls."""
from pathlib import Path
import shutil
import time

import pytest

from experiments.planar_tasks.baseline.runtime import run_script


@pytest.fixture
def work(tmp_path):
    (tmp_path / "check.py").write_text("print('ready')")
    result = run_script(tmp_path, "check.py", lambda request: {}, 3)
    if result["status"] != "ok":
        message = result.get("error", "") + result["stderr"]
        unavailable = ("Operation not permitted", "No permissions", "Creating new namespace failed")
        if not shutil.which("bwrap") or any(x in message for x in unavailable):
            pytest.skip("bwrap namespaces unavailable: " + message)
        pytest.fail(message)
    return tmp_path


def run(work, source, handler=lambda request: {}, timeout=3, **kwargs):
    (work / "script.py").write_text(source)
    return run_script(work, "script.py", handler, timeout, **kwargs)


def test_rpc_and_normal_output(work):
    source = Path("experiments/planar_tasks/baseline/client.py")
    shutil.copyfile(source, work / "client.py")
    calls = []
    def handler(request):
        calls.append(request)
        return "<svg/>" if request["method"] == "render" else {"count": len(calls)}
    result = run(work, "from client import DevelopmentEnv\ne=DevelopmentEnv()\nprint('hello')\nprint(e.reset('demo'))\nprint(e.step({'move':1}))\nprint(e.observe())\nprint(e.render())\n", handler)
    assert result["status"] == "ok", result
    assert result["rpc_calls"] == 4
    assert [x["method"] for x in calls] == ["reset", "step", "observe", "render"]
    assert "hello" in result["stdout"] and "<svg/>" in result["stdout"]
    assert "__COIMAGINE_RPC__" not in result["stdout"]


def test_isolation(work, monkeypatch):
    monkeypatch.setenv("COIMAGINE_SECRET_TEST", "secret")
    source = '''import os, socket
assert 'COIMAGINE_SECRET_TEST' not in os.environ
assert not os.path.exists('/home/zilitang')
assert not os.path.exists('/home/zilitang/repo/coimagine')
open('/work/written', 'w').write('yes')
try:
    socket.create_connection(('127.0.0.1', 9), timeout=.1)
except OSError:
    pass
else:
    raise AssertionError('loopback unexpectedly connected')
print('isolated')
'''
    result = run(work, source)
    assert result["status"] == "ok", result
    assert (work / "written").read_text() == "yes"


def test_timeout_and_output_limit(work):
    assert run(work, "while True: pass", timeout=.15)["status"] == "timeout"
    result = run(work, "import os\nwhile True: os.write(2,b'x'*4096)", max_output_bytes=1024)
    assert result["status"] == "output_limit"
    stdout_result = run(work, "import os\nos.write(1,b'x'*2048)\nwhile True: pass", max_output_bytes=1024)
    assert stdout_result["status"] == "output_limit"
    assert len(result["stdout"].encode()) + len(result["stderr"].encode()) <= 1024


def test_rpc_error_and_handler_deadline(work):
    result = run(work, "print('__COIMAGINE_RPC__bad', flush=True)")
    assert result["status"] == "error" and "JSON" in result["error"]
    def delayed(request):
        time.sleep(.15)
        return {}
    result = run(work, "print('__COIMAGINE_RPC__{}',flush=True)\ninput()", delayed, timeout=.1)
    assert result["status"] == "timeout"


def test_path_escape(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "_outside.py")
    outside.write_text("raise AssertionError('never executed')")
    (tmp_path / "escape.py").symlink_to(outside)
    for name in ("../outside.py", "/etc/passwd", "escape.py", "./escape.py"):
        result = run_script(tmp_path, name, lambda request: {}, 1)
        assert result["status"] == "error" and result["exit_code"] is None
