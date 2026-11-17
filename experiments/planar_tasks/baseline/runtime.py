"""Linux execution boundary for development scripts (no host Python imports).

Per-file limits and an ephemeral tmpfs are safeguards, not an aggregate disk quota.
Handlers are synchronous trusted host code and must enforce their own deadline.
The runtime checks the deadline immediately after each handler returns.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path
import selectors
import shutil
import signal
import subprocess
import time
from typing import Callable

RPC_PREFIX = b"__COIMAGINE_RPC__"
MAX_MESSAGE_BYTES = 1024 * 1024


def run_script(workspace: Path, script: str, handler: Callable[[dict], dict],
               timeout: float, max_output_bytes: int = 1048576) -> dict:
    started = time.monotonic()
    result = dict(status="error", exit_code=None, stdout="", stderr="",
                  wall_seconds=0.0, rpc_calls=0)
    proc = None
    selector = selectors.DefaultSelector()
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        if not math.isfinite(timeout) or timeout <= 0 or max_output_bytes <= 0:
            raise ValueError("timeout and max_output_bytes must be positive")
        workspace = Path(workspace)
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("workspace must be a real directory")
        workspace = workspace.resolve()
        relative = Path(script)
        if relative.is_absolute() or not relative.parts or any(p in ("", "..", ".") for p in script.split("/")):
            raise ValueError("script must be a normalized relative file path")
        target = (workspace / relative).resolve()
        if not target.is_relative_to(workspace) or not target.is_file():
            raise ValueError("script must resolve to a file within workspace")
        bwrap, prlimit = shutil.which("bwrap"), shutil.which("prlimit")
        if not bwrap or not prlimit:
            raise RuntimeError("bwrap and prlimit are required; no unsandboxed fallback")
        command = [prlimit, f"--cpu={max(1, math.ceil(timeout))}", "--as=536870912",
                   "--fsize=16777216", "--nofile=64", "--", bwrap,
                   "--unshare-all", "--die-with-parent", "--new-session", "--cap-drop", "ALL", "--clearenv"]
        for directory in ("/usr", "/lib", "/lib64"):
            if Path(directory).exists():
                command += ["--ro-bind", directory, directory]
        command += ["--proc", "/proc", "--dev", "/dev",
                    "--tmpfs", "/tmp", "--bind", str(workspace), "/work",
                    "--chdir", "/work", "--setenv", "PATH", "/usr/bin:/bin",
                    "--setenv", "PYTHONDONTWRITEBYTECODE", "1", "/usr/bin/python3", "-u", str(relative)]
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, start_new_session=True)
        for stream, name in ((proc.stdout, "stdout"), (proc.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, name)
        os.set_blocking(proc.stdin.fileno(), False)
        pending = bytearray()
        line = bytearray()
        busy = False
        status = None

        def invoke(request):
            try:
                response = {"ok": True, "result": handler(request)}
                data = json.dumps(response, allow_nan=False).encode() + b"\n"
            except Exception as exc:
                data = json.dumps({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}).encode() + b"\n"
            return data

        def capture(name, data):
            nonlocal status
            remaining = max_output_bytes - sum(map(len, captured.values()))
            captured[name].extend(data[:max(0, remaining)])
            if len(data) > remaining:
                status = "output_limit"

        def process_line(data):
            nonlocal busy, status
            if not data.startswith(RPC_PREFIX):
                capture("stdout", data)
                return
            if busy:
                raise ValueError("RPC requests must be synchronous")
            request = json.loads(data[len(RPC_PREFIX):])
            if not isinstance(request, dict):
                raise ValueError("RPC request must be a JSON object")
            result["rpc_calls"] += 1
            busy = True
            response = invoke(request)
            if time.monotonic() - started >= timeout:
                status = "timeout"
                return
            if len(response) > MAX_MESSAGE_BYTES:
                raise ValueError("RPC response exceeds message limit")
            pending.extend(response)
            selector.register(proc.stdin, selectors.EVENT_WRITE, "stdin")
            busy = False

        while status is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                status = "timeout"
                break
            if proc.poll() is not None and not selector.get_map():
                if line:
                    process_line(bytes(line))
                    line.clear()
                status = "ok" if proc.returncode == 0 else "error"
                break
            for key, _ in selector.select(min(0.02, remaining)):
                if key.data == "stdin":
                    try:
                        count = os.write(key.fd, pending)
                    except BrokenPipeError:
                        selector.unregister(key.fileobj)
                        pending.clear()
                        continue
                    del pending[:count]
                    if not pending:
                        selector.unregister(key.fileobj)
                    continue
                data = os.read(key.fd, 65536)
                if not data:
                    selector.unregister(key.fileobj)
                    continue
                if key.data == "stderr":
                    capture("stderr", data)
                    continue
                line.extend(data)
                while b"\n" in line and status is None:
                    end = line.index(b"\n") + 1
                    chunk = bytes(line[:end])
                    del line[:end]
                    if chunk.startswith(RPC_PREFIX) and len(chunk) > MAX_MESSAGE_BYTES:
                        raise ValueError("RPC request exceeds message limit")
                    process_line(chunk)
                if (line and not RPC_PREFIX.startswith(line) and not line.startswith(RPC_PREFIX)
                        and len(line) > max_output_bytes - sum(map(len, captured.values()))):
                    capture("stdout", bytes(line))
                    line.clear()
                if len(line) > MAX_MESSAGE_BYTES:
                    if line.startswith(RPC_PREFIX):
                        raise ValueError("RPC request exceeds message limit")
                    capture("stdout", bytes(line))
                    line.clear()
        result["status"] = status
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if proc is not None:
            # Always kill descendants, including after the entry script exits.
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait()
            result["exit_code"] = proc.returncode
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
        selector.close()
        result["stdout"] = captured["stdout"].decode("utf-8", errors="replace")
        result["stderr"] = captured["stderr"].decode("utf-8", errors="replace")
        result["wall_seconds"] = time.monotonic() - started
    return result
