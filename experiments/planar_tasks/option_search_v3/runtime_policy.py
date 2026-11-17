"""Persistent JSONL policy process, with bounded I/O and no implicit fallback.

This is an IPC boundary, NOT a security sandbox. Only run reviewed local programs.
An actual blind coding-agent experiment needs filesystem/network isolation too.
"""
import json
import os
import selectors
import signal
import subprocess
import time
import resource


class JsonlPolicy:
    def __init__(self, command, stderr_path, *, rpc_handler=None):
        self.rpc_handler = rpc_handler
        self.start_usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        self.exit_info=None
        self.stderr = open(stderr_path, 'xb')
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=self.stderr, start_new_session=True, bufsize=0)
        except BaseException:
            self.stderr.close()
            raise
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.pending = b''
        self.last_response = None

    def request(self, message, timeout):
        self.last_response = None
        payload = memoryview((json.dumps(message, allow_nan=False)+'\n').encode())
        deadline = time.monotonic()+timeout
        with selectors.DefaultSelector() as sel:
            sel.register(self.process.stdin, selectors.EVENT_WRITE)
            sel.register(self.process.stdout, selectors.EVENT_READ)
            while True:
                if b'\n' in self.pending and not payload:
                    line, self.pending = self.pending.split(b'\n', 1)
                    if line.startswith(b'__COIMAGINE_RPC__'):
                        try:
                            if self.rpc_handler is None:
                                raise ValueError('runtime prediction service unavailable')
                            result = self.rpc_handler(json.loads(line[len(b'__COIMAGINE_RPC__'):]), deadline)
                            reply = dict(ok=True, result=result)
                        except Exception as exc:
                            reply = dict(ok=False, error=dict(type=type(exc).__name__, message=str(exc)))
                        payload = memoryview((json.dumps(reply, allow_nan=False)+'\n').encode())
                        sel.register(self.process.stdin, selectors.EVENT_WRITE)
                        continue
                    self.last_response = line.decode('utf-8')
                    return json.loads(self.last_response)
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('policy response deadline exceeded')
                for key, _ in sel.select(remaining):
                    if key.fileobj is self.process.stdin:
                        try:
                            count = os.write(key.fd, payload)
                        except BlockingIOError:
                            continue
                        payload = payload[count:]
                        if not payload:
                            sel.unregister(self.process.stdin)
                    else:
                        try:
                            chunk = os.read(key.fd, 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            raise RuntimeError('policy exited without a complete response')
                        self.pending += chunk
                        if len(self.pending) > 1024*1024:
                            raise ValueError('policy response exceeds 1 MiB')

    def close(self):
        harness_termination=self.process.poll() is None
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self.process.wait(timeout=1.)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=1.)
        self.process.stdin.close()
        self.process.stdout.close()
        self.stderr.close()
        usage=resource.getrusage(resource.RUSAGE_CHILDREN)
        code=self.process.returncode
        self.exit_info=dict(returncode=code,signal=(-code if code is not None and code<0 else None),harness_termination=harness_termination,waited_children_user_seconds=usage.ru_utime-self.start_usage.ru_utime,waited_children_system_seconds=usage.ru_stime-self.start_usage.ru_stime,attribution='waited process tree delta; signal alone does not identify CPU/OOM cause')

