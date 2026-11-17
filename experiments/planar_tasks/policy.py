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


class JsonlPolicy:
    def __init__(self, command, stderr_path, *, rpc_handler=None):
        self.rpc_handler = rpc_handler
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


def main():
    """Protocol fixtures, NOT research baselines or synthesized policies."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--rotate-smoke', action='store_true')
    args = parser.parse_args()
    for line in sys.stdin:
        request = json.loads(line)
        pair, ticks = [44, 44], 4
        if args.rotate_smoke:
            # One fixed CW action; if currently unavailable, brake. No truth access.
            if [36, 36] in request['observation']['legal_pairs']:
                pair, ticks = [36, 36], 20
        print(json.dumps(dict(a0=pair[0], a1=pair[1], ticks=ticks)), flush=True)


if __name__ == '__main__':
    main()
