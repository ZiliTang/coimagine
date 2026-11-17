"""Bounded private IPC to heuristic worker. Prediction RPC remains in the parent."""
import json
import os
import selectors
import subprocess
import sys
import time


class HookClient:
    def __init__(self):
        self.process=None;self.pending=b'';self.failed=None;self.stats={}

    def request(self,payload,deadline):
        if self.failed:raise RuntimeError('heuristic worker disabled: '+self.failed)
        if self.process is None:
            self.process=subprocess.Popen([sys.executable,'-u','hook_worker.py'],stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,stderr=sys.stderr,bufsize=0)
            os.set_blocking(self.process.stdin.fileno(),False);os.set_blocking(self.process.stdout.fileno(),False)
        data=memoryview((json.dumps(payload,allow_nan=False)+'\n').encode())
        try:
            with selectors.DefaultSelector() as sel:
                sel.register(self.process.stdin,selectors.EVENT_WRITE);sel.register(self.process.stdout,selectors.EVENT_READ)
                while True:
                    if not data and b'\n' in self.pending:
                        line,self.pending=self.pending.split(b'\n',1);reply=json.loads(line)
                        if not reply['ok']:raise ValueError(str(reply['error']))
                        self.stats=reply.get('resources',{});return reply['result']
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise TimeoutError('heuristic worker deadline')
                    for key,_ in sel.select(remaining):
                        if key.fileobj is self.process.stdin:
                            try:n=os.write(key.fd,data)
                            except BlockingIOError:continue
                            data=data[n:]
                            if not data:sel.unregister(self.process.stdin)
                        else:
                            chunk=os.read(key.fd,65536)
                            if not chunk:raise RuntimeError('heuristic worker exited')
                            self.pending+=chunk
                            if len(self.pending)>1048576:raise ValueError('heuristic response limit')
        except Exception as exc:
            self.failed=str(exc);self.close();raise

    def begin(self,context,deadline):return self.request(dict(method='begin',context=context),deadline)
    def register(self,node_id,node,deadline):return self.request(dict(method='register',id=node_id,node=node),deadline)
    def rank(self,node_id,candidates,deadline):return self.request(dict(method='rank',id=node_id,candidates=candidates),deadline)
    def close(self):
        if self.process is not None:
            if self.process.poll() is None:self.process.kill()
            self.process.wait()
            self.process.stdin.close();self.process.stdout.close();self.process=None
