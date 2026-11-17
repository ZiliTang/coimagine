"""Cached public projection and persistent hermetic broker, explicit opt-in.

Cache keys are content-derived (including mutations), not observation identities.
Each window rebuilds the exact bounded memory in an isolated context; scoring
queries fork that prepared state. No incremental author-memory approximation and no authored pickle enters the host.
"""
import hashlib
import ast
import json
import os
from pathlib import Path
import pickle
import select
import signal
import struct
import subprocess
import tempfile
import time
from copy import deepcopy
from experiments.planar_tasks.sandbox import policy_command
from experiments.planar_tasks.relational_search.contract import public_observation, validate_action, canonical, digest, content_key

VERSION = 'forked-window-feature-worker-v3'


def validate_source(source):
    """Compatibility gate, NOT the security sandbox (bubblewrap still is).

    Cached functions must be deterministic public-data transforms. Reflective
    runtime/IO dependence belongs to the legacy fresh-interpreter path.
    """
    allowed = {'math','json','collections','itertools','functools','operator','copy','statistics'}
    forbidden = {'open','exec','eval','compile','__import__','globals','locals','vars',
                 'getattr','setattr','delattr','input','breakpoint'}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node,ast.Import) and any(a.name not in allowed for a in node.names):
            raise ValueError('cached runtime requires deterministic public-data imports; use legacy runtime')
        if isinstance(node,ast.ImportFrom) and (node.level or node.module not in allowed):
            raise ValueError('cached runtime requires deterministic public-data imports; use legacy runtime')
        if isinstance(node,ast.Name) and (node.id in forbidden or node.id.startswith('__')):
            raise ValueError('cached runtime disallows reflective/IO dependence; use legacy runtime')
        if isinstance(node,ast.Attribute) and node.attr.startswith('__'):
            raise ValueError('cached runtime disallows reflective/IO dependence; use legacy runtime')


class FastProgram:
    def __init__(self, source, max_names=24, timeout=30., *, typed=False):
        if not isinstance(source, str) or len(source.encode()) > 49152:
            raise ValueError('source at most48KiB')
        if typed:
            from .typed_compiler import validate_source as validate_typed
            validate_typed(source)
        else:validate_source(source)
        if '_PREFIX_MAP' in source or '_COMPLETE_FEATURES' in source:
            raise ValueError('fast runtime requires public author API, not adapter internals')
        self.source, self.max_names, self.timeout = source, max_names, timeout
        self.typed = typed; self.compilation = None
        self.process = None; self.directory = None; self.calls = 0
        self.cache = {}; self.raw_cache = {}; self.remote = set(); self.history_keys = None; self.prepared_for = None

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                # Outer process group, bwrap's die-with-parent kills its namespace.
                try: os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            self.process.wait(); self.process.stdin.close(); self.process.stdout.close()
            self.process = None
        if self.directory is not None:
            self.directory.cleanup(); self.directory = None
        self.remote = set()

    def __del__(self):
        try: self.close()
        except Exception: pass

    def start(self):
        self.close()
        self.directory = tempfile.TemporaryDirectory(prefix='coimagine_feature_broker_')
        path = Path(self.directory.name)/'worker.py'
        artifacts=();extra=''
        if self.typed:
            from .typed_compiler import compile_source
            directory,self.compilation=compile_source(self.source)
            artifacts=((directory/'semantics.so','semantics.so'),)
            extra=', compiled_path="/work/semantics.so"'
        path.write_text(Path(__file__).with_name('feature_worker.py').read_text()+'\nserve('+repr(self.source)+extra+')\n')
        self.process = subprocess.Popen(policy_command(path,artifacts=artifacts), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL, start_new_session=True, bufsize=0)

    def read(self, count, deadline):
        data = bytearray(); fd = self.process.stdout.fileno()
        while len(data) < count:
            remaining = deadline-time.monotonic()
            if remaining <= 0 or not select.select([fd], [], [], remaining)[0]:
                raise TimeoutError('feature worker deadline')
            chunk = os.read(fd, count-len(data))
            if not chunk: raise ValueError('feature worker exited')
            data.extend(chunk)
        return bytes(data)

    def prepare(self, history):
        keys = []; new_cache = {}; new_raw = {}
        for event in history.events:
            key = content_key(event).hex()
            if key not in self.cache:
                projected = dict(observation=public_observation(event['observation']), action=event['action'])
                self.cache[key] = canonical(projected)
                self.raw_cache[key] = canonical(event)
            keys.append(key); new_cache[key] = self.cache[key]; new_raw[key] = self.raw_cache[key]
        self.cache = new_cache; self.raw_cache = new_raw
        if keys != self.history_keys:
            self.history_hash = hashlib.sha256(('['+','.join(new_raw[k] for k in keys)+']').encode()).hexdigest()
            self.history_keys = keys
        self.observation = json.loads(self.cache[keys[-1]])['observation']

    def evaluate(self, history, extras, mapping):
        if self.prepared_for is not history:
            self.prepare(history)
        keys = self.history_keys
        o = self.observation
        supplied = []
        for a in extras:
            validate_action(a, o)
            if a not in supplied: supplied.append(deepcopy(a))
        if len(supplied) > 128: raise ValueError('host candidate limit')
        supplied.sort(key=digest)
        if self.process is None or self.process.poll() is not None or self.calls % 256 == 0:
            self.start()  # bound broker cumulative CPU / lifetime
        request = dict(keys=keys, new=[(k,self.cache[k]) for k in keys if k not in self.remote],
                       candidates=supplied, mapping=mapping, max_names=self.max_names)
        packet = pickle.dumps(request, protocol=4)
        self.calls += 1
        try:
            deadline = time.monotonic()+self.timeout
            # Nonblocking write also bounds a stalled/dead worker on a large history.
            fd=self.process.stdin.fileno();os.set_blocking(fd,False)
            remaining=memoryview(struct.pack('!I',len(packet))+packet)
            while remaining:
                wait=deadline-time.monotonic()
                if wait<=0 or not select.select([], [fd], [], wait)[1]:raise TimeoutError('feature worker write deadline')
                remaining=remaining[os.write(fd,remaining):]
            size=struct.unpack('!I',self.read(4,deadline))[0]
            if size>1024*1024:raise ValueError('feature reply limit')
            result=json.loads(self.read(size,deadline))
            if 'error' in result:raise ValueError('action semantic worker: '+result['error'])
            result=result['result'];self.remote=set(keys)
        except BaseException:
            self.close();raise
        legal=[];invalid=[]
        for a in result['proposals']:
            try:validate_action(a,o)
            except (ValueError,TypeError,KeyError) as exc:invalid.append(dict(action=a,error=str(exc)));continue
            if a not in legal:legal.append(a)
        actions=[];values=[]
        for a,v in zip(result['actions'],result['values']):
            try:validate_action(a,o)
            except (ValueError,TypeError,KeyError):continue
            if a not in actions:actions.append(a);values.append(v)
        raw_first=result['proposals'][0] if result['proposals'] else None
        return dict(names=result['names'],proposals=legal,invalid_proposals=invalid,actions=actions,values=values,
                    raw_first=raw_first,first_legal=raw_first is not None and raw_first in legal)
