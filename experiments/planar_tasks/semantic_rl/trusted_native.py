"""Explicitly trusted, frozen-source execution for training collectors.

This is NOT a sandbox. The normal author runtime stays isolated. A caller must
pin the exact source digest and verify parity before admitting a training source.
C compilation remains isolated; host ABI artifacts have their own cache identity.
"""
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
import json
import marshal
from types import SimpleNamespace
from pathlib import Path
import shutil
import subprocess
import sysconfig
import tempfile
import time
from functools import lru_cache
from .stream_features import StreamProgram
from .typed_compiler import compile_source, sha, ROOT
from experiments.planar_tasks.relational_search.contract import digest

CACHE=ROOT/'outputs/toolchains/semantic-trusted-native-v1'
VERSION='trusted-stream-native-v1'


@lru_cache(maxsize=1)
def host_identity():
    include=Path(sysconfig.get_path('include')).resolve()
    return include,dict(soabi=sysconfig.get_config_var('SOABI'),
        headers={str(p.relative_to(include)):sha(p) for p in sorted(include.rglob('*.h'))},gcc=sha('/usr/bin/gcc'))


def compile_host(source):
    base,original=compile_source(source)
    include,identity=host_identity()
    key=digest(dict(version=VERSION,original=original['key'],host=identity))
    CACHE.mkdir(parents=True,exist_ok=True);dest=CACHE/key
    with (CACHE/(key+'.lock')).open('w') as lock:
        deadline=time.monotonic()+90
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('host native build lock deadline')
                time.sleep(.05)
        if dest.exists():
            manifest=json.loads((dest/'manifest.json').read_text())
            if manifest['key']!=key or manifest['binary_sha256']!=sha(dest/'semantics.so'):
                raise ValueError('trusted native cache integrity mismatch')
            return dest,manifest
        with tempfile.TemporaryDirectory(prefix='build_',dir=CACHE) as temp:
            p=Path(temp)
            command=['prlimit','--as=1073741824','--cpu=60','--fsize=33554432','--',
                'bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
                '--setenv','PATH','/usr/bin:/bin']
            for path in ('/usr','/lib','/lib64'):
                if Path(path).exists():command+=['--ro-bind',path,path]
            command+=['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
                '--ro-bind',str(include),'/include','--ro-bind',str(base/'_coimagine_semantics.c'),'/source.c',
                '--bind',str(p),'/build','--chdir','/build','/usr/bin/gcc','-shared','-fPIC','-O2',
                '-fno-fast-math','-I/include','/source.c','-o','/build/semantics.so']
            started=time.monotonic()
            with (p/'compiler.log').open('wb') as log:
                result=subprocess.run(command,stdout=log,stderr=log,timeout=max(.1,deadline-time.monotonic()),check=False)
            if result.returncode or not (p/'semantics.so').exists():
                raise ValueError('host ABI compilation failed: '+(p/'compiler.log').read_text(errors='replace')[-5000:])
            manifest=dict(key=key,runtime=VERSION,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                host_soabi=identity['soabi'],binary_sha256=sha(p/'semantics.so'),original_build=original,
                compile_wall_seconds=time.monotonic()-started,boundscheck=True,fast_math=False)
            (p/'manifest.json').write_text(json.dumps(manifest,indent=2));shutil.copytree(p,dest)
    return dest,manifest


def owned_copy(value):
    """C-level copy of internal plain data, preserving independent ownership.

    The fallback retains deepcopy semantics for non-plain optional extensions.
    This is not an author-data deserializer: bytes originate here in the host.
    """
    try:return marshal.loads(marshal.dumps(value))
    except (ValueError,TypeError):return deepcopy(value)


def worker_adapter():
    # Reuse the exact trusted validation/packing code of the isolated worker.
    # This executes repository-owned Python only; authored code is the pinned .so.
    from . import feature_worker
    namespace=dict(feature_worker.__dict__)
    # Do not patch the process-global copy module. update_memory and
    # prepare_context still receive distinct event/observation trees.
    namespace['copy']=SimpleNamespace(deepcopy=owned_copy)
    exec(compile(Path(__file__).with_name('stream_worker.py').read_text(),'<trusted stream adapter>','exec'),namespace)
    return namespace['stream_request']


class TrustedStreamProgram(StreamProgram):
    def __init__(self,source,*,trusted_source_sha256,max_names=64):
        if trusted_source_sha256!=digest(source):raise ValueError('explicit matching trusted source digest required')
        super().__init__(source,max_names=max_names)
        self.native_module=None;self.adapter=worker_adapter()

    def start(self):
        directory,self.compilation=compile_host(self.source)
        spec=importlib.util.spec_from_file_location('_coimagine_semantics',directory/'semantics.so')
        self.native_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.native_module)

    def request(self,value):
        if self.native_module is None:self.start()
        self.calls+=1
        # Preserve transport ownership/validation semantics. Copy one current
        # event or small numeric request, never an old raw history window.
        result=self.adapter(self.native_module.__dict__,owned_copy(value))
        return json.loads(json.dumps(result,allow_nan=False))
