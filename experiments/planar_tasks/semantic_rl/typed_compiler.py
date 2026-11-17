"""Versioned pure-Python Cython contract and hermetic, content-addressed build.

The AST gate is an interface/compatibility check, not a native-code sandbox.
Both compilation and loading run under bubblewrap. Never import an authored
extension into the simulator, collector, or optimizer process.
"""
import ast
import fcntl
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time

API = 'typed-batch-v1'
STREAM_API = 'typed-stream-v1'
ROOT = Path(__file__).resolve().parents[3]
TOOLCHAIN = ROOT/'outputs/toolchains/cython-3.0.12-cp310'
CACHE = ROOT/'outputs/toolchains/semantic-native-cache-v1'
VERSION = 'cython-typed-batch-runtime-v1'


def source_api(source):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SEMANTIC_API' for t in node.targets):
            if ast.literal_eval(node.value) not in (API, STREAM_API):
                raise ValueError('unknown SEMANTIC_API')
            return ast.literal_eval(node.value)
    return None


def typed_source(source):return source_api(source) is not None


def stream_config(source):
    if source_api(source) != STREAM_API:raise ValueError("SEMANTIC_API must be typed-stream-v1")
    constants={}
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ("NAMES","STATE_NAMES","HISTORY_LENGTH"):
                    constants[target.id]=ast.literal_eval(node.value)
    for key,capacity in (("NAMES",64),("STATE_NAMES",32)):
        names=constants.get(key)
        if not isinstance(names,list) or not 1<=len(names)<=capacity or len(set(names))!=len(names) or any(type(n) is not str or not 0<len(n)<=100 for n in names):
            raise ValueError("literal unique bounded "+key+" required")
    length=constants.get("HISTORY_LENGTH",8)
    if type(length) is not int or not 1<=length<=32:raise ValueError("HISTORY_LENGTH must be 1..32")
    return dict(version=STREAM_API,history_length=length,state_names=constants["STATE_NAMES"],state_capacity=32,
                memory="explicit-incremental-8192bytes",raw="compact-entity-geometry-v1")


def validate_source(source):
    from .fast_features import validate_source as legacy_validate
    if not typed_source(source):raise ValueError('SEMANTIC_API must be typed-batch-v1')
    if len(source.encode()) > 49152:raise ValueError('source at most48KiB')
    if re.search(r'^\s*#\s*(cython|distutils)\s*:', source, re.M):
        raise ValueError('compiler directives are host controlled')
    tree = ast.parse(source)
    if source_api(source)==STREAM_API:stream_config(source)
    # Reuse the deterministic-public-data gate, with two explicitly supported imports.
    cleaned = ast.Module(body=[n for n in tree.body if not (
        isinstance(n, ast.Import) and all(a.name in ('cython', 'array') for a in n.names))], type_ignores=[])
    legacy_validate(ast.unparse(cleaned))
    allowed = {'double','int','long','bint','Py_ssize_t','const','cfunc','ccall','locals','returns'}
    functions = {n.name:n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for n in ast.walk(tree):
        if isinstance(n, (ast.Global, ast.Nonlocal, ast.ClassDef)):
            raise ValueError('typed functions require explicit state, no global/class mutation')
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == 'cython' and n.attr not in allowed:
            raise ValueError('unsupported cython operation: '+n.attr)
        if isinstance(n, ast.Import) and any(a.name == 'cython' and a.asname not in (None,'cython') for a in n.names):
            raise ValueError('cython import alias unsupported')
    for name in ('initial_memory','update_memory','prepare_context','evaluate_candidates'):
        if name not in functions:raise ValueError('missing '+name)
    kernel=functions['evaluate_candidates']
    expected=['cython.const[cython.double][:]', 'cython.const[cython.double][:, :]', 'cython.double[:, :]']
    annotations=[ast.unparse(a.annotation) if a.annotation else '' for a in kernel.args.args]
    if annotations != expected or [a.arg for a in kernel.args.args]!=['context','candidates','out']:
        raise ValueError('evaluate_candidates requires const double context[:], const double candidates[:,:], double out[:,:]')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def toolchain_identity():
    if not (TOOLCHAIN/'Cython').is_dir():
        raise RuntimeError('missing pinned Cython toolchain; see TYPED_SEMANTICS.md')
    # Includes the pinned compiler package, host-controlled builder and C headers.
    files=list((TOOLCHAIN/'Cython').rglob('*'))+[TOOLCHAIN/'cython.py',Path(__file__).with_name('typed_builder.py')]
    files += list(Path('/usr/include/python3.10').rglob('*.h'))
    files += [Path('/usr/bin/python3'),Path('/usr/bin/gcc')]
    hashes={str(p.relative_to(TOOLCHAIN)) if p.is_relative_to(TOOLCHAIN) else str(p):sha(p)
            for p in files if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def compile_source(source, timeout=90.):
    validate_source(source)
    identity=toolchain_identity()
    key=hashlib.sha256((VERSION+'\n'+identity+'\n'+source).encode()).hexdigest()
    CACHE.mkdir(parents=True,exist_ok=True)
    destination=CACHE/key
    with (CACHE/(key+'.lock')).open('w') as lock:
        deadline=time.monotonic()+timeout
        while True:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:
                if time.monotonic()>deadline:raise TimeoutError('native build lock deadline')
                time.sleep(.05)
        if destination.exists():
            manifest=json.loads((destination/'manifest.json').read_text())
            if manifest['key'] != key or manifest['binary_sha256'] != sha(destination/'semantics.so'):
                raise ValueError('native cache integrity mismatch')
            return destination,manifest
        if not shutil.which('bwrap') or not shutil.which('prlimit'):
            raise RuntimeError('bubblewrap/prlimit required for compilation')
        with tempfile.TemporaryDirectory(prefix='semantic_build_',dir=CACHE) as temp:
            p=Path(temp);(p/'_coimagine_semantics.py').write_text(source)
            command=['prlimit','--as=1073741824','--cpu=60','--fsize=33554432','--',
                     'bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL','--clearenv',
                     '--setenv','PATH','/usr/bin:/bin','--setenv','PYTHONHASHSEED','0']
            for root in ('/usr','/lib','/lib64'):
                if Path(root).exists():command += ['--ro-bind',root,root]
            command += ['--proc','/proc','--dev','/dev','--tmpfs','/tmp',
                        '--ro-bind',str(TOOLCHAIN),'/toolchain',
                        '--ro-bind',str(Path(__file__).with_name('typed_builder.py')),'/builder.py',
                        '--bind',str(p),'/build','--chdir','/build','/usr/bin/python3','-I','/builder.py']
            started=time.monotonic()
            with (p/'compiler.log').open('wb') as log:
                subprocess.run(command,stdout=log,stderr=log,timeout=max(.1,deadline-time.monotonic()),check=False)
            if not (p/'build.json').exists() or not (p/'semantics.so').exists():
                raise ValueError('native compilation failed: '+(p/'compiler.log').read_text(errors='replace')[-12000:])
            manifest=dict(key=key,runtime=VERSION,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                          toolchain_sha256=identity,binary_sha256=sha(p/'semantics.so'),
                          compile_wall_seconds=time.monotonic()-started,build=json.loads((p/'build.json').read_text()))
            (p/'manifest.json').write_text(json.dumps(manifest,indent=2))
            # Publish only after successful compilation; generated author code was never imported here.
            shutil.copytree(p,destination)
    return destination,manifest
