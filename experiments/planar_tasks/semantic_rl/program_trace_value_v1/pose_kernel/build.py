"""Build the repository-owned C kernel once per host/source/compiler identity."""
from functools import lru_cache
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sysconfig
import tempfile

from ...typed_compiler import ROOT
from . import PoseKernel

SOURCE = Path(__file__).with_name('pose_route.c')
FLAGS = ('-std=c99', '-shared', '-fPIC', '-O2', '-fno-fast-math', '-ffp-contract=off')
CACHE = ROOT/'outputs/toolchains/cap009-pose'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def default_kernel():
    cc = Path('/usr/bin/gcc')
    helpers = SOURCE.with_name('shared_helpers.c')
    header = SOURCE.with_name('cap_heap.h')
    identity = dict(source=sha(SOURCE), helpers=sha(helpers), heap=sha(header), compiler=sha(cc), flags=list(FLAGS),
                    platform=sysconfig.get_platform(), libc=list(platform.libc_ver()))
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    dest = CACHE/key
    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE/(key+'.lock')).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not dest.exists():
            with tempfile.TemporaryDirectory(dir=CACHE) as staging:
                stage = Path(staging)
                subprocess.run([cc, *FLAGS, SOURCE, helpers, '-lm', '-o', stage/'pose.so'],
                               check=True, timeout=60)
                manifest = dict(identity=identity, binary_sha256=sha(stage/'pose.so'))
                (stage/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
                stage.rename(dest)
        manifest = json.loads((dest/'manifest.json').read_text())
        if manifest['identity'] != identity or sha(dest/'pose.so') != manifest['binary_sha256']:
            raise ValueError(f'CAP009 kernel cache integrity check failed: {dest}')
        return PoseKernel(dest/'pose.so')
