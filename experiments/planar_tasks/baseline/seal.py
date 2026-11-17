"""Content-based infrastructure release lock; creating it is an explicit operation."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
BASE=Path(__file__).resolve().parent
LOCK=BASE/'release_v1_4_0.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths=set((ROOT/'src/coimagine').rglob('*.py'))
    paths.update(p for directory in (ROOT/'experiments').glob('option*') for p in directory.rglob('*.py'))
    paths.update(BASE.glob('*.py'))
    paths.add(BASE/'prompt.txt')
    paths.add(BASE/'prediction_api.md')
    paths.update(p for p in (BASE.parent/'prediction_sol_20261110').iterdir() if p.suffix in ('.py', '.txt'))
    paths.add(BASE/'README.md')
    paths.add(BASE/'smoke_config.json')
    paths.add(BASE/'development_config.json')
    paths.add(BASE/'flash_high_config.json')
    paths.add(BASE/'evaluation_config.json')
    paths.add(ROOT/'docs/research/Agentic_Options_Baseline_v1.md')
    paths.update(BASE.parent/name for name in ('__init__.py','interface.py','tasks.py','policy.py','runner.py'))
    paths.update(BASE.parent/name for name in ('deepseek_agent.py','agentic.py','sandbox.py'))
    paths.update((ROOT/'tests').glob('test_baseline_*.py'))
    paths.add(ROOT/'tests/test_option_rl_v13_recovery.py')
    return sorted(p for p in paths if '__pycache__' not in p.parts)


def environment():
    result={'host_python':platform.python_version(),'packages':{},'executables':{}}
    for name in ('numpy','scipy','pymunk'):
        try: result['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: result['packages'][name]=None
    for name in ('bwrap','prlimit','python3'):
        path=shutil.which(name) if name!='python3' else '/usr/bin/python3'
        if not path:raise RuntimeError(f'{name} unavailable')
        result['executables'][name]={'path':str(Path(path).resolve()),'sha256':digest(path)}
    return result


def build_release(path=LOCK):
    from . import VERSION
    from ..interface import contract
    from ..tasks import task_catalog
    path=Path(path)
    if path.exists():raise FileExistsError('release already frozen; use a new version/path')
    release={'version':VERSION,'created_unix':time.time(),
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'dirty':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'files':{str(p.relative_to(ROOT)):digest(p) for p in sources()},
        'environment':environment(),'contract':contract(),
        'development_tasks':{n:t.sha256 for n,t in task_catalog().items()}}
    path.write_text(json.dumps(release,indent=2)+'\n')
    return release


def verify_release(path=LOCK):
    path=Path(path)
    data=json.loads(path.read_text())
    from . import VERSION
    if data['version']!=VERSION:raise ValueError('release version mismatch')
    current={str(p.relative_to(ROOT)) for p in sources()}
    if current!=set(data['files']):raise ValueError('release source set changed')
    for name,expected in data['files'].items():
        if digest(ROOT/name)!=expected:raise ValueError('release hash changed: '+name)
    if environment()!=data['environment']:raise ValueError('release runtime environment changed')
    return data
