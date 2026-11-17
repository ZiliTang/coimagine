"""Versioned author-facing knowledge, separate from frozen execution releases."""
from pathlib import Path
import json
from experiments.planar_tasks.baseline import seal
from experiments.planar_tasks.option_search_v3 import build

HERE = Path(__file__).resolve().parent
LOCK = HERE / 'release_v1.json'
GUIDE = HERE / 'option_guide.md'
VERSION = 'public-option-knowledge-v1'
FILES = ('__init__.py', 'knowledge.py', 'session.py', 'driver.py', 'option_guide.md', 'README.md')
SOURCES = (
    'experiments/planar_tasks/interface.py',
    'experiments/option_rl_v21/spec.py', 'experiments/option_rl_v21/system.py',
    'experiments/option_rl_v21/yield_skill.py', 'experiments/option_rl_v21/event_env.py',
    'experiments/option_rl_v19/spec.py', 'experiments/option_rl_v19/system.py',
    'experiments/option_rl_v19/unknown_shape.py',
    'experiments/option_rl_v13/controller.py', 'experiments/option_rl_v13/mode_env.py',
    'experiments/option_rl_v13/push_controller.py',
    'src/coimagine/random_start/skills.py', 'src/coimagine/random_start/effect_push.py',
)


def freeze():
    seal.verify_release(); build.verify()
    if LOCK.exists():
        raise FileExistsError('knowledge release already frozen')
    release = dict(version=VERSION, baseline_sha256=seal.digest(seal.LOCK),
                   framework_sha256=seal.digest(build.LOCK),
                   files={n: seal.digest(HERE / n) for n in FILES},
                   tests_sha256=seal.digest(seal.ROOT / 'tests/test_public_option_knowledge.py'),
                   controller_sources={n: seal.digest(seal.ROOT / n) for n in SOURCES})
    LOCK.write_text(json.dumps(release, indent=2) + '\n')
    return release


def verify():
    seal.verify_release(); build.verify()
    release = json.loads(LOCK.read_text())
    if (release['version'] != VERSION or release['baseline_sha256'] != seal.digest(seal.LOCK)
            or release['framework_sha256'] != seal.digest(build.LOCK)):
        raise ValueError('knowledge dependencies changed')
    for name, expected in release['files'].items():
        if seal.digest(HERE / name) != expected:
            raise ValueError('knowledge source changed: ' + name)
    for name, expected in release['controller_sources'].items():
        if seal.digest(seal.ROOT / name) != expected:
            raise ValueError('documented controller changed: ' + name)
    if seal.digest(seal.ROOT / 'tests/test_public_option_knowledge.py') != release['tests_sha256']:
        raise ValueError('knowledge tests changed')
    return release


def install(session, method):
    verify()
    workspace = session.workspace
    (workspace / 'option_guide.md').write_bytes(GUIDE.read_bytes())
    instruction = '\nRead option_guide.md: shared public controller behavior, including continuation, switching and feedback. It is supplied equally to every backend.\n'
    path = workspace / 'instructions.txt'
    path.write_text(path.read_text() + instruction)
    path = workspace / 'contract.json'
    contract = json.loads(path.read_text())
    contract['public_skill_knowledge'] = dict(version=VERSION, path='option_guide.md', sha256=seal.digest(GUIDE))
    path.write_text(json.dumps(contract, indent=2) + '\n')
    # Host copy cannot be overwritten by author scripts. Package/manifest contains provenance.
    profile = dict(version=VERSION, method=method, release_sha256=seal.digest(LOCK),
                   files={name: seal.digest(workspace / name)
                          for name in ('option_guide.md', 'instructions.txt', 'contract.json')})
    (session.path / 'public_knowledge.json').write_text(json.dumps(profile, indent=2) + '\n')
    session.state['public_knowledge_sha256'] = seal.digest(session.path / 'public_knowledge.json')
    session._save()
    session.event('public_knowledge_installed', {}, profile)


def verify_session(path):
    verify(); path = Path(path)
    state = json.loads((path / 'state.json').read_text())
    profile_path = path / 'public_knowledge.json'
    if seal.digest(profile_path) != state.get('public_knowledge_sha256'):
        raise ValueError('knowledge profile changed')
    profile = json.loads(profile_path.read_text())
    if profile['release_sha256'] != seal.digest(LOCK):
        raise ValueError('knowledge release mismatch')
    for name, expected in profile['files'].items():
        p = path / 'workspace' / name
        if p.is_symlink() or seal.digest(p) != expected:
            raise ValueError('public knowledge material changed: ' + name)
    return profile
