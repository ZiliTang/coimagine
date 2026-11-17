"""Independent authoring release, preserving every prior execution/knowledge seal."""
from pathlib import Path
import json
from experiments.planar_tasks.authoring import knowledge
from experiments.planar_tasks.baseline import seal
HERE=Path(__file__).resolve().parent
LOCK=HERE/'release_v2.json'
FILES=('__init__.py','profile.py','release.py','reports.py','context.py','session.py',
       'driver.py','batch_client.py','experiment_api.md','README.md')
TEST='tests/test_authoring_v2.py'


def freeze():
    knowledge.verify()
    if LOCK.exists():raise FileExistsError('already frozen')
    r=dict(version='experiment-authoring-v2',knowledge_release_sha256=seal.digest(knowledge.LOCK),
           files={n:seal.digest(HERE/n) for n in FILES},tests_sha256=seal.digest(seal.ROOT/TEST))
    LOCK.write_text(json.dumps(r,indent=2)+'\n');return r


def verify():
    knowledge.verify();r=json.loads(LOCK.read_text())
    if r['knowledge_release_sha256']!=seal.digest(knowledge.LOCK):raise ValueError('knowledge changed')
    for n,h in r['files'].items():
        if seal.digest(HERE/n)!=h:raise ValueError('authoring source changed: '+n)
    if seal.digest(seal.ROOT/TEST)!=r['tests_sha256']:raise ValueError('tests changed')
    return r
