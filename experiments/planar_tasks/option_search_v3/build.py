"""Freeze the search implementation and package only the replaceable heuristic."""
from pathlib import Path
import json
import shutil
from dataclasses import asdict
from experiments.planar_tasks.baseline import seal
from .search import Budget

HERE=Path(__file__).resolve().parent
LOCK=HERE/'release_v3.json'
FILES=('search.py','model.py','policy.py','hook_client.py','hook_worker.py','geometry.py','estimator.py',
       'heuristic.py','build.py','session.py','evaluation.py','driver.py','policy_host.py','prompt.txt','README.md','runner.py','runtime_policy.py')
CORE=('search.py','model.py','policy.py','hook_client.py','hook_worker.py','geometry.py','estimator.py')


def freeze():
    seal.verify_release()
    if LOCK.exists():raise FileExistsError('framework release already frozen')
    release=dict(version='option-search-v3',baseline_release_sha256=seal.digest(seal.LOCK),
        files={n:seal.digest(HERE/n) for n in FILES},tests_sha256=seal.digest(seal.ROOT/'tests/test_option_search_v3.py'),
        estimator_origin='prediction_sol_runtime_20261110_01/frozen/package/geom.py',
        output_language='actual legal_pairs x integer ticks 1..40; feedback after every command',
        guarantee='structural action/trace support, not equal finite-budget search coverage or physical safety')
    LOCK.write_text(json.dumps(release,indent=2)+'\n');return release


def verify():
    release=json.loads(LOCK.read_text());seal.verify_release()
    if release['baseline_release_sha256']!=seal.digest(seal.LOCK):raise ValueError('baseline changed')
    for name,expected in release['files'].items():
        if seal.digest(HERE/name)!=expected:raise ValueError('framework source changed: '+name)
    if seal.digest(seal.ROOT/'tests/test_option_search_v3.py')!=release['tests_sha256']:raise ValueError('framework tests changed')
    return release


def build_package(path,heuristic=None,budget=None):
    verify();path=Path(path);path.mkdir(parents=True,exist_ok=False)
    for name in CORE:
        shutil.copyfile(HERE/name,path/name)
    shutil.copyfile(heuristic or HERE/'heuristic.py',path/'heuristic.py')
    compile((path/'heuristic.py').read_text(),'heuristic.py','exec')
    shutil.copyfile(seal.BASE/'predictor_client.py',path/'predictor_client.py')
    shutil.copyfile(HERE/'policy_host.py',path/'__policy_host__.py')
    config=asdict(budget or Budget());(path/'planner_config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest=dict(kind='fixed_option_search_package',framework_release_sha256=seal.digest(LOCK),
        baseline_release_sha256=seal.digest(seal.LOCK),replaceable_files=['heuristic.py'],budget=config,
        files_sha256={p.name:seal.digest(p) for p in path.iterdir() if p.is_file()})
    (path/'package_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest


def verify_package(path):
    path=Path(path);verify();manifest=json.loads((path/'package_manifest.json').read_text())
    if manifest['framework_release_sha256']!=seal.digest(LOCK):raise ValueError('package framework release mismatch')
    if manifest['baseline_release_sha256']!=seal.digest(seal.LOCK):raise ValueError('package baseline release mismatch')
    trusted={name:HERE/name for name in CORE}
    trusted.update({'predictor_client.py':seal.BASE/'predictor_client.py','__policy_host__.py':HERE/'policy_host.py'})
    for name,source in trusted.items():
        if seal.digest(path/name)!=seal.digest(source):raise ValueError('fixed framework modified: '+name)
    config=json.loads((path/'planner_config.json').read_text())
    Budget(**config)
    if config!=manifest['budget']:raise ValueError('budget manifest mismatch')
    if manifest['replaceable_files']!=['heuristic.py']:raise ValueError('only heuristic.py is replaceable')
    actual={p.name for p in path.iterdir() if p.is_file() and p.name!='package_manifest.json'}
    if actual!=set(trusted)|{'heuristic.py','planner_config.json'}:raise ValueError('unexpected package file set')
    if any(p.is_symlink() or not p.is_file() for p in path.iterdir()):raise ValueError('package must contain regular files only')
    if actual!=set(manifest['files_sha256']):raise ValueError('package file set changed')
    for name,digest in manifest['files_sha256'].items():
        if seal.digest(path/name)!=digest:raise ValueError('package content changed: '+name)
    return manifest
