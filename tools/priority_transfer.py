"""Portable, hash-checked export for a development-only CAP teacher-priority study.

Original artifacts are never rewritten. New selections and runtime paths are
explicit derivatives. No LLM calls, training or rollout occurs during export.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
VERSION = 'cap-priority-transfer-v1'
SEEDS = (17, 29, 43)


def read(p):
    return json.loads(Path(p).read_text())


def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_name(p.name + '.pending')
    t.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    t.replace(p)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(base, rel):
    q = PurePosixPath(rel)
    if q.is_absolute() or '..' in q.parts or not q.parts:
        raise ValueError('expected safe relative bundle path')
    p = (Path(base) / str(q)).resolve()
    if not p.is_relative_to(Path(base).resolve()):
        raise ValueError('path escapes bundle')
    return p


def select(rows):
    """Rank only the complete development matrix; keep both policies successful."""
    by = defaultdict(dict)
    for r in rows:
        if r.get('split') != 'development':
            raise ValueError('teacher ranking accepts development only')
        if r['teacher'] in by[r['case_id']]:
            raise ValueError('duplicate teacher/case')
        by[r['case_id']][r['teacher']] = r
    names = sorted({r['teacher'] for r in rows})
    if not by or any(set(v) != set(names) for v in by.values()):
        raise ValueError('incomplete development coverage matrix')
    wins = Counter(r['teacher'] for r in rows if r['success'])
    ranking = sorted(names, key=lambda n: (-wins[n], n))
    rank = {n: i for i, n in enumerate(ranking)}
    arms = {'fastest': [], 'priority': []}
    for case in sorted(by):
        candidates = [r for r in by[case].values() if r['success']]
        if not candidates:
            continue
        arms['fastest'].append(min(candidates, key=lambda r: (r['ticks'], r['teacher'])))
        arms['priority'].append(min(candidates, key=lambda r: (rank[r['teacher']], r['ticks'])))
    return dict(ranking=[dict(teacher=n, success=wins[n], cases=len(by)) for n in ranking], arms=arms)


def entries(bundle):
    rows = read(Path(bundle) / 'library.json')
    return [dict(name=r['name'], groups=r['groups'], source=inside(bundle, r['policy']).read_text()) for r in rows]


def source_hashes():
    # Include the actual runtime tree, including untracked dependencies if any.
    files = list((ROOT/'experiments').rglob('*.py')) + list((ROOT/'src/coimagine').rglob('*.py'))
    files += [ROOT/'src/coimagine/random_start/lidar_kernel.c', Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def export(destination):
    from experiments.planar_tasks.authoring_v2 import random_room_coverage_v2 as V
    dest = Path(destination).resolve()
    if dest.exists():
        raise FileExistsError('new bundle directory required')
    dest.mkdir(parents=True)
    author = V.O.AUTHOR
    scaling = ROOT/'outputs/semantic_rl/random_room_gda_scaling_20261117_03'
    refs = [author, V.O.PARENT, V.BASE/'collection', scaling/'collection']
    for p in refs:
        V.Campaign(p).verify_frozen()
    state = read(author/'campaign.json'); rows = []; fixtures = {}
    for case in state['tasks']:
        fixtures[case] = read(author/'tasks'/case/'fixture.json')
    for member in state['library']:
        for case in state['tasks']:
            p = author/'runs'/member['runs'][case]
            r = read(p/'summary.json')
            rows.append(dict(case_id=case, teacher=member['name'], success=bool(r['success']),
                             ticks=r['ticks'], split='development', origin=str(p/'result.json.gz')))
    for base in [V.BASE, scaling]:
        for case in read(base/'parent/development.json')['cases']:
            if case['id'] in fixtures: raise ValueError('duplicate root')
            fixtures[case['id']] = case['fixture']
        for r in read(base/'collection/result.json')['rows']:
            rows.append(dict(case_id=r['case_id'], teacher=r['condition'], success=bool(r['success']),
                             ticks=r['ticks'], split='development', origin=r['result_path']))
    selected = select(rows)
    assert len(fixtures) == 760 and len(rows) == 7600
    assert all(len(v) == 570 for v in selected['arms'].values())
    # The fastest arm must reproduce the published570 selections exactly.
    old = read(V.O.PARENT/'demonstrations.json') + read(V.BASE/'data_expanded/selected.json') + read(scaling/'data_expanded/selected.json')
    expected = {r['task']: (r['teacher'], str(Path(r['result_path']).resolve())) for r in old}
    assert len(expected) == 570
    for r in selected['arms']['fastest']:
        assert expected[r['case_id']] == (r['teacher'], str(Path(r['origin']).resolve()))
    imported = {}; relative = {}
    for arm in selected['arms'].values():
        for r in arm:
            origin = r['origin']
            if origin not in relative:
                digest = sha(origin); rel = f'trajectories/{digest}.json.gz'; p = dest/rel
                p.parent.mkdir(exist_ok=True)
                if not p.exists():shutil.copyfile(origin, p)
                relative[origin] = dict(path=rel, sha256=digest)
                imported[rel] = dict(original_path=origin, sha256=digest)
    portable = dict(ranking=selected['ranking'], arms={})
    for arm, rs in selected['arms'].items():
        portable['arms'][arm] = [dict(case_id=r['case_id'], teacher=r['teacher'], ticks=r['ticks'], **relative[r['origin']]) for r in rs]
    library = []
    for e in V.entries():
        rel = f'policies/{e["name"]}.py'; p = dest/rel;p.parent.mkdir(exist_ok=True);p.write_text(e['source'])
        library.append(dict(name=e['name'], groups=e['groups'], policy=rel, sha256=sha(p)))
    save(dest/'library.json', library);save(dest/'selections.json', portable)
    save(dest/'development.json', dict(cases=[dict(id=k,fixture=v) for k,v in sorted(fixtures.items())],
                                    coverage=[{k:v for k,v in r.items() if k!='origin'} for r in rows]))
    save(dest/'evaluation.json', dict(cases=read(scaling/'parent/fresh_test.json')['cases'],
        interpretation='These200 outcomes have already been inspected; diagnostic reuse, NOT fresh blind test. No outcomes are exported.'))
    changed = sum(a['teacher'] != b['teacher'] for a,b in zip(portable['arms']['fastest'],portable['arms']['priority']))
    save(dest/'protocol.json', dict(version=VERSION, ranking_population=760, demonstrations_per_arm=570,
        ranking='complete760 development roots only; descending success count, name tie-break',
        change='teacher selection on ALL570 solvable roots; old270 are also eligible for reassignment',
        fastest_matches_existing570=True, changed_teachers=changed, unique_trajectories=len(imported),
        seeds=list(SEEDS),updates=42000,arm='GDA',sampling='unchanged library_learning uniform decision-boundary minibatches',
        caveat='Different trajectory lengths change boundary counts; report counts and fit accuracy, do not claim a pure label-only intervention.',
        fit_wall_seconds=7200,feature_workers_default=4,evaluation_workers_default=8,
        max_ticks=1800,max_decisions=300,rollout_wall_seconds=1800,evaluation_episodes=1200,
        no_llm=True,no_new_collection=True,no_rl=True,relative_runtime_paths=True))
    provenance = dict(code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        python=sys.version, packages={n:importlib.metadata.version(n) for n in ['numpy','torch','pymunk','matplotlib','PyYAML']},
        original_seals={str(p):sha(p/'manifest.json') for p in refs}, trajectories=imported,
        original_records_unchanged=True, subset_export_not_full_original_campaign=True)
    save(dest/'provenance.json', provenance);save(dest/'runtime_sources.json',source_hashes())
    shutil.copytree(ROOT/'outputs/toolchains/cython-3.0.12-cp310',dest/'toolchain',
                    ignore=shutil.ignore_patterns('__pycache__','*.pyc','bin'))
    # Cross-machine feature/probability golden check: no simulator execution.
    import numpy as np
    f = V.LibraryFeatures(entries(dest),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
    sample = portable['arms']['fastest'][0]
    try:
        result = V.unpack(inside(dest,sample['path']));tr=next(V.O.extracted_rows(result,f,max_samples=1))
        golden={}
        for j,p in enumerate(tr['panels']):
            golden[f'context{j}']=p['attribution_context'];golden[f'candidates{j}']=p['attribution_candidates']
        np.savez_compressed(dest/'feature_golden.npz',**golden)
        save(dest/'feature_golden.json',dict(sample=sample,indices=tr['indices'],history_hash=tr['history_hash'],layout=f.layout))
    finally:f.close()
    shutil.copyfile(ROOT/'docs/development/Priority_Transfer_20261117.md',dest/'README.md')
    manifest={str(p.relative_to(dest)):sha(p) for p in sorted(dest.rglob('*')) if p.is_file()}
    save(dest/'manifest.json',dict(version=VERSION,files=manifest))
    print(json.dumps(dict(bundle=str(dest),files=len(manifest),bytes=sum(p.stat().st_size for p in dest.rglob('*') if p.is_file()),changed=changed,unique_trajectories=len(imported),ranking=selected['ranking']),ensure_ascii=False),flush=True)


def verify(bundle, check_features=True):
    bundle=Path(bundle).resolve();m=read(bundle/'manifest.json')
    if m['version']!=VERSION:raise ValueError('bundle version')
    for rel,digest in m['files'].items():
        if sha(inside(bundle,rel))!=digest:raise ValueError('bundle checksum: '+rel)
    for rel,digest in read(bundle/'runtime_sources.json').items():
        if sha(inside(ROOT,rel))!=digest:raise ValueError('runtime source mismatch: '+rel)
    if check_features:
        import numpy as np
        from experiments.planar_tasks.authoring_v2 import random_room_coverage_v2 as V
        f=V.LibraryFeatures(entries(bundle),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
        try:
            info=read(bundle/'feature_golden.json');arrays=np.load(bundle/'feature_golden.npz')
            tr=next(V.O.extracted_rows(V.unpack(inside(bundle,info['sample']['path'])),f,max_samples=1))
            assert tr['indices']==info['indices'] and tr['history_hash']==info['history_hash'] and f.layout==info['layout']
            for j,p in enumerate(tr['panels']):
                for key,field in [('context','attribution_context'),('candidates','attribution_candidates')]:
                    np.testing.assert_array_equal(p[field],arrays[f'{key}{j}'])
        finally:f.close()
    return dict(status='passed',files=len(m['files']),bundle_sha256=sha(bundle/'manifest.json'),features=check_features)


def bootstrap(bundle):
    """Install only verified pinned compiler files; never overwrite different files."""
    verify(bundle,False)
    target=ROOT/'outputs/toolchains/cython-3.0.12-cp310'
    files=[(rel,digest) for rel,digest in read(Path(bundle)/'manifest.json')['files'].items() if rel.startswith('toolchain/')]
    if not files:raise ValueError('bundle lacks compiler payload')
    for rel,digest in files:
        dst=inside(target,rel.removeprefix('toolchain/'))
        if dst.exists() and sha(dst)!=digest:raise ValueError('existing compiler differs: '+str(dst))
    for rel,digest in files:
        dst=inside(target,rel.removeprefix('toolchain/'));dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copyfile(inside(bundle,rel),dst)
    for name in ['gcc','bwrap','prlimit']:
        if not shutil.which(name):raise RuntimeError('required system tool missing: '+name)
    import sysconfig
    if not (Path(sysconfig.get_path('include'))/'Python.h').exists():
        raise RuntimeError('host Python development headers missing')
    include=subprocess.check_output(['/usr/bin/python3','-c','import sysconfig;print(sysconfig.get_path("include"))'],text=True).strip()
    if not (Path(include)/'Python.h').exists():raise RuntimeError('system Python development headers missing')
    print(json.dumps(dict(status='installed',compiler_files=len(files),path=str(target))),flush=True)


_WORKER={}

def setup(bundle):
    from experiments.planar_tasks.authoring_v2 import random_room_coverage_v2 as V
    bundle=str(Path(bundle).resolve())
    if _WORKER.get('bundle')!=bundle:
        V.CACHE.clear();V.entries=lambda:entries(bundle);_WORKER.clear();_WORKER['bundle']=bundle
    return V


def extract_job(job):
    V=setup(job['bundle']);r=job['row']
    return V.extraction_worker(dict(task=r['case_id'],teacher=r['teacher'],result_path=str(inside(job['bundle'],r['path'])),
        sha256=r['sha256'],path=job['path']))


def parallel(jobs,fn,workers,progress):
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        futures=[pool.submit(fn,j) for j in jobs];rows=[]
        from concurrent.futures import as_completed
        for f in as_completed(futures):
            rows.append(f.result());save(progress,dict(completed=len(rows),total=len(jobs),at=time.time()))
    return rows


def bind_output(bundle,out):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    binding=dict(bundle_manifest=sha(Path(bundle)/'manifest.json'),runtime_sources=read(Path(bundle)/'runtime_sources.json'))
    p=out/'binding.json'
    if p.exists() and read(p)!=binding:raise ValueError('output belongs to different bundle/source')
    save(p,binding)
    return out


def extract(bundle,out,workers):
    verify(bundle,False);out=bind_output(bundle,out);selected=read(Path(bundle)/'selections.json');unique={}
    for rs in selected['arms'].values():
        for r in rs:unique[(r['case_id'],r['sha256'])]=r
    jobs=[dict(bundle=str(Path(bundle).resolve()),row=r,path=str(out/'shards'/r['sha256'])) for r in unique.values()]
    rows=parallel(jobs,extract_job,workers,out/'extract_progress.json');by={(r['task'],r['result_sha256']):r for r in rows}
    V=setup(bundle)
    for arm,rs in selected['arms'].items():
        path=out/('data_'+arm);path.mkdir(exist_ok=True)
        V.combine(path,[by[(r['case_id'],r['sha256'])] for r in rs])
    save(out/'EXTRACTED.json',dict(trajectories=len(rows),arms={a:read(out/('data_'+a)/'extraction.json')['boundaries'] for a in selected['arms']}))


def fit(bundle,out,arm,seed):
    import numpy as np
    from experiments.planar_tasks.semantic_rl.library_learning import train
    V=setup(bundle);verify(bundle,False);out=bind_output(bundle,out);p=out/'fit'/f'{arm}_s{seed}'
    if (p/'complete.json').exists():return read(p/'complete.json')
    p.mkdir(parents=True,exist_ok=True);data=out/('data_'+arm);ex=read(data/'extraction.json')
    ds={k:np.load(data/(k+'.npy'),mmap_mode='r') for k in ['contexts','candidates','offsets','targets']}
    ids=list(range(len(ds['contexts'])));start=time.time()
    model,curve=train(ds,ids,read(data/'layout.json'),'GDA',ex['source_sha'],ex['action_protocol'],updates=42000,seed=seed,wall_seconds=7200,
        progress=lambda row:save(p/'progress.json',row))
    acc=V.I.accuracy(model,ds,ids);model.save(p/'model.pt',dict(arm=arm,seed=seed,updates=42000))
    save(p/'accuracy.json',acc);save(p/'curve.json',curve)
    done=dict(arm=arm,seed=seed,updates=42000,boundaries=len(ids),exact_actions=acc['exact_actions'],seconds=time.time()-start,model_sha256=sha(p/'model.pt'))
    save(p/'complete.json',done);return done


def eval_job(job):
    V=setup(job['bundle']);key=(job['model'],job['model_sha256'])
    if _WORKER.get('model_key')!=key:
        if sha(job['model'])!=job['model_sha256']:raise ValueError('model changed')
        if 'features' in _WORKER:_WORKER['features'].close()
        _WORKER.update(model_key=key,model=V.I.Scorer.load(job['model']),features=V.LibraryFeatures(entries(job['bundle']),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess'))
    V.get_runtime=lambda condition:(_WORKER['features'],_WORKER['model'])
    return V.run_worker(job)


def evaluate(bundle,out,arm,seed,workers):
    V=setup(bundle);verify(bundle,False);out=bind_output(bundle,out);name=f'{arm}_s{seed}';model=out/'fit'/name/'model.pt'
    digest=read(model.parent/'complete.json')['model_sha256'];p=out/'evaluation'/name
    cases=read(Path(bundle)/'evaluation.json')['cases'];jobs=[]
    for case in cases:
        j=V.job_for(case,name,p);j.update(bundle=str(Path(bundle).resolve()),model=str(model),model_sha256=digest);jobs.append(j)
    rows=parallel(jobs,eval_job,workers,p/'progress.json')
    errors=[r for r in rows if r.get('error') or r['stop']=='wall_budget']
    result=dict(condition=name,episodes=len(rows),success=sum(r['success'] for r in rows),rows=rows,errors=errors,interpretation='previously inspected diagnostic200')
    save(p/'result.json',result)
    if errors:raise RuntimeError('evaluation infrastructure/policy errors; inspect result, do not silently drop')
    print(name,result['success'],'/',len(rows),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','bootstrap','verify','extract','fit','evaluate','run'])
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,default=ROOT/'outputs/semantic_rl/priority_teacher_20261117_01')
    p.add_argument('--arm',choices=['fastest','priority'],default='priority');p.add_argument('--seed',type=int,choices=SEEDS,default=17)
    p.add_argument('--workers',type=int,default=4);a=p.parse_args()
    if not 1<=a.workers<=16:raise ValueError('workers must be1..16')
    if a.command=='export':export(a.bundle)
    elif a.command=='bootstrap':bootstrap(a.bundle)
    elif a.command=='verify':print(json.dumps(verify(a.bundle)))
    elif a.command=='extract':extract(a.bundle,a.out,a.workers)
    elif a.command=='fit':print(json.dumps(fit(a.bundle,a.out,a.arm,a.seed)))
    elif a.command=='evaluate':evaluate(a.bundle,a.out,a.arm,a.seed,a.workers)
    else:
        verify(a.bundle);extract(a.bundle,a.out,min(a.workers,4))
        for arm in ['fastest','priority']:
            for seed in SEEDS:fit(a.bundle,a.out,arm,seed)
        for arm in ['fastest','priority']:
            for seed in SEEDS:evaluate(a.bundle,a.out,arm,seed,a.workers)
        save(a.out/'FINISHED.json',dict(status='complete',fits=6,evaluations=1200))

if __name__=='__main__':main()
