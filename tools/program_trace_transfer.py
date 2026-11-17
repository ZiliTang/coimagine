"""Incremental, relocatable evidence handoff; no training, rollout or LLM calls.

Depends on the previously delivered priority_teacher_v1 bundle. Historical JSON
is copied verbatim; only index.json contains executable cross-machine references.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from tools import priority_transfer as P

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'outputs/semantic_rl/da_joint_20261117_01'
HOST = ROOT / 'outputs/semantic_rl_authors/da_joint_20261117_01'
VERSION = 'cap-program-trace-handoff-v1'


def check_export_destination(bundle, dependency):
    bundle = Path(bundle).resolve()
    for protected in (Path(dependency).resolve(), STUDY.resolve(), HOST.resolve()):
        if bundle.is_relative_to(protected):
            raise ValueError('export cannot write inside frozen inputs')
    if bundle.exists():
        raise FileExistsError('new bundle directory required')


def resolve(ref, bundle, dependency):
    roots = {'bundle': Path(bundle), 'priority': Path(dependency)}
    if ref['root'] not in roots:
        raise ValueError('unknown data root')
    return P.inside(roots[ref['root']], ref['path'])


def check_inventory(bundle):
    bundle = Path(bundle)
    manifest = P.read(bundle / 'manifest.json')
    if manifest['version'] != VERSION:
        raise ValueError('handoff version')
    actual = {str(p.relative_to(bundle)) for p in bundle.rglob('*')
              if p.is_file() and p != bundle / 'manifest.json'}
    if actual != set(manifest['files']):
        raise ValueError('payload inventory mismatch')
    for rel, expected in manifest['files'].items():
        if P.sha(P.inside(bundle, rel)) != expected:
            raise ValueError('payload checksum: ' + rel)
    return manifest


def export(bundle, dependency):
    from experiments.planar_tasks.authoring_v2 import da_joint_models as M
    import numpy as np

    bundle, dependency = Path(bundle).resolve(), Path(dependency).resolve()
    check_export_destination(bundle, dependency)
    P.verify(dependency, check_features=False)
    if P.read(HOST / 'study.exit.json')['code'] != 0:
        raise ValueError('study is not complete')
    original = P.read(STUDY / 'artifact_manifest.json')['sha256']
    bundle.mkdir(parents=True)
    copied = {}
    for rel, expected in original.items():
        # Derived feature matrices are reproducible; do not transfer 10+ GB.
        if Path(rel).parts[0] in ('data_base', 'data_repair'):
            continue
        src = P.inside(STUDY, rel)
        if P.sha(src) != expected:
            raise ValueError('sealed study changed: ' + rel)
        dest = bundle / 'evidence/da_joint' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)
        copied[rel] = expected
    shutil.copyfile(STUDY / 'artifact_manifest.json', bundle / 'evidence/original_full_inventory.json')
    for folder in ('data_base', 'data_repair'):
        for name in ('extraction.json', 'layout.json', 'selection.json', 'parity.json'):
            src = STUDY / folder / name
            if src.exists():
                if P.sha(src) != original[f'{folder}/{name}']:
                    raise ValueError('sealed data metadata changed')
                dest = bundle / 'evidence/da_joint' / folder / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
    shutil.copytree(HOST, bundle / 'evidence/host')

    def own(path):
        rel = str(Path(path).resolve().relative_to(STUDY))
        dest = bundle / 'evidence/da_joint' / rel
        if not dest.is_file():
            raise ValueError('missing portable payload: ' + rel)
        return dict(root='bundle', path=str(dest.relative_to(bundle)), sha256=P.sha(dest))

    fastest = {r['case_id']: r for r in P.read(dependency / 'selections.json')['arms']['fastest']}
    base = []
    for r in P.read(STUDY / 'data_base/extraction.json')['rows']:
        old = fastest[r['task']]
        if old['sha256'] != r['result_sha256'] or old['teacher'] != r['teacher']:
            raise ValueError('old270 does not match dependency fastest')
        base.append(dict(case_id=r['task'], teacher=r['teacher'], start_boundary=0,
                         trajectory=dict(root='priority', path=old['path'], sha256=old['sha256'])))
    repairs = [dict(case_id=r['task'].removesuffix('__repair'), teacher=r['teacher'],
                    start_boundary=r['start_boundary'], trajectory=own(r['result_path']))
               for r in P.read(STUDY / 'data_repair/jobs.json')]
    states = []
    for s in P.read(STUDY / 'selection.json')['states']:
        row = dict(s)
        row['reference'] = own(s['reference'])
        row.pop('reference_sha256')
        states.append(row)
    replays = []
    for phase in ('donors', 'branches', 'evaluation'):
        for r in P.read(STUDY / phase / 'result.json')['rows']:
            replays.append(dict(phase=phase, case_id=r['job']['case_id'],
                                condition=r['job']['condition'], kind=r['job'].get('kind'),
                                evaluation=r['job'].get('evaluation'), split=r['job'].get('split'),
                                success=r['success'], ticks=r['ticks'], trajectory=own(r['result_path']),
                                usage='viewed_evaluation_not_training' if phase == 'evaluation'
                                else 'development_evidence_observe_root_split'))
    models = [dict(name=n, mode=n.split('_')[1], checkpoint=own(STUDY / 'fit' / n / 'model.pt'))
              for n in M.names()]
    inputs = P.read(STUDY / 'inputs.json')
    donor_source = M.D.DONOR
    if P.sha(donor_source) != inputs['hashes'][str(donor_source)]:
        raise ValueError('historical diagnostic donor changed')
    donor_dest = bundle / 'evidence/donor/model.pt'
    donor_dest.parent.mkdir(parents=True)
    shutil.copyfile(donor_source, donor_dest)
    donor_ref = dict(root='bundle', path=str(donor_dest.relative_to(bundle)), sha256=P.sha(donor_dest))
    assert len(base) == 270 and len(repairs) == 10 and len(states) == 24 and len(replays) == 1696
    index = dict(version=VERSION, original270=base, repair_suffixes=repairs,
                 handoff_states=states, models=models, diagnostic_donor=donor_ref, replays=replays,
                 development_cases=inputs['cases'], viewed80_cases=inputs['fresh'],
                 note='Historical evidence retains source-PC paths. Resolve only these relative references; never rewrite sealed historical JSON.')
    P.save(bundle / 'index.json', index)
    P.save(bundle / 'dependency.json', dict(name='priority_teacher_v1',
           manifest_sha256=P.sha(dependency / 'manifest.json'), version=P.VERSION,
           reused='10 CAPs, 904 trajectories including the original270,760-case development matrix, compiler toolchain'))
    sources = P.source_hashes()
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = P.sha(__file__)
    P.save(bundle / 'runtime_sources.json', sources)

    # One real boundary is a portable feature/model-input acceptance reference.
    f = M.features('joint')
    try:
        tr = next(M.V.O.extracted_rows(M.V.unpack(resolve(base[0]['trajectory'], bundle, dependency)), f, max_samples=1))
        arrays = {}
        for j, p in enumerate(tr['panels']):
            arrays[f'context{j}'] = p['attribution_context']
            arrays[f'candidates{j}'] = p['attribution_candidates']
        np.savez_compressed(bundle / 'feature_golden.npz', **arrays)
        P.save(bundle / 'feature_golden.json', dict(sample=base[0], indices=tr['indices'],
               history_hash=tr['history_hash'], layout=f.layout, source_sha=f.sha256))
    finally:
        f.close()
    report = ROOT / 'docs/agent_exchange/reviews/DA_Joint_Representation_20261117.md'
    dest = bundle / 'report'; dest.mkdir()
    text = report.read_text().replace('../../../outputs/semantic_rl/da_joint_20261117_01/', '../evidence/da_joint/')
    text = text.replace('../../../outputs/semantic_rl_authors/da_joint_20261117_01/', '../evidence/host/')
    text = text.replace('../tasks/DA_Joint_Representation_20261117.md', 'protocol.md')
    text = text.replace('../../development/DA_Joint_Representation_20261117.md', 'implementation.md')
    (dest / 'results.md').write_text(text)
    shutil.copyfile(ROOT / 'docs/agent_exchange/tasks/DA_Joint_Representation_20261117.md', dest / 'protocol.md')
    shutil.copyfile(ROOT / 'docs/development/DA_Joint_Representation_20261117.md', dest / 'implementation.md')
    plan_name = 'CAP_Program_Trace_Selection_Plan_v6_20261117.md'
    plan = (ROOT / 'docs/research' / plan_name).read_text()
    plan = plan.replace('../agent_exchange/reviews/DA_Joint_Representation_20261117.md', 'results.md')
    plan = plan.replace('../development/Program_Trace_Transfer_20261117.md', '../README.md')
    (dest / 'plan.md').write_text(plan)
    instructions = (ROOT / 'docs/development/Program_Trace_Transfer_20261117.md').read_text()
    instructions = instructions.replace('../research/' + plan_name, 'report/plan.md')
    (bundle / 'README.md').write_text(instructions)
    P.save(bundle / 'provenance.json', dict(
        source_git_parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_identity='runtime_sources.json is authoritative, including new files pending the handoff commit',
        original_inventory_sha256=P.sha(STUDY / 'artifact_manifest.json'),
        original_report_sha256=P.sha(report), copied_study_files=len(copied),
        subset_not_full_campaign=True, omitted='derived .npy arrays and per-shard cache metadata',
        new_program_instrumentation_implemented=False, training_started=False))
    files = {str(p.relative_to(bundle)): P.sha(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    P.save(bundle / 'manifest.json', dict(version=VERSION, files=files))
    print(json.dumps(dict(bundle=str(bundle), files=len(files), bytes=sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()))))


def verify(bundle, dependency, data_only=False):
    bundle, dependency = Path(bundle).resolve(), Path(dependency).resolve()
    manifest = check_inventory(bundle)
    if P.sha(dependency / 'manifest.json') != P.read(bundle / 'dependency.json')['manifest_sha256']:
        raise ValueError('dependency is not the frozen priority bundle')
    # Data-only mode does not assert runtime or feature equivalence.
    if data_only:
        for rel, digest in P.read(dependency / 'manifest.json')['files'].items():
            if P.sha(P.inside(dependency, rel)) != digest:
                raise ValueError('dependency checksum: ' + rel)
    else:
        P.verify(dependency, check_features=False)
        for rel, digest in P.read(bundle / 'runtime_sources.json').items():
            if P.sha(P.inside(ROOT, rel)) != digest:
                raise ValueError('runtime source mismatch: ' + rel)
    index = P.read(bundle / 'index.json')
    refs = [r['trajectory'] for r in index['original270'] + index['repair_suffixes'] + index['replays']]
    refs += [r['reference'] for r in index['handoff_states']] + [r['checkpoint'] for r in index['models']]
    if 'diagnostic_donor' in index:
        refs.append(index['diagnostic_donor'])
    checked = {}
    for ref in refs:
        path = resolve(ref, bundle, dependency)
        if path not in checked:
            checked[path] = P.sha(path)
        if checked[path] != ref['sha256']:
            raise ValueError('index reference checksum')
    if not data_only:
        import numpy as np
        import torch
        from experiments.planar_tasks.authoring_v2 import da_joint_models as M
        from experiments.planar_tasks.semantic_rl.joint_evidence import JointEvidenceFeatures
        torch.set_num_threads(1)
        f = JointEvidenceFeatures(P.entries(dependency), M.V.O.protocol(M.V.O.CFG), 'joint')
        try:
            info = P.read(bundle / 'feature_golden.json')
            tr = next(M.V.O.extracted_rows(M.V.unpack(resolve(info['sample']['trajectory'], bundle, dependency)), f, max_samples=1))
            if tr['indices'] != info['indices'] or tr['history_hash'] != info['history_hash'] or f.layout != info['layout'] or f.sha256 != info['source_sha']:
                raise ValueError('portable feature identity mismatch')
            with np.load(bundle / 'feature_golden.npz') as arrays:
                for j, panel in enumerate(tr['panels']):
                    np.testing.assert_array_equal(panel['attribution_context'], arrays[f'context{j}'])
                    np.testing.assert_array_equal(panel['attribution_candidates'], arrays[f'candidates{j}'])
        finally:
            f.close()
        for entry in index['models']:
            f = JointEvidenceFeatures(P.entries(dependency), M.V.O.protocol(M.V.O.CFG), entry['mode'])
            try:
                model = M.V.I.Scorer.load(resolve(entry['checkpoint'], bundle, dependency))
                expected_layout = dict(info['layout'], evidence_mode=entry['mode'])
                if model.layout != expected_layout or model.semantic_sha256 != f.sha256 or model.action_protocol != f.action_protocol:
                    raise ValueError('model/portable CAP library incompatibility')
                if not all(torch.isfinite(v).all() for v in model.state_dict().values()):
                    raise ValueError('nonfinite model')
            finally:
                f.close()
        from experiments.planar_tasks.semantic_rl.handoff_features import HandoffLibraryFeatures
        f = HandoffLibraryFeatures(P.entries(dependency), M.V.O.protocol(M.V.O.CFG), execution_backend='trusted_inprocess')
        try:
            model = M.V.I.Scorer.load(resolve(index['diagnostic_donor'], bundle, dependency))
            if model.semantic_sha256 != f.sha256 or model.action_protocol != f.action_protocol:
                raise ValueError('historical donor/portable CAP library incompatibility')
        finally:
            f.close()
    return dict(status='verified', files=len(manifest['files']), references=len(checked),
                features_and_12_models_checked=not data_only, historical_donor_checked=not data_only,
                no_physics_or_training=True)


def audit_archive(archive, bundle):
    bundle = Path(bundle)
    expected = dict(P.read(bundle / 'manifest.json')['files'], **{'manifest.json': P.sha(bundle / 'manifest.json')})
    seen = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for entry in stream:
            if entry.isdir():
                continue
            rel = str(Path(entry.name).relative_to(bundle.name))
            if not entry.isfile() or rel not in expected or rel in seen:
                raise ValueError('unexpected archive member')
            h = hashlib.sha256()
            with stream.extractfile(entry) as f:
                for block in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != expected[rel]:
                raise ValueError('archive member checksum')
            seen.add(rel)
    if seen != set(expected):
        raise ValueError('archive missing payload')
    return len(seen)


def pack(bundle, archive):
    bundle, archive = Path(bundle).resolve(), Path(archive).resolve()
    check_inventory(bundle)
    if archive.exists() or archive.is_relative_to(bundle):
        raise ValueError('archive must be new and outside bundle')
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, 'w:gz', compresslevel=1) as out:
        out.add(bundle, arcname=bundle.name)
    files = audit_archive(archive, bundle)
    checksum = P.sha(archive)
    archive.with_name(archive.name + '.sha256').write_text(f'{checksum}  {archive.name}\n')
    return dict(archive=str(archive), files=files, bytes=archive.stat().st_size, sha256=checksum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['export', 'verify', 'pack'])
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--dependency')
    parser.add_argument('--archive')
    parser.add_argument('--data-only', action='store_true')
    args = parser.parse_args()
    if args.command == 'pack':
        if not args.archive: parser.error('--archive required')
        print(json.dumps(pack(args.bundle, args.archive)))
    else:
        if not args.dependency: parser.error('--dependency required')
        if args.command == 'export': export(args.bundle, args.dependency)
        else: print(json.dumps(verify(args.bundle, args.dependency, args.data_only)))


if __name__ == '__main__':
    main()
