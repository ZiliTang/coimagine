"""Submission gate for new direct-proposal CAPs using the v1 audit interface."""
import fcntl
import hashlib
import json
import tempfile
from pathlib import Path

from experiments.planar_tasks.relational_search.motion import CODEBOOK_HASH


PROFILE = 'checked-planar-push-v1'
ROOT = Path(__file__).resolve().parents[3]
CHECKER_FILES = (
    'experiments/planar_tasks/authoring_v2/policy_checks.py',
    'experiments/planar_tasks/authoring_v2/policy_check_specs.py',
    'experiments/planar_tasks/authoring_v2/checked_submission.py',
    'tools/cap_policy_check.py',
    'experiments/planar_tasks/relational_search/contract.py',
    'experiments/planar_tasks/semantic_rl/stream_worker.py',
    'experiments/planar_tasks/semantic_rl/feature_worker.py',
    'experiments/planar_tasks/semantic_rl/stream_history.py',
    'experiments/planar_tasks/semantic_rl/stream_features.py',
    'experiments/planar_tasks/semantic_rl/typed_compiler.py',
    'experiments/planar_tasks/semantic_rl/typed_builder.py',
    'experiments/planar_tasks/sandbox.py',
)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def submit_checked(policy, boundaries, out):
    """Check host-provided observations, then publish policy and evidence together."""
    from tools.cap_policy_check import lint_source, run_boundary

    out = Path(out)
    if out.exists():
        raise FileExistsError(f'submission already exists: {out}')
    if not boundaries:
        raise ValueError('at least one host-provided public boundary is required')
    source_bytes = Path(policy).read_bytes()
    source = source_bytes.decode('utf-8')
    lint_source(source)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.cap-check-', dir=out.parent) as staging:
        folder = Path(staging) / 'submission'
        folder.mkdir()
        (folder / 'policy.py').write_bytes(source_bytes)
        policy_hash = file_hash(folder / 'policy.py')
        cases = []
        for index, path in enumerate(boundaries):
            boundary = json.loads(Path(path).read_bytes())
            report = run_boundary(source, boundary)
            if report['status'] not in ('ok', 'advisory'):
                codes = [f['code'] for f in report['findings'] if f['severity'] == 'error']
                raise ValueError(f'{path}: policy check {report["status"]}: {codes}')
            if report['source_sha256'] != policy_hash or not report['selection_checked']:
                raise ValueError('unchecked source or unbound decision')
            tag = f'{index:04d}'
            write_json(folder / f'{tag}.input.json', boundary)
            write_json(folder / f'{tag}.report.json', report)
            cases.append({'case': tag, 'status': report['status'],
                          'observation_sha256': report['observation_sha256'],
                          'findings': [f['code'] for f in report['findings']]})
        inventory = {p.name: file_hash(p) for p in sorted(folder.iterdir())}
        receipt = {'profile': PROFILE, 'source_sha256': policy_hash, 'codebook_sha256': CODEBOOK_HASH, 'cases': cases,
                   'files': inventory, 'checker': {n: file_hash(ROOT / n) for n in CHECKER_FILES}}
        write_json(folder / 'manifest.json', receipt)
        # Serialize publication only. Concurrent authors may check separate drafts.
        with (out.parent / '.cap-check-publication.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if out.exists():
                raise FileExistsError(f'submission already exists: {out}')
            folder.rename(out)
    return receipt


def verify_checked(submission):
    """Verify that the recorded inputs, decisions, checker and policy still match."""
    folder = Path(submission)
    manifest_file = folder / 'manifest.json'
    if not manifest_file.is_file() or manifest_file.is_symlink():
        raise ValueError('missing submission manifest')
    receipt = json.loads(manifest_file.read_bytes())
    if receipt.get('profile') != PROFILE or not isinstance(receipt.get('cases'), list) or not receipt['cases']:
        raise ValueError('invalid checked CAP profile or empty case inventory')
    if (receipt.get('checker') != {n: file_hash(ROOT / n) for n in CHECKER_FILES}
            or receipt.get('codebook_sha256') != CODEBOOK_HASH):
        raise ValueError('checker version differs from this submission')
    expected = {'policy.py'}
    for index in range(len(receipt['cases'])):
        expected.update((f'{index:04d}.input.json', f'{index:04d}.report.json'))
    if {p.name for p in folder.iterdir()} != expected | {'manifest.json'}:
        raise ValueError('submitted file inventory changed')
    if any((folder / f).is_symlink() or not (folder / f).is_file() for f in expected):
        raise ValueError('invalid file in checked submission')
    actual = {f: file_hash(folder / f) for f in expected}
    if receipt.get('files') != actual or receipt.get('source_sha256') != actual['policy.py']:
        raise ValueError('submitted content changed')
    for index, record in enumerate(receipt['cases']):
        tag = f'{index:04d}'
        report = json.loads((folder / f'{tag}.report.json').read_bytes())
        expected_case = {'case': tag, 'status': report['status'],
                         'observation_sha256': report['observation_sha256'],
                         'findings': [f['code'] for f in report['findings']]}
        if (record != expected_case or report['status'] not in ('ok', 'advisory')
                or report['source_sha256'] != actual['policy.py'] or not report['selection_checked']):
            raise ValueError('recorded decision differs from accepted submission')
    return receipt
