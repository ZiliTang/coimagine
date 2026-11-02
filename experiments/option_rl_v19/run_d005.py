"""Run D005: registry tables, 8-case regression, and one headless session demo."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.option_rl_v17 import contact_timing_audit as A
from experiments.option_rl_v19 import spec as S
from experiments.option_rl_v19.session import TestbedSession
from experiments.option_rl_v16.live import atomic_json
from experiments.option_rl_v15.render_episodes import instance_from_dict


def _child(job):
    A.make_env = S.make_env
    case = job['case']
    if case['kind'] == 'witness':
        return A.run_witness(job['cfg'], case, 'witness')
    return A.run_failure(job['cfg'], case, 'failure', Path(job['d001']), Path(job['d002']), Path(job['panel']))


def _case_row(case, r):
    old = r.get('source') or {}
    source_ok = old.get('ok') if case['kind'] == 'failure' else None
    ni = r.get('noninterference_error')
    if case['kind'] == 'witness':
        status = r.get('status')
    else:
        status = 'ok' if (source_ok and ni is not None and ni <= 1e-9) else 'source_failed'
    return dict(case=case['id'], kind=case['kind'], role=case.get('role'),
                success=bool(r.get('instrumented', {}).get('success')),
                periods=int(r.get('instrumented', {}).get('periods', 0)),
                source_ok=source_ok, raw_error=old.get('raw_error'),
                mask_error=old.get('mask_error'), pose_error=old.get('pose_error'),
                d002_overlap=r.get('d002_overlap_error'), noninterference=ni,
                old_error=r.get('old_error'), status=status)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--d001', type=Path, required=True)
    parser.add_argument('--d002', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    started = time.monotonic()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit('output directory must be new/empty')
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(args.config.read_text())
    d3 = json.loads((ROOT / 'docs/agent_exchange/tasks/D003_config.json').read_text())
    S.write_tables()
    tables = dict(option=S.OPTION_TABLE, state=S.STATE_TABLE, param=S.PARAM_TABLE)
    for p in tables.values():
        (out / p.name).write_bytes(p.read_bytes())
    witness_cases = [c for c in cfg['samples'] if c['kind'] == 'witness']
    failure_cases = [c for c in cfg['samples'] if c['kind'] == 'failure']
    jobs = [dict(case=c, cfg=d3, d001=str(args.d001.resolve()), d002=str(args.d002.resolve()),
                 panel=str(args.panel.resolve())) for c in witness_cases + failure_cases]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        results = list(ex.map(_child, jobs))
    by_case = {r['case']: r for r in results}
    rows = [_case_row(c, by_case[c['id']]) for c in witness_cases + failure_cases]
    A._write_csv(out / 'case_metrics.csv', rows)
    # One headless session on W_N's witness instance, using its command sequence.
    wlog = json.loads((ROOT / 'outputs/option_rl_v14/learnability_002/logs/dual_2.json').read_text())
    inst = instance_from_dict(wlog['instance'])
    session = TestbedSession(inst)
    for a0, a1 in wlog['commands']:
        session.add(a0, a1, 1)
    run = session.run()
    session.save(out / 'session_run.json')
    session.save_overrides(out / 'session_overrides.json')
    source_files = [Path(__file__), ROOT / 'experiments/option_rl_v19/spec.py',
                    ROOT / 'experiments/option_rl_v19/system.py',
                    ROOT / 'experiments/option_rl_v19/session.py',
                    ROOT / 'tests/test_d005_option_system.py',
                    ROOT / 'docs/agent_exchange/tasks/D005_Option_System.md',
                    ROOT / 'docs/agent_exchange/tasks/D005_config.json']
    source_hashes = {str(f.relative_to(ROOT)): hashlib.sha256(f.read_bytes()).hexdigest()
                     for f in source_files if f.exists()}
    snap = out / 'source_snapshot'
    for f in source_files:
        if f.exists():
            dest = snap / f.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(f.read_bytes())
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=1))
    manifest = dict(
        version='v19-option-system-v1', status='complete',
        head=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
        dirty_files=len(subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout.splitlines()),
        source_hashes=source_hashes,
        option_rows=len(S.SPECS),
        state_rows=len(S.STATE_TABLE.read_text().splitlines()) - 1,
        param_rows=len(S.PARAM_TABLE.read_text().splitlines()) - 1,
        option_table_sha256=__import__('hashlib').sha256(S.OPTION_TABLE.read_bytes()).hexdigest(),
        state_table_sha256=__import__('hashlib').sha256(S.STATE_TABLE.read_bytes()).hexdigest(),
        witness_ok=all(by_case[c['id']]['status'] == 'ok' for c in witness_cases),
        failure_ok=all(bool((by_case[c['id']].get('source') or {}).get('ok')) and
                       by_case[c['id']].get('noninterference_error') is not None and
                       by_case[c['id']]['noninterference_error'] <= 1e-9 for c in failure_cases),
        cases=len(results), elapsed_seconds=time.monotonic() - started,
        session=dict(success=run['success'], periods=run['periods'], frames=len(run['frames']),
                     sequence=len(run['sequence'])),
        workers=args.workers, command=sys.argv)
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'summary.json', dict(status='complete', manifest=manifest, rows=rows,
                                           session=dict(success=run['success'], periods=run['periods'],
                                                        frames=len(run['frames']))))
    print(json.dumps(dict(status='complete', option_rows=manifest['option_rows'],
                          state_rows=manifest['state_rows'], param_rows=manifest['param_rows'],
                          witness_ok=manifest['witness_ok'], failure_ok=manifest['failure_ok'],
                          session=manifest['session'], elapsed=round(manifest['elapsed_seconds'], 2)),
                     indent=1))


if __name__ == '__main__':
    main()
