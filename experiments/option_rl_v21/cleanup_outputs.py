"""Remove redundant V1 rollout products after verifying the frozen baseline.

Default is a dry run. Only the finished event V1 run, its aborted launch and
pilot are eligible; old research diagnostics and testbed/user presets stay put.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'outputs/rl_baselines/V1_event_progress_20261103'
RUN = ROOT/'outputs/option_rl_v20/event_12h_20261103_v2/run'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if json.loads((RUN/'result.json').read_text())['status'] != 'wall_budget_reached':
        raise RuntimeError('refuse to prune an unfinished run')
    hashes = json.loads((BASE/'SHA256.json').read_text())
    for name, expected in hashes.items():
        p = BASE/name
        if hashlib.sha256(p.read_bytes()).hexdigest() != expected:
            raise RuntimeError('baseline mismatch: '+name)
    summary = json.loads((BASE/'baseline.json').read_text())
    retained = {f'iteration_{i:06d}.json' for i in summary['evaluations_kept']}
    candidates = [RUN/'snapshots', RUN/'worker_snapshot', RUN/'checkpoint.pt']
    candidates += [p for p in (RUN/'evaluations').glob('*.json') if p.name not in retained]
    # Preserve small configuration/result evidence from the two superseded runs.
    obsolete = [ROOT/'outputs/option_rl_v20/event_12h_20261103',
                ROOT/'outputs/option_rl_v20/pilot_20261103']
    for folder in obsolete:
        if folder.exists():
            for p in folder.rglob('*'):
                if p.is_file() and p.name in ('plan.json', 'manifest.json', 'result.json', 'STOP.json'):
                    dest = BASE/'superseded_run_metadata'/p.relative_to(ROOT/'outputs/option_rl_v20')
                    if args.apply:
                        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, dest)
            candidates.append(folder)
    files = []
    for p in candidates:
        if p.is_symlink():
            raise RuntimeError('refuse symlink: '+str(p))
        for f in (p.rglob('*') if p.is_dir() else (p,)):
            if f.is_file():
                if f.is_symlink():
                    raise RuntimeError('refuse symlink: '+str(f))
                files.append(dict(path=str(f.relative_to(ROOT)), bytes=f.stat().st_size))
    record = dict(applied=args.apply, at=datetime.datetime.now().astimezone().isoformat(),
                  bytes=sum(f['bytes'] for f in files), files=files,
                  kept_baseline=str(BASE.relative_to(ROOT)),
                  note='Pruned dense intermediate evaluations/periodic checkpoints, worker cache and superseded smoke/pilot. No testbed runs or historical diagnostic reports removed.')
    report = ROOT/'outputs/rl_baselines/V1_cleanup_20261103.json'
    report.write_text(json.dumps(record, indent=2)+'\n')
    if args.apply:
        for p in candidates:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink(missing_ok=True)
    print(json.dumps({k: v for k, v in record.items() if k != 'files'}, indent=2))
    print('files:', len(files))


if __name__ == '__main__':
    main()
