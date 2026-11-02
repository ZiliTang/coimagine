"""Independent outer supervisor for the D003 read-only audit.

The audit and all its workers run in one new process session/group.  The
supervisor is not in that group.  On the hard deadline it sends SIGTERM to the
whole group, gives a short grace period, then SIGKILLs any survivor and records
an explicit incomplete supervisor manifest.
"""
from __future__ import annotations
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run_supervisor(command, out, *, deadline=300., grace=2.):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(out.suffix + '.run.log')
    started = time.time()
    with log_path.open('wb') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        result = dict(supervisor_pid=os.getpid(), child_pid=child.pid,
                      command=[str(c) for c in command], started_at=started,
                      deadline=deadline, grace=grace, status='running')
        (out / 'supervisor.json').write_text(json.dumps(result, indent=1) + '\n')
        timed_out = False
        try:
            code = child.wait(timeout=deadline)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                code = child.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait()
        ended = time.time()
        result.update(exit_code=code, timed_out=timed_out, ended_at=ended,
                      elapsed=ended - started,
                      status='timeout' if timed_out else ('complete' if code == 0 else 'failed'))
        (out / 'supervisor.json').write_text(json.dumps(result, indent=1) + '\n')
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--d001', type=Path, required=True)
    parser.add_argument('--d002', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deadline', type=float, default=300.)
    parser.add_argument('--grace', type=float, default=2.)
    parser.add_argument('--reuse-witness', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    audit = ROOT / 'experiments/option_rl_v17/contact_timing_audit.py'
    command = [sys.executable, '-u', str(audit),
               '--out', str(out), '--config', str(args.config.resolve()),
               '--d001', str(args.d001.resolve()), '--d002', str(args.d002.resolve()),
               '--panel', str(args.panel.resolve()), '--workers', str(args.workers)]
    if args.reuse_witness:
        command.append('--reuse-witness')
    result = run_supervisor(command, out, deadline=args.deadline, grace=args.grace)
    print(json.dumps(result, indent=1))
    if result['status'] != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
