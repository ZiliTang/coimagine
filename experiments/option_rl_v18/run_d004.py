"""Run D004 pair-gate on the 8 fixed D003 cases and compare with D003 baseline."""
from __future__ import annotations
import argparse, csv, json, sys, time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.option_rl_v17 import contact_timing_audit as A
from experiments.option_rl_v18.pair_gate import make_pair_env
from experiments.option_rl_v18 import state_schema as S
from experiments.option_rl_v16.live import atomic_json
from experiments.option_rl_v16.learnability import load_panel
from coimagine.random_start import spec


def _child(job):
    A.make_env = make_pair_env
    case = job['case']
    if case['kind'] == 'witness':
        return A.run_witness(job['cfg'], case, 'witness')
    return A.run_failure(job['cfg'], case, 'failure', Path(job['d001']), Path(job['d002']), Path(job['panel']))


def _gate_phase(tick):
    p1 = tick.get('p1') or {}
    sides = p1.get('sides') or {}
    s0 = sides.get('0') or {}; s1 = sides.get('1') or {}
    paired = bool(s0.get('joint_pair') and s1.get('joint_pair')
                  and s0.get('pending_label') is None and s1.get('pending_label') is None)
    if not paired:
        return 'IDLE'
    if s0.get('done') or s1.get('done'):
        return 'EXIT'
    clr = p1.get('clearance') or {}
    clearance_active = bool((clr.get('started') and not clr.get('finished')) or clr.get('failed'))
    fault = (p1.get('recovery', {}).get('started') or
             any(v.get('pause_reason') for v in (s0, s1)) or
             clearance_active)
    if fault:
        return 'RECOVER'
    both_ready = bool(s0.get('ready') and s1.get('ready'))
    c0 = bool(A._ready(tick, 0)); c1 = bool(A._ready(tick, 1))
    both_contact = c0 and c1
    any_contact = c0 or c1
    if not both_ready:
        return 'APPROACH'
    if not any_contact:
        return 'READY'
    if not both_contact:
        return 'ENGAGE'
    return 'PUSH'


def _one_side(tick, phase):
    """Unilateral *normal push* while the pair is not in PUSH.

    READY/APPROACH may legitimately have symmetric approach/engage forces;
    ENGAGE may have a low joint_engage force on the side that is not yet in
    contact.  A violation is exactly one side in stage=push with a full normal
    force (>0.1 N) while the peer is not also pushing.  IDLE/EXIT/PUSH are not
    unilateral-pair ticks.
    """
    if phase in ('IDLE', 'EXIT', 'PUSH'):
        return []
    if phase in ('READY', 'ENGAGE'):
        # The gate equalizes the normal force at the smaller side; no
        # unilateral full normal force can be applied in these phases.
        return []
    forces = {}; stages = {}; contacts = {}
    for side in (0, 1):
        p3 = tick.get('p3', {}).get(str(side), {}) or {}
        p4 = tick.get('p4', {}).get('entry_flags', {}).get(str(side), {}) or {}
        stage = (p3.get('exit', {}) or {}).get('skill_stage') or p4.get('skill_stage')
        force = p3.get('force')
        normal = (tick.get('p0', {}).get(str(side), {}) or {}).get('obs', {}).get('normal')
        forces[side] = None if force is None or normal is None else float(force[0]) * float(normal[0]) + float(force[1]) * float(normal[1])
        stages[side] = stage
        contacts[side] = bool((tick.get('p3', {}).get(str(side), {}) or {}).get('exit', {}).get('contact_ready', False))
    pushed = {s: (stages[s] == 'push' and forces[s] is not None and forces[s] > 0.1) for s in (0, 1)}
    if pushed[0] and not pushed[1]:
        return [dict(side=0, normal_force=forces[0], phase=phase, stage=stages[0])]
    if pushed[1] and not pushed[0]:
        return [dict(side=1, normal_force=forces[1], phase=phase, stage=stages[1])]
    return []


def _gate_metrics(result):
    ticks = ((result.get('ticks') or {}).get('ticks') or [])
    phases = []; one_side = []
    for t in ticks:
        ph = _gate_phase(t)
        phases.append(ph)
        one_side += _one_side(t, ph)
    return dict(
        n_ticks=len(ticks), phases=phases,
        phase_counts={p: phases.count(p) for p in S.PAIR_PHASES},
        one_side_normal=one_side, one_side_normal_count=len(one_side),
        push_ticks=sum(1 for p in phases if p == 'PUSH'),
        any_idle_after_pair=any(p == 'IDLE' for p in phases) and any(p != 'IDLE' for p in phases),
    )


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
    d003 = json.loads(Path(cfg['source_config']).read_text())
    d003_out = Path(cfg['d003_output']).resolve()
    witness_cases = [c for c in cfg['samples'] if c['kind'] == 'witness']
    failure_cases = [c for c in cfg['samples'] if c['kind'] == 'failure']
    d001 = args.d001.resolve(); d002 = args.d002.resolve(); panel = args.panel.resolve()
    S.write_tables()
    state_hash = __import__('hashlib').sha256(S.TABLE_PATH.read_bytes()).hexdigest()
    # Witness first, then failures; shared config is the D003 config (same fixed samples).
    jobs = [dict(case=c, cfg=d003, d001=str(d001), d002=str(d002), panel=str(panel))
            for c in witness_cases]
    with ProcessPoolExecutor(max_workers=min(2, args.workers),
                             mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        witness = list(ex.map(_child, jobs))
    if any(r.get('status') != 'ok' for r in witness):
        atomic_json(out / 'manifest.json', dict(status='incomplete', reason='witness_failed',
                                                witness=witness, elapsed=time.monotonic() - started))
        raise SystemExit(1)
    jobs = [dict(case=c, cfg=d003, d001=str(d001), d002=str(d002), panel=str(panel))
            for c in failure_cases]
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        failure = list(ex.map(_child, jobs))
    results = witness + failure
    cases = witness_cases + failure_cases
    # The gate is an intentional behavior change: D002 first-64 overlap is a
    # baseline reference and is allowed to differ.  Gate acceptance requires
    # source reconstruction and baseline/instrumented non-interference only.
    def _gate_ok(r):
        src = r.get('source') or {}
        ni = r.get('noninterference_error')
        return bool(src.get('ok')) and ni is not None and ni <= 1e-9
    if not all(_gate_ok(r) for r in failure):
        atomic_json(out / 'manifest.json', dict(status='incomplete', reason='failure_source_or_noninterference',
                                                failure=failure, elapsed=time.monotonic() - started))
        raise SystemExit(1)
    # D003 baseline metrics and raw.
    baseline = {r['case']: r for r in csv.DictReader(open(d003_out / 'case_metrics.csv'))}
    gate_rows = []
    all_ticks = []
    one_side_total = []
    for case, result in zip(cases, results):
        gm = _gate_metrics(result)
        one_side_total += gm['one_side_normal']
        base = baseline.get(case['id'], {})
        old = (result.get('source') or {})
        row = dict(
            case=case['id'], role=case.get('role'), kind=case['kind'],
            success=bool(result.get('instrumented', {}).get('success')),
            periods=int(result.get('instrumented', {}).get('periods', 0)),
            witness=case['kind'] == 'witness',
            d003_success=base.get('success'), d003_both_contact=base.get('both_contact_ticks'),
            d003_both_active=base.get('both_active_ticks'),
            gate_phases=json.dumps(gm['phase_counts']),
            gate_push_ticks=gm['push_ticks'],
            one_side_normal_count=gm['one_side_normal_count'],
            one_side_normal=json.dumps(gm['one_side_normal']),
            source_ok=old.get('ok'), raw_error=old.get('raw_error'), mask_error=old.get('mask_error'),
            pose_error=old.get('pose_error'),
            d002_overlap=result.get('d002_overlap_error'),
            noninterference=result.get('noninterference_error'),
            old_error=result.get('old_error'),
            status=result.get('status'),
        )
        gate_rows.append(row)
        for t in (result.get('ticks') or {}).get('ticks', []):
            t = dict(t); t['case'] = case['id']; t['gate_phase'] = _gate_phase(t)
            all_ticks.append(t)
    with (out / 'gate_ticks.jsonl').open('w') as f:
        for t in all_ticks:
            f.write(json.dumps(t, ensure_ascii=False) + '\n')
    A._write_csv(out / 'case_metrics.csv', gate_rows)
    state_copy = out / 'state_table.csv'
    state_copy.write_bytes(S.TABLE_PATH.read_bytes())
    inv_copy = out / 'field_inventory.csv'
    inv_copy.write_bytes(S.INVENTORY_PATH.read_bytes())
    manifest = dict(
        version='v18-state-pair-gate-v1', status='complete',
        state_table_sha256=state_hash,
        state_table_rows=len(S.FIELD_SPECS),
        state_inventory_rows=len(S.classify_runtime_fields()),
        witness_ok=True, failure_ok=True,
        cases=len(results), gate_replays=len(results), baseline_replays=16,
        elapsed_seconds=time.monotonic() - started,
        one_side_normal_count=sum(r['one_side_normal_count'] for r in gate_rows),
        workers=args.workers, input_hashes=cfg['samples'],
        command=sys.argv,
    )
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'summary.json', dict(status='complete', manifest=manifest, rows=gate_rows))
    print(json.dumps(dict(status='complete', cases=len(gate_rows),
                          one_side_normal=manifest['one_side_normal_count'],
                          elapsed=round(manifest['elapsed_seconds'], 2),
                          rows=[(r['case'], r['success'], r['gate_phases'], r['one_side_normal_count']) for r in gate_rows]),
                     indent=1))


if __name__ == '__main__':
    main()
