"""Analysis-only correction for D003 (no physical replay).

The 16 formal replays are frozen in raw_ticks.jsonl / baseline/*.json.  This
script only recomputes labels and event timestamps after the review of the
analysis code: (1) BRAKE_HOLD must not be treated as a push, and (2) events and
blocker intervals must use the global control tick / t_start, not list index.
"""
from __future__ import annotations
import csv, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.option_rl_v17.contact_timing_audit import (
    VERSION, analyse_case, _minimal_candidate, _write_csv, _timelines, _replay,
    _source_snapshot, _case_instance_goal, _git, spec, _sha256)
from experiments.option_rl_v16.live import atomic_json
from experiments.option_rl_v16.learnability import load_panel


def main():
    started = time.monotonic()
    out = Path('outputs/agent_exchange/D003_20261029_1843').resolve()
    cfg_path = Path('docs/agent_exchange/tasks/D003_config.json')
    cfg = json.loads(cfg_path.read_text())
    panel = Path(cfg['input_paths']['panel']).resolve()
    witness_cases = [c for c in cfg['samples'] if c['kind'] == 'witness']
    failure_cases = [c for c in cfg['samples'] if c['kind'] == 'failure']
    witness_saved = json.loads((out / 'witness.json').read_text())
    witness_results = witness_saved['results']
    raw = [json.loads(line) for line in (out / 'raw_ticks.jsonl').read_text().splitlines() if line.strip()]
    ticks_by = {}
    for tick in raw:
        ticks_by.setdefault(tick['case'], []).append(tick)
    metric_rows = {r['case']: r for r in csv.DictReader(open(out / 'case_metrics.csv'))}
    repro = json.loads((out / 'reproduction.json').read_text())
    repro_fail = {r['case']: r for r in repro['failures']}

    witness_analysis = []
    for r in witness_results:
        case = next(c for c in witness_cases if c['id'] == r['case'])
        inst = _case_instance_goal(cfg, case, panel)
        witness_analysis.append(analyse_case(case, r, inst.goal_centre))
    failure_results = []
    failure_analysis = []
    for case in failure_cases:
        m = metric_rows[case['id']]
        ticks = ticks_by.get(case['id'], [])
        base_path = out / 'baseline' / f"{case['id']}.json"
        baseline_rows = json.loads(base_path.read_text())['rows'] if base_path.exists() else []
        rr = repro_fail.get(case['id'], {})
        source = rr.get('source') or {}
        source_ok = bool(source.get('ok', m.get('source_ok') == 'True'))
        result = dict(
            case=case['id'], kind='failure', root=case['root'],
            baseline=dict(success=m.get('baseline_success') == 'True', termination=None,
                          periods=len(baseline_rows), reward=None, rows=len(baseline_rows)),
            instrumented=dict(success=m.get('success') == 'True', termination=m.get('termination') or None,
                              periods=int(m.get('periods', 0)), reward=None, rows=len(ticks)),
            ticks=dict(case=case['id'], run='failure', ticks=ticks),
            source=source, baseline_source=source,
            noninterference_error=float(m['noninterference_error']) if m.get('noninterference_error') not in (None, '', 'None') else None,
            d002_overlap_error=float(m['d002_overlap_error']) if m.get('d002_overlap_error') not in (None, '', 'None') else None,
            old_error=None,
            status=m.get('status') or ('ok' if source_ok else 'reproduction_failed'),
        )
        failure_results.append(result)
        inst = _case_instance_goal(cfg, case, panel)
        failure_analysis.append(analyse_case(case, result, inst.goal_centre))

    all_cases = witness_cases + failure_cases
    all_results = witness_results + failure_results
    all_analyses = witness_analysis + failure_analysis
    metrics = [a['metrics'] for a in all_analyses]
    goals = {}
    for case in all_cases:
        inst = _case_instance_goal(cfg, case, panel)
        goals[case['id']] = inst.goal_centre
    display_cases = [{**case, **result} for case, result in zip(all_cases, all_results)]

    events_rows = []
    blocker_rows = []
    for ana, case, result in zip(all_analyses, all_cases, all_results):
        events = ana.get('events') or {}
        labels = ana.get('labels') or []
        ticks = (result.get('ticks') or {}).get('ticks', [])

        def tick_at(i):
            return int(ticks[i]['tick']) if ticks and i is not None and 0 <= i < len(ticks) else None

        def time_at(i):
            return float(ticks[i]['t_start']) if ticks and i is not None and 0 <= i < len(ticks) else None

        for i in events.get('onsets', []):
            events_rows.append(dict(case=case['id'], event='both_contact_onset', tick=tick_at(i), time=time_at(i)))
        for i in events.get('losses', []):
            events_rows.append(dict(case=case['id'], event='both_contact_loss', tick=tick_at(i), time=time_at(i)))
        for i, side in events.get('ready_updates', []):
            events_rows.append(dict(case=case['id'], event='contact_ready_update', tick=tick_at(i), time=time_at(i), side=side))
        for i in events.get('activation_ticks', []):
            events_rows.append(dict(case=case['id'], event='joint_active_transition', tick=tick_at(i), time=time_at(i)))
        for label in labels:
            i = label.get('tick', label.get('start'))
            events_rows.append(dict(case=case['id'], event=label['kind'], tick=tick_at(i), time=time_at(i),
                                    details=label))
        for b in ana.get('blockers') or []:
            blocker_rows.append(dict(case=case['id'], start_tick=tick_at(b['start']), end_tick=tick_at(b['end']),
                                     start_time=time_at(b['start']), end_time=time_at(b['end']),
                                     duration_seconds=(b['end'] - b['start'] + 1) * spec.CONTROL_DT,
                                     blockers=b['blockers']))
    _write_csv(out / 'case_metrics.csv', metrics)
    _write_csv(out / 'events.csv', events_rows)
    _write_csv(out / 'blocker_intervals.csv', blocker_rows)
    candidate = _minimal_candidate(metrics)
    reproduction = dict(
        witness_ok=bool(witness_saved.get('ok')), failure_ok=all(r['status'] == 'ok' for r in failure_results),
        errors=[], analysis_only_retry=True,
        witness=[dict(case=r['case'], status=r.get('status'), old_error=r.get('old_error'),
                      noninterference_error=r.get('noninterference_error'),
                      old_success=(r.get('source') or {}).get('old_success'),
                      old_rows=(r.get('source') or {}).get('old_rows'),
                      stable_ticks=a['metrics']['stable_ticks'],
                      both_active_ticks=a['metrics']['both_active_ticks'])
                 for r, a in zip(witness_results, witness_analysis)],
        failures=[dict(case=r['case'], status=r.get('status'), source=r.get('source'),
                       noninterference_error=r.get('noninterference_error'),
                       d002_overlap_error=r.get('d002_overlap_error')) for r in failure_results],
    )
    atomic_json(out / 'reproduction.json', reproduction)
    _timelines(out, display_cases, all_analyses)
    _replay(out, display_cases, goals)
    source_hashes = _source_snapshot(out, cfg)
    post = Path(__file__).resolve()
    source_hashes[str(post.relative_to(ROOT))] = _sha256(post)
    snapshot = out / 'source_snapshot'
    dest = snapshot / str(post.relative_to(ROOT))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(post.read_bytes())
    atomic_json(out / 'source_hashes.json', source_hashes)
    status = 'complete' if (witness_saved.get('ok') and all(r['status'] == 'ok' for r in failure_results)) else 'incomplete'
    manifest = dict(version=VERSION, status=status, witness_ok=bool(witness_saved.get('ok')),
                    failure_ok=all(r['status'] == 'ok' for r in failure_results),
                    errors=[], cases=len(all_results), replays=16, physical_replays=16,
                    formal_physical_seconds=5.75 + 10.52,
                    analysis_only_retry=True,
                    analysis_retry_reason='corrected BRAKE_HOLD push misclassification and global-tick event/blocker timestamps; raw physical ticks and baseline trajectories unchanged',
                    elapsed_seconds=time.monotonic() - started,
                    workers=4, input_hashes=cfg['input_sha256'], source_hashes=source_hashes,
                    head=_git('rev-parse', 'HEAD'),
                    dirty_files=len(_git('status', '--porcelain').splitlines()),
                    candidate=candidate, command=sys.argv)
    atomic_json(out / 'manifest.json', manifest)
    atomic_json(out / 'summary.json', dict(version=VERSION, status=status, candidate=candidate,
                                           reproduction=reproduction, metrics=metrics, errors=[],
                                           manifest=manifest))
    print(json.dumps(dict(status=status, candidate=candidate,
                          labels={m['case']: [l['kind'] for l in m['labels']] for m in metrics},
                          elapsed=round(manifest['elapsed_seconds'], 1)), indent=1))


if __name__ == '__main__':
    main()
