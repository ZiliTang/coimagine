"""Separate physical contact, distance keeping, and paused joint recovery."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .post_push_follow_audit import audit as follow_audit
else:
    from post_push_follow_audit import audit as follow_audit


def audit(record):
    result = follow_audit(record)
    cfg, rows = record["config"], record["rows"]
    recoveries = []
    events = record["boundary"]["execution"]["events"]
    for start in [e for e in events if e["event"] == "recovery_started"]:
        at = start["time"]
        # Row timestamps are step END; phase decisions read prior-step reports.
        previous = next(r for r in rows if abs(r["time"]-at) < 1e-8)
        assert any((previous[f"r{i}"].get("effect") or {}).get("recovery_requested") for i in (0, 1))
        finish = next((e for e in events if e["event"] in ("recovery_resumed", "recovery_failed") and e["time"] > at), None)
        during = [r for r in rows if at < r["time"] <= (finish["time"] if finish else float('inf')) + 1e-9]
        assert all(r[f"r{i}"]["stage"] in ("recover", "done") for r in during for i in (0, 1))
        if finish and finish["event"] == "recovery_resumed":
            ready = [r for r in rows if finish["time"]-cfg["recovery_ready_seconds"]-1e-9 <= r["time"] <= finish["time"]+1e-9]
            assert all(r[f"r{i}"]["effect"]["recovery_ready"] for r in ready[-2:] for i in (0, 1))
            assert all(r[f"r{i}"]["gap"] <= cfg["recovery_resume_gap"] for r in ready[-2:] for i in (0, 1))
        recoveries.append({"start_s": at, "end_s": finish["time"] if finish else None,
                           "duration_s": finish["time"]-at if finish else None,
                           "result": finish["event"] if finish else "window_ended",
                           "requested": start["requested"],
                           "max_observed_gap_m": max(r[f"r{i}"]["gap"] for r in during for i in (0, 1)
                                                      if r[f"r{i}"]["gap"] is not None)})
    for row in rows:
        for i in (0, 1):
            info = row[f"r{i}"]
            pulse = info.get("command_disturbance")
            assert math.hypot(*row["forces"][i]) <= 15 + 1e-9
            if pulse:
                assert record["case"].get("disturbance")
                expected = [pulse["nominal_force"][j]+pulse["added_force"][j] for j in (0, 1)]
                size = math.hypot(*expected)
                if size > 15:
                    expected = [v*15/size for v in expected]
                assert math.dist(expected, row["forces"][i]) < 1e-8
    result.update(recovery=record["metric"]["variant"], recoveries=recoveries,
                  diagnostic_disturbance=record["case"].get("disturbance"))
    return result


def main(out):
    m = json.loads((out/'manifest.json').read_text())
    records = [json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))]
    results = []
    for r in records:
        if r['metric']['status'] == 'completed':
            r['manifest_actions'] = m['actions']
            results.append(audit(r))
    report = {"version": "joint-contact-recovery-audit-v1", "results": results,
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('joint_contact_recovery_audit.py', 'normal_travel_audit.py', 'post_push_follow_audit.py')},
              "contract": "actual contact and gap keeping are distinct; no physical contact is inferred from proximity"}
    (out/'recovery_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# 按间隙联合恢复审计', '', '实际接触率和距离内保持分开报告；恢复运动不记入主动/随动接触行程。', '',
             '| 场景 | 恢复 | 退出 | 恢复耗时 s | 随动实际接触 A/B | 随动阈值内 A/B | 最大随动间隙 A/B mm |',
             '|---|---|---|---|---|---|---|']
    for r in results:
        fmt = lambda key: ' / '.join('—' if s[key] is None else f'{s[key]:.3f}' for s in r['follow_stats'])
        gap = ' / '.join('—' if s['max_follow_gap_m'] is None else f"{s['max_follow_gap_m']*1000:.2f}" for s in r['follow_stats'])
        durations = ' / '.join('未恢复' if e['duration_s'] is None else f"{e['duration_s']:.2f}" for e in r['recoveries']) or '无触发'
        lines.append(f"| {r['case']} | {r['recovery']} | {' / '.join(r['exits'])} | {durations} | "
                     f"{fmt('contact_fraction')} | {fmt('near_fraction')} | {gap} |")
    (out/'RECOVERY_AUDIT.md').write_text('\n'.join(lines)+'\n')
    plot(records, out)
    print(json.dumps({'audited': len(results), 'samples': sum(r['checked_samples'] for r in results),
                      'max_error_m': max(r['max_reconstruction_error_m'] for r in results),
                      'recovery_events': sum(len(r['recoveries']) for r in results)}))


def plot(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['N_E_0.5_0.1', 'same_N_0.1_0.5_outward_pulse', 'N_E_0.5_0.1_outward_pulse']
    names = [n for n in names if any(r['metric']['case'] == n for r in records)]
    fig, axes = plt.subplots(2, len(names), figsize=(4.5*len(names),7), squeeze=False, constrained_layout=True)
    for col, name in enumerate(names):
        side = 0 if name.startswith('same_N') else 1
        for enabled, color in ((False, '#c08a38'), (True, '#218d70')):
            r = next(r for r in records if r['metric']['case']==name and r['metric']['variant']==enabled)
            selected = [v for v in r['rows'] if (v[f'r{side}'].get('effect') or {}).get('follow_started')]
            t = [v['time'] for v in selected]
            axes[0,col].plot(t, [v[f'r{side}']['gap']*1000 for v in selected], color=color,
                             label='joint gap recovery' if enabled else 'binary loss abort')
            long = 1-side
            active = [v for v in r['rows'] if v[f'r{long}'].get('effect')]
            axes[1,col].plot([v['time'] for v in active], [v[f'r{long}']['effect']['normal_travel'] for v in active],
                             color=color, label='long side active / '+('recover' if enabled else 'abort'))
            if enabled:
                events = r['boundary']['execution']['events']
                for e in events:
                    if e['event']=='recovery_started':
                        end = next((v['time'] for v in events if v['event']=='recovery_resumed' and v['time']>e['time']), None)
                        if end:
                            for ax in axes[:,col]:ax.axvspan(e['time'],end,alpha=.12,color='#6b8ac2')
        axes[0,col].axhline(10, linestyle='--', color='#a34a4a', label='pause threshold 10 mm')
        axes[0,col].axhline(3, linestyle=':', color='#6e7a8a', label='resume threshold 3 mm')
        axes[0,col].set(title=name, ylabel='Short side observed gap / mm')
        axes[1,col].set(xlabel='Time / s', ylabel='Long side active travel / m')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Proximity is distinct from contact; larger gaps pause both robots for reapproach')
    fig.savefig(out/'joint_recovery.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
