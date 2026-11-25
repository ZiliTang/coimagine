"""Apply the host-defined effect claim to the preregistered paired trajectory subset."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

from tools.cap_checked_pair_evaluate import DEFAULT_OUT, ROLES, read, sha, verify_panel
from tools.cap_effect_contract import PLANAR_CLAIM, audit_planar_record
from tools.cap_entry_residual_study import immutable


def audit_subset(out):
    plan = verify_panel(out)
    subset = read(out / 'full_trace_subset.json')['rooms']
    records = read(out / 'records.json')
    expected = {s: min(k for k, row in records.items() if row['stratum'] == s)
                for s in plan['counts']}
    if subset != expected:
        raise ValueError('room subset differs from the prespecified selection')
    result = read(out / 'result.json')
    compositions = read(out / 'compositions.json')['arms']
    rows = []
    summary = {}
    for role in ROLES:
        totals = {s: Counter() for s in ('precondition', 'outcome', 'continuation')}
        for stratum, room in subset.items():
            compact = read(out / 'runs' / role / (room + '.json'))
            trace = out / 'full_traces' / role / (room + '.json.gz')
            if sha(trace) != compact['full_trace_sha256']:
                raise ValueError(f'full trace changed for {role} in {room}')
            full = json.loads(gzip.decompress(trace.read_bytes()))
            if (full['result'] != compact['result'] or compact['source_sha256'] != compositions[role]['sha256']
                    or compact['content_sha256'] != records[room]['content_sha256']):
                raise ValueError(f'trace and evaluation identity disagree for {role} in {room}')
            for check in audit_planar_record(full, PLANAR_CLAIM):
                verdict = check['audit']
                row = dict(role=role, room=room, stratum=stratum, tick=check['tick'],
                           end_tick=check['end_tick'])
                row['target_gain_m'] = verdict['outcome']['metrics']['goal_gain_m']['evidence'][0]
                for stage, counts in totals.items():
                    row[stage] = verdict[stage]['verdict']
                    counts[row[stage]] += 1
                rows.append(row)
        summary[role] = {s: dict(c) for s, c in totals.items()}
    immutable(out / 'effect_audit_summary.json', dict(panel=result['panel'], rooms=subset,
              scope='ordinary non-pending goal PUSH, host-defined 2 cm example, next executed command or goal',
              claim=PLANAR_CLAIM, counts=summary))
    with (out / 'effect_audit_rows.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['role', 'room', 'stratum', 'tick', 'end_tick',
                                                  'target_gain_m', 'precondition', 'outcome', 'continuation'])
        writer.writeheader()
        writer.writerows(rows)
    return summary


if __name__ == '__main__':
    print(json.dumps(audit_subset(Path(DEFAULT_OUT)), indent=2))
