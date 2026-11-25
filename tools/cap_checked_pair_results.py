"""Export room-paired outcomes and stage-level checks for the research journal."""
import csv
import gzip
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

from tools import cap_checked_pair_evaluate as ev
from tools.cap_safe_nominal_permits import audit


OUT = ev.DEFAULT_OUT
DOC = ev.ROOT / 'docs/research/key_findings/2026-11-24_checker_author_pair'


def first_change(room, old, new):
    row = dict(room=room, success=new['result']['success'], changed=False)
    for a, b in zip(old['steps'], new['steps']):
        if a['tick'] != b['tick'] or a['public_history'] != b['public_history']:
            raise ValueError(f'unmatched public prefix in {room}')
        if a['command'] != b['command']:
            row.update(changed=True, tick=a['tick'], parent_push=a['target_push'], checked_push=b['target_push'],
                       parent_gain_m=a['goal_gain_m'], checked_gain_m=b['goal_gain_m'],
                       parent_periods=a['end']-a['tick'], checked_periods=b['end']-b['tick'])
            return row
    if old['result'] != new['result'] or len(old['steps']) != len(new['steps']):
        raise ValueError(f'different terminal results on identical controls in {room}')
    return row


def export():
    results = ev.read(OUT / 'result.json')
    sealed = ev.read(OUT / 'compositions.json')
    records = ev.read(OUT / 'records.json')
    if results['execution_count'] != len(ev.ROLES) * len(records):
        raise ValueError('room-paired export requires all frozen runs')
    extra = ev.read(OUT / 'dev_selected/result.json')
    extra_label = ev.read(OUT / 'dev_selected/protocol.json')['source']['composition']
    changes = []
    with (DOC / 'rooms.csv').open('w', newline='') as file:
        names = {role: sealed['arms'][role]['composition'] for role in ev.ROLES}
        writer = csv.writer(file)
        writer.writerow(['room', 'stratum', 'content_sha256', *names.values(), extra_label])
        for room, fixture in sorted(records.items()):
            rows = {role: ev.read(OUT / 'runs' / role / (room+'.json')) for role in ev.ROLES}
            writer.writerow([room, fixture['stratum'], fixture['content_sha256'],
                             *(int(row['result']['success']) for row in rows.values()), extra['control_rooms'][room]])
            changes.append(first_change(room, rows['control'], rows['checked']))
    with (DOC / 'first_changes.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for r in changes for k in r)))
        writer.writeheader()
        writer.writerows(changes)
    shutil.copyfile(OUT/'dev_selected/result.json', DOC/'selection_result.json')
    shutil.copyfile(OUT/'dev_selected/protocol.json', DOC/'selection_composition.json')
    for name in ['result.json', 'compositions.json', 'effect_audit_summary.json', 'effect_audit_rows.csv']:
        shutil.copyfile(OUT / name, DOC / name)
    return results


def issued_expectations():
    """Reuse the original permit analysis for matching, declared PUSH-endpoint envelopes."""
    from collections import Counter

    subset = ev.read(OUT / 'full_trace_subset.json')['rooms']
    evidence = {}
    with TemporaryDirectory() as tmp:
        for role in ev.ROLES:
            totals, continuation = Counter(), Counter()
            unsupported = []
            for room in subset.values():
                src = OUT / 'full_traces' / role / (room+'.json.gz')
                if ev.sha(src) != ev.read(OUT / 'runs' / role / (room+'.json'))['full_trace_sha256']:
                    raise ValueError('source trajectory changed')
                path = Path(tmp) / (room+'.json')
                path.write_bytes(gzip.decompress(src.read_bytes()))
                try:
                    counts, _, rows = audit(path)
                except (KeyError, TypeError, ValueError) as exc:
                    unsupported.append(dict(room=room, detail=str(exc)))
                    continue
                totals.update(counts)
                continuation.update(row['followup_120'] for row in rows)
            evidence[role] = dict(counts=dict(totals), followup=dict(continuation),
                                  rooms_with_unrecognized_schema=unsupported)
    ev.immutable(OUT / 'issued_permits.json', dict(rooms=subset, roles=evidence,
                 measure='truth endpoints vs matching issued PUSH motion envelope; original permit schema; individual pushes within a room are dependent'))
    shutil.copyfile(OUT / 'issued_permits.json', DOC / 'issued_permits.json')
    return evidence


if __name__ == '__main__':
    export()
    print(json.dumps(issued_expectations(), ensure_ascii=False, indent=2))
