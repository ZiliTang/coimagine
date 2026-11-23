import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from experiments.planar_tasks.relational_search.contract import digest

B = Path('outputs/transfers/cap_trace_value_dev_v1')
A = Path('outputs/transfers/cap_trace_value_target_acceptance_20261118')

def read(name):
    return json.loads((B / name).read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

rows = read('index.json')
inventory = read('inventory.json')
assert [{k: v for k, v in r.items() if k != 'path'} for r in rows] == inventory
assert read('gaps.json') == []
assert all(not r['issues'] and not r['missing_fields'] for r in rows)
assert len({r['path'] for r in rows}) == 5320
assert all((B / r['path']).stat().st_size == r['bytes'] for r in rows)
per_cap = {}
for cap in read('protocol.json')['caps']:
    selected = [r for r in rows if r['cap_id'] == cap]
    assert len(selected) == 760
    per_cap[cap] = dict(
        success=sum(r['success'] for r in selected),
        failure=sum(not r['success'] for r in selected),
        anomalous=sum(r['completion'] != 'complete' for r in selected),
        decisions=sum(r['decisions'] for r in selected),
        observation_boundaries=sum(r['boundaries'] for r in selected),
        missing_fields=sorted({f for r in selected for f in r['missing_fields']}),
        stop_counts=dict(Counter(r['stop'] for r in selected)),
    )
summary = dict(complete=True, per_cap=per_cap, records=len(rows),
               decision_boundaries=sum(r['decisions'] for r in rows),
               observation_boundaries=sum(r['boundaries'] for r in rows))
assert summary == read('audit_summary.json')
assert (summary['decision_boundaries'], summary['observation_boundaries']) == (369726, 375046)

old = Path('outputs/transfers/priority_teacher_v1')
old_manifest = json.loads((old / 'manifest.json').read_text())
old_files = {h: old / rel for rel, h in old_manifest['files'].items() if rel.endswith('.json.gz')}
overlap = [r for r in rows if r['sha256'] in old_files]
assert len(overlap) == 810
assert all(sha(old_files[r['sha256']]) == r['sha256'] for r in overlap)
assert all(r['success'] for r in overlap)

identities = read('used_scene_identities.json')
assert all(set(r).issubset({'case_id', 'fixture_sha256', 'content_sha256', 'role', 'provenance'}) for r in identities)
assert all(len(r['fixture_sha256']) == len(r['content_sha256']) == 64 for r in identities)
development = [r for r in identities if r['role'] == 'development760']
assert len(development) == len({r['case_id'] for r in development}) == 760
by_id = {r['case_id']: r for r in development}
for scene in read('scenes.json'):
    r = by_id[scene['case_id']]
    assert r['fixture_sha256'] == digest(scene['fixture'])
    assert r['content_sha256'] == digest({k: v for k, v in scene['fixture'].items() if k != 'scene_id'})

budget_rows = [r for r in rows if r['stop'] == 'decision_budget']
assert len(budget_rows) == 64
budget_details = []
for r in budget_rows:
    with gzip.open(B / r['path'], 'rt') as f:
        d = json.load(f)
    s = d['summary']
    assert s['decisions'] == 300 and s['stop'] == 'decision_budget'
    assert s['success'] is False and s.get('error') is None and s['termination'] is None
    assert len(d['events']) == 300 and len(d['public_rows']) == 301
    assert sum(e['periods'] for e in d['events']) == s['ticks'] < 1800
    assert all(e['periods'] == e['end_tick'] - e['start_tick'] for e in d['events'])
    budget_details.append(dict(case_id=r['case_id'], cap_id=r['cap_id'], ticks=s['ticks'],
                               original_status=s['status'], original_bootstrap=s.get('bootstrap'),
                               task_cost_seconds=s.get('task_cost_seconds'),
                               final_event_reason=d['events'][-1]['reason'],
                               events_shorter_than_command=sum(e['periods'] < e['executed']['ticks'] for e in d['events'])))

runtime = read('runtime_sources.json')
assert all(sha(Path(rel)) == value for rel, value in runtime.items())
assert all(sha(B / r['path']) == r['sha256'] for r in read('library.json'))
result = dict(
    verified=True, index_inventory_equal=True, audit_summary_recomputed=summary,
    original_gzip_overlap=810, additional_original_runs=4510,
    source_files_matched=len(runtime),
    scene_identity_rows=len(identities),
    identity_roles=dict(Counter(r['role'] for r in identities)),
    unique_fixture_hashes=len({r['fixture_sha256'] for r in identities}),
    unique_content_hashes=len({r['content_sha256'] for r in identities}),
    development_content_hashes=len({r['content_sha256'] for r in development}),
    development_content_hashes_verified=True,
    identity_registry_contains_no_result_fields=True,
    stop_counts=dict(Counter(r['stop'] for r in rows)),
    termination_counts=dict(Counter(str(r['termination']) for r in rows)),
    decision_budget_records=budget_details,
    decision_budget_original_statuses=dict(Counter(r['original_status'] for r in budget_details)),
    decision_budget_tick_range=[min(r['ticks'] for r in budget_details), max(r['ticks'] for r in budget_details)],
    note='Data acceptance only; no CAP execution, trace recomputation, simulation, or training.',
)
(A / 'supplemental_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
print(json.dumps({k: v for k, v in result.items() if k not in ('decision_budget_records', 'audit_summary_recomputed')}, ensure_ascii=False, indent=2))
