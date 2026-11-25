"""Rerun two endpoint-only PUSHes on a fixed development donor and compare its archive."""

import argparse
from pathlib import Path
from time import perf_counter

from tools.guarded_calibration_v1 import cap_guarded_calibration_forks as forks


ROOM = 'diag_dev_nat_013'
ANCHOR = 'early'


def run():
    rows_path = forks.OUT / 'forks' / 'dev' / (ROOM + '.json')
    particles_path = forks.OUT / 'particles' / 'dev' / (ROOM + '.json')
    packet = forks.archive.read(particles_path)
    fixture = forks.development_rooms()[ROOM]
    provenance = dict(collector_sha256=forks.frozen_hash(), policy_sha256=forks.source_lock()['sha256'],
                      content_sha256=forks.archive.content_hash(fixture),
                      trace_sha256=forks.archive.sha(forks.trace_path('dev', ROOM)))
    if any(packet[key] != value for key, value in provenance.items()):
        raise ValueError('development particle differs from the frozen source')
    anchor = next(a for a in packet['anchors'] if a['anchor_id'] == ANCHOR)
    start = perf_counter()
    actual = []
    for p in forks.PROGRESS:
        t = perf_counter()
        out = forks.simulate_development_particle(ROOM, ANCHOR, p)
        wall_s = perf_counter() - t
        followup_ticks = out.pop('followup_ticks')
        if followup_ticks != 0 or 'followup_completed' in out:
            raise ValueError('particle executed a suffix')
        actual.append(dict(progress_m=p, endpoint=out, followup_ticks=followup_ticks, wall_seconds=wall_s))
    timed_seconds = perf_counter() - start

    # The archive with suffix labels is loaded only after both physics runs.
    archive = forks.archive.read(rows_path)
    if any(archive[key] != value for key, value in provenance.items()):
        raise ValueError('reference archive differs from the frozen source')
    for row in actual:
        matches = [r for r in archive['rows'] if r['anchor_id'] == ANCHOR and r['progress_m'] == row['progress_m']]
        if len(matches) != 1 or matches[0]['public_history'] != anchor['public_history']:
            raise ValueError('archive lacks the same unique public prefix and dose')
        expected = matches[0]['outcome']
        row['mismatches'] = {k: dict(actual=v, expected=expected[k]) for k, v in row['endpoint'].items()
                             if v != expected[k]}
    restore_ticks = sum(r['endpoint']['prefix_ticks'] for r in actual)
    push_ticks = sum(r['endpoint']['push_ticks'] for r in actual)
    return dict(scope='single fixed development donor, independently replayed from its own initial state',
                check='endpoint capability only; compatibility with any deployment query is untested',
                room=ROOM, anchor_id=ANCHOR, tick=anchor['tick'], public_history=anchor['public_history'],
                **provenance, archive_sha256=forks.archive.sha(rows_path),
                all_endpoints_exact=all(not r['mismatches'] for r in actual),
                official_calls_added=0, physics_runs=len(actual), restore_ticks=restore_ticks,
                push_ticks=push_ticks, followup_ticks=0, total_physics_ticks=restore_ticks + push_ticks,
                measured_wall_seconds=timed_seconds, results=actual)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=forks.OUT / 'particle_endpoint_check.json')
    args = parser.parse_args()
    result = run()
    forks.write_json(args.out, result)
    print(f"{result['physics_runs']} development runs, {result['total_physics_ticks']} ticks; "
          f"both endpoints exactly reproduced: {result['all_endpoints_exact']}")
    if not result['all_endpoints_exact']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
