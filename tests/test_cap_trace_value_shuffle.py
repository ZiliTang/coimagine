from collections import Counter

from experiments.planar_tasks.authoring_v2.cap_trace_value_shuffle import schedule


def test_stratified_schedule_covers_every_root_and_pairs_exploration_by_visit():
    cases = [dict(case_id=f'{n}_{i}', movable_count=n) for n in range(2, 6) for i in range(50)]
    order = [c['case_id'] for c in reversed(cases)]
    rows = schedule(cases, order)
    assert rows == schedule(cases, order)
    assert rows != schedule(cases, order, seed=10)
    assert [r['round'] for r in rows] == list(range(5, 41))
    for cycle in range(9):
        blocks = rows[cycle*4:(cycle+1)*4]
        flat = [item for b in blocks for item in b['cases']]
        assert Counter(i['case_id'] for i in flat) == Counter(order)
        for block in blocks:
            counts = Counter(int(i['case_id'].split('_')[0]) for i in block['cases'])
            assert sorted(counts.values()) == [12, 12, 13, 13]
        for item in flat:
            assert item['action_seed'] == 170000+(cycle+1)*200+order.index(item['case_id'])
    assert rows[0]['cases'] != rows[4]['cases']
