"""Recompute A/B strata and frozen-actor coverage under the current pair profile."""
import csv

from tools import cap_union_trend as study

d, e = study.d, study.e
OUT = study.OUT.parent / 'cap_pair_fixed72_local_20261123_01'
PRIOR = OUT.parent / 'cap_lidar_gate_20261123_01'
DOC = study.DOC.parent / '2026-11-23_lidar_gate'


def score(item):
    key, fixture = item
    a = e.simulate(fixture, 'A', 'stall')['result']
    old = PRIOR / 'natural' / (key + '.json')
    if not old.exists():
        old = PRIOR / 'remaining' / old.name
    reference = e.read(old)['local_fixed']
    assert all(a[k] == reference[k] for k in ('success', 'history', 'ticks'))
    b = e.simulate(fixture, 'B', 'stall')['result']
    actor = d.episode(fixture, 'semantic', label=False)[0]
    e.save(OUT/'rooms'/(key+'.json'),
           {arm:{k:r[k] for k in ('success','ticks','history')} for arm,r in [('A',a),('B',b),('actor',actor)]})


def main():
    profile = d.init()
    tested = (PRIOR/'local_cap.py').read_text()
    assert profile['A'] == e.perception.fingerprint(tested)
    assert e.base.digest(tested) == e.read(PRIOR/'protocol.json')['local']
    d.load('semantic')
    panel = d.rooms(study.FRESH)
    identities = {k:e.s.phase.input_hash(f) for k,f in panel.items()}
    assert identities == e.read(study.OUT/'confirmation_ids.json')
    e.save(OUT/'protocol.json',dict(**profile, executor=e.g.prep.prior.EXECUTOR,
                                   rooms=identities, actor=d.source.graph.file_sha(d.OUT/'fit/semantic.npz')))
    e.pool(score,panel.items(),16)
    with (DOC/'default_pair_rooms.csv').open('w',newline='') as fp:
        writer=csv.writer(fp)
        writer.writerow(('room','content_sha256','A','B','actor'))
        for k in sorted(panel):
            rec=e.read(OUT/'rooms'/(k+'.json'))
            writer.writerow((k,identities[k],*(rec[m]['success'] for m in ('A','B','actor'))))


if __name__=='__main__':
    main()
