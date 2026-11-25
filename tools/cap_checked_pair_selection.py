"""Secondary paired comparison selected solely by completed development results."""
from pathlib import Path

from tools import cap_checked_pair_author as author
from tools import cap_checked_pair_evaluate as ev


def development_winner(arm):
    base = author.read(author.OUT / 'conformance/result.json')['rooms']
    wins = {k for k, r in base.items() if r['success']}
    scores = [(len(wins), 0, 0, 'parent')]
    ledger = author.OUT / 'ledger' / arm
    for f in sorted(ledger.glob('trial*/result.json')):
        r = author.read(f)
        if r['status'] == 'complete' and set(r['results']) == set(base):
            new = {k for k, row in r['results'].items() if row['success']}
            name = f.parent.name
            scores.append((len(new), -len(wins-new), -int(name[5:]), name))
    return max(scores)[-1]


def run(workers=8):
    out = ev.DEFAULT_OUT
    dest = out / 'dev_selected'
    names = {arm: development_winner(arm) for arm in author.ARMS}
    if names != {'control':'trial01', 'checker':'trial03'}:
        raise ValueError('scores differ from the two development selections registered before outcome inspection')
    ev.verify_panel(out)
    arms = ev.read(out / 'arms.json')
    if arms['checked']['nomination'] != names['checker']:
        raise ValueError('different checked arm')
    role = 'control'
    ref = Path(role) / names[role]
    sub = author.OUT / 'submissions' / ref
    receipt = author.check_receipt(sub)
    arms[role] = dict(campaign=str(author.OUT/'sealed'/ref),sha256=receipt['source_sha256'],
                      submission=str(sub),composition='P常规换位候选.v1@G程序.v1')
    sources, verified = ev.verify_sources(arms)
    records, fixtures = ev.read(out/'records.json'), ev.read(out/'fixtures.json')
    commit = dict(selections=names, source=verified[role], comparison=verified['checked'],
                  protocol_sha256=ev.sha(ev.ROOT / 'docs/research/key_findings/2026-11-24_checker_author_pair/selection_sensitivity.md'),
                  panel_sha256=ev.sha(out/'panel.json'), runner_sha256=ev.sha(Path(ev.__file__)))
    ev.immutable(dest/'protocol.json', commit)
    jobs = [(k, f, verified[role]['push_takeover_callback'], verified[role]['sha256'],
             records[k]['content_sha256'], None) for k, f in fixtures.items()]
    ev.safe.host.old.collect(sources[role], jobs, ev.execution, workers, dest/'runs')
    if not (out / 'result.json').exists():
        return
    rows = {r:{k:ev.read(out/'runs'/r/(k+'.json')) for k in records}
            for r in ('parent','detour','checked')}
    rows[role] = {k:ev.read(dest/'runs'/(k+'.json')) for k in records}
    for r, data in rows.items():
        for k, row in data.items():
            if (row['status'] != 'complete' or row['room'] != k or row['source_sha256'] != verified[r]['sha256']
                    or row['content_sha256'] != records[k]['content_sha256']):
                raise ValueError(f'wrong or incomplete result for {r}, {k}')
    if ev.verify_sources(arms)[1] != verified or ev.sha(Path(ev.__file__)) != commit['runner_sha256']:
        raise ValueError('source or runner changed')
    ev.immutable(dest/'result.json',dict(panel=ev.PANEL, rooms=len(records),
                 successes={r:sum(v['result']['success'] for v in data.values()) for r,data in rows.items()},
                 paired={a+'_minus_'+b:ev.compare(records,rows,a,b) for a,b in
                         [('checked','control'),('control','parent'),('control','detour')]},
                 control_rooms={k:v['result']['success'] for k,v in rows[role].items()}))


if __name__=='__main__':
    run()
