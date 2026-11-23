"""Compare the actual closed-loop executors on matched, perturbed PUSH prefixes."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from itertools import groupby

import numpy as np
import torch

from tools import cap_option_semantic_factorial as study
from tools import push_contract_bench as bench
from tools import push_feasibility_probe as probe

DEST = study.DOC / 'boundary_separation'
OUT = study.OUT / 'boundary_separation'


def execute_prefix(case):
    arms = ('coupled', 'split_precision')
    room = case['room']
    fixture = (bench.HELD if case['split'] == 'stress' else bench.DEV)[room]
    cmd = case['command']
    robot, axis = study.push_phases.pusher(cmd)
    axis = np.asarray(axis)
    tangent = np.array([-axis[1], axis[0]])
    records = bench.archive(case['sid'], room)
    rows = []
    for offset, (normal, side) in bench.OFFSETS.items():
        old = bench.read(bench.OUT / 'runs' / 'base' / offset / (case['id'] + '.json'))
        if old['status'] != 'executed':
            continue
        feature = bench.read(probe.OUT / 'inputs' / offset / (case['id'] + '.json'))
        row = dict(room=room, prefix=case['id'], split=case['split'], offset=offset,
                   features=feature['full'])
        for arm in arms:
            with bench.world(fixture) as (w, _):
                for step in records[:case['index']]:
                    w.step(step['a'], remaining_ticks=study.remaining(w))
                    assert w.env.total_periods == step['tick']
                assert w.history.sha256 == old['prefix']
                if offset != 'zero':
                    assert bench.perturb(w, robot, normal * axis + side * tangent) is None
                bench.validate_action(cmd, study.observe(w))
                start = np.array(bench.position(w.env, cmd['target_id'])[:2])
                hit = []
                original = w.detector.update

                def measure(packet):
                    status = original(packet)
                    hit.append(w.env.active_target == cmd['target_id'] and packet['robots'][robot]['contacting_object'])
                    return status

                w.detector.update = measure
                counts = Counter()
                study.tick_action(w, cmd, arm, lambda history: None, counts)
                delta = np.array(bench.position(w.env, cmd['target_id'])[:2]) - start
                along = float(delta @ axis)
                lateral = abs(float(delta @ tangent))
                contact = any(all(hit[k:k+3]) for k in range(len(hit)-2))
                passed = int(contact and along >= bench.CONTACT_M and lateral <= max(bench.SIDE_M, .75 * along))
                row[arm] = passed
                if arm == 'coupled':
                    assert passed == old['effect'] and np.isclose(along, old['along'], atol=1e-8)
                else:
                    row['retimed'] = counts['retimed']
        rows.append(row)
    return rows


def auc(y, scores, weight):
    order = np.argsort(scores)
    y, scores, weight = y[order], scores[order], weight[order]
    _, idx = np.unique(scores, return_index=True)
    pos = np.add.reduceat(weight*y, idx)
    neg = np.add.reduceat(weight*(1-y), idx)
    den = pos.sum()*neg.sum()
    return float(pos @ (neg.cumsum()-.5*neg)/den) if den else np.nan


def analysis(rows, splits=('validation', 'stress')):
    arms = ('coupled', 'split_precision')
    train = [r for r in rows if r['split'] == 'development']
    def xs(data):
        return np.array([r['features'] for r in data])
    predictors = {arm: probe.fit(xs(train), np.array([r[arm] for r in train]),
                                np.array([r['room'] for r in train])) for arm in arms}
    result = {}
    rng = np.random.default_rng(20261123)
    for split in splits:
        part = [r for r in rows if r['split'] == split]
        rooms, group = np.unique([r['room'] for r in part], return_inverse=True)
        scores = [predictors[a](xs(part)) for a in arms]
        ys = [np.array([r[a] for r in part]) for a in arms]
        summary = dict(branches=len(part), rooms=len(rooms), retimed=sum(r['retimed'] for r in part))
        neighborhoods = []
        for _, it in groupby(sorted(range(len(part)), key=lambda i:part[i]['prefix']), key=lambda i:part[i]['prefix']):
            ids = list(it)
            if len(ids) < 2:
                continue
            left, right = np.triu_indices(len(ids), 1)
            values = [float(np.mean(y[ids][left] != y[ids][right])) for y in ys]
            neighborhoods.append((group[ids[0]], *values))
        near = np.array(neighborhoods)
        for arm, y, s in zip(arms, ys, scores):
            flips, local, sizes = 0, [], []
            for _, it in groupby(sorted(range(len(part)), key=lambda i:part[i]['prefix']), key=lambda i:part[i]['prefix']):
                ids = list(it)
                if len(ids)<2:
                    continue
                sizes.append(len(ids))
                if len(np.unique(y[ids])) == 2:
                    flips += 1
                    local.append(auc(y[ids], s[ids], np.ones(len(ids))))
            summary[arm] = dict(success=int(y.sum()), auc=auc(y,s,np.ones(len(y))),
                                multi_branch_prefixes=len(sizes), mixed_prefixes=flips,
                                within_mixed_prefix_auc=float(np.mean(local)) if local else None)
        diffs = []
        stability_diffs = []
        for _ in range(5000):
            draws = rng.multinomial(len(rooms), np.full(len(rooms), 1/len(rooms)))
            weight = draws[group]
            diffs.append(auc(ys[1], scores[1], weight)-auc(ys[0], scores[0], weight))
            neighbors_weight = draws[near[:,0].astype(int)]
            if neighbors_weight.sum():
                stability_diffs.append(np.average(near[:,2]-near[:,1], weights=neighbors_weight))
        summary['auc_difference_ci95'] = np.nanquantile(diffs,[.025,.975]).tolist()
        summary['neighbor_disagreement'] = dict(coupled=float(near[:,1].mean()), split_precision=float(near[:,2].mean()),
            difference_ci95=np.quantile(stability_diffs,[.025,.975]).tolist())
        result[split] = summary
    return result


def main():
    torch.set_num_threads(1)
    cases = [c for f in ('protocol.json', 'protocol_stress.json') for c in bench.read(bench.OUT/f)['cases']
             if c['kind'] != 'dual']
    path = OUT/'paired.json'
    if not path.exists():
        with ProcessPoolExecutor(10) as pool:
            rows = [r for batch in pool.map(execute_prefix, cases) for r in batch]
        assert len(rows) == 580
        bench.save(path, rows)
    results = analysis(bench.read(path))
    bench.save(DEST/'results.json', results)
    print(results)


if __name__ == '__main__':
    main()
