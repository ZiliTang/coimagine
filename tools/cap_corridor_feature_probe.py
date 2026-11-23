"""Paired test of six current corridor features at the frozen 40-epoch budget."""
import csv

import numpy as np
import torch

from tools import cap_corridor_subtypes as study
from tools import cap_s025_initial as base


def plot_summary():
    from matplotlib import pyplot as plt

    result = study.read(study.OUT/'FINISHED.json')
    kinds = study.KINDS
    groups = result['types']['truth']['held']
    caps = ['s025', 's027', 's041', 's045', 's066']
    counts = np.array([[groups[k]['wins'][c] for c in caps]+[groups[k]['union']] for k in kinds])
    totals = np.array([groups[k]['n'] for k in kinds])
    confusion = result['recognition']['combined']
    observed = (*kinds, 'unknown')
    matches = np.array([[confusion[k].get(o,0) for o in observed] for k in kinds])
    fig, axs = plt.subplots(1,2,figsize=(11.2,3.7),layout='constrained')
    for ax, cells, ns, cols, title in [
        (axs[0], counts, totals, [*caps, 'Union'], 'A. CAP successes in 200 held-out rooms'),
        (axs[1], matches, matches.sum(1), observed, 'B. Public corridor estimate in 400 held-out rooms')]:
        frac = cells/ns[:,None]
        ax.imshow(frac,vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(range(len(cols)),cols)
        ax.set_yticks(range(len(kinds)), [f'{k} (n={n})' for k,n in zip(kinds,ns)])
        ax.set_title(title,fontsize=11,pad=12)
        for (i,j), n in np.ndenumerate(cells):
            ax.text(j,i,str(n),ha='center',va='center',color='white' if frac[i,j]>.52 else '#203449')
    axs[0].set_ylabel('Fixture corridor type')
    axs[0].set_xlabel('Frozen CAP or per-room union')
    axs[1].set_xlabel('Estimate using current public observations')
    fig.savefig(study.DOC/'corridor.png',dpi=160)
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    with (study.DOC/'rooms.csv').open() as file:
        rows = list(csv.DictReader(file))
    with (base.DOC/'rooms.csv').open() as file:
        previous = list(csv.DictReader(file))
    assert [(r['room'], r['panel']) for r in rows] == [(r['room'], r['panel']) for r in previous]
    geo = np.load(study.OUT/'compact_initial.npz')['x']
    names = study.FEATURE_NAMES[:6]
    semantic = np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float32)
    panel = np.array([r['panel'] for r in rows])
    objects = np.array([int(r['objects']) for r in rows])
    y = np.array([int(r['s025']) for r in rows])
    train = np.flatnonzero(panel == 'development')
    result = dict(epoch=40, seed=base.SEED, added=list(names), scores={}, paired={})
    scores = {}
    for arm, features in [('geometry', geo), ('plus_semantics', np.column_stack([geo, semantic]))]:
        pred = base.train_curve('compact_linear', features, y, train, np.arange(len(y)))[40]
        if arm == 'geometry':
            assert np.allclose(pred, [float(r['compact_linear']) for r in previous], rtol=0, atol=1e-7)
        scores[arm] = pred
        result['scores'][arm] = {}
        for split in ['development', 'original', 'current', 'combined']:
            mask = panel != 'development' if split == 'combined' else panel == split
            result['scores'][arm][split] = dict(n=int(mask.sum()), auc=base.auc(y[mask], pred[mask]),
                                                brier=float(np.mean((y[mask]-pred[mask])**2)))
    rng = np.random.default_rng(base.SEED)
    strata = [np.flatnonzero((panel == p) & (objects == n)) for p in ['original','current'] for n in range(2,6)]
    a, b = scores.values()
    diff = []
    for _ in range(4000):
        i = np.concatenate([rng.choice(s, len(s)) for s in strata])
        diff.append([base.auc(y[i], b[i])-base.auc(y[i], a[i]),
                     np.mean((b[i]-y[i])**2-(a[i]-y[i])**2)])
    for j, metric in enumerate(['auc','brier']):
        result['paired'][metric] = dict(delta=result['scores']['plus_semantics']['combined'][metric]-
                                        result['scores']['geometry']['combined'][metric],
                                        ci=np.percentile(np.asarray(diff)[:,j], [2.5,97.5]).tolist())
    study.save(study.DOC/'feature_probe.json', result)
    plot_summary()


if __name__ == '__main__':
    main()
