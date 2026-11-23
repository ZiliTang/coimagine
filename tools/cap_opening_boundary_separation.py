"""Compare opening CAP ranking before/after changing PUSH on identical expert pairs."""
import numpy as np
import torch

from tools import cap_initial_specialization_probe as opening
from tools import cap_option_semantic_factorial as study
from tools import cap_option_semantic_factorial_report as report

DEST = study.DOC / 'boundary_separation'


def main():
    torch.set_num_threads(1)
    protocol = study.read(study.OUT/'protocol.json')
    study.RULES = protocol['rules']
    rooms, features = {}, {}
    for fold in ('development', 'held'):
        fixtures = study.read(study.OUT/(fold+'.json'))
        rooms[fold] = sorted(fixtures)
        features[fold] = np.stack([opening.features(fixtures[k])['evidence'] for k in rooms[fold]])
    results = {}
    for bank, models in protocol['libraries'].items():
        labels, scores, priors = {}, {}, {}
        for executor in study.CONTROLLERS:
            folds = {}
            for fold, ids in rooms.items():
                folds[fold] = np.array([[r['success'] for r in report.read_rows(
                    study.OUT/'singles'/fold/executor/m['sha'], ids)] for m in models]).T
            labels[executor] = folds['held']
            priors[executor] = folds['development'].mean(axis=0)
            scores[executor] = opening.fit_linear(features['development'], folds['development'], features['held'])
        room_values, room_ids, npair = [], [], 0
        for k, room in enumerate(rooms['held']):
            vals = []
            for i in range(len(models)):
                for j in range(i):
                    diffs = [int(labels[e][k,i])-int(labels[e][k,j]) for e in study.CONTROLLERS]
                    if 0 in diffs:
                        continue
                    row = []
                    for e, d in zip(study.CONTROLLERS,diffs):
                        for vector in (scores[e][k], priors[e]):
                            sign = np.sign((vector[i]-vector[j])*d)
                            row.append((sign+1)/2)
                    vals.append(row)
            if vals:
                room_values.append(np.mean(vals, axis=0))
                room_ids.append(room)
                npair += len(vals)
        a = np.array(room_values)
        d = a[:,2]-a[:,0]
        extra = (a[:,2]-a[:,3])-(a[:,0]-a[:,1])
        rng = np.random.default_rng(20261123)
        panel = np.array([k[:6] for k in room_ids])
        ix = np.concatenate([rng.choice(g,(5000,len(g))) for p in sorted(set(panel))
                             if len(g:=np.flatnonzero(panel==p))],axis=1)
        results[bank] = dict(rooms=len(a), pairs=npair, columns=['coupled_evidence','coupled_prior',
                             'split_evidence','split_prior'], accuracy=a.mean(axis=0).tolist(),
                             difference=float(d.mean()), ci95=np.quantile(d[ix].mean(axis=1),[.025,.975]).tolist(),
                             improvement_over_prior_difference=float(extra.mean()),
                             improvement_over_prior_ci95=np.quantile(extra[ix].mean(axis=1),[.025,.975]).tolist())
    study.save(DEST/'opening.json', results)
    print(results)


if __name__ == '__main__':
    main()
