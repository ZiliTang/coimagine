"""Paired action ranking using only the public prefix and the frozen actor's proposals."""
import csv
import gzip
import json

import numpy as np
import torch

from experiments.planar_tasks.relational_search.motion import ACTIONS
from experiments.planar_tasks.semantic_rl.candidate_semantics import candidate_semantics, direction
from tools.cap_semantic_fork_prefixes import OUT, DOC

ARMS = ['actor','actions','semantics']
RIDGE = .1


def encoding(action, row, i):
    result = dict(actor_logit=row['logits'][i], votes=sum(row['sources'][i]), ticks=action['ticks'],
                  target_present=float(action['target_id'] is not None),
                  task_target=float(action['target_id']==row['current']['goal']['target_id']))
    for j,source in enumerate(row['sources'][i]):
        result[f'cap{j}'] = source
    for j in range(2):
        a = ACTIONS[action[f'a{j}']]
        kind = getattr(a,'effect_kind',a.kind)
        for k in ('move','rotate','translate','reposition','wait'):
            result[f'r{j}_{k}'] = float(kind==k)
        result[f'r{j}_speed'] = a.speed
        result[f'r{j}_dx'],result[f'r{j}_dy'] = direction(a)
        result[f'r{j}_turn'] = getattr(a,'rotation_sign',0) or {'CW':-1,'CCW':1}.get(a.direction,0)
    return result


def prepare(row):
    if 'unchanged' in row:
        return row
    raw = [encoding(a,row,i) for i,a in enumerate(row['actions'])]
    semantic = [candidate_semantics(row['current'],a,row['previous'],row['prior_action']) for a in row['actions']]
    for arm,features in [('actions',raw), ('semantics',[a|b for a,b in zip(raw,semantic)])]:
        row[arm+'_names'] = list(features[0])
        row[arm+'_x'] = np.array([list(f.values()) for f in features])
    row['extra'] = semantic
    row['pairs'] = [(i,j) for i,y in enumerate(row['outcomes']) if y
                    for j,z in enumerate(row['outcomes']) if not z]
    return row


def learn(rows, arm):
    examples = [r for r in rows if r.get('pairs')]
    # Scale only from development. Each matched winner-minus-loser vector is one ranking example.
    diffs = [r[arm+'_x'][i]-r[arm+'_x'][j] for r in examples for i,j in r['pairs']]
    scale = np.maximum(np.sqrt(np.mean(np.square(diffs),axis=0)),.05)
    x = torch.tensor(np.asarray(diffs)/scale,dtype=torch.float64)
    w = torch.zeros(x.shape[1],dtype=torch.float64,requires_grad=True)
    opt = torch.optim.LBFGS([w],max_iter=100,line_search_fn='strong_wolfe',tolerance_grad=1e-9)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.softplus(-(x@w)).mean() + .5*RIDGE*w.square().sum()
        loss.backward()
        return loss
    opt.step(closure)
    return w.detach().numpy()/scale


def evaluate(r, models):
    out = dict(room=r['room'],split=r['split'],n=0,pairs=0,step=r.get('step'),prefix=r.get('prefix'))
    if 'unchanged' in r:
        for arm in ARMS:
            out[arm] = r['unchanged']
        out['upper'] = r['unchanged']
        return out
    y = r['outcomes']
    out.update(n=len(y),pairs=len(r['pairs']),upper=max(y),push=any(
        ACTIONS[a[f'a{i}']].kind=='push' and getattr(ACTIONS[a[f'a{i}']],'effect_kind','')=='translate'
        for a in r['actions'] for i in range(2)))
    for arm in ARMS:
        scores = np.array(r['logits']) if arm=='actor' else r[arm+'_x'] @ models[arm]
        ix = max(range(len(scores)),key=lambda j:(scores[j],j==r['pref'],-j))
        if arm=='actor':
            assert ix==r['pref']
        out[arm+'_choice'] = ix
        out[arm] = y[ix]
        wins = sum(float(scores[i]>scores[j]) + .5*float(scores[i]==scores[j]) for i,j in r['pairs'])
        out[arm+'_pairwins'] = wins
        if r['pairs']:
            out[arm+'_rank'] = wins/len(r['pairs'])
    return out


def summary(rows):
    mixed = [r for r in rows if r['pairs']]
    result = dict(rooms=len(rows), forks=sum(r['n']>0 for r in rows), mixed=len(mixed),
                  opposite_pairs=sum(r['pairs'] for r in mixed), upper=sum(r['upper'] for r in rows))
    for arm in ARMS:
        result[arm] = dict(success=sum(r[arm] for r in rows), mixed_success=sum(r[arm] for r in mixed),
                           pairwins=sum(r[arm+'_pairwins'] for r in mixed),
                           room_rank=np.mean([r[arm+'_rank'] for r in mixed]))
    rng = np.random.default_rng(17)
    for a,b in [('actions','semantics'),('actor','semantics'),('actor','actions')]:
        deltas = np.array([r[b]-r[a] for r in rows])
        rankings = np.array([r[b+'_rank']-r[a+'_rank'] for r in mixed])
        def interval(values):
            bs = rng.choice(values,(5000,len(values)),replace=True).mean(1)
            return np.percentile(bs,[2.5,97.5]).tolist()
        result[b+'_vs_'+a] = dict(rescued=int((deltas>0).sum()),lost=int((deltas<0).sum()),
                                  success_ci=interval(deltas),rank_ci=interval(rankings))
    return result


def contrast_counts(rows, decisions):
    mixed = [r for r in rows if r.get('pairs')]
    pairs = [(r['extra'][i],r['extra'][j]) for r in mixed for i,j in r['pairs']]
    evidence = {}
    for key in pairs[0][0]:
        delta = np.array([a[key]-b[key] for a,b in pairs])
        evidence[key] = dict(winner_higher=int((delta>1e-6).sum()), loser_higher=int((delta < -1e-6).sum()))
    changed = [r for r in rows if 'pairs' in r and
               decisions[r['room']]['semantics_choice'] != decisions[r['room']]['actions_choice']]
    both_box = [(a,b) for a,b in pairs if min(a['repeat_box_observed'],b['repeat_box_observed'])>0]
    sweep_diff = sum(any(abs(a[k]-b[k])>1e-6 for k in ('sweep_static','sweep_movable')) for a,b in pairs)
    return dict(changed=len(changed), mixed_changed=sum(bool(r['pairs']) for r in changed),
                sweep_differs=sweep_diff, both_repeat_box_observed=len(both_box),
                both_repeat_box_progress_differs=sum(bool(abs(a['repeat_box_progress']-b['repeat_box_progress'])>1e-6) for a,b in both_box),
                any_feature_differs=sum(any(abs(a[k]-b[k])>1e-6 for k in a) for a,b in pairs),
                feature_contrasts=evidence)


def write_table(path, rows):
    with path.open('w',newline='') as file:
        w = csv.DictWriter(file,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


def main():
    torch.set_num_threads(1)
    with gzip.open(OUT/'prefixes.jsonl.gz','rt') as file:
        rows = [prepare(json.loads(line)) for line in file]
    dev = [r for r in rows if r['split']=='development']
    held = [r for r in rows if r['split']=='held']
    rng = np.random.default_rng(17)
    groups = np.empty(len(dev),int)
    for objects in range(2,6):
        same = [j for j,r in enumerate(dev) if int(r['room'].split('_')[-2])==objects]
        groups[rng.permutation(same)] = np.arange(len(same))%5
    results = []
    for g in range(5):
        models = {arm:learn([r for i,r in enumerate(dev) if groups[i]!=g],arm) for arm in ARMS[1:]}
        results.extend(evaluate(r,models) for i,r in enumerate(dev) if groups[i]==g)
    models = {arm:learn(dev,arm) for arm in ARMS[1:]}
    results.extend(evaluate(r,models) for r in held)
    DOC.mkdir(exist_ok=True,parents=True)
    write_table(DOC/'rooms.csv',results)
    out = {split:summary([r for r in results if r['split']==split]) for split in ('development','held')}
    result_by_id = {r['room']:r for r in results}
    for split, part in [('development',dev),('held',held)]:
        out[split]['contrasts'] = contrast_counts(part,result_by_id)
    candidates = []
    for r in rows:
        for j,a in enumerate(r.get('actions',[])):
            info = result_by_id[r['room']]
            candidates.append(dict(room=r['room'],split=r['split'],candidate=j,**a,
                actor_logit=r['logits'][j],sources=' '.join(str(i) for i,v in enumerate(r['sources'][j]) if v),
                success=r['outcomes'][j], **{f'{arm}_chosen':int(info[arm+'_choice']==j) for arm in ARMS}, **r['extra'][j]))
    write_table(DOC/'candidates.csv',candidates)
    out['weights'] = {arm:dict(zip(next(r[arm+'_names'] for r in dev if 'pairs' in r),
                                  models[arm].tolist())) for arm in ARMS[1:]}
    (DOC/'result.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__=='__main__':
    main()
