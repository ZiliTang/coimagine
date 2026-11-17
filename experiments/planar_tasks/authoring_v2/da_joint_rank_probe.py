"""Read-only conditional ranking probe on already collected branch outcomes.

Labels are conditional on the original frozen donor continuation. They never
enter features, normalizers, model selection or the fresh evaluation set.
"""
import numpy as np
import torch
from . import da_joint_models as M
from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS


def main():
    V=M.V;torch.set_num_threads(1)
    models={n:V.I.Scorer.load(M.BASE/'fit'/n/'model.pt') for n in M.names()}
    states=V.read(M.BASE/'selection.json')['states'];branches=V.read(M.BASE/'branches/result.json')['rows']
    f=M.features('joint');rows=[]
    try:
        for s in states:
            ref=V.unpack(s['reference']);h=f.new_history()
            for r in ref['public_rows'][:s['cut']+1]:
                h.append(r['observation'],r['action'])
                with f.decision(h):pass
            tree=ActionTree(ref['public_rows'][s['cut']]['observation'])
            actions=[s['donor_action']]+[r['action'] for r in s['alternatives']]
            outcomes={r['job']['kind']:r['success'] for r in branches if r['job']['case_id']==s['case_id']}
            labels=[outcomes['identity']]+[outcomes[f'action{i}'] for i in range(len(s['alternatives']))]
            scores={n:[] for n in models}
            with f.decision(h),torch.no_grad():
                for action in actions:
                    prefix={};values={n:0. for n in models}
                    for key in FIELDS:
                        panel=f.panel(h,tree,prefix);chosen=dict(prefix,**{key:action[key]});j=panel['choices'].index(chosen)
                        for name,model in models.items():
                            mode=name.split('_')[1]
                            layout=dict(f.layout,evidence_mode=mode)
                            if layout!=model.layout:raise ValueError('probe/runtime layout mismatch')
                            xs=M.ModeCandidates(panel['attribution_candidates'],mode,10)[:]
                            q=model.logits(panel['attribution_context'],xs)
                            values[name]+=float(q[j]-q.logsumexp(0))
                        prefix=chosen
                    for name in models:scores[name].append(values[name])
            rows.append(dict(case_id=s['case_id'],split=s['split'],cut=s['cut'],actions=actions,
                labels=labels,scores=scores,note='joint action log-probabilities over full legal conditional trees'))
    finally:f.close()
    summary={}
    for name in models:
        summary[name]={}
        for split in ('repair_train','handoff_holdout'):
            rr=[r for r in rows if r['split']==split and any(r['labels'])]
            summary[name][split]=dict(n=len(rr),top_success=sum(r['labels'][int(np.argmax(r['scores'][name]))] for r in rr),
                n_with_failed_competitor=sum(not all(r['labels']) for r in rr),
                clear_conflict_top_success=sum(r['labels'][int(np.argmax(r['scores'][name]))] for r in rr if not all(r['labels'])))
    V.save(M.HOST/'rank_probe.json',dict(rows=rows,summary=summary,training_usage='none; read-only diagnostic'))
    print(summary)


if __name__=='__main__':main()
