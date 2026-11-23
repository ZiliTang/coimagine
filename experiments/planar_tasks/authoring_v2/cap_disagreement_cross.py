"""Swap only the first distinct joint option, then follow the designated scorer."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from . import cap_first_disagreement as study
from . import cap_disagreement_features as features
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save


class SwapOnce:
    def __init__(self,model,index,slot):
        self.model=model
        self.index=index
        self.slot=slot
        self.position=0
        if hasattr(model,'reset'):
            model.reset()

    def values(self,*args):
        scores=self.model.values(*args)
        if self.position==self.index:
            scores=scores.clone()
            scores.fill_(0.)
            scores[:,self.slot]=1.
        self.position+=1
        return scores

    def executed(self,source):
        if hasattr(self.model,'executed'):
            self.model.executed(source)


def run(job):
    case,row,first,suffix=job
    models=study.fast_runtime.W.MODELS
    base=models[study.GA if suffix=='G' else study.ATT]
    alias='first_option_override'
    models[alias]=SwapOnce(base,row['step']-1,row[first]['slot'])
    folder=study.OUT/'cross'/case['case_id']/(first+suffix)
    s=study.scale.family.worker(dict(case=case,condition=first+suffix,checkpoint=alias,epsilon=0.,directory=str(folder)))
    r=features.load(folder/'result.json.gz')
    i=row['step']-1; d=r['decisions'][i]; action=d['action'];exp=row[first]
    assert r['public_rows'][i]['history_hash']==row['hash']
    assert [action['a0'],action['a1'],action['ticks'],action['target_id']]==[*exp['codes'],exp['ticks'],exp['target']]
    return dict(case=case['case_id'],first=first,suffix=suffix,success=int(s['success']),ticks=s['ticks'])


def main():
    roots={r['parent'] for r in read(study.OUT/'neighbors_panel.json')}
    cases={r['case_id']:r for r in read(study.scale.OUT/'evaluation_panel.json')}
    rows=read(study.OUT/'discovery_prefixes.json')
    jobs=[(cases[r['id']],r,a,b) for r in rows if r['id'] in roots for a,b in [('G','T'),('T','G')]]
    assert len(jobs)==48
    with ProcessPoolExecutor(6,mp_context=mp.get_context('spawn'),initializer=study.initialize) as pool:
        result=list(pool.map(run,jobs))
    save(study.OUT/'cross.json',result)


if __name__=='__main__':
    main()
