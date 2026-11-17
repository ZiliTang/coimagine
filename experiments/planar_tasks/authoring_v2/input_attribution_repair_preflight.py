"""Cache/live feature equivalence and four short inference interface checks."""
from . import input_attribution_study as V1
from . import input_attribution_study_v2 as V2
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces
import numpy as np
import torch

def main():
    torch.set_num_threads(1);source=(V1.PARENT/'inputs/semantics.py').read_text();ds=V1.dataset()
    ex=V1.S.read(V1.DATA/'extraction.json');layout=I.upgraded_layout(V1.S.read(V1.DATA/'layout.json'))
    def features():return I.AttributionFeatures(source,fast_embedding=True,action_protocol=ex['action_protocol'],execution_backend='trusted_inprocess',trusted_source_sha256=ex['source_sha'])
    f=features()
    try:
        d=V1.S.read(V1.PARENT/'demonstrations.json')[0];r=V1.S.unpack(d['artifact']['path']);r['split']='train'
        traces=demonstration_traces(r,f,split='train',mode='duration')
        for i,tr in enumerate(traces):
            assert tr['indices']==ds['targets'][i].tolist()
            for k,p in enumerate(tr['panels']):
                a,b=ds['offsets'][i,k:k+2]
                np.testing.assert_array_equal(p['attribution_context'],ds['contexts'][i])
                np.testing.assert_array_equal(p['attribution_candidates'],I.append_duration_encoding(ds['candidates'][a:b]))
        assert f.layout==layout
    finally:f.close()
    m=I.Scorer.load(V2.HOST/'probe/norm1_onehot1.pt');runner=V1.T.load_frozen_runner(V1.PARENT/'inputs/frozen_runner.py')
    case=V1.S.read(V1.PARENT/'panels/anchors.json')['cases'][0];summaries={}
    for arm in I.ARMS:
        torch.manual_seed(17)
        model=I.Scorer(layout,arm,(m.context_mean,m.context_std),(m.candidate_mean,m.candidate_std),ex['source_sha'],ex['action_protocol'])
        f=features()
        try:
            r,_=runner(case['fixture'],source,model=model,features=f,config=V1.S.EventConfig(mode='duration'),max_ticks=4,max_decisions=4,wall_seconds=30,deterministic=True,seed=2026111702,fast_feedback=True,training_fastpath=False,record_details=True)
            assert not r['summary']['error'];summaries[arm]=r['summary']
            V1.S.pack(V2.HOST/f'preflight_{arm}.json.gz',r)
        finally:f.close()
    V1.S.save(V2.HOST/'PREFLIGHT.json',dict(cache_live_boundaries=len(traces),cache_live_panels=4*len(traces),exact_cache_match=True,parameters=sum(p.numel() for p in model.parameters()),summaries=summaries,new_ticks=sum(r['ticks'] for r in summaries.values())))
    print('PREFLIGHT passed',len(traces),'boundaries;4 arms;16 physics ticks')

if __name__=='__main__':main()
