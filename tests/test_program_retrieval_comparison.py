import numpy as np
from experiments.planar_tasks.authoring_v2 import program_retrieval_comparison as C
from experiments.planar_tasks.semantic_rl.program_retrieval import RetrievalIndex


def test_exact_270_demonstration_manifest_and_no_repair_suffixes():
    demos=C.demo_manifest()
    assert len(demos)==270
    assert len({d['task'] for d in demos})==270
    assert all('__repair' not in d['task'] for d in demos)
    assert all(not d['task'].startswith('fresh_test') for d in demos)


def test_same_demo_initial_retrieval_does_not_invent_failed_labels():
    labels=np.full((2,3),np.nan);labels[0,2]=1;labels[1,0]=1
    index=RetrievalIndex([[0],[1]],labels,['CAP1','CAP2','CAP3'],{},evidence='initial_outcomes')
    a=index.select([0])
    assert a['cap']=='CAP3' and a['scores']==[None,None,1.]
    assert index.select([1])['cap']=='CAP1'
