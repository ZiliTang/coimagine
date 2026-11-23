import numpy as np
from tools.cap_regional_actor_replay import sample_history
from tools.cap_regional_expert_suffix import summarize


def test_history_sampling_balances_rooms_before_pairs():
    groups=[np.array([0]),np.array([1,2])]
    picks=sample_history(groups,np.random.default_rng(17),12000)
    counts=np.bincount(picks)/len(picks)
    np.testing.assert_allclose(counts,[.5,.25,.25],atol=.02)


def test_pair_deltas_are_distinct_from_same_action_successes():
    row={'available':1,'actor_base':{'success':0},'actor_alt':{'success':1},
         'expert_base':{'success':1,'complete':True,'supporters':[0,1]},
         'expert_alt':{'success':1,'complete':True,'supporters':[2]}}
    result=summarize([row,{'available':0}])
    assert result['actor_only']==0 and result['expert_only']==1
    assert result['actor_only_rescue']==1 and result['pure']==1
