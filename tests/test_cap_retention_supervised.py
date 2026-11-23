import numpy as np
from experiments.planar_tasks.authoring_v2.cap_retention_supervised import choose


def test_switch_gate_compares_distinct_actions_and_requires_two_consecutive_votes():
    packet=dict(legal=[True]*3,proposals=['a','a','b'])
    for mode in ('guard','free','once'):
        assert choose(mode,np.array([.4,.8,.6]),packet,0,2)==(0,None,'a','hold')
    strong=np.array([.4,.5,.8])
    weak=np.array([.4,.3,.49])
    assert choose('once',strong,packet,0,None)==(0,None,'a','hold')
    assert choose('free',strong,packet,0,None)==(2,None,'b','switch')
    assert choose('guard',strong,packet,0,None)==(0,2,'a','hold')
    assert choose('guard',weak,packet,0,2)==(0,None,'a','hold')
    assert choose('guard',strong,packet,0,2)==(2,None,'b','switch')
    packet['legal'][0]=False
    assert choose('once',strong,packet,0,None)==(2,None,'b','forced')
