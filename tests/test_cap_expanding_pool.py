from experiments.planar_tasks.semantic_rl.sparse_branches import alternatives
from tools.cap_expanding_pool import request_key, select_points


def test_sparse_neighbors_group_commands_and_keep_time_and_target():
    base=dict(target_id='box',a0=4,a1=44,ticks=30)
    shorter=dict(base,ticks=10)
    other_target=dict(base,target_id='other')
    proposals=dict(A=base,B=shorter,C=dict(shorter),D=other_target,E=dict(base))
    star={'A':('B','C','D','E'),'B':('A',)}
    assert list(alternatives(proposals,'A',star).values())==[['B','C'],['D']]
    assert list(alternatives(proposals,'B',star).values())==[['A']]


def test_revisit_prefers_unused_prefix_in_each_temporal_third():
    points=[[i,f'h{i}',('box',1,2,40),'B'] for i in range(9)]
    used={request_key(p) for i,p in enumerate(points) if i%3!=2}
    assert select_points(points,used,2,'room')==[2,5,8]
    assert select_points([],set(),1,'room')==[]
