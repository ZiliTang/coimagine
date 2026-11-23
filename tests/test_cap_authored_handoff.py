from collections import Counter
from tools import cap_authored_handoff as exp
from tools import cap_authored_handoff_graph as graph


def test_joint_budgets_and_conditional_sources():
    # B can route both successor states to success with different subsequent experts.
    n=1000
    counts={(0,0):Counter({(graph.SUCCESS,1,1):600,(graph.FAIL,1,1):400}),
            (0,1):Counter({(1,1,1):450,(2,1,1):450,(graph.FAIL,1,1):100}),
            (1,0):Counter({(graph.SUCCESS,1,2):n}),(1,1):Counter({(graph.FAIL,1,1):n}),
            (2,0):Counter({(graph.FAIL,1,1):n}),(2,1):Counter({(graph.SUCCESS,1,2):n})}
    m,s=graph.fit_tables(counts);p=graph.solve(m,s,tmax=3,dmax=4)
    assert p[2,0,3]==1 and p[2,0,2]==0 and p[1,0,3]==0
    assert p[2,1,3]==0 and p[2,2,3]==1
    assert p[2,3,3]==0


def test_stall_needs_an_observed_flag_and_remaining_room_budget():
    v={'flags':[1,0,0,0,0,0],'tick':180}
    assert exp.eligible(v,5,'stall') and not exp.eligible(v,4,'stall')
    assert not exp.eligible(dict(v,tick=1151),5,'stall')


def test_default_center_shifts_points_planes_and_memory_together():
    from experiments.planar_tasks.semantic_rl import cap_pair_perception as perception

    # The inner fitter observes the translated inputs, allowing the wrapper's
    # coordinate contract to be checked without a compiled expert or campaign.
    source = """def dot(a,b):
    return a[0]*b[0]+a[1]*b[1]

def compact_center(points,faces,previous):
    assert points == [(-1.,0.),(1.,0.)]
    assert faces == [((0.,1.),(-1.,0.),0.,-1.,1.)]
    assert previous == [0.,2.,1.,0.,1.,.25]
    return previous
"""
    assert perception.source_for_profile(source, perception.LEGACY) == source
    namespace = {}
    exec(perception.source_for_profile(source), namespace)
    estimate = namespace['compact_center']
    last = [5.,6.,1.,0.,1.,.25]
    assert estimate([], [], last) == last
    assert estimate([(4.,4.),(6.,4.)], [((0.,1.),(-1.,0.),4.,-6.,-4.)], last) == last
