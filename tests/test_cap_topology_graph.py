import pymunk

from tools.cap_topology_graph import straight_blocked


def test_inflated_approach_distinguishes_hit_and_miss():
    wall=pymunk.Segment(pymunk.Body(body_type=pymunk.Body.STATIC),(0.,0.),(1.,0.),.05)
    wall.cache_bb()
    assert not straight_blocked((.5,1.),(.5,2.),[wall])
    assert straight_blocked((.5,1.),(.5,-1.),[wall])
    assert straight_blocked((-.5,.15),(1.5,.15),[wall])
