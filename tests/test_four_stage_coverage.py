from experiments.planar_tasks.authoring_v2.four_stage.coverage import realized_coverage


def test_smaller_actual_motion_is_covered_but_endpoint_can_miss():
    bound=dict(r0=[-.1,.1,-.1,.1],r1=[-.1,.1,-.1,.1],target_boundary=.1)
    actual=dict(r0=[0.,.02,0.,0.],r1=[0.,0.,0.,0.],target_boundary=0.)
    forecast=dict(branches=[dict(motion=bound,updates=dict(legal=True,x=[0.,1.]))])
    r=realized_coverage(forecast,actual,dict(legal=False,x=.5))
    assert r["root_motion_covered"]
    assert r["joint_endpoint_covered"] is False


def test_joint_coverage_does_not_mix_different_branch_components():
    zero=dict(r0=[0.,0.,0.,0.],r1=[0.,0.,0.,0.],target_boundary=0.)
    motion=dict(r0=[0.,.1,0.,0.],r1=[0.,0.,0.,0.],target_boundary=0.)
    forecast=dict(branches=[dict(motion=zero,updates=dict(legal=True)),
                            dict(motion=motion,updates=dict(legal=False))])
    r=realized_coverage(forecast,motion,dict(legal=True))
    assert r["root_motion_covered"]
    assert r["joint_endpoint_covered"] is False
