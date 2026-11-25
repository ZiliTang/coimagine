from experiments.planar_tasks.authoring_v2.four_stage.execution import stopping_certificate,public_stop_prefix_covered


def row():
    return dict(index=4, selected="push", command=dict(ticks=4), event=dict(periods=4),
                decision=dict(checks=[dict(candidate=dict(option="push"), verdict=dict(
                    allowed=True, reason="calibrated_model", model_id="v0",
                    forecast=dict(motion_horizon="command_then_brake40"),
                    process=[dict(allowed=True)]))]))


def test_carried_tail_preserves_conditional_provenance():
    certificate = stopping_certificate([row()], dict(allowed=False), "enforce")
    assert certificate["certified"]
    assert certificate["source"] == "carried_empirical_command_then_brake40"
    assert certificate["model_id"] == "v0"


def test_unmatched_command_or_uncovered_tail_is_not_certified():
    r = row()
    r["event"]["periods"] = 0
    assert not stopping_certificate([r], dict(allowed=False), "enforce")["certified"]
    r = row()
    r["decision"]["checks"][0]["verdict"]["forecast"] = {}
    assert not stopping_certificate([r], dict(allowed=False), "enforce")["certified"]
    assert not stopping_certificate([row()], dict(allowed=False), "shadow")["certified"]


def test_carried_braking_consumes_the_original_budget_without_resetting_it():
    r=row();v=r["decision"]["checks"][0]["verdict"]
    v.update(reason="carried_stop_plan",carried_stop=dict(certified=True,remaining_ticks=8,
             source="original_plan",boundary=0))
    a=stopping_certificate([r],dict(allowed=False),"enforce")
    assert a["certified"] and a["remaining_ticks"]==4
    v["carried_stop"]=a
    b=stopping_certificate([r],dict(allowed=False),"enforce")
    assert not b["certified"] and b["remaining_ticks"]==0


def test_observed_endpoint_contradiction_retracts_the_tail():
    r=row()
    r["decision"]["checks"][0]["verdict"]["forecast"]["branches"]=[dict(updates=dict(seen=True))]
    result=stopping_certificate([r],dict(allowed=False),"enforce",symbols=dict(seen=False))
    assert not result["certified"] and result["observed_contradiction"]


def test_public_robot_positions_can_retract_a_carried_path_bound():
    plan=dict(origin_robots=[[0.,0.],[1.,1.]],motion_modes=[dict(
        r0=[0.,.1,0.,.1],r1=[0.,0.,0.,0.],target_boundary=0.)])
    o=dict(robots=[dict(own=dict(x=.11,y=0.)),dict(own=dict(x=1.,y=1.))])
    assert not public_stop_prefix_covered(plan,o)
