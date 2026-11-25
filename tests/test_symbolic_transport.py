import math

from experiments.planar_tasks.semantic_rl import guarded_transport as transport


def planner():
    effects=[dict(room=str(i),pair=[44,7],dose=dose,route=route,angle=math.pi/4,
                  forward=.08,push_sweep=dose*.6,endpoint_loss=dose*.5,full_sweep=dose*.8)
             for i in range(10) for route in transport.ROUTES for dose in transport.DOSES]
    return transport.TransportPlanner(dict(effects=effects,gap_margin=.01,geometry_profile=transport.GEOMETRY_PROFILE))


def test_monotone_clearance_and_successor_dependent_route():
    plan=planner()
    cmd=dict(a0=44,a1=7,target_id='box')
    def case(gap,route='ccw',dose=.1):
        return plan.check(cmd,dose,route,dict(bounds=[gap,gap+.03],angle=math.pi/4),True)
    outcomes=[case(i/100)['admitted'] for i in range(70)]
    assert outcomes==sorted(outcomes) and any(outcomes) and not all(outcomes)
    assert case(.25)['admitted'] and not case(.25,'cw')['admitted']
    assert case(.25)['status']=='conditional' and not case(.25,dose=.4)['admitted']
    assert case(.25)['post']['route_entry_gap'][0]==190
    assert not plan.check(cmd,.1,'ccw',None,True)['admitted']
    assert not plan.check(cmd,.1,'ccw',dict(bounds=[1,1],angle=-math.pi/2),True)['admitted']
    assert not plan.check(cmd,.1,'ccw',dict(bounds=[1,1],angle=math.pi/4),False)['admitted']


def test_swept_path_can_block_even_with_a_wide_endpoint():
    plan=planner()
    for r in plan.model['effects']:
        r['endpoint_loss']=0.
        r['push_sweep']=.5
    case=plan.check(dict(a0=44,a1=7,target_id='box'),.1,'ccw',
                    dict(bounds=[.40,.45],angle=math.pi/4),True)
    assert not case['admitted']
    assert case['post']['route_entry_gap'][0]>case['route_width_mm']
    assert any(i.get('metric')=='push_min_gap' for i in case['issues'])
