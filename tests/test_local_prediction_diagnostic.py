"""Pure diagnostic-selection/ranking tests: no model or physics runs."""
from copy import deepcopy
from experiments.planar_tasks.local_prediction_diagnostic.diagnostic import candidates, select_boundaries, rank, hypothesis


def test_candidates_are_visible_legal_distinct_and_include_original():
    obs=dict(legal_pairs=[[3,46],[11,46],[46,27],[34,34],[44,44]])
    expected=candidates(obs,dict(a0=3,a1=46,ticks=20))
    assert len(expected)==4 and expected[0]['pair']==[3,46]
    assert len({tuple(c['pair']) for c in expected})==4
    assert all(c['pair'] in obs['legal_pairs'] for c in expected)
    other=deepcopy(obs);other['hidden_truth']='different'
    assert candidates(other,dict(a0=3,a1=46,ticks=20))==expected


def test_boundary_selection_uses_estimates_after_first_pass_and_no_replacement():
    rows=[dict(time=t,state=dict(mode=mode,pose=[x,1,0]),action=dict(a0=3,a1=46,ticks=20))
          for t,mode,x in [(0,'goal',1),(2,'lateral',1.2),(4,'pass',1.3),(6,'pass',1.3),(8,'pass',1.3),(10,'pass',1.3)]]
    assert select_boundaries(rows)==[(2,'first_lateral_to_pass'),(4,'first_post_pass_estimated_stall')]
    assert select_boundaries(rows[:2])==[]


def test_ranking_penalizes_incomplete_and_preserves_ties():
    rows=[dict(pred=dict(quality=dict(eligible=e,waypoint_progress=v))) for e,v in [(True,0),(False,10),(True,.2),(True,.2)]]
    assert rank(rows,'pred')==[2,3,0,1]


def test_hypothesis_keeps_legal_estimate_not_truth():
    own=dict(x=1,y=1,angle=0,vx=.1,vy=0,omega=0)
    obs=dict(robots=[dict(arena=[5,4],goal_xy=[4,2],goal_radius=.15,own=own),dict(own=own)])
    state=dict(pose=[2,2,.4],map=[[2.5,1],[3,3]],true_pose=[999,999,999])
    h=hypothesis(obs,state)
    assert h['object']==[2,2,.4,0,0,0]
    assert h['robots'][0]==[1,1,0,.1,0,0]
    assert h['obstacles']==[[2.75,2,.25,1,0]]
