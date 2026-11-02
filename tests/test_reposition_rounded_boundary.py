"""Own-scan finite-boundary control, censoring, signed travel and matched corners."""
from dataclasses import asdict, replace
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.boundary_follow import RoundedBoundary
from coimagine.random_start.effect_push import ACTIONS, EffectPushConfig, make_effect_skill
from coimagine.random_start.endpoint_tracking import EndpointObservation
from coimagine.random_start.env import RandomStartEnv
from coimagine.random_start.skills import RepositionSkill
from experiments.option_testbench.effect_push_eval import (
    comparison_config, instance_from_dict, reposition_corner_cases, run_one)
from experiments.option_testbench.reposition_combinations_audit import audit


def corner_observation():
    case = next(c for c in reposition_corner_cases() if c["name"]=="arc_rp_CCW_corner_1.00")
    case["instance"]["robots"][0].update(x=3.15,y=1.65)
    return RandomStartEnv(instance_from_dict(case["instance"])).boundary_observations()[0]


def test_corner_uses_finite_segments_instead_of_nearest_infinite_line():
    obs=corner_observation();b=RoundedBoundary().update(obs,spec.CONTROL_DT)
    assert b is not None
    assert math.dist(b.point,(3.,1.875))<.04
    assert b.outward[0]>.3 and b.outward[1]<-.5
    assert .23<b.distance<.32
    assert b.mode=="corner_arc"


def test_boundary_query_ignores_hidden_labels_and_teammate_private_position():
    obs=corner_observation()
    altered=replace(obs,teammate=replace(obs.teammate,x=-999.,y=888.),
                    scan=replace(obs.scan,raw_labels=("changed",)*len(obs.scan.valid)))
    assert RoundedBoundary().update(obs,.05)==RoundedBoundary().update(altered,.05)


@pytest.mark.parametrize("kind,uncertainty",[("shadow_censored",0.),("range_censored",0.),("silhouette_interval",.2)])
def test_censored_or_wide_endpoint_cannot_create_a_corner(monkeypatch,kind,uncertainty):
    obs=corner_observation()
    # A legal observed line, with its perpendicular foot outside the observed span.
    patch=next(p for p in obs.patches if abs(p.normal_world[1])>.9)
    t=patch.tangent_world;q=[p[0]*t[0]+p[1]*t[1] for p in patch.points_world]
    own_q=obs.own.x*t[0]+obs.own.y*t[1]
    assert own_q<min(q) or own_q>max(q)
    def endpoints(*args):
        return tuple(EndpointObservation(i,None if kind.endswith('censored') else p,p,p,kind,uncertainty)
                     for i,p in enumerate((patch.points_world[0],patch.points_world[-1])))
    monkeypatch.setattr('coimagine.random_start.boundary_follow.observe_endpoints',endpoints)
    assert RoundedBoundary().update(replace(obs,patch=patch,patches=(patch,)),.05) is None


@pytest.fixture(scope="module",params=["arc_rp_CCW_corner_1.00","arc_rp_CW_corner_1.00","arc_rp_CW_solo_1.00"])
def pair(request):
    case=next(c for c in reposition_corner_cases() if c['name']==request.param)
    records=[]
    for enabled in (False,True):
        values=asdict(comparison_config('reposition_corners',enabled,{}));feedback=values.pop('feedback')
        records.append(run_one(case,feedback,values,'reposition_corners'))
    return records


def test_rounding_preserves_budget_and_independently_reaches_signed_target(pair):
    old,new=pair
    assert old['case']['actions']==new['case']['actions']
    assert old['boundary']['execution']['requested_time_budgets']==new['boundary']['execution']['requested_time_budgets']
    assert new['metric']['exits'][0]=='reposition_complete'
    assert audit(new)['options'][0]['independent_signed_travel_m']>=1.
    assert new['metric']['seconds']<old['metric']['seconds']


def test_matched_corner_has_smaller_command_jumps_without_force_override(pair):
    def jumps(record):
        vectors=[r['r0']['tangent_world'] for r in record['rows'] if r['r0']['kind']=='reposition' and r['r0']['tangent_world']]
        return [abs(math.atan2(a[0]*b[1]-a[1]*b[0],a[0]*b[0]+a[1]*b[1])) for a,b in zip(vectors,vectors[1:])]
    old,new=pair
    assert max(jumps(old))>math.radians(60)
    assert max(jumps(new))<math.radians(10)
    assert all(math.hypot(*f)<=spec.MAX_FORCE+1e-9 for r in new['rows'] for f in r['forces'])


def test_factory_retains_explicit_legacy_default_and_opt_in():
    action=next(a for a in ACTIONS if a.kind=='reposition')
    assert make_effect_skill(action).boundary_mode=='legacy_face'
    assert make_effect_skill(action,config=EffectPushConfig(reposition_boundary='rounded_scan')).boundary_mode=='rounded_scan'
    with pytest.raises(ValueError):RepositionSkill(action,boundary_mode='hidden_polygon')
