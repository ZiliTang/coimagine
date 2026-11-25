from copy import deepcopy

from experiments.planar_tasks.authoring_v2.four_stage.compiler import Extractor
from experiments.planar_tasks.authoring_v2.four_stage.gateway import Gateway


def setup():
    metrics = dict(seen=dict(unit="bool"), legal_a=dict(unit="bool"), legal_b=dict(unit="bool"))
    host = dict(metrics=metrics, options={
        a: dict(requires=["var", "legal_" + a], writes=list(metrics), frame=[])
        for a in ("a", "b")})
    source = '''
def decide(z):
    candidates = []
    if z["seen"]:
        candidates.append(("a", 10))
    candidates.append(("b", -10))
    return candidates
'''
    program = Extractor(source, host)
    z = dict(seen=True, legal_a=True, legal_b=True)
    prepared = dict(symbols=z, public_context=dict(symbols=deepcopy(z), observation={"public": True}),
                    commands={a: {"name": a} for a in ("a", "b")},
                    reports={a: dict(legal=True, allowed=True) for a in ("a", "b")})
    return program, prepared


class Model:
    def __init__(self):
        self.calls = []

    def predict(self, context, action):
        self.calls.append((context, action))
        context["symbols"]["seen"] = False
        return dict(supported=True, calibrated=True, model_id="test", branches=[
            dict(updates=dict(seen=False, legal_a=False, legal_b=True))])


def test_one_step_only_uses_raw_public_context_and_does_not_mutate_real_values():
    program, prepared = setup()
    model = Model()
    gateway = Gateway(program, model=model)
    result = gateway.decide(prepared)
    assert result["selected"] == "a"
    assert len(model.calls) == 1
    assert model.calls[0][1] == "a"  # no recursive prediction of successor b
    assert model.calls[0][0]["observation"] == {"public": True}
    assert prepared["symbols"]["seen"] is True
    assert prepared["public_context"]["symbols"]["seen"] is True


def test_broad_host_proof_does_not_spend_unnecessary_model_calls():
    program, prepared = setup()
    model = Model()
    gateway = Gateway(program, model=model, invariants={"legal_b": True})
    assert gateway.decide(prepared)["selected"] == "a"
    assert not model.calls
    assert gateway.ledger["hard_admissions"] == 1


def test_current_full_check_can_fall_back_to_negative_score():
    program, prepared = setup()
    prepared["reports"]["a"]["allowed"] = False
    gateway = Gateway(program, invariants={"legal_b": True})
    result = gateway.decide(prepared)
    assert result["selected"] == "b"
    assert len(result["checks"]) == 2


def test_shadow_records_unresolved_forecast_without_enforcing_it():
    program, prepared = setup()
    gateway = Gateway(program, mode="shadow")
    result = gateway.decide(prepared)
    assert result["selected"] == "a"
    assert not result["checks"][0]["verdict"]["allowed"]


def test_no_certified_candidate_is_explicit_not_unchecked_brake():
    program, prepared = setup()
    for report in prepared["reports"].values():
        report["allowed"] = False
    gateway = Gateway(program, mode="author")
    assert gateway.decide(prepared)["selected"] is None
    assert gateway.ledger["no_admissible"] == 1


def test_online_success_does_not_drop_existing_bad_mode():
    program, prepared = setup()
    class BadModel:
        def predict(self, context, action):
            return dict(supported=False, calibrated=False, branches=[
                dict(updates=dict(seen=False, legal_a=False, legal_b=False))])
    class Online:
        def run(self, context, action, command):
            return dict(admissible_evidence=True, branches=[
                dict(updates=dict(seen=False, legal_a=False, legal_b=True))],
                cost=dict(branches=2, ticks=8, seconds=.1))
    gateway = Gateway(program, model=BadModel(), rollout=Online(), max_queries=1)
    result = gateway.decide(prepared)
    assert result["selected"] is None
    assert gateway.ledger["online_queries"] == 1
    assert gateway.ledger["online_branches"] == 2


def test_current_startup_predicate_cannot_be_overridden_by_stale_report():
    program, prepared = setup()
    prepared["symbols"]["legal_a"] = False
    gateway = Gateway(program, mode="author")
    assert gateway.decide(prepared)["selected"] == "b"


def test_broad_proof_is_reused_only_after_universal_check():
    program,prepared=setup()
    original=program.policy.check;calls=[]
    def counted(current,*args,**kwargs):
        calls.append(current)
        return original(current,*args,**kwargs)
    program.policy.check=counted
    gateway=Gateway(program,invariants={"legal_b":True})
    assert gateway.decide(prepared)["selected"]=="a"
    prepared["symbols"]["legal_b"]=False
    assert gateway.decide(prepared)["selected"]=="a"
    assert calls==[{"legal_a":True}]


def test_unresolved_prior_path_does_not_spend_futile_online_query():
    program,prepared=setup()
    for report in prepared["reports"].values():report["allowed"]=False
    class UnusedOnline:
        def run(self,*args):
            raise AssertionError("new samples cannot remove an earlier unresolved path")
    gateway=Gateway(program,model=Model(),rollout=UnusedOnline())
    result=gateway.decide(prepared)
    assert result["selected"] is None
    assert gateway.ledger["online_queries"]==0
