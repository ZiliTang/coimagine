import pytest

from tools import cap_guarded_calibration_loop as loop


def outcome(forward=.023, static=False, entry=True, completed=False):
    return dict(forward_m=forward, object_static=static, endpoint_followup_available=entry,
                followup_completed=completed)


class Predictions:
    def __init__(self, values, local=True):
        self.values = values
        self.status = "local" if local else "pooled"

    def predict(self, progress, features):
        return dict(samples=[dict(weight=w, outcome=o) for w, o in self.values], status=self.status,
                    local_support_rooms=3, mean={}, interval={}, variance={})


def test_joint_outcomes_and_physical_continuation_are_distinct():
    rule = loop.read(loop.RULE)
    assert loop.contract_verdicts([outcome(), outcome(.0199), outcome(static=True),
                                   outcome(entry=False, completed=True)], rule) == [True, False, False, False]
    model = Predictions([(.5, outcome(.01, entry=True)), (.5, outcome(.04, entry=False))])
    result = loop.estimate(model, .10, {}, rule)
    assert result["model_decision"] == "reject"
    assert result["predicted_contract_risk"] == 1


@pytest.mark.parametrize(("weight","local","decision"), [
    (.2, True, "admit"), (.5, True, "query"), (.8, True, "reject"), (0, False, "query")])
def test_empirical_gate_requires_local_support(weight, local, decision):
    result = loop.estimate(Predictions([(weight,outcome(.01)), (1-weight,outcome())],local),
                           .10, {}, loop.read(loop.RULE))
    assert result["model_decision"] == decision


def test_gate_refuses_privileged_decision_input():
    with pytest.raises(ValueError, match="public"):
        loop.dispatch(None, .2, dict(public_history="h", original_action={},features={},room="test"),
                      {}, None, None)
