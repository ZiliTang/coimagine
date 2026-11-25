import json

import pytest

from tools.cap_guarded_calibration_metrics import paired_changes, summarize


def candidate(room, dose, admitted, completed, ok, source="model", calls=0, **override):
    values = dict(room=room, anchor_id="first", progress_m=dose, admitted=admitted,
                  followup_completed=completed, actual_contract_ok=ok, source=source,
                  calls=calls, probe_requested=source != "model", actual_forward_m=.1,
                  predicted_forward_m=.12, predicted_contract_risk=.25)
    return dict(values, **override)


def panel():
    return [candidate("a", .1, True, True, False),
            candidate("a", .2, True, False, True, "fallback", 2),
            candidate("b", .1, False, True, True, "insufficient", 0,
                      predicted_forward_m=None, predicted_contract_risk=None)]


def test_metrics_keep_candidate_boundary_and_admission_denominators_distinct():
    s = summarize(panel())
    assert (s["n_rooms"], s["n_boundaries"], s["n_candidates"]) == (2, 2, 3)
    assert (s["admitted"], s["useful"], s["failed"], s["rejected_but_completed"]) == (2, 1, 1, 1)
    assert s["coverage"] == 1/3 and s["failure_rate"] == .5
    assert s["calls"] == 2 and s["calls_per_100_boundaries"] == 100
    assert s["issued_contract_violations"] == 1
    assert s["contract_brier"] == (.75**2 + .25**2)/2
    assert s["forward_mae_m"] == pytest.approx(.02)
    assert s["n_forward_predictions"] == s["n_risk_predictions"] == 2
    masked = s["by_mode"]["insufficient"]
    assert masked["failure_rate"] is None and masked["contract_brier"] is None
    assert masked["forward_mae_m"] is None and masked["n_risk_predictions"] == 0
    assert sum(s["sources"].values()) == 3
    json.dumps(s, allow_nan=False)


def test_full_abstention_and_missing_model_predictions_are_undefined():
    v0 = panel()
    v2 = [dict(r, admitted=False, predicted_contract_risk=None, predicted_forward_m=None)
          for r in v0]
    s = summarize(v2)
    assert s["failure_rate"] is None
    assert s["coverage"] == 0 and s["n_risk_predictions"] == 0
    assert s["contract_brier"] is None and s["forward_mae_m"] is None
    out = paired_changes({"v0": v0, "v2": v2}, draws=50)
    failure = out["metrics"]["failure_rate"]
    assert failure["defined_draws"] == 0 and failure["difference"] is None and failure["interval"] is None
    assert out["metrics"]["coverage"]["difference"] == pytest.approx(-1/3)
    json.dumps(out, allow_nan=False)


def test_bootstrap_carries_all_candidates_in_a_room_together():
    initial = panel()
    changed = [dict(r, admitted=(r["room"] == "b")) for r in reversed(initial)]
    result = paired_changes({"v0": initial, "v2": changed})
    # Both rooms must appear for F to exist under both policies.
    stats = result["metrics"]["failure_rate"]
    assert 0 < stats["defined_draws"] < result["draws"]
    assert stats["interval"] == [-.5, -.5]
    assert result["metrics"]["coverage"]["difference"] == 0
    json.dumps(result, allow_nan=False)


def test_comparison_rejects_missing_candidates_changed_labels_and_duplicate_identifiers():
    initial = panel()
    with pytest.raises(ValueError, match="identical"):
        paired_changes({"v0": initial, "v2": initial[:-1]})
    with pytest.raises(ValueError, match="label"):
        paired_changes({"v0": initial, "v2": [dict(initial[0], actual_contract_ok=True), *initial[1:]]})
    with pytest.raises(ValueError, match="duplicate"):
        summarize(initial + initial[:1])
