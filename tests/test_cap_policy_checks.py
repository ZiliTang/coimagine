from copy import deepcopy

import pytest

from experiments.planar_tasks.authoring_v2.policy_checks import check_boundary


SINGLE = [7, 46]
DUAL = [7, 7]
MENU = [SINGLE, DUAL]


def packet():
    return {
        "version": 1,
        "claim": "abstain",
        "selected": None,
        "proposed": [SINGLE],
        "reviews": [
            {"pair": SINGLE, "verdict": "reject", "peer_robots": [1]},
            {"pair": DUAL, "verdict": "pass", "peer_robots": []},
        ],
        "widths": [],
        "peer_predictions": [{"pair": SINGLE, "robot": 1, "model": "trajectory_set", "scope": "hypothesis"}],
    }


def findings(report):
    return {f["code"]: f["severity"] for f in report["findings"]}


def width_check(source, lower, upper, required, conclusion, **extra):
    report = packet()
    report["widths"] = [dict(pair=DUAL, source=source, lower_m=lower, upper_m=upper,
                             required_m=required, conclusion=conclusion, **extra)]
    return findings(check_boundary(MENU, report, selected=None))


@pytest.mark.parametrize("source,low,high,need,verdict,allowed", [
    ("visible_overlap", .151, None, .42, "insufficient", False),
    ("observed_lower_bound", .42, None, .42, "sufficient", True),
    ("certified_upper_bound", None, .41, .42, "insufficient", True),
    ("certified_upper_bound", None, .42, .42, "insufficient", False),
    ("certified_interval", .40, .43, .42, "unknown", True),
    ("visible_overlap", .10, .15, .42, "insufficient", False),
    ("certified_interval", .43, .44, .42, "sufficient", True),
])
def test_width_inference_direction(source, low, high, need, verdict, allowed):
    result = width_check(source, low, high, need, verdict)
    assert ("bound_direction" not in result) == allowed


def test_partial_visibility_can_support_honest_abstention():
    result = width_check("visible_overlap", .151, None, .42, "unknown", pruned=True)
    assert result["partial_width_pruning"] == "warning"
    assert "bound_direction" not in result


def test_unguaranteed_peer_exactness_and_hypothetical_veto():
    report = packet()
    peer = report["peer_predictions"][0]
    peer.update(model="stationary", scope="exact", used_to_veto=True)
    assert findings(check_boundary(MENU, report))["unsupported_peer_guarantee"] == "error"
    peer["scope"] = "hypothesis"
    assert findings(check_boundary(MENU, report))["single_yield_hypothesis"] == "warning"
    peer["used_to_veto"] = False
    assert "single_yield_hypothesis" not in findings(check_boundary(MENU, report))


def test_exhaustion_is_stronger_than_partial_search():
    report = packet()
    honest = check_boundary(MENU, report, selected=None)
    assert honest["omitted_passing"] == [DUAL] and honest["selection_checked"]
    assert honest["status"] == "advisory"
    assert findings(honest) == {"omitted_passing": "warning"}
    report["claim"] = "no_passing_push"
    categorical = check_boundary(MENU, report)
    assert categorical["status"] == "violation" and not categorical["selection_checked"]
    assert findings(categorical)["unsupported_exhaustion"] == "error"


def test_greedy_valid_selection_can_skip_passing_alternatives():
    report = packet()
    report.update(claim="selected", selected=SINGLE)
    report["reviews"][0]["verdict"] = "pass"
    result = check_boundary(MENU, report, selected=SINGLE)
    assert result["status"] == "ok"
    assert result["omitted_passing"] == [DUAL]
    assert findings(check_boundary(MENU, report, selected=None))["selection_mismatch"] == "error"
    report["reviews"][0]["verdict"] = "reject"
    assert findings(check_boundary(MENU, report, selected=SINGLE))["selected_not_permitted"] == "error"


def test_unresolved_physical_outcomes_can_have_no_passing_pairs():
    report = packet()
    report["claim"] = "no_passing_push"
    report["reviews"][1]["verdict"] = "unknown"
    partial = check_boundary(MENU, report, selected=None)
    assert findings(partial)["uncertified_partial_search"] == "error"
    report["proposed"] = MENU
    result = check_boundary(MENU, report, selected=None)
    assert result["status"] == "ok" and result["counts"]["unknown"] == 1


@pytest.mark.parametrize("update,match", [
    (lambda r: r.pop("proposed"), "missing"),
    (lambda r: r["reviews"].pop(), "missing"),
    (lambda r: r["reviews"].append(r["reviews"][0]), "duplicate"),
    (lambda r: r["reviews"][0].pop("peer_robots"), "missing"),
    (lambda r: r["peer_predictions"].clear(), "missing peer"),
    (lambda r: r["peer_predictions"].append(r["peer_predictions"][0]), "duplicate peer"),
    (lambda r: r["peer_predictions"][0].update(robot=True), "invalid peer"),
    (lambda r: r["proposed"].append([2, 46]), "outside"),
    (lambda r: r["proposed"].append([44, 46]), "no translating PUSH"),
    (lambda r: r.update(version=True), "version"),
])
def test_invalid_rows_cannot_masquerade_as_passing_checks(update, match):
    report = packet()
    update(report)
    with pytest.raises(ValueError, match=match):
        check_boundary(MENU, report)


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_malformed_evidence_rejected(value):
    with pytest.raises(ValueError, match="nonnegative distance"):
        width_check("visible_overlap", value, None, .42, "unknown")


def test_empty_menu_and_deterministic_order():
    empty = dict(version=1, claim="no_passing_push", selected=None, proposed=[], widths=[],
                 reviews=[], peer_predictions=[])
    assert check_boundary([], empty, selected=None)["status"] == "ok"
    p = packet()
    q = deepcopy(p)
    q["reviews"].reverse()
    assert check_boundary(MENU, p) == check_boundary(MENU[::-1], q)
