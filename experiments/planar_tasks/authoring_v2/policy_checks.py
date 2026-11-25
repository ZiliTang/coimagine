"""Check one authored decision against its public, finite target-PUSH menu.

The host supplies the complete menu and the executed proposal. The checker consumes
plain data returned by an isolated Python policy. It never imports a policy.
A review verdict refers to the policy's declared geometric test, not physical safety.
See check_boundary for the version 1 wire format.
"""

import math


_MISSING = object()
YIELD_CLEARANCE = 46
WIDTH_SOURCES = ("visible_overlap", "observed_lower_bound", "certified_upper_bound", "certified_interval")
PEER_MODELS = ("stationary", "co_moving", "trajectory_set", "unknown")
PEER_SCOPES = ("exact", "hypothesis")


def _field(record, name):
    if not isinstance(record, dict) or name not in record:
        raise ValueError(f"missing field {name!r}")
    return record[name]


def _list(value, name):
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a list")
    return value


def _one_of(value, choices, name):
    if value not in choices:
        raise ValueError(f"invalid {name}: {value!r}")
    return value


def _flag(record, name):
    value = record.get(name, False)
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _distance(record, name, *, optional=False):
    value = _field(record, name)
    if value is None and optional:
        return None
    if type(value) not in (float, int) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, nonnegative distance in metres")
    return value


def _pair(value, menu=None):
    value = _list(value, "action pair")
    if len(value) != 2 or any(type(action) is not int or action < 0 for action in value):
        raise ValueError(f"invalid action pair: {value!r}")
    pair = tuple(value)
    if min(pair) >= 32:
        raise ValueError(f"pair {pair} has no translating PUSH")
    if menu is not None and pair not in menu:
        raise ValueError(f"pair {pair} outside the supplied target menu")
    return pair


def _pairs(values, name, menu=None):
    result = [_pair(value, menu) for value in _list(values, name)]
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate action pair in {name}")
    return result


def _width(row, pair, add):
    source = _one_of(_field(row, "source"), WIDTH_SOURCES, "width source")
    conclusion = _one_of(_field(row, "conclusion"), ("sufficient", "insufficient", "unknown"), "width conclusion")
    lower = _distance(row, "lower_m", optional=True)
    upper = _distance(row, "upper_m", optional=True)
    required = _distance(row, "required_m")
    pruned = _flag(row, "pruned")
    if lower is not None and upper is not None and upper < lower:
        raise ValueError(f"inverted width interval for {pair}")
    needs_lower = source != "certified_upper_bound"
    needs_upper = source in ("certified_upper_bound", "certified_interval")
    if (lower is not None) != needs_lower or (upper is not None) != needs_upper:
        add("error", "bound_direction", "Reported interval endpoints disagree with their evidence source.", pair)
        return
    sufficient = lower is not None and lower >= required
    insufficient = upper is not None and upper < required
    if (conclusion == "insufficient" and not insufficient) or (conclusion == "sufficient" and not sufficient):
        add("error", "bound_direction", "Evidence does not justify the stated width conclusion.", pair)
    if pruned and conclusion == "unknown":
        add("warning", "partial_width_pruning", "A candidate was pruned with unresolved width evidence.", pair)


def check_boundary(menu, packet, *, selected=_MISSING):
    """Audit one public target and one decision time; return a JSON-ready report.

    menu: complete list of [a0, a1] representing public target translation PUSH
      pairs. Compare reviews at a common, documented duration convention.
    selected: independently obtained target translation pair, or None for
      another action. Omitting this keyword leaves binding unverified.
    packet (all six body fields plus version are required):
      version: 1
      claim: selected | abstain | no_passing_push (whole menu), or
        no_proposed_push (examined subset). no_certified_push aliases no_passing_push.
        Version 1 has no independently verified pruning certificate, so an
        executed whole-menu claim requires considering all its pairs.
      selected: [a0,a1] or None.
      proposed: distinct pairs actually considered for the committed decision.
      reviews: exactly one {pair, verdict: pass|reject|unknown, peer_robots:[0,1] or
        a subset} per menu pair. A peer list explicitly declares whose future
        motion enters that review; [] means none. Optional reason is descriptive.
      widths: {pair,source,lower_m,upper_m,required_m,conclusion,pruned?} rows.
        visible_overlap and observed_lower_bound require only the lower endpoint.
        certified_upper_bound requires only the upper, certified_interval both.
        Use None for absent endpoints. Conclusions: sufficient, insufficient,
        unknown. pruned=True marks an actual omission. Certified bounds rely
        on the stated physical derivation; this function verifies logical direction.
      peer_predictions: one {pair,robot,model,scope,used_to_veto?} for each
        declared peer. model: stationary|co_moving|trajectory_set|unknown.
        scope: exact|hypothesis. used_to_veto means used to reject an actually
        considered option. This profile supplies no exact peer dynamics.

    Invalid/incomplete packets raise ValueError. Counts separate omitted passing
    alternatives from inconsistent decisions. A clean audit remains conditional
    on the policy faithfully exposing its computations and the supplied inputs.
    """
    if type(_field(packet, "version")) is not int or packet["version"] != 1:
        raise ValueError("expected policy checks version 1")
    ordered = _pairs(menu, "public menu")
    legal = set(ordered)
    proposed = set(_pairs(_field(packet, "proposed"), "proposed", legal))
    chosen = _field(packet, "selected")
    if chosen is not None:
        chosen = _pair(chosen, legal)
    claim = _one_of(_field(packet, "claim"),
                    ("selected", "abstain", "no_passing_push", "no_certified_push", "no_proposed_push"),
                    "decision claim")
    issues = {}

    def add(severity, code, message, *pairs):
        item = issues.setdefault((severity, code), {"severity": severity, "code": code,
                                                   "message": message, "pairs": set()})
        item["pairs"].update(pairs)

    verdicts = {}
    expected_peers = set()
    for row in _list(_field(packet, "reviews"), "reviews"):
        pair = _pair(_field(row, "pair"), legal)
        if pair in verdicts:
            raise ValueError(f"duplicate review for {pair}")
        verdicts[pair] = _one_of(_field(row, "verdict"), ("pass", "reject", "unknown"), "review verdict")
        for robot in _list(_field(row, "peer_robots"), "peer_robots"):
            if type(robot) is not int or robot not in (0, 1) or (pair, robot) in expected_peers:
                raise ValueError(f"invalid or repeated peer robot for {pair}")
            expected_peers.add((pair, robot))
    if verdicts.keys() != legal:
        raise ValueError(f"reviews missing public menu pairs: {sorted(legal - verdicts.keys())}")

    for row in _list(_field(packet, "widths"), "widths"):
        _width(row, _pair(_field(row, "pair"), legal), add)

    seen_peers = set()
    for row in _list(_field(packet, "peer_predictions"), "peer_predictions"):
        pair = _pair(_field(row, "pair"), legal)
        robot = _field(row, "robot") if "robot" in row else _field(row, "peer")
        if type(robot) is not int or robot not in (0, 1):
            raise ValueError(f"invalid peer robot for {pair}")
        if (pair, robot) not in expected_peers or (pair, robot) in seen_peers:
            raise ValueError(f"undeclared or duplicate peer prediction for {pair}, robot {robot}")
        seen_peers.add((pair, robot))
        model = _one_of(_field(row, "model"), PEER_MODELS, "peer model")
        scope = _one_of(_field(row, "scope"), PEER_SCOPES, "peer model scope")
        veto = _flag(row, "used_to_veto")
        if veto and (pair not in proposed or verdicts[pair] == "pass"):
            raise ValueError(f"peer veto for an unexamined or passing pair: {pair}")
        if scope == "exact":
            add("error", "unsupported_peer_guarantee", "The option contract provides no exact peer motion model.", pair)
        elif pair[robot] == YIELD_CLEARANCE and model == "stationary" and veto:
            add("warning", "single_yield_hypothesis", "A stationary prediction for a yielding peer affected a veto.", pair)
    if seen_peers != expected_peers:
        raise ValueError(f"missing peer predictions: {sorted(expected_peers - seen_peers)}")

    passed = {pair for pair in legal if verdicts[pair] == "pass"}
    omitted = passed - proposed
    if (chosen is not None) != (claim == "selected"):
        add("error", "selection_mismatch", "The selected pair and decision claim disagree.")
    if selected is not _MISSING:
        actual = None if selected is None else _pair(selected, legal)
        if actual != chosen:
            add("error", "selection_mismatch", "The report disagrees with the actual selected action.")
    if chosen is not None and (chosen not in proposed or chosen not in passed):
        add("error", "selected_not_permitted", "The selected push was unexamined or did not pass its review.", chosen)
    if claim in ("no_passing_push", "no_certified_push"):
        if passed:
            add("error", "unsupported_exhaustion", "Passing pairs contradict the claim about the full menu.", *sorted(passed))
        elif proposed != legal:
            add("error", "uncertified_partial_search", "The executed search did not cover its claimed scope.",
                *sorted(legal - proposed))
    elif claim == "no_proposed_push" and proposed & passed:
        add("error", "unsupported_exhaustion", "Passing proposed pairs contradict the claim about the examined subset.",
            *sorted(proposed & passed))
    if chosen is None and omitted:
        add("warning", "omitted_passing", "The policy abstained with passing pairs outside the proposed subset.", *sorted(omitted))

    findings = [{**row, "pairs": [list(p) for p in sorted(row["pairs"])]} for _, row in sorted(issues.items())]
    status = "advisory" if findings else "ok"
    if any(x["severity"] == "error" for x in findings):
        status = "violation"
    return {"status": status, "findings": findings, "selection_checked": selected is not _MISSING,
            "counts": {"legal": len(legal), "proposed": len(proposed), "passed": len(passed),
                       "rejected": sum(v == "reject" for v in verdicts.values()),
                       "unknown": sum(v == "unknown" for v in verdicts.values()), "omitted_passing": len(omitted)},
            "omitted_passing": [list(p) for p in sorted(omitted)]}
