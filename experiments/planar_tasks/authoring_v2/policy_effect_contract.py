"""Offline checks of declared metric intervals against independently supplied evidence.

The adapter supplies evidence at three distinct times. A missing bound means
the measurement does not establish that side. An interval is supported only
when all values still possible under the evidence satisfy the declaration.
"""
import math
from collections.abc import Mapping


def _interval(value, *, expectation):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError("each interval needs two bounds")
    low, high = value
    for bound in value:
        if bound is not None and (type(bound) not in (float, int) or not math.isfinite(bound)):
            raise ValueError("bounds must be finite numbers or null")
    if low is not None and high is not None and low > high:
        raise ValueError("interval bounds are reversed")
    if expectation and low is None and high is None:
        raise ValueError("an expectation must restrict at least one bound")
    return low, high


def _verdict(expected, observed):
    low, high = _interval(expected, expectation=True)
    lo, hi = _interval(observed, expectation=False)
    if (low is not None and hi is not None and hi < low
            or high is not None and lo is not None and lo > high):
        return "contradicted"
    if ((low is None or lo is not None and low <= lo)
            and (high is None or hi is not None and hi <= high)):
        return "supported"
    return "unknown"


def _all(verdicts):
    if "contradicted" in verdicts:
        return "contradicted"
    return "unknown" if "unknown" in verdicts else "supported"


def _group(requirements, measured):
    if not isinstance(requirements, Mapping) or not requirements:
        raise ValueError("a claim needs at least one named requirement")
    checks = {}
    for name, interval in requirements.items():
        if name not in measured:
            raise ValueError(f"metric not provided by this adapter at this stage: {name!r}")
        checks[name] = {"claim": list(interval), "evidence": list(measured[name]),
                        "verdict": _verdict(interval, measured[name])}
    return {"verdict": _all([v["verdict"] for v in checks.values()]), "metrics": checks}


def audit_contract(claim, measurements):
    """AND within a claim, OR over declared continuations. No physical model here.

    Each adapter must publish its stage-specific metric names and units. It
    supplies [None, None] for an unobserved metric. This is an evidence audit,
    never an execution-time filter that can read future observations.
    """
    stages = {"precondition", "outcome", "continuation"}
    if set(claim) != stages or set(measurements) != stages:
        raise ValueError("a contract needs precondition, outcome and continuation stages")
    result = {k: _group(claim[k], measurements[k]) for k in ("precondition", "outcome")}
    declared = claim["continuation"]
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError("a continuation needs a named witness")
    witnesses = {name: _group(req, measurements["continuation"]) for name, req in declared.items()}
    verdicts = [w["verdict"] for w in witnesses.values()]
    feasible = "supported" if "supported" in verdicts else (
        "unknown" if "unknown" in verdicts else "contradicted")
    result["continuation"] = {"verdict": feasible, "witnesses": witnesses}
    result["verdict"] = _all([result[s]["verdict"] for s in stages])
    return result
