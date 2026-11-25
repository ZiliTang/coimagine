"""Frozen candidate-panel metrics and paired uncertainty for the PUSH calibration pilot."""

from collections import Counter, defaultdict
from math import fsum, isfinite

import numpy as np


def _index(rows):
    result = {}
    for row in rows:
        key = (row["room"], row["anchor_id"], row["progress_m"])
        if key in result:
            raise ValueError(f"duplicate candidate {key!r}")
        for flag in ("admitted", "followup_completed", "actual_contract_ok", "probe_requested"):
            if type(row[flag]) is not bool:
                raise ValueError(f"{flag} needs a measured boolean for candidate {key!r}")
        if type(row["calls"]) is not int or row["calls"] < 0:
            raise ValueError("calls must count actual, nonnegative simulation invocations")
        if not isfinite(row["actual_forward_m"]):
            raise ValueError("missing observed forward displacement")
        prediction = row["predicted_forward_m"]
        if prediction is not None and not isfinite(prediction):
            raise ValueError("invalid predicted displacement")
        risk = row["predicted_contract_risk"]
        if risk is not None and not 0 <= risk <= 1:
            raise ValueError("risk must be null or a probability")
        result[key] = row
    return result


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def _counts(rows):
    admitted = sum(r["admitted"] for r in rows)
    useful = sum(r["admitted"] and r["followup_completed"] for r in rows)
    failed = admitted - useful
    calls = sum(r["calls"] for r in rows)
    n_boundaries = len({(r["room"], r["anchor_id"]) for r in rows})
    return dict(n_rooms=len({r["room"] for r in rows}), n_boundaries=n_boundaries,
                n_candidates=len(rows), admitted=admitted, useful=useful, failed=failed,
                rejected_but_completed=sum(not r["admitted"] and r["followup_completed"] for r in rows),
                coverage=_ratio(useful, len(rows)), failure_rate=_ratio(failed, admitted),
                calls=calls, calls_per_100_boundaries=_ratio(100 * calls, n_boundaries),
                probes_requested=sum(r["probe_requested"] for r in rows),
                issued_contract_violations=sum(r["admitted"] and not r["actual_contract_ok"] for r in rows))


def _errors(rows):
    forward = [r["predicted_forward_m"] - r["actual_forward_m"]
               for r in rows if r["predicted_forward_m"] is not None]
    residuals = [r["predicted_contract_risk"] - int(not r["actual_contract_ok"])
                 for r in rows if r["predicted_contract_risk"] is not None]
    return dict(n_forward_predictions=len(forward),
                forward_mae_m=_ratio(fsum(abs(v) for v in forward), len(forward)),
                n_risk_predictions=len(residuals),
                contract_brier=_ratio(fsum(v*v for v in residuals), len(residuals)))


def summarize(rows):
    """C counts useful candidates; F counts failed follow-ups among admissions.

    K counts simulated branches per 100 distinct decision boundaries offered on
    this panel. Branches used to measure candidate labels are excluded by callers.
    A missing prediction is omitted from scoring and its denominator is exposed.
    """
    rows = list(_index(rows).values())
    groups = {"model": [], "fallback": [], "insufficient": []}
    for r in rows:
        if r["source"] not in groups:
            raise ValueError(f"unexpected decision source {r['source']!r}")
        groups[r["source"]].append(r)
    return dict(**_counts(rows), **_errors(rows),
                sources=dict(sorted(Counter(r["source"] for r in rows).items())),
                by_mode={k: {**_counts(v), **_errors(v)} for k, v in groups.items()})


def paired_changes(per_version, baseline="v0", final="v2", *, draws=4000, seed=17):
    """Final minus baseline, resampling rooms with replacement and retaining each full room."""
    if type(draws) is not int or draws < 1:
        raise ValueError("draws must be positive")
    left, right = _index(per_version[baseline]), _index(per_version[final])
    if not left or left.keys() != right.keys():
        raise ValueError("versions require identical nonempty room/anchor/progress candidates")
    rooms = defaultdict(list)
    for key in sorted(left):
        for field in ("followup_completed", "actual_contract_ok", "actual_forward_m"):
            if left[key][field] != right[key][field]:
                raise ValueError(f"unpaired physical label {field} at {key}")
        rooms[key[0]].append(key)
    versions = [
        [_counts([idx[k] for k in keys]) for keys in rooms.values()]
        for idx in (left, right)
    ]
    n = len(rooms)
    # One shared draw for both versions, all candidates and all metrics.
    weights = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=draws)
    metrics = {}
    for name, numerator, denominator, scale in [
        ("coverage", "useful", "n_candidates", 1),
        ("failure_rate", "failed", "admitted", 1),
        ("calls_per_100_boundaries", "calls", "n_boundaries", 100),
    ]:
        point = []
        repeats = []
        for counts in versions:
            nums = np.array([c[numerator] for c in counts], dtype=int)
            dens = np.array([c[denominator] for c in counts], dtype=int)
            point.append(_ratio(scale * int(nums.sum()), int(dens.sum())))
            den = weights @ dens
            repeats.append(np.divide(scale * (weights @ nums), den,
                                      out=np.full(draws, np.nan), where=(den > 0)))
        differences = repeats[1] - repeats[0]
        defined = differences[np.isfinite(differences)]
        metrics[name] = dict(baseline=point[0], final=point[1],
                             difference=point[1]-point[0] if None not in point else None,
                             interval=np.quantile(defined, [.025, .975]).tolist() if len(defined) else None,
                             defined_draws=len(defined), undefined_draws=draws-len(defined))
    return dict(baseline=baseline, final=final, n_rooms=n, n_candidates=len(left), seed=seed, draws=draws,
                method="paired room percentile bootstrap, pooled candidate ratios; intervals use defined draws",
                metrics=metrics)
