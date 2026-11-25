"""Fit a scalar correction at a frozen recovery program's proposed entry.

Rows are paired interventions on an identical public prefix. The label is terminal
success(defer one opportunity) minus success(take now). Both branches revert to
the original gate thereafter. A positive score favours deferral. Offline rates
give each represented room equal weight. They describe one-opportunity edits,
not the closed-loop effect of applying the gate repeatedly.
"""
from collections import Counter
import math


RIDGE = 1.0
Z_LIMIT = 3.0
THRESHOLDS = (0.0, 0.05, 0.15)
ZERO = {"kind": "zero"}
ALWAYS_DEFER = {"kind": "always_defer"}


def _features(names, features):
    if set(features) != set(names):
        raise ValueError("gate feature keys do not match its frozen input schema")
    values = [float(features[name]) for name in names]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("gate inputs must be finite")
    return values


def _advantage(gate, features):
    values = _features(gate["features"], features)
    z_limit = gate["z_limit"]
    score = gate["intercept"]
    for x, mu, scale, coefficient in zip(
        values, gate["center"], gate["scale"], gate["weights"], strict=True
    ):
        z = max(-z_limit, min(z_limit, (x - mu) / scale))
        score += coefficient * z
    return score


def should_defer(gate, features):
    """Pure-Python deployment rule. Strict ties preserve the author's entry."""
    if gate["kind"] == "zero":
        return False
    if gate["kind"] == "always_defer":
        return True
    if gate["kind"] != "linear":
        raise ValueError(f"unknown gate kind: {gate['kind']}")
    return _advantage(gate, features) > gate["threshold"]


def _check(rows):
    for row in rows:
        if not isinstance(row["room"], str) or not row["room"]:
            raise ValueError("a paired row needs its room identity")
        if row["take"] not in (0, 1) or row["defer"] not in (0, 1):
            raise ValueError("paired terminal labels must be complete binary outcomes")


def _room_weights(rows):
    counts = Counter(row["room"] for row in rows)
    return [1.0 / (len(counts) * counts[row["room"]]) for row in rows]


def _fit_linear(rows):
    import numpy as np

    names = sorted(rows[0]["features"])
    if not names:
        raise ValueError("no public entry features")
    x = np.asarray([_features(names, row["features"]) for row in rows], dtype=float)
    w = np.asarray(_room_weights(rows))
    y = np.asarray([int(row["defer"]) - int(row["take"]) for row in rows])
    center = w @ x
    scale = np.sqrt(w @ ((x - center) ** 2))
    scale[scale < 1e-8] = 1.0
    z = np.clip((x - center) / scale, -Z_LIMIT, Z_LIMIT)
    z_mean = w @ z
    y_mean = float(w @ y)
    centered = z - z_mean
    # Normalised weights give the same regularisation when rooms are repeated.
    gram = centered.T @ (w[:, None] * centered)
    weights = np.linalg.solve(gram + RIDGE * np.eye(len(names)),
                              centered.T @ (w * (y - y_mean)))
    return dict(kind="linear", features=names, center=center.tolist(),
                scale=scale.tolist(), weights=weights.tolist(), z_limit=Z_LIMIT,
                intercept=float(y_mean - z_mean @ weights))


def _summary(rows, gate):
    w = _room_weights(rows)
    labels = Counter()
    defer_weight = take_rate = chosen_rate = best_rate = 0.0
    won = lost = deferred = kept_better_take = 0
    for row, weight in zip(rows, w):
        a, b = int(row["take"]), int(row["defer"])
        labels[(a, b)] += 1
        reject = should_defer(gate, row["features"])
        chosen = b if reject else a
        deferred += reject
        kept_better_take += int(not reject and a > b)
        won += int(chosen > a)
        lost += int(chosen < a)
        defer_weight += weight * reject
        take_rate += weight * a
        chosen_rate += weight * chosen
        best_rate += weight * max(a, b)
    return dict(
        rooms=len({r["room"] for r in rows}), opportunities=len(rows),
        take_only=labels[(1, 0)], defer_only=labels[(0, 1)],
        both_success=labels[(1, 1)], both_fail=labels[(0, 0)],
        deferred=deferred, better_takes_retained=kept_better_take,
        gained_against_take=won, lost_against_take=lost,
        room_mean_defer_fraction=defer_weight if rows else None,
        room_mean_chosen_success=chosen_rate if rows else None,
        room_mean_delta_vs_take=(chosen_rate - take_rate) if rows else None,
        room_mean_best_observed_single_fork=best_rate if rows else None,
    )


def _ranking(rows, gate):
    """Conditional ranking of defer-only versus take-only outcomes; ties count half."""
    weighted = sorted(
        (_advantage(gate, r["features"]), int(r["defer"]) - int(r["take"]), weight)
        for r, weight in zip(rows, _room_weights(rows)) if r["take"] != r["defer"]
    )
    from itertools import groupby
    neg = pos = concordant = 0.0
    for _, group in groupby(weighted, key=lambda item: item[0]):
        tied = list(group)
        n = sum(w for _, d, w in tied if d < 0)
        p = sum(w for _, d, w in tied if d > 0)
        concordant += p * (neg + 0.5 * n)
        neg += n
        pos += p
    return concordant / (neg * pos) if neg and pos else None


def train(train_rows, validation_rows):
    """Return a frozen experimental gate and a separate validation adoption choice.

    Only the fit split sets coefficients and normalization. The validation split
    chooses among three fixed deferral thresholds. A tie favours the highest
    threshold. The learned candidate is always returned for independent testing.
    """
    fit, val = list(train_rows), list(validation_rows)
    _check(fit)
    _check(val)
    if {r["room"] for r in fit} & {r["room"] for r in val}:
        raise ValueError("a room crosses fit and validation splits")
    if not fit:
        raise ValueError("no paired training opportunities")
    linear = _fit_linear(fit)
    candidates = [dict(linear, threshold=t) for t in THRESHOLDS]
    grid = [_summary(val, g) for g in candidates]
    idx = max(range(len(grid)), key=lambda i: (
        round(grid[i]["room_mean_delta_vs_take"] or 0.0, 12), THRESHOLDS[i]))
    gate = candidates[idx]
    offline = {}
    for label, rows in [("train", fit), ("validation", val)]:
        models = {k: _summary(rows, g) for k, g in [
            ("zero", ZERO), ("always_defer", ALWAYS_DEFER), ("learned", gate)]}
        models["discordant_room_weighted_auc"] = _ranking(rows, gate)
        offline[label] = models
    metrics = offline["validation"]
    best_constant = max(metrics[m]["room_mean_chosen_success"] or 0.0
                        for m in ("zero", "always_defer"))
    # If always deferring does as well, there is no evidence of selective entry.
    support = ((metrics["learned"]["room_mean_chosen_success"] or 0.0)
               > best_constant + 1e-12)
    return dict(gate=gate, adopted_gate=gate if support else dict(ZERO),
                validation_supports_selective_gate=support, offline=offline,
                validation_thresholds=[dict(threshold=t, **v)
                                       for t, v in zip(THRESHOLDS, grid)],
                fit_protocol=dict(label="defer_success_minus_take_success",
                                  room_weight="uniform_room_then_uniform_opportunity",
                                  ridge=RIDGE, thresholds=list(THRESHOLDS),
                                  z_clip=Z_LIMIT))
