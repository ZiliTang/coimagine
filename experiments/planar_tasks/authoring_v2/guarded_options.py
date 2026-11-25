"""One-command static verification and dispatch over finite public observation cells.

The profile is supplied by the host. Each cell is ONE reading of interval-valued
sensors, not all possible individual noiseless readings inside those intervals.
A guard fires only when every bound is established by that public reading.
Options and next observation symbols are closed enumerations. Within a host
outcome branch we propagate integer boxes. Correlations can make a rejected box
conservative. Proof is relative to the completeness and bounds of the host model.
"""
from copy import deepcopy

STOP = "STOP"


def _keys(record, keys, where):
    if not isinstance(record, dict) or set(record) != set(keys):
        raise ValueError(f"{where} expects exactly {sorted(keys)}")


def _interval(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"expected two integer bounds, got {value!r}")
    lo, hi = value
    if any(v is not None and type(v) is not int for v in value):
        raise ValueError("bounds must be integers in the metric's declared unit, or null")
    if lo is not None and hi is not None and lo > hi:
        raise ValueError("reversed interval")
    return lo, hi


def _box(box, units):
    if not isinstance(box, dict):
        raise ValueError("expected a metric-to-interval object")
    for key, value in box.items():
        if key not in units:
            raise ValueError(f"undeclared metric {key!r}")
        lo, hi = _interval(value)
        if units[key] == "flag" and any(v not in (None, 0, 1) for v in (lo, hi)):
            raise ValueError(f"invalid flag bound for {key}")


def _bound_check(observed, required):
    """Return status and one violating finite model endpoint when available."""
    low, high = observed
    lower, upper = required
    # Even a partially known box can contain a concrete violating endpoint.
    for value in (low, high):
        if value is not None and ((lower is not None and value < lower)
                                  or (upper is not None and value > upper)):
            return "counterexample", value
    if (lower is not None and low is None) or (upper is not None and high is None):
        return "unknown", None
    return "proved", None


def _obligations(kind, required, actual):
    issues = []
    for metric, bounds in required.items():
        interval = actual.get(metric, [None, None])
        status, witness = _bound_check(interval, bounds)
        if status != "proved":
            issues.append(dict(property=kind, metric=metric, status=status,
                               actual=list(interval), required=list(bounds), endpoint=witness))
    return issues


def _sum(a, b):
    return None if a is None or b is None else a + b


def _neg(a):
    return None if a is None else -a


def _post(metric, expr, cell, change, units, initial_names):
    if not isinstance(expr, list) or not expr:
        raise ValueError(f"invalid transfer for {metric}")
    if expr[0] not in ("pre", "effect", "add", "sub"):
        result = list(_interval(expr))
        _box({metric: result}, units)
        return result
    op = expr[0]
    if op in ("pre", "effect"):
        if len(expr) != 2 or not isinstance(expr[1], str):
            raise ValueError(f"expected [{op}, metric]")
        source = expr[1]
        if source not in (initial_names if op == "pre" else change):
            raise ValueError(f"unavailable {op} metric {source!r}")
        if units[source] != units[metric]:
            raise ValueError(f"incompatible units for {metric} and {source}")
        return list((cell if op == "pre" else change).get(source, [None, None]))
    if len(expr) != 3 or not all(isinstance(x, str) for x in expr[1:]) or units[metric] == "flag":
        raise ValueError(f"expected [{op}, numeric pre metric, numeric effect metric]")
    a, b = _post(metric, ["pre", expr[1]], cell, change, units, initial_names)
    c, d = _post(metric, ["effect", expr[2]], cell, change, units, initial_names)
    if op == "sub":
        c, d = _neg(d), _neg(c)
    return [_sum(a, c), _sum(b, d)]


def _status(issues):
    for status in ("counterexample", "unknown", "conditional"):
        if any(issue["status"] == status for issue in issues):
            return status
    return "proved"


def _validate(program, profile):
    _keys(program, {"rules"}, "program")
    _keys(profile, {"units", "cells", "options"}, "host profile")
    units, cells, options = (profile[k] for k in ("units", "cells", "options"))
    if not all(isinstance(d, dict) and d for d in (units, cells, options)):
        raise ValueError("expected nonempty units, public cells, host options")
    if any(not isinstance(key, str) or not key for table in (units, cells, options) for key in table):
        raise ValueError("metric, cell and option names must be nonempty strings")
    if any(not isinstance(u, str) or not u for u in units.values()):
        raise ValueError("expected named units")
    if STOP in options:
        raise ValueError("STOP is reserved")
    initial = set()
    for cell in cells.values():
        _box(cell, units)
        initial.update(cell)
    for name, option in options.items():
        if not isinstance(option, dict) or set(option) not in (
            {"requires"}, {"requires", "quality", "outcomes"}
        ):
            raise ValueError(f"unsupported host option format for {name}")
        _box(option["requires"], units)
        if "outcomes" not in option:
            continue
        if option["quality"] not in ("model", "empirical", "unknown"):
            raise ValueError("unknown host model quality")
        if not isinstance(option["outcomes"], list) or not option["outcomes"]:
            raise ValueError("provide all modeled outcomes, including unknown bounds if needed")
        for outcome in option["outcomes"]:
            _keys(outcome, {"observe", "change", "post"}, "host outcome")
            if not isinstance(outcome["observe"], str) or not outcome["observe"]:
                raise ValueError("expected a public observation symbol")
            _box(outcome["change"], units)
            if not isinstance(outcome["post"], dict) or not set(outcome["post"]) <= units.keys():
                raise ValueError("unknown successor metric")
            for metric, expr in outcome["post"].items():
                _post(metric, expr, {}, outcome["change"], units, initial)
    if not isinstance(program["rules"], list) or not program["rules"]:
        raise ValueError("expected ordered rules")
    names = set()
    for r in program["rules"]:
        _keys(r, {"name", "when", "option", "expect", "then"}, "policy rule")
        if not isinstance(r["name"], str) or not r["name"] or r["name"] in names:
            raise ValueError("rule names must be nonempty and distinct")
        names.add(r["name"])
        _box(r["when"], units)
        if set(r["when"]) - initial:
            raise ValueError("guards may read public initial metrics only")
        name = r["option"]
        if not isinstance(name, str) or (name != STOP and (name not in options or "outcomes" not in options[name])):
            raise ValueError(f"invalid primary option {name}")
        _box(r["expect"], units)
        if not isinstance(r["then"], dict) or any(
            not isinstance(symbol, str) or not symbol or not isinstance(action, str)
            or (action not in options and action != STOP)
            for symbol, action in r["then"].items()
        ):
            raise ValueError("invalid continuation dispatch")
        if name == STOP and (r["expect"] or r["then"]):
            raise ValueError("STOP has no productive effect or continuation")
        if name != STOP and r["then"].keys() - {o["observe"] for o in options[name]["outcomes"]}:
            raise ValueError("undeclared successor observation in continuation dispatch")


def _case(cell_id, cell, rules, units, options, initial_names):
    enabled = [r for r in rules if not _obligations("guard", r["when"], cell)]
    report = dict(cell=cell_id, rule=None, option=None, status="no_action", issues=[], branches=[],
                  enabled=[r["name"] for r in enabled])
    if not enabled:
        return report
    r = enabled[0]
    report.update(rule=r["name"], option=r["option"])
    if r["option"] == STOP:
        return report
    option = options[r["option"]]
    issues = report["issues"]
    issues.extend(_obligations("precondition", option["requires"], cell))
    if option["quality"] != "model":
        issues.append(dict(property="model_quality", status=(
            "conditional" if option["quality"] == "empirical" else "unknown")))
    for index, outcome in enumerate(option["outcomes"]):
        symbol, change = outcome["observe"], outcome["change"]
        post = {metric: _post(metric, expr, cell, change, units, initial_names)
                for metric, expr in outcome["post"].items()}
        branch_issues = _obligations("effect", r["expect"], change)
        for metric in change:
            if metric not in r["expect"] or list(r["expect"][metric]) == [None, None]:
                branch_issues.append(dict(property="missing_effect", metric=metric, status="unknown"))
        next_action = r["then"].get(symbol)
        if next_action in (None, STOP):
            branch_issues.append(dict(property="no_productive_continuation", status="counterexample"))
        else:
            branch_issues.extend(_obligations("continuation", options[next_action]["requires"], post))
        report["branches"].append(dict(index=index, observation=symbol, change=change, post=post,
                                       next_option=next_action, issues=branch_issues))
        issues.extend(dict(issue, branch=index, observation=symbol) for issue in branch_issues)
    report["status"] = _status(issues)
    return report


class CompiledPolicy:
    """Take an abstract public cell, issue only verified option tokens at two boundaries."""

    def __init__(self, program, profile):
        program, profile = deepcopy(program), deepcopy(profile)
        _validate(program, profile)
        initial_names = set().union(*(c.keys() for c in profile["cells"].values()))
        self._cases = {key: _case(key, cell, program["rules"], profile["units"], profile["options"],
                                  initial_names) for key, cell in profile["cells"].items()}
        used = {c["rule"] for c in self._cases.values()}
        self._unselected = [r["name"] for r in program["rules"] if r["name"] not in used]

    def check(self):
        cases = list(self._cases.values())
        states = {c["status"] for c in cases}
        status = next(s for s in ("counterexample", "unknown", "conditional", "no_action", "proved")
                      if s in states)
        return dict(status=status, cases=deepcopy(cases), unselected_rules=list(self._unselected))

    def decide(self, cell, *, allow_empirical=False):
        row = self._cases.get(cell)
        if row and (row["status"] == "proved" or (allow_empirical and row["status"] == "conditional")):
            return row["option"]
        return None

    def resume(self, cell, observation_symbol, *, allow_empirical=False):
        if self.decide(cell, allow_empirical=allow_empirical) is not None:
            for branch in self._cases[cell]["branches"]:
                if branch["observation"] == observation_symbol:
                    return branch["next_option"]
        return None


def compile_policy(program, profile):
    return CompiledPolicy(program, profile)
