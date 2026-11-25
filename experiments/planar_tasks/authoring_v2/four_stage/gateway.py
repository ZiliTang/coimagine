"""One-step external checks and first-admissible ranked selection.

Forecast backends see public observations/history, not the simulator object.
Every predicted branch reuses the extracted third-stage program. Checking its
startup selection never calls a second predictor at the hypothetical boundary.
"""
from copy import deepcopy
import time
from ..symbolic_general.core import evaluate


def point_box(symbols):
    return {k: v if type(v) is bool else [v, v] for k, v in symbols.items()}


class Gateway:
    def __init__(self, program, *, model=None, rollout=None, process_check=None,
                 mode="enforce", max_queries=2, timeout_ms=1000, invariants=None):
        if mode not in ("author", "shadow", "enforce"):
            raise ValueError("unknown checking mode")
        self.program = program
        self.model = model
        self.rollout = rollout
        self.process_check = process_check
        self.mode = mode
        self.max_queries = max_queries
        self.timeout_ms = timeout_ms
        self.invariants = deepcopy(invariants or {})
        self._broad_proofs = {}
        self.ledger = dict(boundaries=0, checked=0, rejected=0, forecasts=0,
                           online_queries=0, online_branches=0, online_ticks=0,
                           online_seconds=0., seconds=0., no_admissible=0,
                           hard_admissions=0, empirical_admissions=0,carried_stop_admissions=0)

    def check_successor(self, symbols, action, outcomes, *, quality):
        return self.program.policy.check(point_box(symbols), action, outcomes,
            quality=quality, timeout_ms=self.timeout_ms, enumerate_rules=False)

    def broad_successor(self, symbols, action):
        if action in self._broad_proofs:
            return self._broad_proofs[action]
        option = self.program.host["options"][action]
        requires = option["requires"]
        outcomes = [dict(updates=self.invariants, path_safe=True)]
        if (not option.get("frame") and set(option["writes"]) == set(self.program.host["metrics"])
                and isinstance(requires,list) and len(requires)==2 and requires[0]=="var"
                and self.program.host["metrics"][requires[1]]["unit"]=="bool"):
            # Prove over ALL valid current symbols with this startup bit true.
            # No post variable is framed, so a successful universal proof can
            # be reused at other real boundaries without re-solving point boxes.
            broad=self.program.policy.check({requires[1]:True},action,outcomes,
                quality="model",timeout_ms=self.timeout_ms,enumerate_rules=False)
            if broad["status"]=="proved":
                broad["current_scope"]="all schema states with startup bit true; no framed variables"
                self._broad_proofs[action]=broad
                return broad
        return self.check_successor(symbols,action,outcomes,quality="model")

    def forecast_outcomes(self, prepared, action, forecast):
        outcomes, process = [], []
        for branch in forecast.get("branches", []):
            if self.process_check is None:
                report = prepared["reports"][action]
            else:
                report = self.process_check(prepared, action, branch)
            process.append(report)
            outcomes.append(dict(updates=branch["updates"],
                                 path_safe=True if report["allowed"] else None,
                                 label=branch.get("label", "joint_forecast")))
        return outcomes, process

    def assess(self, prepared, action, *, queries_left):
        z = prepared["symbols"]
        if not evaluate(
                self.program.host["options"][action]["requires"],
                self.program.policy.values(z)):
            return dict(allowed=False, reason="startup", queries=0)
        current = prepared["reports"][action]
        if not current["legal"]:
            return dict(allowed=False, reason="startup", queries=0)
        stop_plan=prepared.get("carried_stop_plan",{})
        if (action=="brake" and not current["allowed"] and stop_plan.get("certified")
                and stop_plan.get("remaining_ticks",0)>=prepared["commands"][action]["ticks"]):
            # BRAKE must still be emitted by the author and pass startup. Its
            # process is a remaining segment of an already admitted stopping
            # tail, not a new optimistic prediction or an implicit extra action.
            broad=self.broad_successor(z,action)
            if broad["status"]=="proved" or self.mode=="author":
                self.ledger["carried_stop_admissions"]+=1
                return dict(allowed=True,reason="carried_stop_plan",queries=0,
                            carried_stop=deepcopy(stop_plan),successor=broad,
                            assurance=stop_plan.get("assurance","conditional prior stopping envelope"))
        if self.mode == "author":
            return dict(allowed=current["allowed"], reason="author_geometry_only",
                        geometry=current, successor="not_checked", queries=0)
        # A reliable broad abstraction can prove successor startup without
        # simulation. Do not manufacture simulation demand for a tautology.
        if current["allowed"]:
            broad = self.broad_successor(z, action)
            if broad["status"] == "proved":
                self.ledger["hard_admissions"] += 1
                return dict(allowed=True, reason="host_bound_and_broad_successor",
                            geometry=current, successor=broad, queries=0,
                            assurance=current.get("quality", "conditional_host_model"))
        forecasts = []
        if self.model is not None:
            self.ledger["forecasts"] += 1
            # A detached public packet prevents forecasts from changing real
            # policy state. No physical world object is accepted here.
            forecasts.append(self.model.predict(deepcopy(prepared["public_context"]), action))
        prior_outcomes, prior_process = [], []
        for forecast in forecasts:
            outcomes, process = self.forecast_outcomes(prepared, action, forecast)
            prior_outcomes.extend(outcomes)
            prior_process.extend(process)
            if (forecast.get("supported") and forecast.get("calibrated") and outcomes
                    and all(p["allowed"] for p in process)):
                report = self.check_successor(z, action, outcomes, quality="empirical")
                if report["status"] == "conditional":
                    self.ledger["empirical_admissions"] += 1
                    return dict(allowed=True, reason="calibrated_model", successor=report,
                                process=process, model_id=forecast.get("model_id"),
                                forecast=forecast, queries=0)
        online = None
        # Online evidence is combined with every existing possible mode below.
        # It cannot rescue an unresolved/unsafe prior path under that rule, so
        # do not spend a query whose result cannot change the current decision.
        prior_path_blocks = any(o["path_safe"] is not True for o in prior_outcomes)
        if queries_left > 0 and self.rollout is not None and not prior_path_blocks:
            self.ledger["online_queries"] += 1
            online = self.rollout.run(deepcopy(prepared["public_context"]), action,
                                      deepcopy(prepared["commands"][action]))
            for field in ("branches", "ticks", "seconds"):
                self.ledger["online_" + field] += online.get("cost", {}).get(field, 0)
            outcomes, process = self.forecast_outcomes(prepared, action, online)
            # Finite nominal successes cannot erase an earlier predicted mode.
            joint = prior_outcomes + outcomes
            if (online.get("admissible_evidence") and outcomes and
                    not online.get("errors") and not online.get("incomplete")):
                report = self.check_successor(z, action, joint, quality="empirical")
                if report["status"] == "conditional":
                    self.ledger["empirical_admissions"] += 1
                    return dict(allowed=True, reason="online_conditional", successor=report,
                                process=prior_process + process, online=online, queries=1)
        return dict(allowed=False, reason="unresolved_prediction_or_process",
                    process=prior_process, forecasts=forecasts, online=online,
                    online_skipped_due_to_prior_path=prior_path_blocks, queries=int(online is not None))

    def decide(self, prepared):
        self.ledger["boundaries"] += 1
        started = time.monotonic()
        checks, selected = [], None
        remaining = self.max_queries
        for candidate in self.program.candidates(prepared["symbols"]):
            name = candidate["option"]
            verdict = self.assess(prepared, name, queries_left=remaining)
            remaining -= verdict["queries"]
            self.ledger["checked"] += 1
            self.ledger["rejected"] += int(not verdict["allowed"])
            checks.append(dict(candidate=candidate, verdict=verdict))
            # Shadow mode logs predictive vetoes but does not enforce them.
            allowed = (prepared["reports"][name]["allowed"]
                       if self.mode == "shadow" else verdict["allowed"])
            if allowed:
                selected = name
                break
        self.ledger["no_admissible"] += int(selected is None)
        self.ledger["seconds"] += time.monotonic() - started
        return dict(selected=selected, checks=checks, mode=self.mode,
                    command=deepcopy(prepared["commands"][selected]) if selected else None)
