"""Finite public-compatible nominal rollouts, never execution-state forks.

Nominal samples alone are diagnostic. An independently calibrated error adapter
must cover cold controller state, latent velocity and omitted geometry before
these samples may support an empirical admission. Unknown is still checked by
the real public evidence map, never the empty parts of a synthetic scene.
"""
from copy import deepcopy
import math
import time
from types import SimpleNamespace

from ..symbolic_general.online import SyntheticWorld
from ..symbolic_unknown.geometry import rectangle_pose
from experiments.planar_tasks.relational_search.contract import digest, public_observation
from experiments.planar_tasks.relational_search.estimator import estimate
from .physical import command_and_stop


def compatibility(real, synthetic, tolerance=.03):
    mismatches, free_violations, support = [], 0, 0
    for r, s in zip(real["robots"], synthetic["robots"]):
        a, b = r["scan"], s["scan"]
        for valid, d, cat, entity, sv, sd, sc, se in zip(
                a["valid"], a["distances"], a["category"], a["entity_ids"],
                b["valid"], b["distances"], b["category"], b["entity_ids"]):
            if sv and sd < d-tolerance:
                free_violations += 1
            if valid:
                same = sv and cat == sc and (cat != 1 or entity == se)
                mismatches.append(abs(d-sd) if same else 100.)
                support += int(entity == real["goal"]["target_id"])
    return dict(compatible=bool(support >= 3 and free_violations == 0 and
                               max(mismatches, default=100.) <= tolerance),
                target_rays=support, free_violations=free_violations,
                maximum_hit_error=max(mismatches, default=None), tolerance_m=tolerance)


def hypotheses(context):
    o = public_observation(context["observation"])
    pose = rectangle_pose(o)
    if pose is None:
        raise ValueError("unsupported public target pose")
    observations=[public_observation(item) for item in context.get("history",[o])]
    ticks=[item["tick"] for item in observations]
    if any(a>b for a,b in zip(ticks,ticks[1:])) or any(t>o["tick"] for t in ticks):
        raise ValueError("noncausal public geometry history")
    if not observations or observations[-1]["tick"]!=o["tick"]:
        observations.append(o)
    # The estimator only fuses observed geometry. This is deliberately not a
    # command-replay History: that API requires real intervening actions, which
    # cannot be invented from consecutive sensor frames.
    history=SimpleNamespace(events=[dict(observation=item) for item in observations])
    base = estimate(history)
    target = next(e for e in base["entities"] if e["id"] == o["goal"]["target_id"])
    target.update(x=pose.centre[0], y=pose.centre[1], width=pose.width,
                  height=pose.length, angle=math.atan2(pose.normal[1], pose.normal[0]))
    # These are explicit incomplete shape hypotheses. Small hit boxes do NOT
    # satisfy the environment's minimum-component-area prior or complete X(h).
    arena = base["arena"]
    base["obstacles"] = [[p[0], p[1], .003, .003, 0.] for p in context["memory"].get("static", [])
                         if .01 < p[0] < arena[0]-.01 and .01 < p[1] < arena[1]-.01]
    for delta in (0., -.02, .02):
        proposal = deepcopy(base)
        obj = next(e for e in proposal["entities"] if e["id"] == o["goal"]["target_id"])
        obj["x"] += delta*pose.tangent[0]
        obj["y"] += delta*pose.tangent[1]
        yield proposal


def offset_observation(observation, offset):
    o = deepcopy(observation)
    o["tick"] += offset
    for e in o["entities"]:
        e["last_seen_tick"] += offset
    for r in o["robots"]:
        r["tick"] = o["tick"]
        r["time"] = r["elapsed_seconds"] = .05*o["tick"]
        r["scan"]["tick"] = o["tick"]
        r["scan"]["sample_time"] = .05*o["tick"]
    return o


class OnlineRollout:
    def __init__(self, frontend, *, error_adapter=None, max_branches=60,
                 max_ticks=3000, max_seconds=15., per_query=3):
        self.frontend = frontend  # only public evidence; no Environment handle
        self.error_adapter = error_adapter
        self.limits = dict(branches=max_branches, ticks=max_ticks, seconds=max_seconds,
                           per_query=per_query)
        self.ledger = dict(branches=0, ticks=0, seconds=0., constructions=0, incompatible=0)

    def run(self, context, action, command):
        started = time.perf_counter()
        if digest(context) != digest(self.frontend.last_packet["public_context"]):
            raise ValueError("online root is not the current public boundary")
        before = dict(self.ledger)
        branches, details, errors = [], [], []
        incomplete = False
        quote = command["ticks"]+40
        try:
            for index, proposal in enumerate(hypotheses(context)):
                if index >= self.limits["per_query"]:
                    break
                if (self.ledger["branches"] >= self.limits["branches"] or
                        self.ledger["ticks"]+quote > self.limits["ticks"] or
                        self.ledger["seconds"]+time.perf_counter()-started >= self.limits["seconds"]):
                    incomplete = True
                    break
                world = SyntheticWorld(proposal)
                self.ledger["constructions"] += 1
                check = compatibility(context["observation"], world.history.events[-1]["observation"])
                detail = dict(hypothesis_sha256=digest(proposal), compatibility=check)
                details.append(detail)
                if not check["compatible"]:
                    self.ledger["incompatible"] += 1
                    continue
                self.ledger["branches"] += 1
                self.ledger["ticks"] += quote  # retain the reservation if execution fails
                measured = command_and_stop(world, command, brake_chunk=40)
                actual = measured["root"]["event"]["periods"]+measured["stop_ticks"]
                self.ledger["ticks"] -= quote-actual
                cloned = deepcopy(self.frontend)
                cloned.record_command(action)
                end = offset_observation(measured["endpoint"], context["observation"]["tick"])
                future = cloned.prepare(end)["symbols"]
                branches.append(dict(updates={k: v if type(v) is bool else [v, v] for k, v in future.items()},
                                     motion=measured["process"], label="uncalibrated_public_nominal",
                                     hypothesis_sha256=detail["hypothesis_sha256"]))
                detail.update(status="complete", ticks=actual, stopped=measured["stopped"],
                              synthetic_contact=any(measured[k] for k in
                                  ("full_target_static", "full_robot_static", "full_peer")))
        except (ValueError, KeyError, RuntimeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        elapsed = time.perf_counter()-started
        self.ledger["seconds"] += elapsed
        result = dict(branches=branches, details=details, errors=errors, incomplete=incomplete,
                      admissible_evidence=False,
                      cost={k: self.ledger[k]-before[k] for k in ("branches", "ticks", "seconds")},
                      assurance="finite nominal hypotheses; missing scene and cold-state error uncalibrated")
        if self.error_adapter is not None and branches and not errors and not incomplete:
            result.update(self.error_adapter.calibrate(context, action, branches))
        return result
