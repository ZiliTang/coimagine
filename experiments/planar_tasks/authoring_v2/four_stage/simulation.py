"""Four-stage physical episodes: policy inputs and privileged audits separated."""
from collections import Counter
from copy import deepcopy
import time

from .carrier import verify
from .frontend import PublicFrontend
from .gateway import Gateway
from .physical import measure, state
from .process import ProcessChecker
from ..symbolic_general.simulation import observation
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from experiments.planar_tasks.relational_search.contract import digest


def episode(fixture, source, *, model=None, rollout=None, mode="author",
            max_ticks=1800, max_decisions=450, wall_seconds=600, heartbeat=None,
            frontend=None):
    program = verify(source)
    world = Environment(fixture, EventConfig(mode="duration"), fast_feedback=True, record_details=True)
    if frontend is None:
        if 'native-grid-dependencies-v1' in source:
            from .native_frontend import NativeFrontend
            frontend=NativeFrontend(semantic_names=program.host["metrics"])
        else:
            frontend=PublicFrontend()
    gateway = Gateway(program, model=model, rollout=rollout, process_check=ProcessChecker(frontend),
                      mode=mode, invariants={"legal:brake": True})
    started = time.monotonic()
    rows, stopping = [], []
    reason, error = "decision_budget", None
    initial = state(world)
    last_prepared = None
    for index in range(max_decisions):
        if world.env.termination:
            reason = world.env.termination
            break
        # Reserve the final physical stopping tail inside the common tick cap.
        if world.env.total_periods >= max_ticks-40:
            reason = "tick_budget_with_stop_reserve"
            break
        if time.monotonic()-started >= wall_seconds:
            reason = "episode_wall_budget"
            break
        prepared = frontend.prepare(observation(world))
        last_prepared = prepared
        z = frontend.register(prepared)
        # Validate the exact submitted function against the extracted emitter.
        native = program.native_candidates(z)
        extracted = sorted(program.candidates(z), key=lambda r: r["emission"])
        if native != [(r["option"], r["score"]) for r in extracted]:
            raise ValueError("submitted/extracted candidate mismatch")
        decision = gateway.decide(prepared)
        name = decision["selected"]
        if name is None:
            reason = "no_admissible_command"
            break
        try:
            audit = measure(world, decision["command"],
                            remaining=max_ticks-40-world.env.total_periods,
                            regions=prepared["regions"][name])
        except (ValueError, RuntimeError, KeyError) as exc:
            error = repr(exc)
            reason = "execution_error"
            break
        frontend.record_command(name)
        rows.append(dict(index=index, tick=prepared["observation"]["tick"],
                         public=prepared["observation"], public_sha256=digest(prepared["observation"]),
                         public_context=prepared["public_context"], semantic=z, selected=name,
                         command=decision["command"], decision=decision, audit=audit))
        if heartbeat and (index % 10 == 0):
            heartbeat(dict(index=index, ticks=world.env.total_periods, last=name,
                           elapsed=time.monotonic()-started))
        if any(audit[k] for k in ("target_static", "robot_static", "peer")):
            reason = "physical_contact"
            break
        if audit["envelope_violations"]:
            reason = "host_motion_assumption_refuted"
            break
        if audit["event"]["periods"] == 0:
            reason = "zero_tick_command"
            break
        if len(rows) >= 12 and all(r["selected"] == "brake" for r in rows[-12:]):
            reason = "repeated_brake"
            break
    stop_certified = False
    if not world.env.termination:
        prepared = frontend.prepare(observation(world))
        last_prepared = prepared
        # An emergency stop outside a certificate is explicitly counted as
        # guarantee failure, never labeled as an admitted productive candidate.
        stop_certified = prepared["reports"]["brake"]["allowed"]
        remaining = min(40, max_ticks-world.env.total_periods)
        while remaining > 0 and not world.env.termination:
            ticks = min(4, remaining)
            row = measure(world, dict(target_id=None, a0=44, a1=44, ticks=ticks),
                          regions=prepared["regions"]["brake"])
            stopping.append(row)
            actual = row["event"]["periods"]
            remaining -= actual
            if not actual:
                break
    final = state(world)
    audits = [r["audit"] for r in rows] + stopping
    rest = max([final["target_boundary_speed"], *final["robot_speeds"]]) < .005
    import math
    goal = observation(world)["goal"]["xy"]
    summary = dict(success=bool(world.env.success), termination=reason,
        environment_termination=world.env.termination, error=error,
        ticks=world.env.total_periods, decisions=len(rows),
        goal_gain=math.dist(initial["target_xy"], goal)-math.dist(final["target_xy"], goal),
        target_static_commands=sum(a["target_static"] for a in audits),
        robot_static_commands=sum(a["robot_static"] for a in audits),
        peer_contact_commands=sum(a["peer"] for a in audits),
        target_other_commands=sum(a["target_other"] for a in audits),
        host_envelope_violations=sum(len(a["envelope_violations"]) for a in audits),
        actions=dict(Counter(r["selected"] for r in rows)), stopped=rest,
        stop_certified=stop_certified, stop_ticks=sum(a["event"]["periods"] for a in stopping),
        terminal_stop_not_executed=bool(world.env.termination) and not stopping,
        task_cost_seconds=world.env.total_periods*.05 if world.env.success else 2*world.env.config.time_budget,
        wall_seconds=time.monotonic()-started, frontend=frontend.ledger, checking=gateway.ledger)
    return dict(summary=summary, source_sha256=program.source_sha256,
                rows=rows, stopping=stopping, replay=world.replay(),
                assurance="conditional host/empirical checks; task termination and physical rest reported separately")
