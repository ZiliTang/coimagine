"""Public-only policy execution with separate matched physical auditing."""
from collections import Counter
import time

from .carrier import verify
from .gateway import Gateway
from .native_frontend import NativeFrontend
from .process import ProcessChecker
from ..symbolic_general.simulation import observation
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory
from experiments.planar_tasks.relational_search.contract import digest


def stopping_certificate(rows, current_report, mode, *, symbols=None, observation=None):
    """A just-admitted command may already cover its prescribed braking tail.

    This is process-safety provenance, not a claim that residual speed is zero.
    It expires as soon as a different command is executed.
    """
    if current_report["allowed"]:
        return dict(certified=True, source="current_host_brake_envelope",remaining_ticks=40)
    if not rows or mode == "shadow":
        return dict(certified=False, source="no_current_or_carried_certificate")
    row = rows[-1]
    if row["event"]["periods"] != row["command"]["ticks"]:
        return dict(certified=False, source="interrupted_root_command")
    verdict = next(c["verdict"] for c in row["decision"]["checks"]
                   if c["candidate"]["option"] == row["selected"])
    if verdict.get("reason")=="carried_stop_plan" and verdict["allowed"]:
        previous=dict(verdict["carried_stop"])
        previous["remaining_ticks"]-=row["event"]["periods"]
        previous["certified"]=previous["remaining_ticks"]>0
        if previous["certified"] and not public_stop_prefix_covered(previous,observation):
            return dict(certified=False,source="observed_robot_left_carried_envelope",observed_contradiction=True)
        return previous
    forecast = verdict.get("forecast", {})
    if (verdict["allowed"] and forecast.get("motion_horizon") == "command_then_brake40" and
            verdict.get("process") and all(p["allowed"] for p in verdict["process"])):
        if symbols is not None:
            from .coverage import semantic_contains
            if not any(semantic_contains(b["updates"],symbols) for b in forecast["branches"]):
                return dict(certified=False,source="observed_endpoint_outside_forecast",observed_contradiction=True)
        plan=dict(certified=True, source="carried_empirical_command_then_brake40",
                    model_id=verdict.get("model_id"), boundary=row["index"],
                    remaining_ticks=40,
                    assurance="conditional on forecast coverage; stop speed separately audited")
        if "public" in row:
            plan["origin_robots"]=[[r["own"]["x"],r["own"]["y"]] for r in row["public"]["robots"]]
            plan["motion_modes"]=[b["motion"] for b in forecast["branches"]]
        if not public_stop_prefix_covered(plan,observation):
            return dict(certified=False,source="observed_robot_left_carried_envelope",observed_contradiction=True)
        return plan
    if verdict["allowed"] and verdict.get("geometry", {}).get("allowed"):
        return dict(certified=True, source="carried_host_command_and_stop_envelope",
                    boundary=row["index"],remaining_ticks=40, assurance="conditional host motion assumption")
    return dict(certified=False, source="no_current_or_carried_certificate")


def public_stop_prefix_covered(plan, observation):
    if observation is None or "origin_robots" not in plan:
        return True
    for motion in plan["motion_modes"]:
        covered=True
        for i,r in enumerate(observation["robots"]):
            dx=r["own"]["x"]-plan["origin_robots"][i][0]
            dy=r["own"]["y"]-plan["origin_robots"][i][1]
            b=motion[f"r{i}"]
            covered=covered and b[0]-1e-7<=dx<=b[1]+1e-7 and b[2]-1e-7<=dy<=b[3]+1e-7
        if covered:
            return True
    return False


def run(fixture,source,*,model=None,rollout_factory=None,mode="author",
        max_ticks=1800,max_decisions=450,wall_seconds=600,heartbeat=None):
    startup=time.perf_counter()
    program=verify(source)
    frontend=NativeFrontend(semantic_names=program.host["metrics"])
    rollout=rollout_factory(frontend) if rollout_factory is not None else None
    gateway=Gateway(program,model=model,rollout=rollout,mode=mode,
                    process_check=ProcessChecker(frontend),invariants={"legal:brake":True})
    world=Environment(fixture,EventConfig(mode="duration"),fast_feedback=True,record_details=False,history_factory=MemoryHistory)
    startup=time.perf_counter()-startup
    begun=time.perf_counter();timing=dict(frontend=0.,decision=0.,physics=0.)
    rows=[];stop=[];withheld=[];reason="decision_budget"
    for index in range(max_decisions):
        if world.env.termination:
            reason=world.env.termination;break
        if world.env.total_periods>=max_ticks-40:
            reason="tick_budget_with_stop_reserve";break
        if time.perf_counter()-begun>=wall_seconds:
            reason="wall_budget";break
        t=time.perf_counter();prepared=frontend.prepare_history(world.history);timing["frontend"]+=time.perf_counter()-t
        prepared["carried_stop_plan"]=stopping_certificate(rows,dict(allowed=False),mode,
            symbols=prepared["symbols"],observation=prepared["observation"])
        t=time.perf_counter();decision=gateway.decide(prepared);timing["decision"]+=time.perf_counter()-t
        name=decision["selected"]
        if name is None:
            withheld.append(dict(index=index,public=prepared["observation"],
                public_context=prepared["public_context"],semantic=prepared["symbols"],decision=decision))
            reason="no_admissible_command";break
        t=time.perf_counter();event=world.step(decision["command"],remaining_ticks=max_ticks-40-world.env.total_periods);timing["physics"]+=time.perf_counter()-t
        rows.append(dict(index=index,public=prepared["observation"],public_context=prepared["public_context"],
                         semantic=prepared["symbols"],selected=name,command=decision["command"],decision=decision,event=event))
        frontend.record_command(name)
        if heartbeat and index%20==0:
            heartbeat(dict(decisions=len(rows),ticks=world.env.total_periods,elapsed=time.perf_counter()-begun))
        if event["periods"]==0:
            reason="zero_tick_command";break
        # Runtime reactions use feedback, never an evaluator's hidden contacts.
        if any(r["contacting_static"] or r["contacting_teammate"] for r in observation(world)["robots"]):
            reason="public_contact_feedback";break
        if len(rows)>=12 and all(r["selected"]=="brake" for r in rows[-12:]):
            reason="repeated_brake";break
    stop_certified=False
    stop_provenance=dict(certified=False,source="terminal_stop_not_executed")
    if not world.env.termination:
        prepared=frontend.prepare_history(world.history)
        stop_provenance=stopping_certificate(rows,prepared["reports"]["brake"],mode,
            symbols=prepared["symbols"],observation=prepared["observation"])
        stop_certified=stop_provenance["certified"]
        left=min(stop_provenance.get("remaining_ticks",40),40,max_ticks-world.env.total_periods)
        while left>0 and not world.env.termination:
            command=dict(target_id=None,a0=44,a1=44,ticks=min(4,left))
            event=world.step(command);stop.append(dict(command=command,event=event));left-=event["periods"]
            if not event["periods"]:break
    summary=dict(success=bool(world.env.success),termination=reason,environment_termination=world.env.termination,
        ticks=world.env.total_periods,decisions=len(rows),stop_certified=stop_certified,
        stop_provenance=stop_provenance,
        terminal_stop_not_executed=bool(world.env.termination) and not stop,
        startup_seconds=startup,wall_seconds=time.perf_counter()-begun,timing=timing,
        task_cost_seconds=world.env.total_periods*.05 if world.env.success else 2*world.env.config.time_budget,
        actions=dict(Counter(r["selected"] for r in rows)),checking=gateway.ledger,
        geometry=frontend.evidence.ledger,physical_audit="pending separate replay")
    return dict(summary=summary,fixture=fixture,source_sha256=program.source_sha256,source=source,
                rows=rows,withheld=withheld,stopping=stop)


def audit(trace):
    """Replay exact recorded commands; privileged quantities only in this path."""
    from .physical import measure,state,process_bounds
    from .coverage import realized_coverage
    program=verify(trace["source"])
    frontend=NativeFrontend(semantic_names=program.host["metrics"])
    world=Environment(trace["fixture"],EventConfig(mode="duration"),fast_feedback=True,record_details=False)
    audits=[];prediction_coverage=[];carried_coverage=[];pending=None;begun=time.perf_counter();parity=True
    def check_tail(root_index,through):
        root_row=trace["rows"][root_index]
        root_verdict=next(c["verdict"] for c in root_row["decision"]["checks"]
                          if c["candidate"]["option"]==root_row["selected"])
        forecast=root_verdict.get("forecast")
        if forecast is None:
            return
        actual=process_bounds(audits[root_index]["start"],
            [sample for a in audits[root_index:] for sample in a["samples"]])
        coverage=realized_coverage(forecast,actual)
        carried_coverage.append(dict(root_index=root_index,through=through,
            motion_covered=coverage["root_motion_covered"],
            brake_ticks_measured=sum(a["event"]["periods"] for a in audits[root_index+1:])))
    for row in trace["rows"]:
        o=observation(world)
        if digest(o)!=digest(row["public"]):
            raise ValueError("physical audit public-prefix divergence")
        prepared=frontend.prepare(o)
        if pending is not None:
            prediction_coverage.append(dict(index=pending[0],**realized_coverage(
                pending[1],pending[2],prepared["symbols"])))
            pending=None
        measured=measure(world,row["command"],regions=prepared["regions"][row["selected"]])
        verdict=next(c["verdict"] for c in row["decision"]["checks"]
                     if c["candidate"]["option"]==row["selected"])
        if verdict.get("forecast"):
            pending=(row["index"],verdict["forecast"],measured["process"])
        audits.append(measured);frontend.record_command(row["selected"])
        if verdict.get("reason")=="carried_stop_plan":
            check_tail(verdict["carried_stop"]["boundary"],row["index"])
    prepared=frontend.prepare(observation(world)) if not world.env.termination else None
    if pending is not None:
        prediction_coverage.append(dict(index=pending[0],**realized_coverage(
            pending[1],pending[2],prepared["symbols"] if prepared else None)))
    for row in trace["stopping"]:
        audits.append(measure(world,row["command"],regions=prepared["regions"]["brake"]))
    if trace["rows"] and trace["stopping"]:
        last=trace["rows"][-1]
        verdict=next(c["verdict"] for c in last["decision"]["checks"]
                     if c["candidate"]["option"]==last["selected"])
        if verdict.get("reason")=="carried_stop_plan":
            check_tail(verdict["carried_stop"]["boundary"],"final_stop")
        elif verdict.get("forecast"):
            check_tail(last["index"],"final_stop")
    final=state(world)
    summary=dict(public_prefixes_equal=parity,ticks=world.env.total_periods,
        target_static_commands=sum(r["target_static"] for r in audits),
        robot_static_commands=sum(r["robot_static"] for r in audits),
        robot_other_commands=sum(r["robot_other"] for r in audits),
        peer_contact_commands=sum(r["peer"] for r in audits),target_other_commands=sum(r["target_other"] for r in audits),
        host_envelope_violations=sum(len(r["envelope_violations"]) for r in audits),
        empirical_root_motion_misses=sum(not r["root_motion_covered"] for r in prediction_coverage),
        empirical_joint_endpoint_misses=sum(r["joint_endpoint_covered"] is False for r in prediction_coverage),
        empirical_admissions_audited=len(prediction_coverage),
        empirical_stopping_plan_misses=len({r["root_index"] for r in carried_coverage if not r["motion_covered"]}),
        goal_gain=sum(r["goal_gain"] for r in audits),
        stopped=max(final["target_boundary_speed"],*final["robot_speeds"])<.005,
        wall_seconds=time.perf_counter()-begun)
    return dict(summary=summary,rows=audits,prediction_coverage=prediction_coverage,
                carried_coverage=carried_coverage,final=final,
                scope="independent privileged labels; excluded from policy inputs and online K")
