"""Same-prefix physical effect probes with full braking and public target labels."""
import argparse
from copy import deepcopy
from pathlib import Path
import time

from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend, OPTIONS
from experiments.planar_tasks.authoring_v2.four_stage.physical import command_and_stop
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import observation
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from experiments.planar_tasks.relational_search.contract import digest
from tools.cap_symbolic_general import read, save, sha, ROOT


def replay(fixture, prefix):
    world = Environment(fixture, EventConfig(mode="duration"), fast_feedback=True, record_details=False)
    for command in prefix:
        if world.env.termination:
            raise ValueError("terminal prefix")
        world.step(command)
    return world


def collect(trace_path, out, boundary, actions):
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= len(actions) <= 8 or any(a not in OPTIONS for a in actions):
        raise ValueError("one to eight registered actions required")
    trace = read(trace_path)
    if not 0 <= boundary < len(trace["rows"]):
        raise ValueError("boundary absent")
    frontend = PublicFrontend()
    for row in trace["rows"][:boundary]:
        frontend.prepare(row["public"])
        frontend.record_command(row["selected"])
    root = trace["rows"][boundary]
    prepared = frontend.prepare(root["public"])
    prefix = [r["command"] for r in trace["rows"][:boundary]]
    reserved = sum(c["ticks"] for c in prefix) * len(actions) + sum(OPTIONS[a]["ticks"]+40 for a in actions)
    inventory = {str(p.relative_to(ROOT)): sha(p) for p in
                 (ROOT/"experiments/planar_tasks/authoring_v2/four_stage").glob("*.py")}
    source_folder = out.parent / (out.stem + "_sources")
    for relative in inventory:
        dest = source_folder / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT/relative).read_bytes())
    save(source_folder/"manifest.json", inventory)
    results = []
    started = time.monotonic()
    for action in actions:
        row = dict(action=action, boundary=boundary)
        try:
            world = replay(trace["fixture"], prefix)
            root_hash = digest(observation(world))
            if root_hash != root["public_sha256"]:
                raise ValueError("same-prefix public digest mismatch")
            if not prepared["symbols"]["legal:" + action]:
                row.update(status="not_startable", charged_ticks=world.env.total_periods)
            else:
                copied = deepcopy(frontend)
                copied.record_command(action)
                measured = command_and_stop(world, prepared["commands"][action],
                                            regions=prepared["regions"][action])
                after = copied.prepare(measured["endpoint"])
                row.update(status="complete", public_sha256=root_hash,
                    public_context=prepared["public_context"],
                    semantic=prepared["symbols"], next_semantic=after["symbols"],
                    process=measured["process"], measured=measured,
                    charged_ticks=world.env.total_periods)
        except Exception as exc:
            row.update(status="error", error=repr(exc),
                       charged_ticks=sum(c["ticks"] for c in prefix)+OPTIONS[action]["ticks"]+40,
                       charge_is_reservation=True)
        results.append(row)
        save(out.with_name(out.name+".partial"), dict(rows=results, status="running"))
        print(action, row["status"], row.get("process"), flush=True)
    result = dict(scope="development physical probes, never policy inputs",
        trace=str(trace_path), trace_sha256=sha(trace_path), room=trace["summary"]["room"],
        boundary=boundary, reserved_ticks=reserved,
        charged_ticks=sum(r["charged_ticks"] for r in results),
        source_manifest=str(source_folder/"manifest.json"), rows=results,
        seconds=time.monotonic()-started)
    save(out, result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--boundary", type=int, required=True)
    p.add_argument("--actions", nargs="+", required=True)
    a = p.parse_args()
    collect(a.trace, a.out, a.boundary, a.actions)
