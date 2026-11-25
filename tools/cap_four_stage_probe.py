"""Bounded integration probes; no held-out panels or model fitting."""
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import time

from experiments.planar_tasks.authoring_v2.four_stage.compiler import Extractor
from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend, schema
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from experiments.planar_tasks.relational_search.contract import digest
from tools.cap_fault_curriculum_rooms import panel, SEEDS
from tools.cap_symbolic_general import ROOT, save, sha
from tools.cap_unknown_safety import audit_initial


OUT = ROOT / "outputs/semantic_rl/cap_four_stage_v1"
SOURCE = ROOT / "experiments/planar_tasks/authoring_v2/four_stage/pilot_policy.py"


def run(path, *, room=0, decisions=80, source_path=SOURCE):
    if path.exists():
        raise FileExistsError(path)
    source = source_path.read_text()
    policy = Extractor(source, schema())
    SEEDS["four_stage_integration"] = 261125401
    fixtures, records = panel("four_stage_integration", 4)
    key = list(fixtures)[room]
    f = fixtures[key]
    frontend = PublicFrontend()
    initial = audit_initial(f, frontend.contract)
    if not initial["initial_condition_valid"] or not initial["area_prior_valid"]:
        raise ValueError("initial contract not supported")
    folder = path.parent / (path.stem + "_sources")
    sources = [*sorted((ROOT / "experiments/planar_tasks/authoring_v2/four_stage").glob("*.py")),
               Path(__file__)]
    inventory = {}
    for p in sources:
        relative = str(p.relative_to(ROOT))
        dst = folder / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(p.read_bytes())
        inventory[relative] = sha(p)
    save(folder / "manifest.json", inventory)
    world = Environment(f, EventConfig(mode="duration"), fast_feedback=True, record_details=True)
    rows = []
    started = time.monotonic()
    reason = "decision_budget"
    for index in range(decisions):
        if world.env.termination:
            reason = world.env.termination
            break
        o = simulation.observation(world)
        prepared = frontend.prepare(o)
        z = frontend.register(prepared)
        original = policy.native_candidates(z)
        extracted = sorted(policy.candidates(z), key=lambda r: r["emission"])
        if original != [(r["option"], r["score"]) for r in extracted]:
            raise ValueError("native/extracted candidate mismatch")
        chosen = policy.select(z, lambda a: prepared["reports"][a])
        name = chosen["selected"]
        if name is None:
            reason = "no_certified_command"
            break
        frontend.record_command(name)
        measured = simulation.measured_step(world, prepared["commands"][name])
        rows.append(dict(index=index, tick=o["tick"], public=o, public_sha256=digest(o),
                         semantic=z, selected=name, command=prepared["commands"][name],
                         checks=chosen["checks"], audit=measured))
        if any(measured[k] for k in ("object_static", "robot_static", "robot_peer")):
            reason = "physical_contact"
            break
        if time.monotonic() - started > 180:
            reason = "probe_wall_budget"
            break
        if len(rows) >= 8 and all(r["selected"] == "brake" for r in rows[-8:]):
            reason = "repeated_brake"
            break
    summary = dict(room=key, termination=reason, success=bool(world.env.success),
                   ticks=world.env.total_periods, decisions=len(rows),
                   actions=dict(Counter(r["selected"] for r in rows)),
                   goal_gain=sum(r["audit"]["goal_gain"] for r in rows),
                   contact_commands=sum(any(r["audit"][k] for k in ("object_static", "robot_static", "robot_peer")) for r in rows),
                   wall_seconds=time.monotonic() - started, frontend=frontend.ledger)
    save(path, dict(scope="development integration only; prediction gate disabled",
                    summary=summary, fixture=f, fixture_record=records[key],
                    initial_audit=initial, motion_contract=asdict(frontend.contract),
                    source_sha256=policy.source_sha256, source_manifest=str(folder / "manifest.json"),
                    rows=rows, replay=world.replay()))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--room", type=int, default=0, choices=range(4))
    parser.add_argument("--decisions", type=int, default=80)
    args = parser.parse_args()
    if not 1 <= args.decisions <= 450 or not args.name.replace("_", "").isalnum():
        raise ValueError("bounded decisions and simple output name required")
    print(run(OUT / "integration" / (args.name + ".json.gz"), room=args.room, decisions=args.decisions), flush=True)


if __name__ == "__main__":
    main()
