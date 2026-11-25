"""Prepare immutable four-stage author packages and supervised physical batches."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time
import traceback

from experiments.planar_tasks.authoring_v2.four_stage import campaign
from experiments.planar_tasks.authoring_v2.four_stage.carrier import build, checked_boundary
from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend
from experiments.planar_tasks.authoring_v2.four_stage.execution import run, audit
from experiments.planar_tasks.authoring_v2.four_stage.native_frontend import NativeFrontend
from experiments.planar_tasks.authoring_v2.four_stage.native_geometry import build as build_native,identity
from experiments.planar_tasks.authoring_v2.four_stage.carrier import verify
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import observation
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig
from tools.cap_fault_curriculum_rooms import panel, SEEDS
from tools.cap_symbolic_general import ROOT, save, read, sha


BASE = ROOT / "outputs/semantic_rl/cap_four_stage_v1"


def prepare(name, source_path):
    build_native()
    rooms = BASE / "rooms.json"
    if not rooms.exists():
        splits = dict(author=(261125411, 8), fit=(261125421, 48),
                      calibration=(261125431, 16), confirmation=(261125441, 32))
        groups, records = {}, {}
        for split, (seed, count) in splits.items():
            SEEDS["four_stage_" + split] = seed
            groups[split], record = panel("four_stage_" + split, count)
            records.update(record)
        if len({r["content_sha256"] for r in records.values()}) != len(records):
            raise ValueError("room content overlap")
        save(rooms, dict(splits=splits, rooms=groups, records=records,
            outcome_filtering=False, confirmation_exposure="fixture generation only; no outcomes"))
    groups = read(rooms)["rooms"]
    protocol = dict(author_round=name, development_rooms=list(groups["author"]),
        max_ticks=1800, max_decisions=450, wall_seconds=600, workers=4,
        stop_reserve_ticks=40, first_round_mode="author", second_round_mode="enforce",
        model_versions=[0,1,2], fit_rooms_per_increment=16, alpha=.10,
        delta_C=.05, delta_F=.05, online_queries_per_boundary=2,
        confirmation_rooms=32, calibration_rooms=16,
        paired_candidate_rule="Fixed public prefixes; all emitted PUSH candidates plus highest ranked MOVE and ORBIT and BRAKE, deduplicated; no hidden-state pruning.",
        C="admitted and full-process-safe and declared physical successor completed / all tested candidates",
        F="admitted but declared physical successor not completed / admitted; undefined if none",
        K="executed online simulation branches per 100 decision boundaries",
        task_cost="elapsed physical seconds on success; 180 seconds on failure",
        identity="Current conversational CAP author; prior history visible; no isolated author or separately metered model call claimed.",
        geometry=dict(identity(),resolution=.01,scan_stride_ticks=16,
                      registration="extract policy dependencies; other candidate geometry on demand",
                      occupied="cells with static hits; count is conservative and may duplicate across body envelopes"),
        fixtures_sha256=sha(rooms), rules="No outcome-based room filtering; confirmation excluded from fitting and author revision.")
    package = BASE / "authors" / name
    source = build(source_path.read_text(),lean=True)
    program=verify(source)
    c = campaign.create(package, next(iter(groups["author"].values())), source, protocol)
    bounds = BASE / "checks" / name
    for i, fixture in enumerate(list(groups["author"].values())[:3]):
        world = Environment(fixture, EventConfig(mode="duration"), fast_feedback=True)
        prepared = NativeFrontend(semantic_names=program.host["metrics"]).prepare(observation(world))
        for action in prepared["commands"]:
            prepared["reports"][action]
        permissions = {a:r["allowed"] for a,r in prepared["reports"].items()}
        save(bounds / f"boundary_{i}.json", checked_boundary(source, prepared, permissions))
    print(dict(package=str(c.path), checks=str(bounds)), flush=True)


def worker(job):
    path = Path(job["out"])
    save(path / "started.json", dict(at=time.time(), reservation_ticks=1800))
    try:
        result = run(job["fixture"], job["source"], mode="author",
            heartbeat=lambda v: save(path / "heartbeat.json", v))
        physical=audit(result)
        save(path / "physical_audit.json.gz",physical)
        result["summary"]["physical_audit"]=physical["summary"]
        result.update(fixture=job["fixture"], fixture_record=job["record"])
        result["summary"]["room"] = job["room"]
        save(path / "result.json.gz", result)
        save(path / "summary.json", result["summary"])
        return result["summary"]
    except Exception:
        error = dict(room=job["room"], traceback=traceback.format_exc(),
                     reservation_retained=True, charged_ticks=1800)
        save(path / "error.json", error)
        return error


def batch(name, count):
    package = BASE / "authors" / name
    receipt = campaign.verify_frozen(package)
    source = (package / "policy.py").read_text()
    panels = read(BASE / "rooms.json")
    out = BASE / "author_runs" / name
    chosen = list(panels["rooms"]["author"].items())[:count]
    jobs = []
    for room, fixture in chosen:
        if (out / room / "summary.json").exists():
            continue
        if (out / room / "started.json").exists():
            raise ValueError("interrupted run needs explicit new run version: " + room)
        jobs.append(dict(out=str(out / room), room=room, fixture=fixture,
                         record=panels["records"][room], source=source))
    save(out / f"reservation_{time.time_ns()}.json", dict(rooms=[j["room"] for j in jobs],
         ticks=1800*len(jobs), freeze=receipt, source_sha256=sha(package / "policy.py")))
    with ProcessPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(worker, j) for j in jobs]):
            row = future.result()
            print(row, flush=True)
    rows = [read(out / room / "summary.json") for room, _ in chosen if (out / room / "summary.json").exists()]
    save(out / "batch.json", dict(rows=rows, requested=len(chosen), completed=len(rows),
         ticks=sum(r["ticks"] for r in rows), success=sum(r["success"] for r in rows)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prepare", "freeze", "batch"])
    p.add_argument("--name", required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--count", type=int, default=4, choices=range(1,9))
    a = p.parse_args()
    if a.command == "prepare":
        prepare(a.name, a.source)
    elif a.command == "freeze":
        print(campaign.freeze(BASE / "authors" / a.name, BASE / "checks" / a.name / "checked"))
    else:
        batch(a.name, a.count)
