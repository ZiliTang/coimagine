"""Replay public development prefixes and compare physical fork labels to checks."""
from pathlib import Path
import argparse

from tools.cap_symbolic_general import ROOT, read, save
from experiments.planar_tasks.authoring_v2.four_stage.carrier import build, verify
from experiments.planar_tasks.authoring_v2.four_stage.native_frontend import NativeFrontend
from experiments.planar_tasks.authoring_v2.four_stage.prediction import JointModel
from experiments.planar_tasks.authoring_v2.four_stage.process import ProcessChecker


BASE = ROOT / "outputs/semantic_rl/cap_four_stage_v1"


def diagnose(model_name, source_path, room="diag_four_stage_fit_001", curriculum="leader"):
    model = JointModel(read(BASE / "models" / model_name / "model.json.gz"))
    program = verify(build(Path(source_path).read_text(), lean=True))
    if model.host != program.host:
        raise ValueError("policy/model registered symbols differ")
    frontend = NativeFrontend(semantic_names=program.host["metrics"])
    checker = ProcessChecker(frontend)
    root = BASE / "data/warmup_v0/fit" / room / curriculum
    trajectory = read(root / "prefix.json.gz")["rows"]
    results = []
    for index, row in enumerate(trajectory):
        prepared = frontend.prepare(row["public"])
        if index in (0,4,12,24,48,96,160,240):
            action = "push_r0_1_4"
            physical = read(root / "branches" / f"p{index:04d}_{action}.json.gz")
            if physical["status"] == "complete":
                forecast = model.predict(prepared["public_context"], action)
                reports = [checker(prepared, action, b) for b in forecast["branches"]]
                results.append(dict(prefix=index, action=action, host=prepared["reports"][action],
                    predicted_process=reports, supported=forecast["supported"],
                    calibrated=forecast["calibrated"], support_rooms=forecast["support_rooms"],
                    motion=forecast["branches"][0]["motion"] if forecast["branches"] else None,
                    actual_motion=physical["process"],
                    root_gain=physical["measured"]["root"]["goal_gain"],
                    actual_static_contact=physical["measured"]["full_target_static"]))
        frontend.record_command(row["selected"])
    out = BASE / "shadow" / f"{model_name}_{room}_{curriculum}.json"
    if out.exists():
        raise FileExistsError(out)
    save(out, dict(model_id=model.sha256, room=room, curriculum=curriculum, rows=results,
                   scope="already-used development room diagnosis; not held-out evidence"))
    for r in results:
        print(dict(prefix=r["prefix"], supported=r["supported"], calibrated=r["calibrated"],
                   host_allowed=r["host"]["allowed"], process=r["predicted_process"][:1],
                   actual_gain=r["root_gain"]), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--room", default="diag_four_stage_fit_001")
    p.add_argument("--curriculum", default="leader")
    a=p.parse_args()
    diagnose(a.model, a.source, a.room, a.curriculum)
