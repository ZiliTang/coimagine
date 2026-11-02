"""D007 lateral-transfer calibration runner.

Headless only: no training, no default change, no testbed server restart.
Runs the frozen D007 matrix and stores raw ticks + paired CSV + summary.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / 'src'):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coimagine.continuous2d.scenes import Pose
from experiments.option_rl_v16.learnability import load_panel
from experiments.option_rl_v17 import contact_timing_audit as A
from experiments.option_rl_v19.system import OptionSystemEnv
from experiments.option_rl_v13.push_controller import ContinuousPush
from experiments.option_rl_v14.render_episodes import instance_from_dict

PANEL = ROOT / "outputs/option_rl_v14/learnability_002/panel.json"
CFG = ROOT / "docs/agent_exchange/tasks/D007_config.json"
OUT = ROOT / "outputs/agent_exchange/D007_20261030_001"
CONTROL_DT = 0.05


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cfg():
    return json.loads(CFG.read_text())


def scenes():
    train, _ = load_panel(PANEL)
    return {i.name: i for i in train}


def preset_instance(preset, scene_map):
    base = scene_map[preset["scene"]]
    goal = (preset.get("scene_meta") or {}).get("goal") or {}
    return replace(
        base,
        object_pose=Pose(preset["object"]["x"], preset["object"]["y"],
                         preset["object"].get("angle", 0.0)),
        robot_poses=tuple(Pose(r["x"], r["y"], r.get("angle", 0.0))
                          for r in preset["robots"]),
        goal_centre=(float(goal.get("x", base.goal_centre[0])),
                     float(goal.get("y", base.goal_centre[1]))),
        goal_radius=float(goal.get("radius", base.goal_radius)),
    )


def mirror_instance(inst, width=5.0):
    def mp(p):
        return Pose(width - p.x, p.y, wrap(math.pi - p.angle))
    obstacles = []
    for b in inst.obstacles:
        nb = replace(b, x=width - b.x, angle=wrap(math.pi - b.angle))
        obstacles.append(nb)
    return replace(
        inst, name=f"{inst.name}_mirror", obstacles=tuple(obstacles),
        object_pose=mp(inst.object_pose),
        robot_poses=tuple(mp(p) for p in inst.robot_poses),
        goal_centre=(width - inst.goal_centre[0], inst.goal_centre[1]),
    )


def make_env(inst, overrides, profile):
    env = OptionSystemEnv(inst, overrides=overrides or {})
    base_factory = env.factory

    def factory(action):
        skill = base_factory(action)
        if (isinstance(skill, ContinuousPush) and action.kind == "push"
                and getattr(action, "effect_kind", None) == "translate"):
            skill.config = replace(skill.config, **profile)
        return skill

    for slot in env.slots:
        slot.factory = factory
    return env


def _score(env):
    try:
        cond = env.success_conditions()
    except Exception:
        cond = {}
    return dict(success=bool(getattr(env, "success", False)),
                termination=getattr(env, "termination", None),
                periods=int(getattr(env, "total_periods", 0)),
                reward=float(getattr(env, "reward_total", 0.0)),
                conditions=cond)


def run_ticks(inst, overrides, profile, sequence=None, commands=None,
              diagnostic=None, record=True):
    """Run a fresh env.  diagnostic=(side, offset, t0, t1) or None."""
    env = make_env(inst, overrides, profile)
    rec = A.Recorder("d007", "d007") if record else None
    instrum = A.Instrumentation(env, rec).install() if record else None
    trace = []
    try:
        if commands is not None:
            ticks = [(int(a0), int(a1)) for a0, a1 in commands]
        else:
            ticks = []
            for seg in sequence:
                ticks.extend([(int(seg["a0"]), int(seg["a1"]))] *
                             int(seg.get("decisions", 1)))
        for a0, a1 in ticks:
            if env.termination is not None:
                break
            t = env.total_periods * CONTROL_DT
            for side in (0, 1):
                skill = env.slots[side].skill
                if hasattr(skill, "diagnostic_tangent_offset"):
                    off = 0.0
                    if diagnostic and side == diagnostic[0]:
                        s0, s1, off_arg = diagnostic[1], diagnostic[2], diagnostic[3]
                        if s0 <= t < s1:
                            off = off_arg
                    skill.diagnostic_tangent_offset = float(off)
            env.joint_step(a0, a1, record=record)
            if not record:
                o = env.object_state(); rs = env.robot_states()
                trace.append(dict(
                    t=float(t), object=[o.x, o.y, o.angle, o.vx, o.vy, o.omega],
                    robots=[[r.x, r.y, r.angle, r.vx, r.vy, r.omega] for r in rs],
                    forces=getattr(env, "gate_post", None),
                ))
    finally:
        if instrum is not None:
            instrum.restore()
    out = _score(env)
    if record:
        ticks_data = rec.finish(env, out["success"], out["termination"], out["reward"])
        out["ticks"] = ticks_data["ticks"]
        out["raw_ticks"] = len(ticks_data["ticks"])
    else:
        out["trace"] = trace
    return out


def load_preset(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-runs", type=int, default=36)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    t0 = time.monotonic()
    cfg = load_cfg(); scene_map = scenes()
    prof = {k: v["push_overrides"] for k, v in cfg["profiles"].items()}

    # baseline presets
    cases = {}
    for c in cfg["cases"]:
        p = load_preset(ROOT / c["preset"])
        cases[c["id"]] = dict(preset=p, inst=preset_instance(p, scene_map),
                              sequence=c["sequence"])
    cases["adjacent_mirror"]["inst"] = mirror_instance(cases["adjacent"]["inst"])

    rows = []
    raw = {}
    count = 0

    def save_run(tag, case_id, profile_key, result, meta):
        nonlocal count
        count += 1
        rid = f"{count:03d}_{tag}"
        data = dict(id=rid, tag=tag, case=case_id, profile=profile_key,
                    created=time.time(), meta=meta, result=result)
        raw[rid] = data
        # remove ticks from summary rows but keep raw file
        rows.append(dict(rid=rid, tag=tag, case=case_id, profile=profile_key,
                         success=int(result["success"]),
                         termination=str(result.get("termination")),
                         periods=result.get("periods"),
                         reward=round(result.get("reward", 0.0), 6)))
        return rid

    # 1) baseline 12
    for case_id in ("adjacent", "adjacent_mirror", "same_face", "same_face_closer"):
        c = cases[case_id]
        for pk in ("A", "B", "C"):
            r = run_ticks(c["inst"], c["preset"].get("overrides") or {}, prof[pk],
                          sequence=c["sequence"], record=True)
            save_run("baseline", case_id, pk, r, dict(sequence=c["sequence"]))
            print("baseline", case_id, pk, "periods", r["periods"], "success", r["success"], flush=True)
            if count >= args.max_runs:
                break
        if count >= args.max_runs:
            break

    # 2) probes 16 (only if budget remains)
    probes = []
    if count < args.max_runs:
        for case_id in ("adjacent", "adjacent_mirror"):
            c = cases[case_id]
            for pk in ("B", "C"):
                for side in (0, 1):
                    for off in (-0.04, 0.04):
                        r = run_ticks(c["inst"], c["preset"].get("overrides") or {}, prof[pk],
                                      sequence=c["sequence"], record=True,
                                      diagnostic=(side, 2.0, 3.0, off))
                        rid = save_run("probe", case_id, pk, r,
                                       dict(side=side, offset=off, t0=2.0, t1=3.0,
                                            sequence=c["sequence"]))
                        probes.append(dict(rid=rid, case=case_id, profile=pk, side=side,
                                           offset=off))
                        print("probe", case_id, pk, side, off, "periods", r["periods"], flush=True)
                        if count >= args.max_runs:
                            break
                    if count >= args.max_runs:
                        break
                if count >= args.max_runs:
                    break
            if count >= args.max_runs:
                break

    # 3) witnesses 6
    witnesses = []
    if count < args.max_runs:
        for wid in ("W_N", "W_E"):
            log = json.loads((ROOT / ("outputs/option_rl_v14/learnability_002/logs/"
                                      + ("dual_2.json" if wid == "W_N" else "dual_0.json"))).read_text())
            w_inst = instance_from_dict(log["instance"])
            cmds = log["commands"]
            for pk in ("A", "B", "C"):
                r = run_ticks(w_inst, {}, prof[pk], commands=cmds, record=True)
                rid = save_run("witness", wid, pk, r, dict(commands=cmds, instance=log["instance"]))
                witnesses.append(dict(rid=rid, case=wid, profile=pk))
                print("witness", wid, pk, "periods", r["periods"], "success", r["success"], flush=True)
                if count >= args.max_runs:
                    break
            if count >= args.max_runs:
                break

    # 4) noninterference 2 (no instrumentation)
    nonint = []
    if count < args.max_runs:
        for pk in ("A", "C"):
            c = cases["adjacent"]
            r = run_ticks(c["inst"], c["preset"].get("overrides") or {}, prof[pk],
                          sequence=c["sequence"], record=False)
            rid = save_run("nonint", "adjacent", pk, r, dict(sequence=c["sequence"], logging=False))
            nonint.append(dict(rid=rid, profile=pk))
            print("nonint", pk, "periods", r["periods"], flush=True)
            if count >= args.max_runs:
                break

    # save raw + summary
    (out / "runs").mkdir()
    for rid, data in raw.items():
        (out / "runs" / f"{rid}.json").write_text(json.dumps(data, indent=1))
    with (out / "matrix.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())) if rows else None
        if w:
            w.writeheader(); w.writerows(rows)
    summary = dict(
        status="complete", runs=count, elapsed_seconds=time.monotonic() - t0,
        rows=rows, probes=probes, witnesses=witnesses, nonint=nonint,
        cases={k: dict(preset=c["preset"].get("name"), sequence=c["sequence"])
               for k, c in cases.items()},
        profiles=cfg["profiles"], source_hashes=cfg["source_sha256"],
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "manifest.json").write_text(json.dumps(dict(
        task="D007", started=time.time() - summary["elapsed_seconds"],
        finished=time.time(), elapsed_seconds=summary["elapsed_seconds"],
        head=os.popen("git rev-parse HEAD").read().strip(),
        dirty=len(os.popen("git status --short").read().splitlines()),
        runs=count, budget=cfg["budget"], source_hashes=cfg["source_sha256"],
    ), indent=1))
    print("DONE", count, "runs", summary["elapsed_seconds"], "s", out.resolve().relative_to(ROOT))


if __name__ == "__main__":
    main()
