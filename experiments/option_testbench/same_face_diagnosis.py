"""Bounded diagnostic ablations, with no production controller/mask changes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from experiments.option_testbench import effect_push_eval as bench
from coimagine.random_start import spec
from coimagine.random_start.effect_push import VERSION


def panel():
    original = next(c for c in bench.joint_cases() if c["name"] == "same_face_N_NW")
    result = []
    for equal in (False, True):
        for same_direction in (False, True):
            c = json.loads(json.dumps(original))
            ordinal = (2 if equal else 1) + (2 if same_direction else 0)
            name = f"{ordinal:02d}_{'equal_gap' if equal else 'original'}_{'N_N' if same_direction else 'N_NW'}"
            c["name"] = c["instance"]["name"] = name
            c["diagnostic"] = {"equal_true_initial_gap": equal, "same_nominal_direction": same_direction}
            if same_direction:
                c["actions"] = [14, 10]
            if equal:
                pose = c["instance"]["object"]
                n = (-math.sin(pose["angle"]), math.cos(pose["angle"]))
                t = (-n[1], n[0])
                for i, offset in enumerate((-.25, .25)):
                    r = c["instance"]["robots"][i]
                    r["x"] = pose["x"] - .405 * n[0] + offset * t[0]
                    r["y"] = pose["y"] - .405 * n[1] + offset * t[1]
            result.append(c)
    return result


def measure(record):
    rows = record["rows"]
    angles = record["case"]["instance"]["object"]["angle"]
    t = np.asarray((-math.cos(angles), -math.sin(angles)))  # offline canonical face tangent
    separation, per_robot = [], []
    for row in rows:
        es = [row[f"r{i}"].get("effect") for i in (0, 1)]
        if all(e and e.get("contact_target") for e in es):
            separation.append(float(np.linalg.norm(np.asarray(es[0]["contact_target"]) - es[1]["contact_target"])))
    for i in (0, 1):
        valid = [row for row in rows if row[f"r{i}"].get("effect")]
        es = [row[f"r{i}"]["effect"] for row in valid]
        target = np.asarray([e["contact_target"] for e in es])
        steps = np.linalg.norm(np.diff(target, axis=0), axis=1)
        # Object is stationary in the failed baseline; moving ablations use pre-step truth.
        errors = []
        previous_pose = record["initial_object"][:3]
        for row in rows:
            e = row[f"r{i}"].get("effect")
            if e and e.get("estimated_centre"):
                errors.append(float((np.asarray(e["estimated_centre"]) - previous_pose[:2]) @ t))
            previous_pose = row["true"]["object"][:3]
        per_robot.append({"initial_gap_m": valid[0][f"r{i}"]["gap"],
                          "target_step_max_m": float(steps.max()) if len(steps) else 0.,
                          "target_step_p95_m": float(np.percentile(steps, 95)) if len(steps) else 0.,
                          "anchor_switches": sum(bool(e.get("endpoint") and e["endpoint"]["anchor_switched"]) for e in es),
                          "observed_endpoint_counts": sorted(set(e["endpoint"]["observed_count"] for e in es if e.get("endpoint"))),
                          "avoid_fraction": sum(row[f"r{i}"]["avoidance_active"] for row in valid) / len(valid),
                          "centre_tangent_rmse_m": float(np.sqrt(np.mean(np.square(errors)))),
                          "first_contact_s": next((row["time"] for row in rows if row[f"r{i}"]["contacting_object"]), None)})
    return {"case": record["metric"]["case"], "contact_correction": record["metric"]["feedback"],
            "seconds": record["metric"]["seconds"], "exits": record["metric"]["exits"],
            "actual_displacement_xy": record["metric"]["actual_displacement_xy"],
            "min_target_separation_m": min(separation), "final_target_separation_m": separation[-1],
            "robots": per_robot}


def run(out):
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a fresh output directory")
    (out / "logs").mkdir(parents=True)
    cases = panel()
    overrides = {"endpoint_tracking": True, "fast_approach": True, "translation_completion": "normal_reference"}
    manifest = {"version": VERSION, "env_version": spec.ENV_VERSION,
                "comparison": "contact_feedback", "cases": cases,
                "diagnostic_scope": "4 explicit geometries/actions x contact-point correction off/on",
                "branch_seconds": 10, "batch_seconds": 60,
                "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in (Path(__file__).resolve(), Path(bench.__file__),
                                            ROOT / "src/coimagine/random_start/effect_push.py",
                                            ROOT / "src/coimagine/random_start/endpoint_tracking.py",
                                            ROOT / "src/coimagine/random_start/env.py",
                                            ROOT / "src/coimagine/random_start/lidar.py")}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ROUND_PLAN.md").write_text(
        "# Same-face diagnosis\n\nFreeze four explicit scene/action variants before running.\n"
        "Paired variable: contact-point correction only; keep sensing, physics and all safety checks.\n"
        "Separate unequal original gaps from an explicitly declared equal-gap scene; same-direction\n"
        "control preserves each geometry. Retain all failures. 10 s/branch, 60 s total.\n"
        "No production controller, candidate mask, reward or historical output changes.\n")
    started = time.monotonic()
    for case in cases:
        for feedback in (False, True):
            path = out / "logs" / f"{case['name']}__{'on' if feedback else 'off'}.json"
            task = {"case": case, "feedback": feedback, "overrides": overrides,
                    "comparison": "contact_feedback", "path": str(path)}
            remaining = 60 - (time.monotonic() - started)
            try:
                if remaining <= 0:
                    raise subprocess.TimeoutExpired("batch", 60)
                proc = subprocess.run([sys.executable, str(Path(bench.__file__)), "--worker"],
                                      input=json.dumps(task), text=True, capture_output=True,
                                      cwd=ROOT, timeout=min(10, remaining))
                if proc.returncode:
                    raise RuntimeError(proc.stderr[-3000:])
            except (subprocess.TimeoutExpired, RuntimeError) as exc:
                path.write_text(json.dumps({"metric": {"case": case["name"], "feedback": feedback,
                                                      "status": "incomplete", "error": str(exc)}}))
    bench.render(out)
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    summaries = [measure(r) for r in records if r["metric"]["status"] == "completed"]
    (out / "diagnosis.json").write_text(json.dumps(summaries, indent=2) + "\n")
    plot(records, out)
    print(json.dumps({"seconds": time.monotonic() - started, "results": summaries}, indent=2))


def plot(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for feedback, color in ((False, "#218d70"), (True, "#d38b32")):
        r = next(r for r in records if r["metric"]["case"] == "01_original_N_NW" and r["metric"]["feedback"] == feedback)
        rows = [v for v in r["rows"] if all(v[f"r{i}"].get("effect") for i in (0, 1))]
        separation = [np.linalg.norm(np.asarray(v["r0"]["effect"]["contact_target"]) - v["r1"]["effect"]["contact_target"]) for v in rows]
        axes[0].plot([v["time"] for v in rows], separation, color=color, label=f"correction {'on' if feedback else 'off'}")
    axes[0].axhline(2 * spec.ROBOT_RADIUS, color="#a43e46", ls="--", label="2 robot radii: 0.36 m")
    axes[0].axhline(2 * spec.ROBOT_RADIUS + spec.OBSTACLE_AVOID_MARGIN, color="#6c7d91", ls=":", label="avoidance onset: 0.48 m")
    axes[0].set_xlabel("Time / s")
    axes[0].set_ylabel("Requested contact-point separation / m")
    axes[0].set_title("Original geometry and occlusion retained")
    r = next(r for r in records if r["metric"]["case"] == "01_original_N_NW" and r["metric"]["feedback"])
    pose = r["case"]["instance"]["object"]
    t = np.asarray((-math.cos(pose["angle"]), -math.sin(pose["angle"])))
    for i, color in ((0, "#4f93db"), (1, "#e59a41")):
        rows = [v for v in r["rows"] if v[f"r{i}"].get("effect")]
        target = [np.dot(np.asarray(v[f"r{i}"]["effect"]["contact_target"]) - (pose["x"], pose["y"]), t) for v in rows]
        own = [np.dot(np.asarray(v["true"]["robots"][i][:2]) - (pose["x"], pose["y"]), t) for v in rows]
        axes[1].plot([v["time"] for v in rows], target, color=color, label=f"robot {i} target")
        axes[1].plot([v["time"] for v in rows], own, color=color, ls="--", label=f"robot {i} actual")
    axes[1].set_xlabel("Time / s")
    axes[1].set_ylabel("Face tangent coordinate / m")
    axes[1].set_title("Targets converge inward while avoidance holds robots apart")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.savefig(out / "diagnosis.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
