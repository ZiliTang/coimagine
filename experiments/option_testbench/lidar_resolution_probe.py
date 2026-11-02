"""Read-only lidar-resolution diagnostic; never changes controller/sensor defaults.

Compare actual ray casting with interpolation inside an already fitted face, and a
causal endpoint EMA used ONLY as a diagnostic. No synthetic sample is a new return.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from coimagine.random_start import lidar, spec


def densify_face(patch, origin, factor=4, polar_linear=False):
    """Interior-only derived points. Never joins classes, faces or missing rays.

    Input must be a single fitted patch. Callers must retain its original support
    count/confidence. These samples are for this diagnostic, NOT LidarScan returns.
    """
    points = []
    normal = np.asarray(patch.normal_world)
    origin = np.asarray(origin)
    raw = np.asarray(patch.points_world)
    plane = float(np.mean((raw - origin) @ normal))
    for a, b in zip(raw[:-1], raw[1:]):
        points.append(tuple(a))
        ar, br = a - origin, b - origin
        aa, ba = math.atan2(ar[1], ar[0]), math.atan2(br[1], br[0])
        delta = (ba - aa + math.pi) % (2 * math.pi) - math.pi
        for j in range(1, factor):
            f = j / factor
            angle = aa + f * delta
            ray = np.asarray((math.cos(angle), math.sin(angle)))
            radius = ((1 - f) * np.linalg.norm(ar) + f * np.linalg.norm(br)
                      if polar_linear else plane / float(normal @ ray))
            points.append(tuple(origin + radius * ray))
    points.append(tuple(raw[-1]))
    return points


def endpoints(points, normal):
    t = np.asarray((-normal[1], normal[0]))
    n = np.asarray(normal)
    raw = np.asarray(points)
    q = raw @ t
    plane = float(np.mean(raw @ n))
    return (float(q.min()), float(q.max()), plane)


def run(out, frames=100):
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    dt, tau = spec.CONTROL_DT, .15
    alpha = dt / (tau + dt)
    angle = math.radians(8)
    normal = np.asarray((-math.sin(angle), math.cos(angle)))
    tangent = np.asarray((-normal[1], normal[0]))
    centre = np.asarray((2.5, 2.0))
    robot = centre - .405 * normal
    manifest = {
        "version": "lidar-resolution-probe-v1", "frames": frames, "dt": dt,
        "scenarios": ["static_phase_sweep", "moving_then_stop", "teammate_shadow"],
        "methods": ["72", "288", "72_geometric_4x", "72_polar_linear_4x", "72_endpoint_ema"],
        "factor": 4, "ema_tau_seconds": tau, "ema_alpha": alpha,
        "truth_use": "scene generation, offline face selection check and errors only",
        "selection": "same observable normal as first patch, dot>cos(20deg); no true face id",
        "motion": "lateral 0.12 m/s until t=2s, then stop; yaw oscillation +/-3deg period 2s",
        "sensor_defaults_unchanged": True, "not_controller_integrated": True,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__).resolve(), ROOT / "src/coimagine/random_start/lidar.py")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "ROUND_PLAN.md").write_text(
        "# Read-only lidar resolution probe\n\n"
        "Freeze 3 scripted scan sequences, compare 72/288 actual rays, interior-only\n"
        "4x planar/polar interpolation and a causal EMA. Report jitter, bias, lag and\n"
        "timing. Preserve missing frames and shadow boundaries. No physics rollout,\n"
        "reward change, controller replacement or RL training. Inputs: manifest.json.\n")
    records = []
    timing = {m: [] for m in manifest["methods"]}
    for scenario in manifest["scenarios"]:
        previous_normal = {72: None, 288: None}
        smooth = None
        for i in range(frames):
            now = i * dt
            displacement = .12 * min(now, 2.0) if scenario == "moving_then_stop" else 0.
            c = centre + displacement * tangent
            heading = math.pi / 2 + angle + math.radians(3) * math.sin(math.pi * now)
            peer = ((c[0] + robot[0]) / 2, (c[1] + robot[1]) / 2, 0.) \
                if scenario == "teammate_shadow" and now >= 2 else (4.4, .5, 0.)
            robots = ((*robot, heading), peer)
            true_q = float(c @ tangent)
            frame = {"scenario": scenario, "t": now, "heading": heading,
                     "object_centre": list(c), "true_centre_q": true_q, "methods": {}}
            for rays in (72, 288):
                started = time.perf_counter()
                scan = lidar.scan_lidar((5., 4.), (), (*c, angle), robots, 0, i, now, scan_rays=rays)
                patches = lidar.scan_patches(scan, tuple(robot))
                prev = previous_normal[rays]
                candidates = [p for p in patches if p.valid and (prev is None
                              or float(np.dot(p.normal_world, prev)) > math.cos(math.radians(20)))]
                patch = min(candidates, key=lambda p: p.distance) if candidates else None
                timing[str(rays)].append(time.perf_counter() - started)
                if patch is None:
                    frame["methods"][str(rays)] = None
                    if rays == 72:
                        smooth = None
                        for method in manifest["methods"][2:]:
                            frame["methods"][method] = None
                    continue
                previous_normal[rays] = patch.normal_world
                # A fixed axis is used only for offline comparison, not fitting/filtering.
                def measure(points):
                    lo, hi, _ = endpoints(points, normal)
                    fit = lidar._fit_line(points)
                    return {"span": hi - lo, "low": lo, "high": hi,
                            "centre_error": (lo + hi) / 2 - true_q, "fit_rms": fit[3]}
                frame["methods"][str(rays)] = measure(patch.points_world)
                if rays != 72:
                    continue
                frame["patch"] = asdict(patch)
                frame["scan"] = asdict(scan)
                for polar in (False, True):
                    method = "72_polar_linear_4x" if polar else "72_geometric_4x"
                    started = time.perf_counter()
                    points = densify_face(patch, scan.origin, polar_linear=polar)
                    fit = lidar._fit_line(points)
                    timing[method].append(time.perf_counter() - started)
                    row = measure(points)
                    row["derived_points"] = len(points)
                    row["fit_rms"] = fit[3]
                    frame["methods"][method] = row
                # Diagnostic EMA of observed endpoints, in world coordinates. It deliberately
                # does NOT infer full physical edges or hide shrinkage caused by occlusion.
                started = time.perf_counter()
                raw = np.asarray(patch.points_world)
                q = raw @ np.asarray(patch.tangent_world)
                bounds = np.asarray((raw[q.argmin()], raw[q.argmax()]))
                if smooth is not None:
                    if np.linalg.norm(bounds.mean(axis=0) - smooth.mean(axis=0)) > .12:
                        smooth = None  # conservative gap/change reset
                smooth = bounds if smooth is None else alpha * bounds + (1 - alpha) * smooth
                frame["methods"]["72_endpoint_ema"] = measure(smooth)
                timing["72_endpoint_ema"].append(time.perf_counter() - started)
            records.append(frame)
    summary = {}
    for scenario in manifest["scenarios"]:
        summary[scenario] = {}
        for method in manifest["methods"]:
            rows = [r["methods"].get(method) for r in records if r["scenario"] == scenario]
            valid = [r for r in rows if r]
            jumps = [abs(b["span"] - a["span"]) for a, b in zip(rows[:-1], rows[1:]) if a and b]
            errors = [r["centre_error"] for r in valid]
            summary[scenario][method] = {
                "observed_frames": len(valid), "frames": frames,
                "span_std_m": float(np.std([r["span"] for r in valid])) if valid else None,
                "max_span_jump_m": max(jumps, default=None),
                "centre_rmse_m": float(np.sqrt(np.mean(np.square(errors)))) if valid else None,
                "max_abs_centre_error_m": max(map(abs, errors), default=None),
                "max_fit_rms_m": max((r["fit_rms"] for r in valid), default=None),
            }
    for method, samples in timing.items():
        summary.setdefault("timing", {})[method] = {
            "median_ms": float(np.median(samples)) * 1000,
            "scope": "scan+patch fit" if method in ("72", "288") else "additional postprocessing only"}
    (out / "records.json").write_text(json.dumps(records) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 7), constrained_layout=True)
    for j, scenario in enumerate(manifest["scenarios"]):
        sequence = [r for r in records if r["scenario"] == scenario]
        ts = [r["t"] for r in sequence]
        for method in manifest["methods"]:
            for row, field in enumerate(("span", "centre_error")):
                values = [r["methods"].get(method, {}).get(field, float("nan"))
                          if r["methods"].get(method) else float("nan") for r in sequence]
                axes[row, j].plot(ts, values, label=method, linewidth=1.3)
                axes[row, j].grid(alpha=.25)
        axes[0, j].set_title(scenario)
        axes[0, j].set_ylabel("Observed segment span / m")
        axes[1, j].set_ylabel("Segment midpoint error / m")
        axes[1, j].set_xlabel("Time / s")
        axes[1, j].axhline(0, color="grey", linewidth=.7)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Lidar sampling probe: derived points add no edge observations; EMA trades jitter for lag")
    fig.savefig(out / "comparison.png", dpi=130)
    plt.close(fig)
    print(json.dumps({"out": str(out), "frames": len(records), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args().out.resolve())
