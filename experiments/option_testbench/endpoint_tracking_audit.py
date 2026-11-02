"""Offline-only endpoint coverage/error audit of saved effect-PUSH branch logs.

True object poses are evaluation labels, never inputs to the online tracker.
Measurement time is matched to the PRE-step pose, not the post-step log pose.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def body_point(point, pose):
    dx, dy = point[0] - pose[0], point[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return np.asarray((c * dx + s * dy, -s * dx + c * dy))


def audit(record):
    coverage = [{"active_frames": 0, "fresh_endpoint_frames": 0, "anchor_errors": [],
                 "switches": 0, "target_steps": []} for _ in (0, 1)]
    both_active = both_fresh = 0
    prior_pose = record["initial_object"][:3]
    previous_target = [None, None]
    for row in record["rows"]:
        fresh = [False, False]
        active = [False, False]
        for robot in (0, 1):
            e = row[f"r{robot}"].get("effect")
            if not e:
                continue
            active[robot] = True
            c = coverage[robot]
            c["active_frames"] += 1
            ep = e.get("endpoint")
            fresh[robot] = bool(ep and ep["observed_count"] >= 1 and ep["status"] == "observed")
            c["fresh_endpoint_frames"] += int(fresh[robot])
            target = e.get("contact_target")
            if target:
                local = body_point(target, prior_pose)
                if previous_target[robot] is not None:
                    c["target_steps"].append(float(np.linalg.norm(local - previous_target[robot])))
                previous_target[robot] = local
            if not ep:
                continue
            c["switches"] += int(ep["anchor_switched"])
            if fresh[robot] and ep["anchor"]:
                # Rectangular shape is fixed by the environment; used ONLY here
                # to label the single declared anchor, not select best-of-K.
                normal = e["face_normal"]
                cc, ss = math.cos(prior_pose[2]), math.sin(prior_pose[2])
                local_n = (cc * normal[0] + ss * normal[1], -ss * normal[0] + cc * normal[1])
                axis = int(abs(local_n[1]) > abs(local_n[0]))
                n = np.zeros(2)
                n[axis] = math.copysign(1., local_n[axis])
                t = np.asarray((-n[1], n[0]))
                depth, length = ((1., .25) if axis == 0 else (.25, 1.))
                expected = -n * depth / 2 + t * length / 2 * (-1 if ep["anchor_side"] == 0 else 1)
                error = np.linalg.norm(body_point(ep["anchor"], prior_pose) - expected)
                c["anchor_errors"].append(float(error))
        if all(active):
            both_active += 1
            both_fresh += int(all(fresh))
        prior_pose = row["true"]["object"][:3]
    for c in coverage:
        errors, steps = c.pop("anchor_errors"), c.pop("target_steps")
        c["fresh_endpoint_fraction"] = (c["fresh_endpoint_frames"] / c["active_frames"]
                                        if c["active_frames"] else None)
        c["anchor_rmse_m"] = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
        c["anchor_max_error_m"] = max(errors, default=None)
        c["target_step_p95_m"] = float(np.percentile(steps, 95)) if steps else None
        c["target_step_max_m"] = max(steps, default=None)
        if not record["metric"].get("endpoint_tracking"):
            c["fresh_endpoint_frames"] = c["fresh_endpoint_fraction"] = None
    return {"case": record["metric"]["case"], "variant": record["metric"]["variant"],
            "robots": coverage, "both_active_frames": both_active,
            "both_fresh_frames": both_fresh if record["metric"].get("endpoint_tracking") else None,
            "both_fresh_fraction": (both_fresh / both_active if both_active else None)
                                   if record["metric"].get("endpoint_tracking") else None}


def main(out):
    results = []
    for path in sorted((out / "logs").glob("*.json")):
        record = json.loads(path.read_text())
        if record["metric"]["status"] == "completed":
            results.append(audit(record))
    report = {"version": "endpoint-audit-v1", "truth_use": "offline labels only",
              "timing": "pre-step pose paired with controller's pre-step observation",
              "scope": "per-option tracking; active frames only; not a general visibility guarantee",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "records": results}
    (out / "endpoint_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    plot(out)
    print(json.dumps([{k: r[k] for k in ("case", "variant", "both_fresh_fraction")}
                      for r in results if r["both_active_frames"]], indent=2))


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for column, name in enumerate(("tilt_+8", "cooperative_N")):
        for enabled, color in ((False, "#d38b32"), (True, "#218d70")):
            path = out / "logs" / f"{name}__{'tracking_on' if enabled else 'tracking_off'}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            pose = record["initial_object"][:3]
            ts, targets, counts = [], [], [[], []]
            for row in record["rows"]:
                e = row["r0"].get("effect")
                if e and e.get("contact_target"):
                    ts.append(e["observation_time"])
                    targets.append(body_point(e["contact_target"], pose)[0])
                    for robot in (0, 1):
                        ep = (row[f"r{robot}"].get("effect") or {}).get("endpoint")
                        counts[robot].append(ep["observed_count"] if ep else float("nan"))
                pose = row["true"]["object"][:3]
            axes[0, column].plot(ts, targets, color=color, lw=1.1,
                                 label="tracking on" if enabled else "tracking off")
            if enabled:
                for robot, style in ((0, "-"), (1, "--")):
                    axes[1, column].plot(ts, counts[robot], style, label=f"robot {robot}")
        axes[0, column].set_title(name)
        axes[0, column].set_ylabel("Contact target, object-frame x / m")
        axes[1, column].set_ylabel("Fresh physical endpoint brackets")
        axes[1, column].set_xlabel("Observation time / s")
        axes[1, column].set_ylim(-.1, 2.2)
        for ax in axes[:, column]:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Endpoint-relative contact targets: no teammate private observation sharing")
    fig.savefig(out / "endpoint_comparison.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
