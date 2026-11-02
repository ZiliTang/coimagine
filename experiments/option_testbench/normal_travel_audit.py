"""Recompute body normal travel from saved poses and preceding sensed normals."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(record):
    """Pose snapshots are used offline to independently check own-pose odometry.

    Rows contain POST-physics truth but PRE-physics observations. Shift the pose
    stream by one row before projecting; do not use the controller's step_vector
    or its accumulated values as the reference calculation.
    """
    pose_before = [[p["x"], p["y"]] for p in record["case"]["instance"]["robots"]]
    previous_pose, previous_info = None, [None, None]
    totals = [0., 0.]
    follow_totals, follow_paths = [0., 0.], [0., 0.]
    recovery_totals, recovery_paths = [0., 0.], [0., 0.]
    checked = positive = reverse = 0
    errors = []
    max_angle = math.radians(record["config"]["face_step_deg"])
    for row in record["rows"]:
        for i in (0, 1):
            info = row[f"r{i}"]
            e = info.get("effect")
            old = previous_info[i]
            increment = 0.
            follow_increment = 0.
            recovery_increment = 0.
            if e:
                if old and previous_pose is not None:
                    oe = old["effect"]
                    a, b = oe.get("face_normal"), e.get("face_normal")
                    active = (old["stage"] in ("push", "brake", "follow") and not old["done"]
                              and (record["boundary"]["execution"]["protocol"] != "proportional_normal_v1" or oe["joint_active"]))
                    following = oe.get("follow_started", False) and old["stage"] in ("follow", "brake") and not old["done"]
                    recovering = old["stage"] == "recover" and not old["done"]
                    if following:
                        follow_paths[i] += math.dist(pose_before[i], previous_pose[i])
                    if recovering:
                        recovery_paths[i] += math.dist(pose_before[i], previous_pose[i])
                    elapsed = e["observation_time"] - oe["observation_time"]
                    # dt comes from saved period end timestamps, not from control code.
                    dt = row["time"] - previous_time
                    geometry_valid = (a is not None and b is not None and info["patch_valid"]
                             and 0 < elapsed <= 1.5 * dt + 1e-12
                             and sum(x*y for x, y in zip(a, b)) >= math.cos(max_angle))
                    valid = active and old["contacting_object"] and info["contacting_object"] and geometry_valid
                    if recovering and geometry_valid:
                        recovery_increment = sum((pose_before[i][j] - previous_pose[i][j]) * a[j] for j in (0, 1))
                    if valid:
                        increment = sum((pose_before[i][j] - previous_pose[i][j]) * a[j] for j in (0, 1))
                        positive += increment > 1e-10
                        reverse += increment < -1e-10
                        if following:
                            follow_increment, increment = increment, 0.
                totals[i] += increment
                follow_totals[i] += follow_increment
                recovery_totals[i] += recovery_increment
                errors.extend([abs(increment - e["normal_travel_increment"]),
                               abs(totals[i] - e["normal_travel"])])
                if "follow_normal_travel" in e:
                    errors.extend([abs(follow_increment - e["follow_normal_increment"]),
                                   abs(follow_totals[i] - e["follow_normal_travel"]),
                                   abs(follow_paths[i] - e["follow_path_length"])])
                if "recovery_normal_travel" in e:
                    errors.extend([abs(recovery_increment-e["recovery_normal_increment"]),
                                   abs(recovery_totals[i]-e["recovery_normal_travel"]),
                                   abs(recovery_paths[i]-e["recovery_path_length"])])
                checked += 1
                previous_info[i] = info
            else:
                previous_info[i] = None  # no motion is re-credited after Hold/exit
        previous_pose = pose_before
        pose_before = [p[:2] for p in row["true"]["robots"]]
        previous_time = row["time"]
    execution = record["boundary"]["execution"]
    for i in (0, 1):
        if execution["normal_travel"][i] is not None:
            errors.append(abs(totals[i] - execution["normal_travel"][i]))
        if "follow_normal_travel" in execution:
            errors.append(abs(follow_totals[i] - execution["follow_normal_travel"][i]))
        if "recovery_normal_travel" in execution:
            errors.append(abs(recovery_totals[i]-execution["recovery_normal_travel"][i]))
        end = record["boundary"]["skill_ends"][str(i)]
        if end["exit_reason"] == "normal_travel_complete":
            action = record["case"]["actions"][i]
            assert totals[i] >= next(a["distance"] for a in record["manifest_actions"] if a["index"] == action) - 1e-8
    assert max(errors, default=0.) < 1e-8, max(errors)
    metric = record["metric"]
    return {"case": metric["case"], "measured_completion": metric["variant"],
            "exits": metric["exits"], "independent_normal_travel_m": totals,
            "independent_follow_normal_travel_m": follow_totals,
            "independent_follow_path_m": follow_paths,
            "independent_recovery_normal_travel_m": recovery_totals,
            "independent_recovery_path_m": recovery_paths,
            "reference_budget_m": metric["normal_reference_budget"],
            "object_displacement_xy_m": metric["actual_displacement_xy"],
            "object_effect": metric["actual_effect"], "seconds": metric["seconds"],
            "exit_times_s": [record["boundary"]["skill_ends"][str(i)]["time"] for i in (0, 1)],
            "checked_samples": checked, "positive_intervals": positive, "reverse_intervals": reverse,
            "max_reconstruction_error_m": max(errors, default=0.)}


def main(out):
    manifest = json.loads((out / "manifest.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    results = []
    for r in records:
        if r["metric"]["status"] == "completed":
            r["manifest_actions"] = manifest["actions"]
            results.append(audit(r))
    report = {"version": "normal-travel-audit-v1", "results": results,
              "truth_use": "offline robot-pose consistency and physical-effect labels only",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "travel_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 本体法向行程独立复算", "", "按步前本体位置差和前一有效观测法向复算；物体位移单独列出。",
             "", "| 场景 | 本体实测终止 | 退出 | 实测行程 A/B m | 参考积分 A/B m | 物体效果 | 耗时 s |",
             "|---|---|---|---|---|---:|---:|"]
    for r in results:
        travel = ' / '.join(f'{v:.4f}' for v in r['independent_normal_travel_m'])
        reference = ' / '.join('—' if v is None else f'{v:.4f}' for v in r['reference_budget_m'])
        lines.append(f"| {r['case']} | {r['measured_completion']} | {' / '.join(r['exits'])} | "
                     f"{travel} | {reference} | {r['object_effect']:.4f} | {r['seconds']:.2f} |")
    (out / "TRAVEL_AUDIT.md").write_text('\n'.join(lines) + '\n')
    plot(records, out)
    print(json.dumps({"audited": len(results), "samples": sum(r["checked_samples"] for r in results),
                      "max_error_m": max(r["max_reconstruction_error_m"] for r in results)}))


def plot(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, name in zip(axes, ('scale_0.50', 'same_N_0.5_0.5', 'same_N_0.1_0.5')):
        for enabled, style in ((False, '--'), (True, '-')):
            r = next(r for r in records if r['metric']['case'] == name and r['metric']['variant'] == enabled)
            rows = [row for row in r['rows'] if row['r0'].get('effect')]
            ax.plot([v['time'] for v in rows], [v['r0']['effect']['normal_travel'] for v in rows],
                    color='#218d70', linestyle=style, label='A own travel / measured' if enabled else 'A own travel / clock')
            if enabled:
                ax.plot([v['time'] for v in rows], [v['r0']['effect']['normal_reference_budget'] for v in rows],
                        color='#bb8833', label='A command integral')
                ax.plot([v['time'] for v in r['rows']], [v['true']['object'][1]-r['initial_object'][1] for v in r['rows']],
                        color='#5187b5', label='Object world dy (offline)')
                if name == 'same_N_0.1_0.5':
                    rows_b = [v for v in r['rows'] if v['r1'].get('effect')]
                    ax.plot([v['time'] for v in rows_b], [v['r1']['effect']['normal_travel'] for v in rows_b],
                            color='#9758a5', label='B own travel / measured')
        target = .1 if name == 'same_N_0.1_0.5' else .5
        ax.axhline(target, color='gray', linestyle=':', label='A travel target')
        ax.set(title=name, xlabel='Time / s', ylabel='Distance / m')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Measured body normal travel, commanded travel, and object motion are distinct')
    fig.savefig(out / 'normal_travel.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
