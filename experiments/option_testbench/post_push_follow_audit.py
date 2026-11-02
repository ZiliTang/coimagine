"""Audit the active/follow split, local limits, and physical paired outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

if __package__:
    from .normal_travel_audit import audit as travel_audit
else:
    from normal_travel_audit import audit as travel_audit


def audit(record):
    travel = travel_audit(record)
    cfg = record["config"]
    stats = []
    for i in (0, 1):
        frames = [row for row in record["rows"] if row[f"r{i}"]["stage"] == "follow"]
        infos = [row[f"r{i}"] for row in frames]
        nominal = lambda v: (v.get("command_disturbance") or {}).get("nominal_force", v["force"])
        maximum_force = max((math.hypot(*nominal(v)) for v in infos), default=0.)
        maximum_normal = max((abs(sum(x*y for x, y in zip(nominal(v), v["effect"]["face_normal"])))
                              for v in infos), default=0.)
        maximum_speed = max((math.hypot(*v["desired_velocity"]) for v in infos), default=0.)
        assert maximum_force <= cfg["follow_force"] + 1e-8
        assert maximum_normal <= cfg["follow_normal_force"] + 1e-8
        assert maximum_speed <= cfg["follow_speed"] + 1e-8
        completed = [row[f"r{i}"]["effect"] for row in record["rows"]
                     if (row[f"r{i}"].get("effect") or {}).get("follow_started")]
        if completed:
            # Drive and clock freeze at the phase transition, through joint braking.
            assert max(v["normal_travel"] for v in completed) - min(v["normal_travel"] for v in completed) < 1e-9
            assert max(v["normal_reference_budget"] for v in completed) - min(v["normal_reference_budget"] for v in completed) < 1e-9
        stats.append({"frames": len(frames), "contact_frames": sum(v["contacting_object"] for v in infos),
                      "contact_fraction": sum(v["contacting_object"] for v in infos) / len(infos) if infos else None,
                      "near_fraction": sum(v["gap"] <= cfg.get("recovery_enter_gap", .01) for v in infos) / len(infos) if infos else None,
                      "max_follow_gap_m": max((v["gap"] for v in infos), default=None),
                      "max_command_force_N": maximum_force, "max_command_normal_force_N": maximum_normal,
                      "max_desired_speed_m_s": maximum_speed})
    travel.update(follow=record["metric"]["variant"], follow_stats=stats,
                  rotation_deg=record["metric"]["rotation_deg"],
                  drive_complete_times_s=record["boundary"]["execution"]["drive_complete_times"],
                  min_robot_centre_distance_m=min(math.dist(row["true"]["robots"][0][:2], row["true"]["robots"][1][:2])
                                                  for row in record["rows"]),
                  max_object_penetration_m=record["metric"]["max_penetration"])
    return travel


def main(out):
    manifest = json.loads((out / "manifest.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    results = []
    for r in records:
        if r["metric"]["status"] == "completed":
            r["manifest_actions"] = manifest["actions"]
            results.append(audit(r))
    sensitivity = []
    groups = {}
    for r in records:
        if "case" in r and r["case"].get("perturbation"):
            base = r["case"]["perturbation"]["base"]
            groups.setdefault(base, []).append(r["metric"]["case"])
    for base, variants in groups.items():
        for follow in (False, True):
            selected = [r for r in records if r.get("metric", {}).get("case") in [base, *variants]
                        and r["metric"]["variant"] == follow]
            rotation = [r["metric"]["rotation_deg"] for r in selected]
            displacement = [r["metric"]["actual_displacement_xy"] for r in selected]
            sensitivity.append({"base": base, "follow": follow, "cases": len(selected),
                                "final_angle_range_deg": max(rotation)-min(rotation),
                                "max_final_xy_separation_m": max(math.dist(a, b) for a in displacement for b in displacement)})
    report = {"version": "post-push-follow-audit-v1", "results": results, "local_sensitivity": sensitivity,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "travel_audit_sha256": hashlib.sha256(Path(__file__).with_name("normal_travel_audit.py").read_bytes()).hexdigest(),
              "truth_use": "offline own-pose reconstruction and physical outcomes; never runtime inputs"}
    (out / "follow_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 退出后随动对照：独立复算与物理后果", "",
             "主动行程冻结，随动行程另记；检查指令力限幅不等同于测量真实接触力。所有拒绝与失败保留。", "",
             "| 场景 | 随动 | 退出 | 主动行程 A/B m | 随动法向行程 A/B m | 转角 ° | 耗时 s |",
             "|---|---|---|---|---|---:|---:|"]
    for r in results:
        fmt = lambda values: ' / '.join(f'{v:.4f}' for v in values)
        lines.append(f"| {r['case']} | {r['follow']} | {' / '.join(r['exits'])} | "
                     f"{fmt(r['independent_normal_travel_m'])} | {fmt(r['independent_follow_normal_travel_m'])} | "
                     f"{r['rotation_deg']:.2f} | {r['seconds']:.2f} |")
    lines += ["", "## ±1 cm 局部敏感性（仅三点范围，不是泛化置信区间）", "",
              "| 基础场景 | 随动 | 点数 | 最终角度范围 ° | 最大最终位置间距 m |", "|---|---|---:|---:|---:|"]
    for s in sensitivity:
        lines.append(f"| {s['base']} | {s['follow']} | {s['cases']} | {s['final_angle_range_deg']:.3f} | "
                     f"{s['max_final_xy_separation_m']:.5f} |")
    (out / "FOLLOW_AUDIT.md").write_text('\n'.join(lines) + '\n')
    plot(records, out)
    print(json.dumps({"audited": len(results), "samples": sum(r['checked_samples'] for r in results),
                      "max_travel_reconstruction_error_m": max(r['max_reconstruction_error_m'] for r in results),
                      "sensitivity": sensitivity}))


def plot(records, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = ['same_N_0.1_0.5', 'same_face_N_NW_equal_gap', 'N_E_0.5_0.1']
    names = [n for n in names if any(r.get('metric', {}).get('case') == n for r in records)]
    fig, axes = plt.subplots(2, len(names), figsize=(4.4*len(names), 7), squeeze=False, constrained_layout=True)
    for col, name in enumerate(names):
        for enabled, color in ((False, '#c08a38'), (True, '#218d70')):
            r = next(r for r in records if r['metric']['case'] == name and r['metric']['variant'] == enabled)
            rows = r['rows']; t = [v['time'] for v in rows]
            axes[0, col].plot(t, [math.degrees(v['true']['object'][2]-r['initial_object'][2]) for v in rows],
                              color=color, label='contact follow' if enabled else 'hold / brake')
            # Chosen panels put the shorter scale at A for same_N, B for the others.
            i = 0 if name == 'same_N_0.1_0.5' else 1
            own = [v[f'r{i}'] for v in rows if v[f'r{i}'].get('effect')]
            rt = [v['time'] for v in rows if v[f'r{i}'].get('effect')]
            axes[1, col].plot(rt, [v['effect']['normal_travel'] for v in own], color=color,
                              linestyle='--', label='short side active travel / '+('follow' if enabled else 'hold'))
            if enabled:
                axes[1, col].plot(rt, [v['effect']['normal_travel']+v['effect']['follow_normal_travel'] for v in own],
                                  color=color, label='short side active + follow travel')
        axes[0, col].set_title(name)
        axes[0, col].set_ylabel('Object rotation / degrees (offline)')
        axes[1, col].set(xlabel='Time / s', ylabel='Measured contacted normal travel / m')
        for ax in axes[:, col]:
            ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Post-threshold contact follow: reduced same-face rotation, adjacent-face contact loss remains')
    fig.savefig(out / 'post_follow.png', dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    main(parser.parse_args().out.resolve())
