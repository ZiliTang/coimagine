"""Independently audit reference-dose/time contracts, without scoring a fabricated joint goal."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(record, actions):
    b, m = record["boundary"], record["metric"]
    execution = b["execution"]
    checked = 0
    previous = [0., 0.]
    stopped = [False, False]
    reference = execution["normal_reference_pair"]
    for row in record["rows"]:
        state = row.get("execution")
        if not state:
            continue
        for i in (0, 1):
            assert not (stopped[i] and state["active"][i]), "finished controller restarted"
            stopped[i] |= not state["active"][i]
            increment = state["normal_reference_budget"][i] - previous[i]
            if increment > 1e-9:
                assert state["phase"] == "execute" and row[f"r{i}"]["contacting_object"]
                assert abs(increment - execution["normal_speed_references"][i] * .05) < 1e-8
                checked += 1
            assert increment >= -1e-9
        previous = state["normal_reference_budget"]
    for i in (0, 1):
        if b["skill_ends"][str(i)]["exit_reason"] == "normal_reference_complete":
            target = actions[record["case"]["actions"][i]]["distance"]
            assert previous[i] >= target - 1e-9
            assert previous[i] - target <= execution["normal_speed_references"][i] * .05 + 1e-8
    gap = abs(b["skill_ends"]["0"]["time"] - b["skill_ends"]["1"]["time"])
    return {"case": m["case"], "new_protocol": m["variant"], "normal_reference_pair": reference,
            "normal_reference_budget": execution["normal_reference_budget"],
            "normal_speed_references": execution["normal_speed_references"],
            "checked_contact_increments": checked, "exits": m["exits"],
            "exit_gap_seconds": gap, "seconds": m["seconds"],
            "actual_displacement_xy": m["actual_displacement_xy"], "rotation_deg": m["rotation_deg"],
            "max_penetration": m["max_penetration"], "events": execution["events"]}


def main(out):
    manifest = json.loads((out / "manifest.json").read_text())
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    results = [audit(r, manifest["actions"]) for r in records if r["metric"]["status"] == "completed"]
    report = {"version": "joint-reference-audit-v1", "records": results,
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "physics_scored_against_joint_vector_goal": False}
    (out / "execution_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 联合执行协议审计", "", "只检查参考预算和生命周期；实际位移是模型标签，不是预设联合位移的成功率。",
             "", "| 场景 | 新协议 | 退出0 / 1 | 参考预算0 / 1 | 退出时差s | 实际Δx / Δy m | 转角° | 总耗时s |",
             "|---|---|---|---|---:|---|---:|---:|"]
    for r in results:
        d = r["actual_displacement_xy"]
        dose = r["normal_reference_budget"]
        lines.append(f"| {r['case']} | {r['new_protocol']} | {' / '.join(r['exits'])} | "
                     f"{dose[0]:.4f} / {dose[1]:.4f} | {r['exit_gap_seconds']:.2f} | "
                     f"{d[0]:.4f} / {d[1]:.4f} | {r['rotation_deg']:.2f} | {r['seconds']:.2f} |")
    (out / "EXECUTION.md").write_text("\n".join(lines) + "\n")
    plot(records, out)
    print(json.dumps({"audited": len(results), "checked_contact_increments": sum(r["checked_contact_increments"] for r in results)}))


def plot(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    record = next((r for r in records if r["metric"]["case"] == "same_N_0.1_0.5"
                   and r["metric"]["variant"]), None)
    if record:
        for i, color in ((0, "#4f93db"), (1, "#e59a41")):
            frames = [r for r in record["rows"] if r.get("execution")]
            target = .1 if i == 0 else .5
            axes[0, 0].plot([r["time"] for r in frames],
                            [r["execution"]["normal_reference_budget"][i] / target for r in frames],
                            label=f"robot {i}", color=color, linestyle="-" if i == 0 else "--")
            valid = [r for r in frames if r[f"r{i}"].get("effect")]
            values = [sum(a * b for a, b in zip(r[f"r{i}"]["desired_velocity"],
                                               r[f"r{i}"]["effect"]["face_normal"])) for r in valid]
            axes[0, 1].plot([r["time"] for r in valid], values, color=color, label=f"robot {i} total")
            axes[0, 1].axhline(record["boundary"]["execution"]["normal_speed_references"][i],
                               color=color, linestyle="--", label=f"robot {i} nominal")
    axes[0, 0].set_title("0.1 / 0.5: normalized local reference budgets")
    axes[0, 0].set_ylabel("Consumed reference / requested scale")
    axes[0, 1].set_title("Proportional bias + unscaled contact following")
    axes[0, 1].set_ylabel("Normal velocity reference / m/s")
    for ax, name in zip(axes[1], ("same_N_0.1_0.5", "N_E_0.5_0.1")):
        for enabled, color in ((False, "#d38b32"), (True, "#218d70")):
            r = next((r for r in records if r["metric"]["case"] == name and r["metric"]["variant"] == enabled), None)
            if not r:
                continue
            initial = r["initial_object"]
            xy = [(v["true"]["object"][0] - initial[0], v["true"]["object"][1] - initial[1]) for v in r["rows"]]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], color=color, label="new" if enabled else "legacy")
        ax.set_title(f"{name}: actual physical outcome")
        ax.set_xlabel("Object delta x / m")
        ax.set_ylabel("Object delta y / m")
    for ax in axes.flat:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    axes[0, 0].set_xlabel("Time / s")
    axes[0, 1].set_xlabel("Time / s")
    fig.suptitle("Joint PUSH: independent scales, reference protocol, learned physical consequences")
    fig.savefig(out / "execution.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
