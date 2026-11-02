"""Saved-log checks for local reservations and actual robot-to-robot clearance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(record):
    minimum = float("inf")
    target_minimum = float("inf")
    margins = []
    checked, side = 0, [None, None]
    for row in record["rows"]:
        robots = row["true"]["robots"]
        minimum = min(minimum, math.dist(robots[0][:2], robots[1][:2]))
        es = [row[f"r{i}"].get("effect") for i in (0, 1)]
        if all(e and e.get("contact_target") for e in es):
            target_minimum = min(target_minimum, math.dist(es[0]["contact_target"], es[1]["contact_target"]))
        for i, e in enumerate(es):
            reservation = e.get("same_face") if e else None
            if not reservation or reservation["status"] != "allocated":
                continue
            if side[i] is not None:
                assert side[i] == reservation["side"], "role swapped during the option"
            side[i] = reservation["side"]
            n = e["face_normal"]
            target = e["contact_target"]
            q = -n[1] * target[0] + n[0] * target[1]
            lo, hi = reservation["interval"]
            assert lo - 1e-8 <= q <= hi + 1e-8
            assert .04 - 1e-9 <= reservation["avoidance_margin"] <= .12 + 1e-9
            margins.append(reservation["avoidance_margin"])
            checked += 1
    m = record["metric"]
    if m["variant"]:
        assert minimum > .36, "actual robot overlap in candidate replay"
    return {"case": m["case"], "allocation": m["variant"], "exits": m["exits"],
            "seconds": m["seconds"], "min_actual_centre_distance_m": minimum,
            "min_actual_body_gap_m": minimum - .36,
            "min_target_distance_m": target_minimum if math.isfinite(target_minimum) else None,
            "margin_range_m": [min(margins), max(margins)] if margins else None,
            "checked_local_targets": checked, "fixed_roles": side,
            "max_penetration_m": m["max_penetration"],
            "actual_displacement_xy": m["actual_displacement_xy"], "rotation_deg": m["rotation_deg"]}


def main(out):
    records = [json.loads(p.read_text()) for p in sorted((out / "logs").glob("*.json"))]
    results = [audit(r) for r in records if r["metric"]["status"] == "completed"]
    report = {"version": "same-face-allocation-audit-v1", "results": results,
              "truth_use": "offline physical clearance labels only",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out / "allocation_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    lines = ["# 同面接触分区审计", "", "中心间距、参考预算结束与物体真实后果分别报告，不评分预设的联合位移目标。",
             "", "| 场景 | 分区 | 两侧退出 | 最小中心距 m | 最小实体间隙 m | 总耗时 s |",
             "|---|---|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['case']} | {r['allocation']} | {' / '.join(r['exits'])} | "
                     f"{r['min_actual_centre_distance_m']:.4f} | {r['min_actual_body_gap_m']:.4f} | {r['seconds']:.2f} |")
    (out / "ALLOCATION.md").write_text("\n".join(lines) + "\n")
    plot(records, out)
    print(json.dumps({"audited": len(results), "checked_local_targets": sum(r["checked_local_targets"] for r in results)}))


def plot(records, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    for col, name in enumerate(("same_face_N_NW", "same_face_N_NW_equal_gap")):
        for enabled, color in ((False, "#d38b32"), (True, "#218d70")):
            r = next(r for r in records if r["metric"]["case"] == name and r["metric"]["variant"] == enabled)
            rows = r["rows"]
            centres = [math.dist(v["true"]["robots"][0][:2], v["true"]["robots"][1][:2]) for v in rows]
            axes[0, col].plot([v["time"] for v in rows], centres, color=color, label="regions on" if enabled else "regions off")
            valid = [v for v in rows if all(v[f"r{i}"].get("effect") for i in (0, 1))]
            targets = [math.dist(v["r0"]["effect"]["contact_target"], v["r1"]["effect"]["contact_target"]) for v in valid]
            axes[1, col].plot([v["time"] for v in valid], targets, color=color, label="regions on" if enabled else "regions off")
        axes[0, col].set_title(name)
        axes[0, col].axhline(.36, color="#9a424b", linestyle="--", label="body geometry: 0.36")
        axes[0, col].axhline(.38, color="#b4725b", linestyle=":", label="unchanged sensor abort ~0.38")
        axes[1, col].axhline(.41, color="#647992", linestyle=":", label="nominal target spacing: 0.41")
        axes[0, col].set_ylabel("Actual centre distance / m")
        axes[1, col].set_ylabel("Requested contact-point distance / m")
        axes[1, col].set_xlabel("Time / s")
        for ax in axes[:, col]:
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Same-face roles, feasible contact regions, and speed-dependent working clearance")
    fig.savefig(out / "allocation.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    main(parser.parse_args().out.resolve())
