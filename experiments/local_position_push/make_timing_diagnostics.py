"""C.4.3: three time-aligned V/X diagnostics, plus the per-case time series.

    python experiments/local_position_push/make_timing_diagnostics.py

Central single push, two-robot same-side push and offset rotation: the lexicographically first
baseline root of each class, V and X side by side. Everything is read from the per-control-period
logs of the recorded executions -- nothing is re-simulated.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

#: labels stay ASCII so the figures never depend on a CJK font being installed
CASES = (
    ("solo_central", "central single push"),
    ("dual_same_side", "two-robot same-side push"),
    ("dual_rotate", "offset rotation"),
)


def _arm_rows(run_dir: Path, split: str, root_id: str, arm: str) -> List[dict]:
    path = run_dir / "logs" / split / f"{root_id.replace('/', '_')}__baseline.jsonl"
    if not path.exists():
        return []
    return [row for row in (json.loads(line) for line in path.open()) if row.get("arm") == arm]


def series(rows: List[dict]) -> Dict[str, List[Optional[float]]]:
    out: Dict[str, List[Optional[float]]] = {
        key: [] for key in (
            "t", "phase", "beta", "gap", "error_normal", "error_tangential", "error_norm",
            "force_applied", "force_raw", "force_normal", "force_tangential", "penetration",
            "impulse", "reference_speed", "actual_speed", "object_speed", "object_omega",
            "contact_0", "contact_1", "object_x", "object_y", "robot0_x", "robot0_y",
            "q_x", "q_y", "p_contact_x", "p_contact_y", "p_ref_x", "p_ref_y",
        )
    }
    impulse_total = 0.0
    for row in rows:
        entry = (row.get("observed") or {}).get("0") or {}
        force = row["force"][0]
        object_state = row["true"]["object"]
        robot = row["true"]["robots"][0]
        normal = entry.get("patch_normal")
        error = entry.get("error")
        reference = entry.get("reference")
        raw = entry.get("force_raw")
        magnitude = math.hypot(*force)
        impulse_total += magnitude * 0.05
        out["t"].append(row["time"])
        out["phase"].append(row.get("phase") or row.get("stage"))
        out["beta"].append(entry.get("beta"))
        out["gap"].append(entry.get("gap_signed"))
        if normal and error:
            out["error_normal"].append(error[0] * normal[0] + error[1] * normal[1])
            out["error_tangential"].append(error[0] * (-normal[1]) + error[1] * normal[0])
        else:
            out["error_normal"].append(None)
            out["error_tangential"].append(None)
        out["error_norm"].append(math.hypot(*error) if error else None)
        out["force_applied"].append(magnitude)
        out["force_raw"].append(math.hypot(*raw) if raw else None)
        if normal:
            out["force_normal"].append(force[0] * normal[0] + force[1] * normal[1])
            out["force_tangential"].append(force[0] * (-normal[1]) + force[1] * normal[0])
        else:
            out["force_normal"].append(None)
            out["force_tangential"].append(None)
        out["penetration"].append(row.get("penetration"))
        out["impulse"].append(impulse_total)
        out["reference_speed"].append(math.hypot(*reference) if reference else None)
        out["actual_speed"].append(math.hypot(robot[3], robot[4]))
        out["object_speed"].append(math.hypot(object_state[3], object_state[4]))
        out["object_omega"].append(abs(object_state[5]))
        out["contact_0"].append(1.0 if entry.get("contact") else 0.0)
        out["contact_1"].append(
            1.0 if ((row.get("observed") or {}).get("1") or {}).get("contact") else 0.0)
        out["object_x"].append(object_state[0])
        out["object_y"].append(object_state[1])
        out["robot0_x"].append(robot[0])
        out["robot0_y"].append(robot[1])
        for prefix, key in (("q", "q"), ("p_contact", "p_contact"), ("p_ref", "p_ref")):
            value = entry.get(key)
            out[f"{prefix}_x"].append(value[0] if value else None)
            out[f"{prefix}_y"].append(value[1] if value else None)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--docs-dir", default="docs/local_position_push_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    out_dir = run_dir / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)
    roots = {row["root_id"]: row for row in json.loads(
        (run_dir / f"roots_{args.split}.json").read_text())["roots"]}

    index = []
    for interaction_class, label in CASES:
        candidates = sorted(root for root in roots
                            if root.startswith(f"{args.split}/{interaction_class}/"))
        if not candidates:
            continue
        root_id = candidates[0]
        figure, axes = plt.subplots(4, 2, figsize=(13.0, 12.0), dpi=110)
        collected = {}
        for column, arm in enumerate(("V", "X")):
            rows = _arm_rows(run_dir, args.split, root_id, arm)
            if not rows:
                continue
            data = series(rows)
            collected[arm] = data
            with (out_dir / f"{interaction_class}__{arm}.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(list(data))
                for index_row in range(len(data["t"])):
                    writer.writerow([data[key][index_row] for key in data])
            # trajectory
            axis = axes[0][column]
            axis.plot(data["object_x"], data["object_y"], color="#e8871a", lw=1.6, label="object centre")
            axis.plot(data["robot0_x"], data["robot0_y"], color="#1668c1", lw=1.2, label="robot0")
            if any(value is not None for value in data["p_contact_x"]):
                axis.plot(data["p_contact_x"], data["p_contact_y"], color="tab:red", lw=0.9,
                          ls="--", label="p_contact (X)")
                axis.plot(data["p_ref_x"], data["p_ref_y"], color="tab:green", lw=0.9,
                          ls=":", label="p_ref (X, virtual)")
                axis.plot(data["q_x"], data["q_y"], color="k", lw=0.7, ls="-.", label="q (foot)")
            axis.set_title(f"{label} — {arm}   {root_id}", fontsize=9)
            axis.set_aspect("equal")
            axis.legend(fontsize=6, loc="best")
            # gap and error
            axis = axes[1][column]
            axis.plot(data["t"], data["gap"], color="tab:red", lw=1.1, label="gap_signed [m]")
            axis.plot(data["t"], data["error_norm"], color="tab:blue", lw=1.1, label="|e| [m]")
            axis.plot(data["t"], data["error_normal"], color="tab:cyan", lw=0.8, ls="--",
                      label="e normal")
            axis.plot(data["t"], data["error_tangential"], color="tab:purple", lw=0.8, ls="--",
                      label="e tangential")
            axis.axhline(0.0, color="0.6", lw=0.6)
            axis.legend(fontsize=6)
            axis.set_ylabel("m")
            # force
            axis = axes[2][column]
            axis.plot(data["t"], data["force_applied"], color="k", lw=1.2, label="F applied [N]")
            axis.plot(data["t"], data["force_raw"], color="tab:orange", lw=0.9, ls="--",
                      label="F raw (X only)")
            axis.plot(data["t"], data["force_normal"], color="tab:red", lw=0.8, label="F normal")
            axis.plot(data["t"], data["force_tangential"], color="tab:green", lw=0.8,
                      label="F tangential")
            axis.axhline(15.0, color="0.5", lw=0.7, ls=":")
            axis.axhline(-15.0, color="0.5", lw=0.7, ls=":")
            axis.legend(fontsize=6)
            axis.set_ylabel("N")
            # phase / speeds
            axis = axes[3][column]
            axis.plot(data["t"], data["reference_speed"], color="darkgreen", lw=1.1,
                      label="|v_ref| (V; N/A for X)")
            axis.plot(data["t"], data["actual_speed"], color="tab:blue", lw=1.1, label="|v_robot|")
            axis.plot(data["t"], data["object_speed"], color="#e8871a", lw=1.1, label="|v_object| truth")
            axis.plot(data["t"], data["object_omega"], color="0.4", lw=0.8, label="|w_object| truth")
            axis.plot(data["t"], data["beta"], color="tab:brown", lw=0.9, ls=":", label="beta (X)")
            axis.plot(data["t"], data["contact_0"], color="tab:red", lw=0.6, alpha=0.5, label="contact r0")
            axis.legend(fontsize=6)
            axis.set_xlabel("simulation time [s]")
        figure.tight_layout()
        figure.savefig(out_dir / f"{interaction_class}__diagnostic.png")
        plt.close(figure)
        index.append({"case": interaction_class, "label": label, "root_id": root_id,
                      "csv": {arm: f"{interaction_class}__{arm}.csv" for arm in ("V", "X")},
                      "figure": f"{interaction_class}__diagnostic.png",
                      "push_periods": {arm: sum(1 for value in collected[arm]["phase"]
                                                if value == "push") for arm in collected}})
        print(f"{interaction_class}: {root_id}", flush=True)
    (out_dir / "index.json").write_text(json.dumps({"cases": index}, indent=1, sort_keys=True))
    docs = ROOT / args.docs_dir
    docs.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"cases": len(index)}, indent=1))
    print("DIAGNOSTICS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
