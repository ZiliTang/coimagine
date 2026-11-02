"""Run a bounded rotation_feedback_gain comparison through the D006 testbed.

Every trial is a normal testbed run: it is saved under outputs/testbed/runs/
and therefore appears in the Web run list.  No training, no optimizer step.

Usage:
    .venv/bin/python experiments/option_rl_v19/diag_rotation_feedback.py \
        --preset "相邻侧双推 5s 右上" --gains 0 0.5 1.0
"""
from __future__ import annotations
import argparse
import copy
import json
import math
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/testbed/diagnostics"


def _get(server, path):
    with urllib.request.urlopen(server + path) as r:
        return json.load(r)


def _post(server, path, body):
    req = urllib.request.Request(
        server + path, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def _metrics(run, gain):
    frames, strip = run["frames"], run["strip"]
    o0, o1 = frames[0]["object"], frames[-1]["object"]
    disp = math.hypot(o1[0] - o0[0], o1[1] - o0[1])
    angles = [f["object"][2] for f in frames]
    r1_rel = [f["robots"][1][0] - f["object"][0] for f in frames]
    t1_r1 = []
    t1_obj = []
    for f in frames:
        t = f.get("target")
        if t and t[1]:
            t1_r1.append(t[1][0] - f["robots"][1][0])
            t1_obj.append(t[1][0] - f["object"][0])
    both = sum(1 for s in strip if all(s.get("contact") or [False, False]))
    push = sum(1 for s in strip if s.get("phase") == "PUSH")
    return dict(
        gain=float(gain), run_id=run["id"], name=run.get("name"),
        success=bool(run.get("success")), periods=int(run.get("periods", 0)),
        object_displacement=round(disp, 4),
        angle_end=round(o1[2], 4),
        angle_min=round(min(angles), 4), angle_max=round(max(angles), 4),
        both_contact_ticks=both, push_ticks=push,
        r1_minus_object_x_start=round(r1_rel[0], 4),
        r1_minus_object_x_end=round(r1_rel[-1], 4),
        r1_minus_object_x_min=round(min(r1_rel), 4),
        r1_minus_object_x_max=round(max(r1_rel), 4),
        target_minus_r1_x_max=round(max(t1_r1), 4) if t1_r1 else None,
        target_minus_object_x_max=round(max(t1_obj), 4) if t1_obj else None,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8765")
    ap.add_argument("--preset", default="相邻侧双推 5s 右上")
    ap.add_argument("--gains", nargs="+", type=float, default=[0.0, 0.5, 1.0])
    args = ap.parse_args()

    presets = _get(args.server, "/api/presets")
    item = next((p for p in presets if p.get("name") == args.preset), None)
    if item is None:
        raise SystemExit(f"preset not found: {args.preset}")
    preset = _get(args.server, f"/api/presets/{item['id']}")
    goal = preset.get("scene_meta", {}).get("goal")
    rows = []
    for gain in args.gains:
        overrides = copy.deepcopy(preset.get("overrides") or {})
        overrides.setdefault("PUSH", {})["rotation_feedback_gain"] = float(gain)
        body = dict(
            scene=preset["scene"], object=preset["object"], robots=preset["robots"],
            sequence=preset["sequence"], overrides=overrides, goal=goal,
            name=f"DIAG rotfb {gain:g}")
        run = _post(args.server, "/api/run", body)
        row = _metrics(run, gain)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = OUT / f"rotation_feedback_{stamp}.json"
    path.write_text(json.dumps(dict(
        preset=args.preset, preset_id=item["id"], created=time.time(),
        gains=args.gains, runs=rows), indent=1))
    print(f"\nsummary: {path.relative_to(ROOT)}")
    print("run IDs are also visible in the testbed Runs list.")


if __name__ == "__main__":
    main()
