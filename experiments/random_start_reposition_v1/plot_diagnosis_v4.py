"""Render saved replay logs only; no simulation or new policy selection."""
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import rl
from coimagine.random_start import render


def main():
    out = rl.ROOT / "outputs/random_start_rl_diagnosis_v4"
    report = json.loads((out / "replay.json").read_text())
    names = [r["instance"] for r in report["initial"]["rows"]]
    models = ["initial", "degraded_100", "late_4600"]
    figure, axes = plt.subplots(8, 3, figsize=(14, 20), sharex=True, sharey=True)
    observations = []
    for j, model in enumerate(models):
        for i, name in enumerate(names):
            with gzip.open(out / "logs" / model / f"{name}_0.json.gz", "rt") as f:
                log = json.load(f)
            rows, boundaries = log["rows"], log["boundaries"]
            axis = axes[i, j]
            axis.plot([r["time"] for r in rows], [r["true_goal_distance"] for r in rows])
            axis.axhline(.2, color="green", linestyle="--", linewidth=.7)
            short = [b for b in boundaries if b["periods"] <= 2]
            axis.scatter([b["start_time"] for b in short], [0.04]*len(short), s=2, color="red")
            result = report[model]["rows"][i]
            axis.set_title(f"{model} / {name[-4:]} / {'success' if result['success'] else 'failure'}")
            axis.set_ylim(0, 2.5)
            if j == 0:
                axis.set_ylabel("Goal distance (m)")
            if i == 7:
                axis.set_xlabel("Actual time (s); red = <=0.1s joint skill")
            immediate_not_observed = sum(b["periods"] == 1 and any(
                end["exit_reason"] == "not_observed" for end in b["skill_ends"].values())
                for b in boundaries)
            observations.append(dict(model=model, instance=name, success=result["success"],
                decisions=len(boundaries), short=len(short),
                short_duration_seconds=sum(b["periods"]*.05 for b in short),
                one_tick_with_not_observed=immediate_not_observed,
                min_distance=min(r["true_goal_distance"] for r in rows),
                max_hold_seconds=max(r["hold_seconds"] for r in rows)))
    figure.tight_layout()
    figure.savefig(out / "replay_distances.png", dpi=120)
    plt.close(figure)
    (out / "trajectory_findings.json").write_text(json.dumps(observations, indent=2))
    # Predefined presentation rule: fewest-skill initial success, then compare same instance.
    chosen = min((r for r in report["initial"]["rows"] if r["success"]),
                 key=lambda r: (r["skills_used"], r["instance"]))["instance"]
    contact_sheet = []
    for model in models:
        with gzip.open(out / "logs" / model / f"{chosen}_0.json.gz", "rt") as f:
            log = json.load(f)
        rows = log["rows"]
        meta = dict(log["instance"], title=model + " | diagnostic truth, not policy input")
        picks = sorted(set([0, len(rows)//3, 2*len(rows)//3, len(rows)-1]))
        tiles = []
        for index in picks:
            frame = Image.fromarray(render.render_frame(rows, index, meta))
            path = out / f"key_{model}_{chosen}_{index:04d}.png"
            frame.save(path)
            tiles.append(frame.resize((640, 274)))
        contact_sheet.append(tiles)
    sheet = Image.new("RGB", (4*640, 3*274), "white")
    for j, row in enumerate(contact_sheet):
        for i, tile in enumerate(row):
            sheet.paste(tile, (i*640, j*274))
    sheet.save(out / "keyframes_comparison.png")
    (out / "keyframes_selection.json").write_text(json.dumps(dict(instance=chosen,
        rule="initial successful trajectory with fewest skills; same instance in other checkpoints",
        frame_rule="first post-control frame, one-third, two-thirds, final"), indent=2))


if __name__ == "__main__":
    main()
