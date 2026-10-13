"""C4 render acceptance: check the produced videos against the logs they came from.

    python experiments/local_contact_option/verify_animations.py

Checks, for every expected file: it exists, it has a non-zero frame count, its duration matches
the logged execution plus the single display-only hold second, every case has the same initial
state in all four arms, and the animation index agrees with the execution records.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[2]
ARMS = ("O", "P", "F", "N")
FPS = 20
HOLD = 1.0


def probe(path: Path) -> dict:
    process = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-i", str(path)],
        capture_output=True, text=True,
    )
    text = process.stderr
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", text)
    frames = re.search(r"frame=\s*(\d+)", text)
    streams = re.findall(r"Video: ([^,]+)", text)
    seconds = None
    if duration:
        hours, minutes, rest = duration.groups()
        seconds = int(hours) * 3600 + int(minutes) * 60 + float(rest)
    return {"seconds": seconds, "codec": streams[0] if streams else None,
            "frame_line": bool(frames)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-repair-v2")
    parser.add_argument("--out-root", default="outputs/local_contact_option_repair_v2")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    index = json.loads((run_dir / "animation_index.json").read_text())
    records = [json.loads(line) for line in (run_dir / "executions_test.jsonl").open()]
    by_key = {(row["root_id"], row["variant"], row["arm"]): row for row in records}
    figures = run_dir / "figures"

    problems = []
    checked = 0
    for case in index["cases"]:
        expected = {}
        for arm in ARMS:
            row = by_key.get((case["root_id"], case["variant"], arm))
            if row is None:
                problems.append(f"{case['case']}: no record for arm {arm}")
                continue
            total = row["metrics"]["wall_clock_seconds"] + HOLD
            for folder, name in (("singles", f"{case['case']}__{arm}.mp4"),):
                path = figures / folder / name
                if not path.exists():
                    problems.append(f"missing {path}")
                    continue
                info = probe(path)
                checked += 1
                if info["seconds"] is None or info["seconds"] <= 0:
                    problems.append(f"{path}: zero duration")
                elif abs(info["seconds"] - total) > 0.35:
                    problems.append(f"{path}: duration {info['seconds']:.2f}s vs log "
                                    f"{row['metrics']['wall_clock_seconds']:.2f}s + {HOLD}s hold")
                expected[arm] = row["initial_object"]
        quad = figures / "quads" / f"{case['case']}__quad.mp4"
        gif = figures / "gif" / f"{case['case']}__quad.gif"
        for path in (quad, gif):
            if not path.exists():
                problems.append(f"missing {path}")
            else:
                checked += 1
        if len(set(tuple(round(value, 9) for value in item) for item in expected.values())) != 1:
            problems.append(f"{case['case']}: arms do not share one initial state")

    # index consistency and terminal state
    for row in index["rows"]:
        record = by_key.get((row["root_id"], row["variant"], row["arm"]))
        if record is None:
            problems.append(f"index row without record: {row['case']} {row['arm']}")
            continue
        if abs(record["metrics"]["displacement"] - row["displacement_m"]) > 1e-12:
            problems.append(f"{row['case']} {row['arm']}: displacement mismatch")
        terminal = record.get("terminal_object")
        if terminal is None:
            problems.append(f"{row['case']} {row['arm']}: no terminal snapshot")
        elif not all(math.isfinite(value) for value in terminal):
            problems.append(f"{row['case']} {row['arm']}: non-finite terminal state")

    report = {
        "cases": len(index["cases"]),
        "single_videos": index["single_videos"], "quad_videos": index["quad_videos"],
        "preview_gifs": index["preview_gifs"], "checked_files": checked,
        "fps": index["fps"], "real_time": index["real_time"],
        "problems": problems, "ok": not problems,
    }
    (run_dir / "animation_verification.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    print("VERIFY_OK" if not problems else "VERIFY_FAILED")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
