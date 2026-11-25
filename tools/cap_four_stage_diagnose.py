"""Public evidence replay and per-body withholding diagnostics."""
import argparse
from pathlib import Path
import time

from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend
from tools.cap_symbolic_general import read, save, sha


def diagnose(source, output):
    if output.exists():
        raise FileExistsError(output)
    trace = read(source)
    frontend = PublicFrontend()
    rows = []
    started = time.monotonic()
    selected = {0, min(30, len(trace["rows"])-1), min(50, len(trace["rows"])-1), len(trace["rows"])-1}
    for i, row in enumerate(trace["rows"]):
        prepared = frontend.prepare(row["public"])
        if i in selected:
            action = "push_r0_1_4"
            regions = prepared["regions"][action]
            safe = frontend.evidence.free.buffer(-frontend.evidence.numerical_margin)
            missing = {}
            for body, region in (("robot0", regions["robots"][0]), ("robot1", regions["robots"][1]),
                                 ("target", regions["target"])):
                part = region.difference(safe)
                missing[body] = dict(area=part.area,
                    centroid=None if part.is_empty else [part.centroid.x, part.centroid.y],
                    bounds=None if part.is_empty else list(part.bounds))
            result = dict(boundary=i, tick=row["tick"], action=action,
                          report=prepared["reports"][action], missing=missing)
            rows.append(result)
            save(output.with_suffix(".partial.json"), dict(status="running", rows=rows))
            print(i, row["selected"], {k: round(v["area"], 6) for k, v in missing.items()}, flush=True)
        frontend.record_command(row["selected"])
    packet = dict(status="complete", scope="public evidence replay; one development trajectory",
                  source=str(source), source_sha256=sha(source), rows=rows,
                  seconds=time.monotonic()-started)
    save(output, packet)
    return packet


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    diagnose(args.trace, args.out)
