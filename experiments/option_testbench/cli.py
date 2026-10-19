"""Unified CLI for the option testbench: prepare / run / render / summarize."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for path in (HERE, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import panel as P  # noqa: E402


def run_bounded(command, timeout, cwd):
    """External process-group timeout with a 2 s cleanup grace; never uses pkill."""
    process = subprocess.Popen(command, cwd=str(cwd), start_new_session=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = process.communicate(timeout=timeout)
        return process.returncode, out, err, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            out, err = process.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            out, err = process.communicate()
        return None, out, err, True


def placeholder_log(out_dir, request, status):
    path = out_dir / "logs" / f"{request['branch_id']}.json"
    if path.exists():
        return
    P.write_json(path, {"branch_id": request["branch_id"], "snapshot": request["snapshot"],
                        "actions": request["actions"], "status": status,
                        "env_version": P.spec.ENV_VERSION, "policy": "baseline-option-v1",
                        "contract_version": P.CONTRACT_VERSION,
                        "metric_version": P.METRIC_VERSION, "rows": [], "boundaries": [],
                        "eligibility": [], "skill_boundaries": [],
                        "note": "driver placeholder; no branch log was produced"})


def command_run(args):
    out_dir = Path(args.out)
    requests = json.loads((out_dir / "requests.json").read_text())["requests"]
    indices = args.request_indices or list(range(len(requests)))
    batch_deadline = time.monotonic() + args.total_wall_seconds
    index_rows = []
    for position, request_index in enumerate(indices):
        request = requests[request_index]
        if time.monotonic() >= batch_deadline:
            for pending in indices[position:]:
                placeholder_log(out_dir, requests[pending], "batch_budget_exhausted")
                index_rows.append({"request_index": pending,
                                   "branch_id": requests[pending]["branch_id"],
                                   "status": "batch_budget_exhausted"})
            break
        remaining = max(1.0, min(args.branch_wall_seconds, batch_deadline - time.monotonic()))
        command = [sys.executable, str(HERE / "runner.py"), "single",
                   "--panel-dir", str(out_dir), "--request-index", str(request_index),
                   "--out", str(out_dir), "--branch-wall-seconds",
                   str(args.branch_wall_seconds),
                   "--recovery-horizon-seconds", str(args.recovery_horizon_seconds)]
        started = time.monotonic()
        code, stdout, stderr, timed_out = run_bounded(command, remaining, ROOT)
        if timed_out:
            status = "wall_timeout"
        else:
            log_path = out_dir / "logs" / f"{request['branch_id']}.json"
            status = json.loads(log_path.read_text())["status"] if log_path.exists() else "failed"
        if not (out_dir / "logs" / f"{request['branch_id']}.json").exists():
            placeholder_log(out_dir, request, status)
        index_rows.append({"request_index": request_index, "branch_id": request["branch_id"],
                           "snapshot": request["snapshot"], "status": status,
                           "wall_seconds": time.monotonic() - started,
                           "exit_code": code, "timed_out": timed_out,
                           "stderr_tail": (stderr or "")[-400:]})
        P.write_json(out_dir / "logs" / "index.json",
                     {"batch_wall_seconds": args.total_wall_seconds,
                      "branch_wall_seconds": args.branch_wall_seconds,
                      "branches": index_rows})
    P.write_json(out_dir / "logs" / "index.json",
                 {"batch_wall_seconds": args.total_wall_seconds,
                  "branch_wall_seconds": args.branch_wall_seconds, "branches": index_rows})
    return index_rows


def command_prepare(args):
    result = P.prepare(args.out)
    print(json.dumps({"snapshots": len(result["panel"]) + 1,
                      "static_fixtures": list(result["panel"]),
                      "requests": len(result["requests"]),
                      "legal": all(e["legal"] for e in result["report"].values())}, indent=1))


def command_render(args):
    import renderer
    renderer.render(Path(args.out))
    print(json.dumps({"index.html": str(Path(args.out) / "index.html")}))


def command_summarize(args):
    import metrics
    summary = metrics.summarize(Path(args.out))
    print(json.dumps({key: summary[key] for key in
                      ("requests", "accepted", "request_coverage", "state_coverage",
                       "fulfilled", "violated", "unresolved", "conservative_interval",
                       "effective_fulfillment_coverage")}, indent=1, default=str))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run", "render", "summarize"):
        node = sub.add_parser(name)
        node.add_argument("--out", type=Path,
                          default=ROOT / "outputs/option_testbench_v1/round_000")
        if name == "run":
            node.add_argument("--branch-wall-seconds", type=float, default=10.0)
            node.add_argument("--recovery-horizon-seconds", type=float, default=12.0)
            node.add_argument("--total-wall-seconds", type=float, default=55.0)
            node.add_argument("--request-indices", type=int, nargs="*", default=None)
    args = parser.parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "run":
        rows = command_run(args)
        print(json.dumps({"ran": len(rows),
                          "statuses": {row["status"]: sum(1 for r in rows
                                                          if r["status"] == row["status"])
                                       for row in rows}}, indent=1))
    elif args.command == "render":
        command_render(args)
    else:
        command_summarize(args)


if __name__ == "__main__":
    main()
