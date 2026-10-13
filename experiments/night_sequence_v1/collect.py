"""Night-round data collection: root sets, then real option sequences.

Root layout (task book section 3.3). The new evaluation roots are fixed **before** the train
extension, and every pair of splits is separated by the original near-duplicate standard
(``MIN_ROOT_DISTANCE = 0.02 m``, angle scaled by a 0.5 m lever arm):

* ``d0``   -- the **original** 512 phase-B train roots, reused unchanged by id;
* ``test`` -- 512 new roots under a night salt;
* ``val``  -- 256 new roots under the same night salt, checked against ``d0`` and ``test``;
* ``d1ext``-- 3584 further roots, so ``D1 = d0 + d1ext = 4096`` train roots.

Sequence collection replays ``d0`` exactly (same root, command, snapshot and protocol) and runs the
new roots fresh. Either way one execution is one sequence carrying O0 and the real exit instant T.

    python experiments/night_sequence_v1/collect.py --stage roots
    python experiments/night_sequence_v1/collect.py --stage d0     --workers 12
    python experiments/night_sequence_v1/collect.py --stage eval   --workers 12
    python experiments/night_sequence_v1/collect.py --stage d1     --workers 12
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.sequences import ShardWriter, run_sequence  # noqa: E402
from experiments.option_prediction_mechanism_v1.phase_b import roots_b  # noqa: E402

DATA = ROOT / "outputs/night_sequence_v1"
SALT_NIGHT = "night-sequence-v1/v1"

SPLIT_ROOTS = {"d0": 512, "val": 256, "test": 512, "d1ext": 3584, "d2ext": 12288}
_WORKER: Dict[str, object] = {}


def sample_split(split: str, total: int, *, salt: str, forbidden: Sequence,
                 id_prefix: str = "night") -> Tuple[List, dict]:
    """Deterministic roots for one split under a caller-chosen salt and id prefix."""

    saved = roots_b.SALT_B
    roots_b.SALT_B = salt
    try:
        accepted, report = roots_b.sample_split_roots(
            split, total, forbidden=list(forbidden),
            min_distance=roots_b.MIN_ROOT_DISTANCE)
    finally:
        roots_b.SALT_B = saved
    stamped = []
    for index, root in enumerate(accepted):
        interaction = root.interaction_class
        stamped.append(replace(root, root_id=f"{id_prefix}/{split}/{interaction}/{index:05d}"))
    return stamped, {
        "split": split, "salt": salt, "planned": report.planned, "sampled": report.sampled,
        "per_class": report.per_class, "sample_none": report.sample_none,
        "leakage_rejected": report.leakage_rejected, "unfilled": report.unfilled,
    }


def build_roots() -> dict:
    """Fix evaluation roots first, then extend train. Order matters and is recorded."""

    d0_roots, _reports = roots_b.build_phase_b_roots({"train": 512, "val": 0, "test": 0})
    d0 = d0_roots["train"]
    forbidden: List = list(d0)

    test, test_report = sample_split("test", SPLIT_ROOTS["test"], salt=f"{SALT_NIGHT}/test",
                                     forbidden=forbidden)
    forbidden.extend(test)
    val, val_report = sample_split("val", SPLIT_ROOTS["val"], salt=f"{SALT_NIGHT}/val",
                                   forbidden=forbidden)
    forbidden.extend(val)
    d1ext, ext_report = sample_split("d1ext", SPLIT_ROOTS["d1ext"], salt=f"{SALT_NIGHT}/d1",
                                     forbidden=forbidden)
    forbidden.extend(d1ext)
    d2ext, d2_report = sample_split("d2ext", SPLIT_ROOTS["d2ext"], salt=f"{SALT_NIGHT}/d2",
                                    forbidden=forbidden)

    combined = {"d0": d0, "val": val, "test": test, "d1ext": d1ext, "d2ext": d2ext}
    # D1 = d0 + d1ext by construction, so the guard is applied to the four *disjoint* sets
    leakage = roots_b.leakage_summary({"d0": d0, "val": val, "test": test, "d1ext": d1ext,
                                       "d2ext": d2ext})
    payload = {
        "salt_night": SALT_NIGHT,
        "order": ["d0 (original phase-B train roots, reused by id)",
                  "test (512 new)", "val (256 new)", "d1ext (3584 new)"],
        "split_roots": SPLIT_ROOTS,
        "reports": {"d0": {"split": "d0", "planned": 512, "sampled": len(d0),
                           "note": "original phase-B train roots, ids unchanged",
                           "per_class": _counts(d0)},
                    "test": test_report, "val": val_report, "d1ext": ext_report,
                    "d2ext": d2_report},
        "leakage": leakage,
        "roots": {name: roots_b.root_records(values) for name, values in combined.items()},
        "commands": {name: roots_b.commands_records(values) for name, values in combined.items()},
    }
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / "roots_night.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    # the exact RootSpec objects, so a worker never re-samples and can never drift from the
    # accepted set (rejection sampling depends on the forbidden list, so re-running it with an
    # empty list would silently produce different roots)
    import pickle
    (DATA / "roots_night.pkl").write_bytes(pickle.dumps(combined))
    print(json.dumps({"counts": {name: len(values) for name, values in combined.items()},
                      "min_cross_split_distance": [
                          {"pair": f"{p['split_a']}-{p['split_b']}", "min": p["min_distance"]}
                          for p in leakage["pairs"]],
                      "reports": {name: {"sampled": r.get("sampled"),
                                         "unfilled": r.get("unfilled"),
                                         "leakage_rejected": r.get("leakage_rejected")}
                                  for name, r in payload["reports"].items()}}, indent=1))
    return payload


def _counts(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for root in values:
        out[root.interaction_class] = out.get(root.interaction_class, 0) + 1
    return out


def _prepare_worker(_unused=None) -> None:
    """Load the accepted roots and their command tables once per worker process."""

    import pickle
    from experiments.option_prediction_mechanism_v1.phase_b import roots_b as rb

    roots = pickle.loads((DATA / "roots_night.pkl").read_bytes())
    _WORKER["roots"] = roots
    table: Dict[str, List[dict]] = {}
    commands: Dict[str, List[dict]] = {}
    for name, values in roots.items():
        for root in values:
            table[f"{name}:{root.root_id}"] = [command.as_dict()
                                               for command in rb.commands_for_root(root)]
    _WORKER["commands"] = table


def _work_task(task: Tuple[str, int]) -> List[dict]:
    """One root: run its eight commands and return the finished sequences."""

    split, index = task
    roots = _WORKER["roots"]
    command_table = _WORKER["commands"]
    root = roots[split][index]
    commands = command_table[f"{split}:{root.root_id}"]
    out: List[dict] = []
    for command in commands:
        payload = dict(command)
        payload["index"] = int(payload["command_index"])
        record = run_sequence(root, payload)
        if record is None:
            continue
        record["split"] = split
        record["command"] = payload
        out.append(record)
    return out


def collect(stage: str, workers: int) -> dict:
    import pickle
    payload = pickle.loads((DATA / "roots_night.pkl").read_bytes())
    if stage == "d0":
        splits = [("d0", 0, len(payload["d0"]))]
    elif stage == "eval":
        splits = [("test", 0, len(payload["test"])),
                  ("val", 0, len(payload["val"]))]
    elif stage == "d1":
        splits = [("d1ext", 0, len(payload["d1ext"]))]
    else:
        splits = [("d2ext", 0, len(payload["d2ext"]))]
    import multiprocessing as mp

    started = time.time()
    total_sequences = 0
    shard_rows: List[dict] = []
    for split, low, high in splits:
        writer = ShardWriter(DATA / "sequences", split=split, shard_size=32)
        tasks = [(split, index) for index in range(low, high)]
        with mp.get_context("fork").Pool(processes=workers,
                                         initializer=_prepare_worker) as pool:
            for done, records in enumerate(pool.imap(_work_task, tasks, chunksize=1), start=1):
                for record in records:
                    writer.add(record)
                if done % 128 == 0:
                    print(f"  {split}: {done}/{len(tasks)} roots "
                          f"({time.time() - started:.0f}s)", flush=True)
        writer.flush()
        total_sequences += sum(row["sequences"] for row in writer.shards)
        for row in writer.shards:
            row["split"] = split
        shard_rows.extend(writer.shards)
        print(f"{split}: {sum(r['sequences'] for r in writer.shards)} sequences in "
              f"{len(writer.shards)} shards", flush=True)
    summary = {"stage": stage, "seconds": time.time() - started, "workers": workers,
               "sequences": total_sequences, "shards": shard_rows}
    path = DATA / f"collection_{stage}.json"
    path.write_text(json.dumps(summary, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in summary.items() if k != "shards"}, indent=1))
    print("COLLECT_OK")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=["roots", "d0", "eval", "d1", "d2"])
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.stage == "roots":
        build_roots()
        return 0
    collect(args.stage, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
