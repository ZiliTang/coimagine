"""Differential CAP execution and paired measurement on frozen online-training fixtures."""

import argparse
import gzip
import hashlib
import math
import multiprocessing as mp
import os
import pickle
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tools import cap_regional_cpp as cpp
from tools import cap_regional_id_ppo as ppo

study = ppo.study
AUDIT = ppo.OUT / "cpp_audit"


def compare_memory(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict) and a.keys() == b.keys():
        for key in a:
            compare_memory(a[key], b[key], path + "/" + str(key))
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b):
        for i in range(len(a)):
            compare_memory(a[i], b[i], path + "/" + str(i))
    elif (
        isinstance(a, float)
        and isinstance(b, (int, float))
        and math.isclose(a, b, rel_tol=1e-12, abs_tol=2e-12)
    ):
        return
    elif a != b:
        raise AssertionError(f"{path}: {a!r} != {b!r}")


def paired_case(job):
    cid, fixture, index = job
    env = ppo.environment(fixture)
    reference, port = study.dq.Proposals(), cpp.Proposals()
    try:
        for step in range(300):
            old = reference.panel(env.history)
            event = study.dq.current_event(env.history)
            before = pickle.dumps(event, protocol=5)
            new = port.panel(env.history)
            assert pickle.dumps(event, protocol=5) == before, "observation mutated"
            assert new[-1] == old[-1], f"raw CAPs differ {new[-1]} vs {old[-1]}"
            assert new[-2] == old[-2], "action pool differs"
            for left, right in zip(old[:-2], new[:-2]):
                np.testing.assert_array_equal(left, right)
            compare_memory(reference.memory, port.memory)
            raw = old[-1]
            pool = old[-2]
            ix = index % 10 if index < 20 else (index + step) % 10
            action = raw[ix] if raw[ix] in pool else pool[0]
            env.step(action, remaining_ticks=1800 - env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods >= 1800:
                break
        return dict(
            case_id=cid,
            steps=step + 1,
            success=int(env.env.success),
            ticks=env.env.total_periods,
            history=env.history.sha256,
        )
    except Exception as e:
        with gzip.open(AUDIT / (cid + "_mismatch.pkl.gz"), "wb") as f:
            pickle.dump(dict(step=step, reference=reference.memory, native=port.memory, event=event), f)
        raise RuntimeError(f"{cid} at boundary {step}: {e}") from e
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE = None


def fingerprint(r):
    out = {k: v for k, v in r.items() if k != "shard"}
    with np.load(r["shard"]) as d:
        out["arrays"] = {k: hashlib.sha256(d[k].tobytes()).hexdigest() for k in d.files}
    return out


def run(directory=None):
    global AUDIT
    # Preserve the original timing and backend choice when repeating a completed audit.
    initial_audit = AUDIT
    if directory is not None:
        AUDIT = Path(directory)
        AUDIT.mkdir(parents=True, exist_ok=False)
    elif (ppo.OUT / "cpp_backend.json").exists():
        raise FileExistsError("Use --out with a new directory to repeat this completed audit")
    AUDIT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    frozen = study.read(study.OUT / "frozen.json")
    study.init_models(frozen["sids"])
    native, manifest = cpp.load()
    ctx = np.array([17.0], np.float64)
    candidates = np.zeros((9, 4), np.float64)
    candidates[:, 3] = np.arange(9) * 3
    for mod in study.dq.MODULES:
        a = np.ones((9, 2))
        b = a.copy()
        mod.evaluate_candidates(ctx, candidates, a)
        native.evaluate_candidates(ctx, candidates, b)
        np.testing.assert_array_equal(a, b)
    pool = {c["case_id"]: c["fixture"] for c in study.read(ppo.POOL)["cases"]}
    rng = np.random.default_rng(970634)
    cases = []
    for size in sorted({len(f["movable_objects"]) for f in pool.values()}):
        ids = [k for k in sorted(pool) if len(pool[k]["movable_objects"]) == size]
        cases.extend(rng.choice(ids, 8, replace=False).tolist())
    rng.shuffle(cases)
    with ProcessPoolExecutor(max_workers=16, mp_context=mp.get_context("fork")) as workers:
        evidence = list(workers.map(paired_case, [(cid, pool[cid], i) for i, cid in enumerate(cases)]))
    study.save(AUDIT / "parity.json", evidence)
    print("Matched boundaries", sum(x["steps"] for x in evidence), flush=True)
    ppo.POLICY = ppo.Policy()
    checkpoint = initial_audit / "r05.pt"
    if not checkpoint.exists():
        os.link(ppo.OUT / "latest.pt", checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert len(state["history"]) == 5
    ppo.POLICY.load_state_dict(state["policy"])
    ppo.POLICY.eval()
    ppo.TEMP = study.read(ppo.OUT / "protocol.json")["temperature_calibration"]["temperature"]
    measurements = []
    for block, order in enumerate([("cython", "cpp"), ("cpp", "cython")]):
        items = [(cid, int(rng.integers(2**31))) for cid in cases]
        ref = None
        for name in order:
            ppo.PROPOSALS = cpp.Proposals if name == "cpp" else study.dq.Proposals
            dest = AUDIT / f"bench_{block}_{name}"
            start = time.perf_counter()
            rows = ppo.collect(pool, items, dest, True)
            elapsed = time.perf_counter() - start
            hashes = [fingerprint(r) for r in rows]
            if ref is None:
                ref = hashes
            elif ref != hashes:
                study.save(AUDIT / "bench_mismatch.json", dict(first=ref, second=hashes))
                raise AssertionError("paired PPO samples or training tensors differ")
            measurements.append(
                dict(
                    block=block,
                    backend=name,
                    seconds=elapsed,
                    n=len(items),
                    steps=sum(r["boundaries"] for r in rows),
                )
            )
            shutil.rmtree(dest)
            print(measurements[-1], flush=True)
        study.save(AUDIT / f"benchmark_pair_{block}.json", ref)
    totals = {k: sum(m["seconds"] for m in measurements if m["backend"] == k) for k in ("cython", "cpp")}
    speedup = totals["cython"] / totals["cpp"]
    result = dict(
        key=manifest["key"],
        binary=manifest["binary"],
        sources=manifest["spec"]["original"],
        use_cpp=speedup > 1,
        completed_before_switch=5,
        workers=16,
        measurements=measurements,
        speedup=speedup,
        paired_episodes=len(evidence),
        paired_boundaries=sum(x["steps"] for x in evidence),
        ppo_paired_episodes=len(cases) * 2,
        cases=cases,
        checkpoint_sha256=study.sha(checkpoint),
    )
    study.save((ppo.OUT if directory is None else AUDIT) / "cpp_backend.json", result)
    print("speedup", speedup, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path)
    run(parser.parse_args().out)
