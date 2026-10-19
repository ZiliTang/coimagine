"""Paired old-checkpoint inference after the versioned PUSH clock repair; no training."""
import concurrent.futures
import hashlib
import multiprocessing as mp
import time

import torch

import diagnose_v4 as diagnosis


def main():
    torch.set_num_threads(1)
    out = diagnosis.OUT / "skill_clock_fix"
    out.mkdir(exist_ok=False)
    started = time.time()
    report = dict(environment=diagnosis.rl.E.EnvConfig().as_dict(),
                  weights_trained_under="random-start-time-budget-v3", training_updates=0,
                  scope="same 3 checkpoints and 8 original train instances, deterministic",
                  source_sha256={str(p.relative_to(diagnosis.rl.ROOT)):
                      hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (diagnosis.rl.ROOT / "src/coimagine/random_start").glob("*.py")}, results={})
    with concurrent.futures.ProcessPoolExecutor(max_workers=8,
            mp_context=mp.get_context("spawn"), initializer=diagnosis.runner.worker_init) as executor:
        for name, checkpoint in diagnosis.CHECKPOINTS.items():
            model = diagnosis.load_policy(diagnosis.SOURCE / "checkpoints" / checkpoint)
            episodes = diagnosis.collect(executor, model, diagnosis.tracking(), log_dir=out/"logs"/name)
            report["results"][name] = diagnosis.summary(episodes)
            report["wall_seconds"] = time.time()-started
            diagnosis.save(out / "comparison.json", report)
            print(name, report["results"][name]["success"], flush=True)


if __name__ == "__main__":
    main()
